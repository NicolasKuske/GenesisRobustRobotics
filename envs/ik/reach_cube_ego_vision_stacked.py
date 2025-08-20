# envs/ik/reach_cube_ego_video_stacked.py

import numpy as np
import genesis as gs
import torch
import math
import matplotlib.pyplot as plt
from collections import deque
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat

class ReachCubeEgoVisionStackedEnv:
    def __init__(self, vis, device, num_envs=1, randomize_every=100,
                 noise_config: dict = None,
                 # --- reward/done parity (already added in prior patch) ---
                 success_thresh: float = 0.3001,
                 success_bonus: float = 20.0,
                 report_success_as_done: bool = True,
                 # --- NEW: curriculum controls ---
                 cube_positions=None,            # list of (x,y,z); if None uses 3x3 grid like audio
                 inference_mode: bool = False,   # cycle positions deterministically when True
                 ):
        self.device = device
        self.num_envs = num_envs
        self.randomize_every = randomize_every
        self.episode_count = 0
        self.step_count = 0

        # --- reward/done config (parity with audio) ---
        self.success_thresh = success_thresh
        self.success_bonus = success_bonus
        self.report_success_as_done = report_success_as_done

        # --- Curriculum-by-position setup ---
        if cube_positions is None:
            # Same 3x3 grid as audio env (tweak if off-camera in your setup)
            self.cube_positions = [
                np.array([0.6, -0.3, 0.6]),
                np.array([0.6, -0.3, 0.4]),
                np.array([0.6, -0.3, 0.2]),
                np.array([0.6,  0.0, 0.6]),
                np.array([0.6,  0.0, 0.4]),
                np.array([0.6,  0.0, 0.2]),
                np.array([0.6,  0.3, 0.6]),
                np.array([0.6,  0.3, 0.4]),
                np.array([0.6,  0.3, 0.2]),
            ]
        else:
            # ensure np arrays of shape (3,)
            self.cube_positions = [np.array(p, dtype=float).reshape(3,) for p in cube_positions]

        self.pos_count = len(self.cube_positions)
        self.pos_probs = np.ones(self.pos_count, dtype=np.float32) / self.pos_count  # start uniform
        self.env_pos_idx = np.zeros(self.num_envs, dtype=np.int64)  # filled each reset()
        self.current_cube_pos = None

        # Episode accounting for curriculum (already used by reward patch)
        self._episode_return_per_env = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._done_mask_episode      = torch.zeros(self.num_envs, dtype=torch.bool,   device=self.device)

        # Control flags
        self._episode_active = False
        self._episode_return = 0.0
        self.inference_mode  = inference_mode
        self._infer_cycle_idx = 0  # used only in inference_mode


        # history settings
        self.history_length = 25
        self.sample_offsets = [-21, -16, -11, -6, -1]
        self.image_history = deque(maxlen=self.history_length)
        self.render_every = 5
        self._step_count = 0

        # Store noise configuration
        self.noise_config = noise_config if noise_config else {"visual_noise_level": 0.0}

        # cube position
        self.initial_pos = np.array([-0.9, 0.6, 0.7])[None, :]
        self.current_cube_pos = None

        self.obs_shape = (3 * len(self.sample_offsets), 120, 120)
        self.action_space = 6

        # Genesis scene setup
        self.scene = gs.Scene(show_FPS=False,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 2, 1.5),
                camera_lookat=(0.0, 0.0, 0.2),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=vis,
            vis_options=gs.options.VisOptions(plane_reflection=True),
            renderer=gs.renderers.Rasterizer(),
        )

        # Add reflecting plane
        self.scene.add_entity(
            gs.morphs.Plane(),
            surface=gs.surfaces.Aluminium(ior=10.0)
        )

        # Add walls
        for pos, color, euler in [
            ((4,  0, 1),  (0.9, 0.9, 0.9),  (0, -20,  0)),
            ((-3, 0, 1),  (0.7, 0.7, 0.7),  (0,  20,  0)),
            ((0, -3, 1),  (0.56,0.57,0.58), (0,  20, 90)),
        ]:
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.1, 8, 4),
                    pos=pos,
                    euler=euler,
                    collision=False
                ),
                surface=gs.surfaces.Rough(color=color),
                material=gs.materials.Rigid(gravity_compensation=1.0)
            )

        # Robot and cube
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="assets/xml/franka_emika_panda/panda.xml")
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06), collision=False),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # Camera setup
        self.cams = []
        self.cam_transform = trans_quat_to_T(
            np.array([0.03, 0, 0.03]),
            xyz_to_quat(np.array([185, 0, 90]))
        )
        for _ in range(self.num_envs):
            cam = self.scene.add_camera(res=(120,120), fov=90, GUI=True)
            self.cams.append(cam)

        # Build parallel environments
        env_space = 100.0
        self.scene.build(n_envs=self.num_envs, env_spacing=(env_space, env_space))
        self.envs_idx = np.arange(self.num_envs)

        for cam in self.cams:
            cam.start_recording()

        self._init_robot()


    def _init_robot(self):
        self.motors_dof = torch.arange(7, device=self.device)
        self.fingers_dof = torch.arange(7,9, device=self.device)

        q0 = torch.tensor(
            [0.4, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:,7:9].clone()
        self.end_effector = self.franka.get_link("hand")

        base_pos = torch.tensor([0.2720, -0.1683, 1.0164], device=self.device)
        self.pos = base_pos.unsqueeze(0).repeat(self.num_envs, 1)
        base_quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], device=self.device)
        self.quat = base_quat.unsqueeze(0).repeat(self.num_envs, 1)

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:,:-2], self.motors_dof, self.envs_idx)

    def _render(self):
        imgs = []
        M = int(math.sqrt(self.num_envs))
        env_space = 100.0
        noise_level = self.noise_config.get("visual_noise_level", 0.0)

        for idx, cam in enumerate(self.cams):
            ee_pos = self.end_effector.get_pos(envs_idx=[idx])[0].cpu().numpy()
            ee_quat = self.end_effector.get_quat(envs_idx=[idx])[0].cpu().numpy()
            col = idx // M;
            row = idx % M
            x_off = (col - (M - 1) / 2) * env_space
            y_off = (row - (M - 1) / 2) * env_space
            ee_pos_offset = ee_pos + np.array([x_off, y_off, 0.0])
            ee_T = trans_quat_to_T(ee_pos_offset, ee_quat)
            cam_T = ee_T @ self.cam_transform
            cam.set_pose(transform=cam_T)
            rgb = cam.render()[0]

            # Add Gaussian noise here
            if noise_level > 0.0:
                noise = np.random.normal(0, noise_level, rgb.shape)
                rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)

            img = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
            imgs.append(img)

        return torch.stack(imgs, dim=0)

    def _build_observation(self):
        samples = [self.image_history[i] for i in self.sample_offsets]
        return torch.cat(samples, dim=1)

    def _anti_collapse_probs(self, per_pos_returns: np.ndarray, min_prob: float = 0.05) -> np.ndarray:
        """
        Reverse-rank distribution:
        - Higher-return positions get lower probability next episode;
        - Lower-return positions get higher probability (but never below min_prob).
        """
        R = np.maximum(per_pos_returns.astype(np.float64), 0.0)
        S = R.sum()
        if not np.isfinite(S) or S <= 0.0:
            q = np.ones(self.pos_count, dtype=np.float64) / self.pos_count
            return q.astype(np.float32)

        p = R / S
        order_low_to_high = np.argsort(R, kind='mergesort')  # stable
        p_sorted_high_to_low = np.sort(p)[::-1]  # reverse

        q = np.empty_like(p)
        q[order_low_to_high] = p_sorted_high_to_low  # reverse-rank mapping
        q = q / q.sum()

        if min_prob and min_prob > 0.0:
            q = np.maximum(q, min_prob)
            q = q / q.sum()
        return q.astype(np.float32)

    def reset(self):
        """
        Curriculum-aware reset:
          - INFERENCE: cycle deterministically through positions.
          - TRAINING: update position probabilities from last episode's returns
                      (reverse-rank anti-collapse), then sample positions per env.
        Returns: stacked observation tensor [N, C, H, W].
        """
        # =========================
        # 1) INFERENCE MODE
        # =========================
        if getattr(self, "inference_mode", False):
            self.episode_count += 1
            self._episode_active = True
            self._episode_return = 0.0

            # deterministic cycle over positions
            self._infer_cycle_idx = (getattr(self, "_infer_cycle_idx", 0) + 1) % self.pos_count
            self.env_pos_idx = np.full(self.num_envs, self._infer_cycle_idx, dtype=np.int64)

            one_pos = self.cube_positions[self._infer_cycle_idx].reshape(1, -1)
            self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)

            # Reset robot & place cube
            self._init_robot()
            self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
            self.scene.step()

            print(f"[Inference] Episode {self.episode_count}: pos idx={self._infer_cycle_idx} "
                  f"pos={self.current_cube_pos[0].tolist()}")

            # Histories
            self.image_history.clear()
            first = self._render()
            self.image_history.clear()
            for _ in range(self.history_length):
                self.image_history.append(first.clone())

            # Per-episode accounting
            self._episode_return_per_env.zero_()
            self._done_mask_episode.zero_()
            self._step_count = 0

            return self._build_observation()

        # =========================
        # 2) TRAINING MODE
        # =========================

        # If we just finished an episode, update next-episode position distribution
        if getattr(self, "_episode_active", False):
            # Aggregate returns per position from the last episode
            per_env = self._episode_return_per_env.detach().cpu().numpy()
            per_pos_returns = np.zeros(self.pos_count, dtype=np.float64)
            for i in range(self.pos_count):
                per_pos_returns[i] = per_env[self.env_pos_idx == i].sum()

            # Compute next distribution (reverse-rank) with floor
            self.pos_probs = self._anti_collapse_probs(per_pos_returns, min_prob=0.05)

            # Debug info
            counts = np.bincount(self.env_pos_idx, minlength=self.pos_count)
            pr = np.round(per_pos_returns, 3).tolist()
            pp = np.round(self.pos_probs, 3).tolist()
            print(f"[EP end] per-pos return={pr} | assignment_counts={counts.tolist()} | next p={pp}")

        # Start a new episode
        self.episode_count += 1
        self._episode_active = True
        self._episode_return = 0.0
        self._episode_return_per_env.zero_()
        self._done_mask_episode.zero_()

        # Sample a position for each env from the learned distribution
        self.env_pos_idx = np.random.choice(self.pos_count, size=self.num_envs, p=self.pos_probs)
        self.current_cube_pos = np.stack([self.cube_positions[i] for i in self.env_pos_idx], axis=0)

        # Reset robot & place cube per env
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        print(f"Episode {self.episode_count}: pos_probs={np.round(self.pos_probs, 3).tolist()} "
              f"| assigned idx={self.env_pos_idx.tolist()}")

        # --- PRIME STACKED HISTORY ---
        self.image_history.clear()
        first = self._render()
        for _ in range(self.history_length):
            self.image_history.append(first.clone())
        self._step_count = 0

        return self._build_observation()

    def step(self, actions):
        """
        Apply discrete Cartesian moves, render on a frame-skip schedule, update the
        stacked observation, compute audio-parity reward + success bonus, and set dones on success.
        """
        # --- Discrete Cartesian moves (existing logic retained) ---
        masks = [actions == i for i in range(6)]
        pos = self.pos.clone()
        pos[masks[0], 0] += 0.05;
        pos[masks[1], 0] -= 0.05
        pos[masks[2], 1] += 0.05;
        pos[masks[3], 1] -= 0.05
        pos[masks[4], 2] += 0.05;
        pos[masks[5], 2] -= 0.05

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        # --- Frame-skip: render only every Nth step (existing behavior) ---
        if self._step_count % self.render_every == 0:
            new_frame = self._render()  # [N, 3, H, W]
        else:
            new_frame = self.image_history[-1]
        self._step_count += 1

        # Update history and build stacked observation
        self.image_history.append(new_frame)
        obs = self._build_observation()  # [N, 3*len(offsets), H, W]

        # Optional display every 100 steps (existing)
        self.step_count += 1
        if self.num_envs == 2 and self.step_count % 100 == 0:
            frames = obs[0].cpu().numpy().reshape(len(self.sample_offsets), 3, 120, 120)
            plt.figure(figsize=(8, 8))
            for i in range(len(self.sample_offsets)):
                ax = plt.subplot(2, 3, i + 1)
                img = np.transpose(frames[i], (1, 2, 0))
                ax.imshow(img)
                ax.axis('off')
            plt.suptitle(f"Stacked frames at step {self.step_count}")
            plt.pause(0.1)
            plt.show(block=False)

        # --- Distance & reward (parity with audio env) ---
        obj_pos = self.cube.get_pos()
        gp_l = self.franka.get_link("left_finger").get_pos()
        gp_r = self.franka.get_link("right_finger").get_pos()
        dist = torch.norm(obj_pos - (gp_l + gp_r) / 2, dim=1)  # [N]

        # NEW: audio-style base reward
        base_reward = torch.clamp(torch.exp(-4 * dist), 0.0, 1.0)

        # Success & bonus
        success_mask = (dist <= self.success_thresh)  # [N] bool on device
        if self.success_bonus < 0:
            bonus = torch.zeros_like(dist, dtype=torch.float32, device=self.device)
        else:
            bonus = success_mask.float() * self.success_bonus

        rewards = (base_reward + bonus).to(self.device)

        # Per-episode accumulation until first success (per env)
        active_mask = (~self._done_mask_episode).float()
        self._episode_return_per_env += rewards * active_mask
        self._done_mask_episode |= success_mask  # latch once successful

        # Dones: success -> done (configurable)
        if self.report_success_as_done:
            dones = success_mask.to(self.device)
        else:
            dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Keep pos for next step
        self.pos = pos

        return obs, rewards, dones


if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoVisionStackedEnv(vis=True, device=torch.device("cuda"))
    obs = env.reset()
    for _ in range(200):
        actions = torch.randint(0, 6, (env.num_envs,), device=env.device)
        obs, rewards, dones = env.step(actions)
        if dones.any():
            print("Done!", dones)
            break
