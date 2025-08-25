# envs/ik/reach_cube_ego_video_stacked.py

import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import genesis as gs
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat


class ReachCubeEgoVisionStackedEnv:
    """
    Visual IK reaching environment with a Franka end-effector camera and stacked RGB history.

    This version uses a PARALLEL, ANTI-COLLAPSE curriculum over TWO cube positions:
      - Positions: HARD and EASY (configurable via `hard_pos`, `easy_pos`)
      - Training (inference_mode=False):
          * At each reset, every env samples one of the two positions according to a learned
            anti-collapse distribution (positions with *lower* recent return get *higher* prob).
          * Optional: guarantee that both positions are present in the batch when num_envs >= 2.
      - Inference (inference_mode=True):
          * Deterministic global cycling: all envs share the same position; alternates hard/easy.

    Notes:
      - `easy_episodes` and `hard_episodes` are kept for call-site compatibility but are ignored.
      - Only the cube exists in the scene (no sphere/bunny/dragon).
      - Reward: exp(-4 * distance(fingers_midpoint, cube)) + success bonus inside a threshold.
      - Observations: stacked RGB frames sampled at `sample_offsets`.

    Default positions:
      hard_pos = [0.2,  0.8, 0.2]
      easy_pos = [0.2, -0.8, 0.2]
    """

    # -------------------------
    # Construction
    # -------------------------
    def __init__(
        self,
        vis,
        device,
        num_envs: int = 1,
        randomize_every: int = 100,  # kept for compatibility; not used here
        noise_config: dict | None = None,

        # reward/done parity
        success_thresh: float = 0.3001,
        success_bonus: float = 20.0,
        report_success_as_done: bool = True,

        # ---------- positions & (legacy) schedule args ----------
        easy_episodes: int = 1,  # ignored (kept for compatibility)
        hard_episodes: int = 1,  # ignored (kept for compatibility)
        hard_pos=None,   # defaults to [0.2,  0.8, 0.2] if None
        easy_pos=None,   # defaults to [0.2, -0.8, 0.2] if None

        # ---------- curriculum controls ----------
        inference_mode: bool = False,   # deterministic cycle hard/easy shared across envs
        min_prob: float = 0.05,         # probability floor in anti-collapse
        ensure_each_position: bool = True,  # try to include both positions each episode (if possible)
    ):
        self.device = device
        self.num_envs = num_envs
        self.randomize_every = randomize_every
        self.episode_count = 0
        self.step_count = 0

        # reward config
        self.success_thresh = success_thresh
        self.success_bonus = success_bonus
        self.report_success_as_done = report_success_as_done

        # positions
        hp = np.array(hard_pos if hard_pos is not None else [0.2,  0.8, 0.2], dtype=float).reshape(3,)
        ep = np.array(easy_pos if easy_pos is not None else [0.2, -0.8, 0.2], dtype=float).reshape(3,)
        self.cube_positions = [hp, ep]
        self.pos_count = 2
        self.pos_names = ["hard", "easy"]

        # anti-collapse state
        self.pos_probs = np.ones(self.pos_count, dtype=np.float32) / self.pos_count
        self.env_pos_idx = np.zeros(self.num_envs, dtype=np.int64)
        self.current_cube_pos = None  # (N, 3)
        self.min_prob = float(min_prob)
        self.ensure_each_position = bool(ensure_each_position)

        # per-episode accounting (success bookkeeping)
        self._episode_return_per_env = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._done_mask_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # state flags
        self._episode_active = False
        self._episode_return = 0.0
        self.inference_mode = inference_mode
        self._infer_cycle_idx = 0  # 0 -> hard, 1 -> easy

        # stacked vision history config
        self.history_length = 25
        self.sample_offsets = [-21, -16, -11, -6, -1]
        self.image_history = deque(maxlen=self.history_length)
        self.render_every = 5
        self._step_count = 0

        # noise config
        self.noise_config = noise_config if noise_config else {"visual_noise_level": 0.0}

        # observation / action
        self.obs_shape = (3 * len(self.sample_offsets), 120, 120)
        self.action_space = 6

        # ------------- Scene -------------
        self.scene = gs.Scene(
            show_FPS=False,
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
            vis_options=gs.options.VisOptions(show_world_frame=False, plane_reflection=True),
            renderer=gs.renderers.Rasterizer(),
        )

        # plane
        self.scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Aluminium(ior=10.0))

        # walls (visual background)
        for pos, color, euler in [
            ((4,  0, 1),  (0.9, 0.9, 0.9),  (0, -20,  0)),
            ((-3, 0, 1),  (0.7, 0.7, 0.7),  (0,  20,  0)),
            ((0, -3, 1),  (0.56, 0.57, 0.58), (0,  20, 90)),
        ]:
            self.scene.add_entity(
                gs.morphs.Box(size=(0.1, 8, 4), pos=pos, euler=euler, collision=False),
                surface=gs.surfaces.Rough(color=color),
                material=gs.materials.Rigid(gravity_compensation=1.0),
            )

        # robot
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="assets/xml/franka_emika_panda/panda.xml"))

        # ----- ONLY CUBE -----
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06), collision=False),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        # ---------------------

        # cameras mounted to EE
        self.cams = []
        self.cam_transform = trans_quat_to_T(np.array([0.03, 0, 0.03]), xyz_to_quat(np.array([185, 0, 90])))
        for _ in range(self.num_envs):
            self.cams.append(self.scene.add_camera(res=(120, 120), fov=90, GUI=True))

        # parallelize scene
        self._env_space = 100.0
        self.scene.build(n_envs=self.num_envs, env_spacing=(self._env_space, self._env_space))
        self.envs_idx = np.arange(self.num_envs)

        for cam in self.cams:
            cam.start_recording()

        self._init_robot()

    # -------------------------
    # Robot init
    # -------------------------
    def _init_robot(self):
        self.motors_dof = torch.arange(7, device=self.device)
        self.fingers_dof = torch.arange(7, 9, device=self.device)

        q0 = torch.tensor(
            [-0.2, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:, 7:9].clone()
        self.end_effector = self.franka.get_link("hand")

        base_pos = torch.tensor([0.2720, -0.1683, 1.0164], device=self.device)
        self.pos = base_pos.unsqueeze(0).repeat(self.num_envs, 1)
        base_quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], device=self.device)
        self.quat = base_quat.unsqueeze(0).repeat(self.num_envs, 1)

        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=self.pos, quat=self.quat)
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)

    # -------------------------
    # Rendering / observation
    # -------------------------
    def _render(self) -> torch.Tensor:
        imgs = []
        N = self.num_envs
        cols = int(math.ceil(math.sqrt(N)))
        rows = int(math.ceil(N / cols))
        env_space = self._env_space
        noise_level = float(self.noise_config.get("visual_noise_level", 0.0))

        for idx, cam in enumerate(self.cams):
            ee_pos = self.end_effector.get_pos(envs_idx=[idx])[0].cpu().numpy()
            ee_quat = self.end_effector.get_quat(envs_idx=[idx])[0].cpu().numpy()

            r = idx // cols
            c = idx % cols
            x_off = (c - (cols - 1) / 2) * env_space
            y_off = (r - (rows - 1) / 2) * env_space

            ee_pos_offset = ee_pos + np.array([x_off, y_off, 0.0])
            ee_T = trans_quat_to_T(ee_pos_offset, ee_quat)
            cam_T = ee_T @ self.cam_transform
            cam.set_pose(transform=cam_T)

            rgb = cam.render()[0]  # H×W×3, uint8

            if noise_level > 0.0:
                noise = np.random.normal(0, noise_level, rgb.shape)
                rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)

            img = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0  # 3×H×W
            imgs.append(img)

        return torch.stack(imgs, dim=0)  # N×3×H×W

    def _build_observation(self) -> torch.Tensor:
        samples = [self.image_history[i] for i in self.sample_offsets]
        return torch.cat(samples, dim=1)  # N×(3*k)×H×W

    # -------------------------
    # Curriculum utils
    # -------------------------
    def _anti_collapse_probs(self, per_pos_returns: np.ndarray, min_prob: float) -> np.ndarray:
        """
        Reverse-rank distribution:
          - Positions with higher return -> lower prob next episode;
          - Positions with lower return -> higher prob (bounded below by min_prob).
        """
        R = np.maximum(per_pos_returns.astype(np.float64), 0.0)
        S = R.sum()
        if not np.isfinite(S) or S <= 0.0:
            return (np.ones(self.pos_count, dtype=np.float64) / self.pos_count).astype(np.float32)

        p = R / S
        order_low_to_high = np.argsort(R, kind="mergesort")   # stable
        p_sorted_high_to_low = np.sort(p)[::-1]               # reverse
        q = np.empty_like(p)
        q[order_low_to_high] = p_sorted_high_to_low
        q = q / q.sum()

        if min_prob and min_prob > 0.0:
            q = np.maximum(q, min_prob)
            q = q / q.sum()
        return q.astype(np.float32)

    def _ensure_each_position_present(self, idxs: np.ndarray) -> np.ndarray:
        """
        If possible, make sure at least one env is assigned to each position.
        Only acts when num_envs >= pos_count and a position is missing.
        """
        if self.num_envs < self.pos_count:
            return idxs
        missing = [i for i in range(self.pos_count) if i not in idxs]
        if not missing:
            return idxs
        # Reassign a few envs (choose arbitrary unique env slots) to cover missing positions
        unique_envs = np.unique(np.arange(self.num_envs))
        ptr = 0
        for m in missing:
            if ptr >= len(unique_envs):
                break
            idxs[unique_envs[ptr]] = m
            ptr += 1
        return idxs

    # -------------------------
    # Reset
    # -------------------------
    def reset(self):
        """
        Inference:
          - all envs take the same position, alternating hard/easy deterministically.
        Training:
          - update anti-collapse probabilities from last episode returns and then
            sample per-env positions; optionally guarantee that both positions appear.

        Returns: obs [N, C, H, W]
        """
        # ------ Inference mode ------
        if self.inference_mode:
            self.episode_count += 1
            self._episode_active = True
            self._episode_return = 0.0

            idx = self._infer_cycle_idx % self.pos_count  # 0: hard, 1: easy
            self._infer_cycle_idx = (idx + 1) % self.pos_count

            # shared position across all envs
            self.env_pos_idx = np.full(self.num_envs, idx, dtype=np.int64)
            pos1 = self.cube_positions[idx].reshape(1, 3)
            self.current_cube_pos = np.repeat(pos1, self.num_envs, axis=0)

            # reset robot
            self._init_robot()

            # place cube
            self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
            self.scene.step()

            print(f"[Inference] Episode {self.episode_count}: position='{self.pos_names[idx]}' "
                  f"| pos={self.current_cube_pos[0].tolist()}")

            # prime history
            self.image_history.clear()
            first = self._render()
            for _ in range(self.history_length):
                self.image_history.append(first.clone())

            self._episode_return_per_env.zero_()
            self._done_mask_episode.zero_()
            self._step_count = 0
            return self._build_observation()

        # ------ Training mode ------
        if self._episode_active:
            # update position probabilities from last episode returns
            per_env = self._episode_return_per_env.detach().cpu().numpy()
            per_pos_returns = np.zeros(self.pos_count, dtype=np.float64)
            for i in range(self.pos_count):
                per_pos_returns[i] = per_env[self.env_pos_idx == i].sum()
            self.pos_probs = self._anti_collapse_probs(per_pos_returns, min_prob=self.min_prob)

            counts = np.bincount(self.env_pos_idx, minlength=self.pos_count)
            print(f"[EP end] per-pos return={np.round(per_pos_returns,3).tolist()} "
                  f"| counts={counts.tolist()} "
                  f"| next p={np.round(self.pos_probs,3).tolist()}")

        # start new episode
        self.episode_count += 1
        self._episode_active = True
        self._episode_return = 0.0
        self._episode_return_per_env.zero_()
        self._done_mask_episode.zero_()

        # sample per-env positions
        self.env_pos_idx = np.random.choice(self.pos_count, size=self.num_envs, p=self.pos_probs)
        if self.ensure_each_position:
            self.env_pos_idx = self._ensure_each_position_present(self.env_pos_idx)
        self.current_cube_pos = np.stack([self.cube_positions[i] for i in self.env_pos_idx], axis=0)

        # reset robot
        self._init_robot()

        # place cube
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        # logging
        unique, counts = np.unique(self.env_pos_idx, return_counts=True)
        asg = {self.pos_names[int(u)]: int(c) for u, c in zip(unique, counts)}
        print(f"Episode {self.episode_count}: pos_probs={np.round(self.pos_probs, 3).tolist()} "
              f"| assigned={asg}")

        # prime history
        self.image_history.clear()
        first = self._render()
        for _ in range(self.history_length):
            self.image_history.append(first.clone())
        self._step_count = 0

        return self._build_observation()

    # -------------------------
    # Step
    # -------------------------
    def step(self, actions: torch.Tensor):
        """
        actions: int tensor [N] with values in {0..5} for ±x, ±y, ±z discrete moves.
        Returns: obs [N, C, H, W], rewards [N], dones [N]
        """
        # discrete Cartesian moves
        masks = [actions == i for i in range(6)]
        pos = self.pos.clone()
        pos[masks[0], 0] += 0.05
        pos[masks[1], 0] -= 0.05
        pos[masks[2], 1] += 0.05
        pos[masks[3], 1] -= 0.05
        pos[masks[4], 2] += 0.05
        pos[masks[5], 2] -= 0.05

        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=pos, quat=self.quat)
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        # render with frame skip
        if self._step_count % self.render_every == 0:
            new_frame = self._render()
        else:
            new_frame = self.image_history[-1]
        self._step_count += 1

        # update observation history
        self.image_history.append(new_frame)
        obs = self._build_observation()

        # optional debug plot (only when num_envs==2)
        self.step_count += 1
        if self.num_envs == 2 and self.step_count % 100 == 0:
            frames = obs[0].cpu().numpy().reshape(len(self.sample_offsets), 3, 120, 120)
            plt.figure(figsize=(8, 8))
            for i in range(len(self.sample_offsets)):
                ax = plt.subplot(2, 3, i + 1)
                img = np.transpose(frames[i], (1, 2, 0))
                ax.imshow(img)
                ax.axis("off")
            plt.suptitle(f"Stacked frames at step {self.step_count}")
            plt.pause(0.1)
            plt.show(block=False)

        # distance to cube
        obj_pos = self.cube.get_pos()  # N×3
        gp_l = self.franka.get_link("left_finger").get_pos()
        gp_r = self.franka.get_link("right_finger").get_pos()
        dist = torch.norm(obj_pos - (gp_l + gp_r) / 2, dim=1)  # [N]

        # reward
        base_reward = torch.clamp(torch.exp(-4 * dist), 0.0, 1.0)
        success_mask = (dist <= self.success_thresh)
        if self.success_bonus < 0:
            bonus = torch.zeros_like(dist, dtype=torch.float32, device=self.device)
        else:
            bonus = success_mask.float() * self.success_bonus
        rewards = (base_reward + bonus).to(self.device)

        # episode accounting (until first success per env)
        active_mask = (~self._done_mask_episode).float()
        self._episode_return_per_env += rewards * active_mask
        self._done_mask_episode |= success_mask

        # dones
        if self.report_success_as_done:
            dones = success_mask.to(self.device)
        else:
            dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # keep position for next step
        self.pos = pos
        return obs, rewards, dones


if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoVisionStackedEnv(
        vis=True,
        device=torch.device("cuda"),
        num_envs=16,                 # set >1 to learn both positions in parallel
        inference_mode=False,        # True -> deterministic alternating eval
        hard_pos=[0.2,  0.8, 0.2],
        easy_pos=[0.2, -0.8, 0.2],
        min_prob=0.05,
        ensure_each_position=True,
    )
    obs = env.reset()
    for _ in range(200):
        actions = torch.randint(0, 6, (env.num_envs,), device=env.device)
        obs, rewards, dones = env.step(actions)
        if dones.any():
            print("Done!", dones.nonzero(as_tuple=False).flatten().tolist())
            # typically you'd continue stepping; this break is just a demo
            break
