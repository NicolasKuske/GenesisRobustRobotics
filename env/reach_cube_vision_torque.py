# env/reach_cube_vision_torque.py

import numpy as np
import genesis as gs
import torch
import math
from collections import deque

class ReachCubeVisionTorqueEnv:
    """
    Vision-based continuous-torque environment with curriculum on cube X-axis sampling.

    Curriculum stages (lower bound → 0.6):
      Stage 0: X ∈ [0.4, 0.6]
      Stage 1: X ∈ [0.2, 0.6]
      Stage 2: X ∈ [0.0, 0.6]
      Stage 3: X ∈ [-0.2, 0.6]
      Stage 4: X ∈ [-0.4, 0.6]
      Stage 5: X ∈ [-0.6, 0.6]
      Stage 6: Final stability round with same bounds as Stage 5.
    The agent outputs a 7-dim torque vector (one per joint), scaled by ±1.
    """

    def __init__(
        self,
        vis: bool,
        device: torch.device,
        num_envs: int = 1,
        episodes_per_position: int = 3,
        window_size: int = 4,
        reward_thresholds=None
    ):
        # Device & parallelism
        self.device = device
        self.num_envs = num_envs

        # Curriculum
        self.episodes_per_position = episodes_per_position
        self.window_size = window_size
        self.last_rewards = deque(maxlen=window_size)
        self.reward_thresholds = reward_thresholds or [3, 3, 3, 3, 3.5, 3.5, 3.5]
        self.fixed_x = 0.6
        self.x_bounds = [0.4, 0.2, 0.0, -0.2, -0.4, -0.6]
        self.max_stages = len(self.x_bounds)
        self.x_stage = 0
        self.completed = False

        # Reward shaping
        self.success_thresh = 0.30
        self.success_bonus = 0.1
        self.shaping_type = 'exp'
        self.shaping_coef = 10.0
        self.k = 0.5
        self.dist_offset = 0.0

        # Episode trackers
        self.episode_count = 0
        self.prev_dist = None
        self.episode_reward = 0.0
        self.sum_delta = None
        self.sum_success = None

        # Observation & action dims
        self.obs_shape    = (3, 120, 120)
        self.action_space = 7
        # Per-joint max torque limits
        self.max_torque = torch.tensor([87, 87, 87, 87, 12, 12, 12],
                                       dtype=torch.float32,
                                       device=self.device)

        # —— build scene ——
        self.scene = gs.Scene(
            #show_FPS=False,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 2, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=vis,
        )
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml")
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06)),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # camera grid
        self.cams = []
        env_space = 5.0
        M = int(math.sqrt(self.num_envs))
        assert M*M == self.num_envs, "num_envs must be a perfect square"
        for idx in range(self.num_envs):
            row, col = divmod(idx, M)
            x_off = (col - (M-1)/2) * env_space
            y_off = (row - (M-1)/2) * env_space
            cam = self.scene.add_camera(
                res=(120, 120),
                pos=(2.5 + x_off, 0.5 + y_off, 3.5),
                lookat=(x_off, y_off, 0.35),
                fov=30,
                GUI=True,
            )
            self.cams.append(cam)

        self.scene.build(n_envs=self.num_envs, env_spacing=(env_space, env_space))
        self.envs_idx = np.arange(self.num_envs)
        for cam in self.cams:
            cam.start_recording()

        # Initialize robot to starting pose
        self._init_robot()

    def _init_robot(self):
        # set all joint positions, lock fingers
        q0 = torch.tensor(
            [-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        # store finger positions for later
        self.fixed_finger_pos = q0[:, 7:9].clone()

    def _collect_states(self):
        # render RGB images from each env into a tensor [N, C, H, W]
        batch = torch.empty(
            (self.num_envs, *self.obs_shape),
            dtype=torch.float32,
            device=self.device,
        )
        for i, cam in enumerate(self.cams):
            # cam.render()[0] may have negative strides—make a copy
            rgb = cam.render()[0]
            np_img = rgb.copy()  # now contiguous, positive strides
            img = torch.from_numpy(np_img) \
                       .permute(2, 0, 1) \
                       .float() \
                       .div(255.0)
            batch[i].copy_(img, non_blocking=True)
        return batch


    def _sample_cube_pos(self):
        idx   = min(self.x_stage, self.max_stages)
        lower= self.x_bounds[idx] if idx < self.max_stages else self.x_bounds[-1]
        x     = np.random.uniform(lower, self.fixed_x, (self.num_envs, 1))
        y     = np.random.uniform(-0.6, 0.6,       (self.num_envs, 1))
        z     = np.random.uniform(0.1,  1.0,       (self.num_envs, 1))
        return np.concatenate([x, y, z], axis=1)

    def reset(self):
        if self.episode_count > 0:
            self._process_episode_end()
        self.episode_count += 1

        # reset reward accumulators
        self.sum_delta   = torch.zeros(self.num_envs, device=self.device)
        self.sum_success = torch.zeros(self.num_envs, device=self.device)

        # periodically resample cube
        if (self.episode_count - 1) % self.episodes_per_position == 0:
            self.current_cube_pos = self._sample_cube_pos()

        # reset robot & cube
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

        # measure initial distance
        obj   = self.cube.get_pos()
        gp_l  = self.franka.get_link("left_finger").get_pos()
        gp_r  = self.franka.get_link("right_finger").get_pos()
        self.prev_dist = torch.norm(obj - (gp_l + gp_r)/2, dim=1)
        self.episode_reward = 0.0

        return self._collect_states()

    def step(self, actions: torch.Tensor):
        """
        actions: tensor of shape [N, 7], values in ℝ. We clamp to [-1,1] then scale by max_torque.
        """
        # apply joint torques
        torque = actions.clamp(-1.0, 1.0) * self.max_torque.unsqueeze(0)
        self.franka.control_dofs_force(
            torque.cpu().numpy(),
            dofs_idx_local=np.arange(7),
            envs_idx=self.envs_idx
        )
        # keep fingers fixed
        self.franka.control_dofs_position(
            self.fixed_finger_pos,
            torch.arange(7, 9, device=self.device),
            envs_idx=self.envs_idx
        )

        self.scene.step()

        # collect next state
        states = self._collect_states()

        # compute reward shaping
        obj_pos  = self.cube.get_pos()
        gp_l     = self.franka.get_link("left_finger").get_pos()
        gp_r     = self.franka.get_link("right_finger").get_pos()
        dist_new = torch.norm(obj_pos - (gp_l + gp_r)/2, dim=1)

        if self.shaping_type == 'exp':
            delta = self.shaping_coef * (
                torch.exp(-self.k * (dist_new - self.dist_offset))
                - torch.exp(-self.k * (self.prev_dist - self.dist_offset))
            )
        else:
            delta = self.shaping_coef * (self.prev_dist - dist_new)

        success = (dist_new < self.success_thresh).float()
        bonus   = success * self.success_bonus
        rewards = delta + bonus

        # accumulate for curriculum
        self.sum_delta   += delta
        self.sum_success += bonus

        self.prev_dist     = dist_new
        self.episode_reward += rewards.mean().item()

        dones = success.bool()
        return states, rewards, dones

    def _process_episode_end(self):
        ep_r    = self.episode_reward
        shaping = self.sum_delta.mean().item()
        bonus   = self.sum_success.mean().item()
        print(f"[Episode {self.episode_count}] Shaping: {shaping:.4f}, Bonus: {bonus:.4f}, Total: {ep_r:.4f}")

        self.last_rewards.append(ep_r)
        if len(self.last_rewards) == self.window_size:
            mean_r = sum(self.last_rewards) / self.window_size
            thr    = self.reward_thresholds[min(self.x_stage, len(self.reward_thresholds)-1)]
            print(f"[Curriculum] last {self.window_size}-ep mean: {mean_r:.4f}, threshold: {thr:.4f}")
            if mean_r > thr:
                self._advance_stage()

    def _advance_stage(self):
        self.x_stage += 1
        if self.x_stage <= self.max_stages:
            lb = self.x_bounds[min(self.x_stage, len(self.x_bounds)-1)]
            print(f"Advanced to stage {self.x_stage}: X ∈ [{lb:.2f}, {self.fixed_x:.2f}]")
            self.last_rewards.clear()
        else:
            self.completed = True
            print("Curriculum complete!")

if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeVisionTorqueEnv(vis=True, device=torch.device("cuda"), num_envs=1)
    obs = env.reset()
    for t in range(100):
        # random continuous torques for testing
        actions = torch.randn(env.num_envs, env.action_space, device=env.device)
        obs, reward, done = env.step(actions)
        if done.any():
            print("Done!", done)
            break
