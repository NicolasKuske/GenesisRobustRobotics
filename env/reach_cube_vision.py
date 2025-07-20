import numpy as np
import genesis as gs
import torch
import math
from collections import deque
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat

class ReachCubeVisionEnv:
    """
    Vision-based environment with curriculum on cube X-axis sampling.

    Curriculum stages (lower bound → 0.6):
      Stage 0: X ∈ [0.4, 0.6]
      Stage 1: X ∈ [0.2, 0.6]
      Stage 2: X ∈ [0.0, 0.6]
      Stage 3: X ∈ [-0.2, 0.6]
      Stage 4: X ∈ [-0.4, 0.6]
      Stage 5: X ∈ [-0.6, 0.6]
      Stage 6: Final stability round with same bounds as Stage 5.
    """

    def __init__(
        self,
        vis,
        device,
        num_envs=1,
        episodes_per_position=3,
        window_size=4,
        reward_thresholds=None
    ):
        # Device & envs
        self.device = device
        self.num_envs = num_envs

        # Curriculum parameters
        self.episodes_per_position = episodes_per_position
        self.window_size = window_size
        self.last_rewards = deque(maxlen=window_size)
        # default thresholds for stages 0..6
        if reward_thresholds is None:
            self.reward_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5]
        else:
            self.reward_thresholds = reward_thresholds
        # X-axis curriculum bounds
        self.fixed_x = 0.6
        self.x_bounds = [0.4, 0.2, 0.0, -0.2, -0.4, -0.6]
        self.max_stages = len(self.x_bounds)
        self.x_stage = 0
        self.completed = False

        # Reward shaping parameters
        self.success_thresh = 0.30
        self.success_bonus = 0.1
        self.shaping_type = 'exp'
        self.shaping_coef = 10.0
        self.k = 0.5
        self.dist_offset = 0.0

        # Episode tracking
        self.episode_count = 0
        self.prev_dist = None
        self.episode_reward = 0.0

        # observation/action dims
        self.obs_shape = (3, 120, 120)
        self.action_space = 6

        # Setup scene
        self.scene = gs.Scene(
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
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06)),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # cameras
        self.cams = []
        env_space = 5.0
        M = int(math.sqrt(self.num_envs))
        assert M*M == self.num_envs, "num_envs must be a perfect square"
        for idx in range(self.num_envs):
            row, col = divmod(idx, M)
            x_off = (col - (M-1)/2)*env_space
            y_off = (row - (M-1)/2)*env_space
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

        # init robot
        self._init_robot()

    def _init_robot(self):
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        q0 = torch.tensor(
            [-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:, 7:9].clone()
        self.end_effector = self.franka.get_link("hand")
        base_pos = torch.tensor([0.2720, -0.1683, 1.0164], device=self.device)
        base_quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], device=self.device)
        self.pos = base_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.quat = base_quat.unsqueeze(0).repeat(self.num_envs, 1)
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)

    def _collect_states(self):
        states = []
        for cam in self.cams:
            rgb = cam.render()[0]
            img = torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255.0
            states.append(img)
        return torch.stack(states, dim=0)

    def _sample_cube_pos(self):
        """Random cube position based on current curriculum stage."""
        idx = min(self.x_stage, self.max_stages)
        lower = self.x_bounds[idx] if idx < self.max_stages else self.x_bounds[-1]
        x = np.random.uniform(lower, self.fixed_x, (1,1))
        y = np.random.uniform(-0.6, 0.6, size=(1,1))
        z = np.random.uniform(0.1, 1.0, size=(1,1))
        return np.concatenate([x, y, z], axis=1)

    def _process_episode_end(self):
        ep_r = self.episode_reward
        self.last_rewards.append(ep_r)
        if len(self.last_rewards) == self.window_size:
            mean_r = sum(self.last_rewards)/self.window_size
            thr = self.reward_thresholds[min(self.x_stage, len(self.reward_thresholds)-1)]
            if mean_r > thr:
                self._advance_stage()
        self.episode_reward = 0.0

    def _advance_stage(self):
        self.x_stage += 1
        if self.x_stage <= self.max_stages:
            lb = self.x_bounds[min(self.x_stage, len(self.x_bounds)-1)]
            print(f"Advanced to stage {self.x_stage}: X ∈ [{lb:.2f}, {self.fixed_x:.2f}]")
            self.last_rewards.clear()
        else:
            self.completed = True
            print("Curriculum complete!")

    def reset(self):
        if self.episode_count > 0:
            self._process_episode_end()
        self.episode_count += 1
        if (self.episode_count-1) % self.episodes_per_position == 0:
            one_pos = self._sample_cube_pos()
        else:
            one_pos = self.current_cube_pos[:1]
        self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)

        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

        # init distance for shaping
        obj = self.cube.get_pos()
        gp_l = self.franka.get_link("left_finger").get_pos()
        gp_r = self.franka.get_link("right_finger").get_pos()
        dist = torch.norm(obj - (gp_l + gp_r)/2, dim=1)
        self.prev_dist = dist

        self.episode_reward = 0.0
        return self._collect_states()

    def step(self, actions):
        masks = [actions == i for i in range(6)]
        pos = self.pos.clone()
        pos[masks[0], 0] += 0.05; pos[masks[1], 0] -= 0.05
        pos[masks[2], 1] += 0.05; pos[masks[3], 1] -= 0.05
        pos[masks[4], 2] += 0.05; pos[masks[5], 2] -= 0.05
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        states = self._collect_states()
        obj_pos = self.cube.get_pos()
        gp_l = self.franka.get_link("left_finger").get_pos()
        gp_r = self.franka.get_link("right_finger").get_pos()
        dist_new = torch.norm(obj_pos - (gp_l + gp_r)/2, dim=1)

        # shaping reward
        if self.shaping_type == 'exp':
            delta = self.shaping_coef * (
                torch.exp(-self.k * (dist_new - self.dist_offset))
                - torch.exp(-self.k * (self.prev_dist - self.dist_offset))
            )
        else:
            delta = self.shaping_coef * (self.prev_dist - dist_new)
        success = (dist_new < self.success_thresh).float()
        bonus = success * self.success_bonus
        rewards = delta + bonus

        # update trackers
        self.prev_dist = dist_new
        self.episode_reward += rewards.mean().item()

        dones = success.bool()
        self.pos = pos
        return states, rewards, dones

if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeVisionEnv(vis=True, device=torch.device("cuda"))
    for ep in range(10):
        obs = env.reset()
        for t in range(50):
            actions = torch.randint(0,6,(env.num_envs,), device=env.device)
            _, r, done = env.step(actions)
            if done.any(): break
        print(f"Episode {ep} total reward: {env.episode_reward:.4f}")
