# envs/reach_cube_position_IKsimple_curriculum.py

import numpy as np
import genesis as gs
import torch
from collections import deque

class ReachCubePositionEnv:

    def __init__(
            self, vis, device, num_envs=1,
            episodes_per_position=3,
            window_size=4,
            reward_thresholds=[2, 2.5, 3, 3, 3.5, 3.5, 3.5]
    ):
        self.device = device
        self.num_envs = num_envs

        # add these two lines explicitly:
        self.state_dim = 6  # [cube_xyz, gripper_xyz]
        self.action_space = 6  # six discrete ±x/±y/±z moves

        # curriculum params
        self.episodes_per_position = episodes_per_position
        self.window_size = window_size
        self.reward_thresholds = reward_thresholds
        self.last_episode_rewards = deque(maxlen=self.window_size)

        # shaping parameters
        self.success_thresh = 0.30
        self.success_bonus = 0.1
        self.shaping_type = "exp"
        self.shaping_coef = 10.0
        self.k = 0.5
        self.dist_offset = 0.0

        # tracking variables
        self.prev_dist = None
        self.sum_delta = None
        self.sum_success = None

        self.fixed_x = 0.6
        self.dynamic_x = False
        self.min_x_dynamic = self.fixed_x
        self.max_x_dynamic = self.fixed_x
        self.x_stage = 0
        self.max_stages = int((self.fixed_x - (-0.6)) / 0.2)

        self.completed = False

        self.success_thresh = 0.3
        self.success_bonus = 50.0

        self.episode_count = 0
        self.sum_rewards = None

        self.current_cube_pos = self._sample_random_pos()

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 2, 1.5), camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30, res=(960, 640), max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=vis,
            vis_options=gs.options.VisOptions(plane_reflection=True),
            renderer=gs.renderers.Rasterizer(),
        )

        # scene entities setup (floor, walls, robot, cube)
        self.scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Aluminium(ior=10.0))

        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="assets/xml/franka_emika_panda/panda.xml"))
        self.cube = self.scene.add_entity(gs.morphs.Box(size=(0.06, 0.06, 0.06)),
                                          material=gs.materials.Rigid(gravity_compensation=1.0))

        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)

        self.build_env()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

    def _sample_random_pos(self):
        x = np.random.uniform(self.min_x_dynamic, self.max_x_dynamic, (self.num_envs, 1)) if self.dynamic_x else np.full((self.num_envs, 1), self.fixed_x)
        y = np.random.uniform(-0.6, 0.6, (self.num_envs, 1))
        z = np.random.uniform(0.1, 0.7, (self.num_envs, 1))
        return np.concatenate([x, y, z], axis=1)

    def build_env(self):
        q0 = torch.tensor([-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.01, 0.01], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:, 7:9].clone()
        self.end_effector = self.franka.get_link("hand")

        pos = torch.tensor([0.2720, -0.1683, 1.0164], device=self.device).repeat(self.num_envs, 1)
        quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], device=self.device).repeat(self.num_envs, 1)

        self.pos = pos
        self.quat = quat

        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=pos, quat=quat)
        self.franka.control_dofs_position(qpos[:, :-2], torch.arange(7, device=self.device), self.envs_idx)

    def reset(self):
        if self.episode_count > 0:
            shaping = self.sum_delta.cpu().mean().item()
            bonus = self.sum_success.cpu().mean().item()
            print(f"[Episode {self.episode_count}] Mean shaping reward: {shaping:.4f}")
            print(f"[Episode {self.episode_count}] Mean bonus reward:   {bonus:.4f}")

            # Update running reward window
            self.last_episode_rewards.append(shaping + bonus)

            if len(self.last_episode_rewards) == self.window_size and self.x_stage < self.max_stages:
                threshold = self.reward_thresholds[self.x_stage]
                mean_reward = np.mean(self.last_episode_rewards)
                print(f"[Curriculum] Mean reward: {mean_reward:.4f} (threshold: {threshold:.4f})")
                if mean_reward > threshold:
                    self.x_stage += 1
                    self.dynamic_x = True
                    self.min_x_dynamic = self.fixed_x - 0.2 * self.x_stage
                    self.last_episode_rewards.clear()
                    print(
                        f"[Curriculum] Advanced to stage {self.x_stage}, X-range: [{self.min_x_dynamic:.2f}, {self.max_x_dynamic:.2f}]")

        self.episode_count += 1

        # Reset reward trackers
        self.sum_delta = torch.zeros(self.num_envs, device=self.device)
        self.sum_success = torch.zeros(self.num_envs, device=self.device)

        # Resample cube position periodically
        if (self.episode_count - 1) % self.episodes_per_position == 0:
            self.current_cube_pos = self._sample_random_pos()
            print(f"[Episode {self.episode_count}] New cube positions:\n{self.current_cube_pos}")

        # Build environment and set cube
        self.build_env()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

        # Construct initial observation
        obj_pos = self.cube.get_pos()
        grip_pos = (
                           self.franka.get_link("left_finger").get_pos() +
                           self.franka.get_link("right_finger").get_pos()
                   ) / 2
        state = torch.cat([obj_pos, grip_pos], dim=1)

        # Initialize previous distance
        self.prev_dist = torch.norm(obj_pos - grip_pos, dim=1)

        return state

    def step(self, actions):
        # Define move offsets
        move_offsets = torch.tensor([
            [0.05, 0, 0], [-0.05, 0, 0],
            [0, 0.05, 0], [0, -0.05, 0],
            [0, 0, 0.05], [0, 0, -0.05]
        ], device=self.device)

        # Compute new positions from actions
        pos = self.franka.get_link("hand").get_pos() + move_offsets[actions]

        # Inverse kinematics and control
        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=pos, quat=self.quat)
        self.franka.control_dofs_position(qpos[:, :-2], torch.arange(7, device=self.device), self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, torch.arange(7, 9, device=self.device), self.envs_idx)
        self.scene.step()

        # Observe new state
        obj_pos = self.cube.get_pos()
        grip_pos = (
                           self.franka.get_link("left_finger").get_pos() +
                           self.franka.get_link("right_finger").get_pos()
                   ) / 2
        state = torch.cat([obj_pos, grip_pos], dim=1)

        # Reward shaping and success bonus
        dist_new = torch.norm(obj_pos - grip_pos, dim=1)
        dist_old = self.prev_dist

        # Exponential shaping reward
        delta = self.shaping_coef * (
                torch.exp(-self.k * (dist_new - self.dist_offset)) -
                torch.exp(-self.k * (dist_old - self.dist_offset))
        )

        # Success bonus
        success_mask = dist_new < self.success_thresh
        bonus = success_mask.float() * self.success_bonus

        # Total reward and done
        reward = delta + bonus
        done = success_mask

        # Update trackers
        self.sum_delta += delta
        self.sum_success += bonus
        self.prev_dist = dist_new

        return state, reward, done

