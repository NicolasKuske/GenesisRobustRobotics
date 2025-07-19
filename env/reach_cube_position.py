#env/reach_cube_position.py


import numpy as np
import genesis as gs
import torch
from collections import deque

class ReachCubePositionEnv:
    # Build the scene & entities
    def __init__(
        self,
        vis,
        device,
        num_envs=1,
        episodes_per_position=3,
        window_size=4,
        reward_thresholds=[2, 2.5, 2.5, 2.5, 2.5, 2.5, 3]
    ):
        self.device = device
        self.num_envs = num_envs

        # observation & action dimensions
        self.state_dim = 6  # [cube_x, cube_y, cube_z, gripper_x, gripper_y, gripper_z]
        self.action_space = 6  # six discrete moves: ±x, ±y, ±z

        # episodes per position before resampling cube
        self.episodes_per_position = episodes_per_position

        # Y and Z sampling ranges
        self.min_y, self.max_y = -0.6, 0.6
        self.min_z, self.max_z = 0.1, 0.7

        # curriculum running window size and threshold
        self.window_size = window_size
        self.last_episode_rewards = deque(maxlen=self.window_size)
        self.reward_thresholds = reward_thresholds

        # X-axis curriculum variables
        self.fixed_x = 0.6
        self.dynamic_x = False
        self.min_x_dynamic = self.fixed_x
        self.max_x_dynamic = self.fixed_x
        self.x_stage = 0
        self.max_stages = int((self.fixed_x - (-0.6)) / 0.2)

        # flag for final window after last stage
        self.ready_for_checkpoint = False
        self.completed = False

        # success and shaping parameters
        self.success_thresh = 0.30
        self.success_bonus = 0.1
        self.shaping_type = "exp"
        self.shaping_coef = 10.0
        self.k = 0.5
        self.dist_offset = 0.0

        # trackers
        self.prev_dist = None
        self.episode_count = 0
        self.sum_delta = None
        self.sum_success = None

        # initial cube position
        self.current_cube_pos = self._sample_random_pos()

        # build the Genesis scene
        self.scene = gs.Scene(
            show_FPS=False,
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
            vis_options=gs.options.VisOptions(plane_reflection=True),
            renderer=gs.renderers.Rasterizer(),
        )

        # add floor and walls
        self.scene.add_entity(
            gs.morphs.Plane(), surface=gs.surfaces.Aluminium(ior=10.0)
        )
        for pos, color, euler in [
            ((4, 0, 1), (0.9, 0.9, 0.9), (0, -20, 0)),
            ((-3, 0, 1), (0.7, 0.7, 0.7), (0, 20, 0)),
            ((0, -3, 1), (0.56, 0.57, 0.58), (0, 20, 90)),
        ]:
            self.scene.add_entity(
                gs.morphs.Box(size=(0.1, 8, 4), pos=pos, euler=euler, collision=False),
                surface=gs.surfaces.Rough(color=color),
                material=gs.materials.Rigid(gravity_compensation=1.0)
            )

        # add Franka robot and cube
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml")
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # finalize scene
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)

        # initial placement
        self.build_env()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

    def _sample_random_pos(self):
        # X-coordinate
        if self.dynamic_x:
            x = np.random.uniform(self.min_x_dynamic, self.max_x_dynamic, (self.num_envs, 1))
        else:
            x = np.full((self.num_envs, 1), self.fixed_x)
        # Y-coordinate from two disjoint intervals
        y_low = np.random.uniform(self.min_y, -0.15, (self.num_envs, 1))
        y_high = np.random.uniform(0.15, self.max_y, (self.num_envs, 1))
        mask = np.random.rand(self.num_envs, 1) < 0.5
        y = np.where(mask, y_low, y_high)
        # Z-coordinate
        z = np.random.uniform(self.min_z, self.max_z, (self.num_envs, 1))
        return np.concatenate([x, y, z], axis=1)

    def build_env(self):
        # reset to ready joint pose
        q0 = torch.tensor([
            -1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.01, 0.01
        ], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:, 7:9].clone()

        # IK target for end effector
        self.end_effector = self.franka.get_link("hand")
        pos = torch.tensor([0.2720, -0.1683, 1.0164], device=self.device)
        quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], device=self.device)
        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(
            qpos[:, :-2], torch.arange(7, device=self.device), self.envs_idx
        )
        self.franka.control_dofs_position(
            self.fixed_finger_pos, torch.arange(7, 9, device=self.device), self.envs_idx
        )

    def reset(self):
        # report shaping and bonus and apply curriculum
        if self.episode_count > 0:
            shaping = self.sum_delta.cpu().mean().item()
            bonus = self.sum_success.cpu().mean().item()
            print(f"[Episode {self.episode_count}] Mean shaping reward: {shaping:.4f}")
            print(f"[Episode {self.episode_count}] Mean bonus reward:   {bonus:.4f}")

            # update running window
            self.last_episode_rewards.append(shaping + bonus)
            if len(self.last_episode_rewards) == self.window_size:
                mean_reward = np.mean(self.last_episode_rewards)

                # still climbing stages?
                if self.x_stage < self.max_stages:
                    threshold = self.reward_thresholds[self.x_stage]
                    print(
                        f"[Curriculum] Mean of last {self.window_size} episodes: "
                        f"{mean_reward:.4f}  (need > {threshold:.4f} to advance from stage {self.x_stage})"
                    )
                    if mean_reward > threshold:
                        # advance into the next X‐range
                        self.x_stage += 1
                        self.dynamic_x = True
                        self.min_x_dynamic = self.fixed_x - 0.2 * self.x_stage
                        self.last_episode_rewards.clear()

                        # announce the new stage
                        if self.x_stage == self.max_stages:
                            # just hit the final stage
                            print(
                                f"[Env] Stage {self.x_stage}: X ∈ "
                                f"[{self.min_x_dynamic:.2f}, {self.max_x_dynamic:.2f}]"
                            )
                            print(
                                f"[Env] Final stage reached. Now running "
                                f"X ∈ [{self.min_x_dynamic:.2f}, {self.max_x_dynamic:.2f}]. "
                                f"In this stage, mean > {self.reward_thresholds[-1]:.4f} to complete."
                            )
                        else:
                            # still an intermediate stage
                            print(
                                f"[Env] Stage {self.x_stage}: X ∈ "
                                f"[{self.min_x_dynamic:.2f}, {self.max_x_dynamic:.2f}]"
                            )

                else:
                    # already in final stage → enforce 3.0 one‐shot
                    final_thresh = self.reward_thresholds[-1]
                    print(
                        f"[Curriculum] (final) Mean of last {self.window_size} episodes: "
                        f"{mean_reward:.4f}  (need > {final_thresh:.4f} to complete)"
                    )
                    if mean_reward > final_thresh:
                        print("[Env] Final threshold satisfied. Curriculum complete.")
                        self.completed = True

        # prepare for next episode
        self.episode_count += 1
        self.sum_delta = torch.zeros(self.num_envs, device=self.device)
        self.sum_success = torch.zeros(self.num_envs, device=self.device)

        # resample cube position periodically
        if (self.episode_count - 1) % self.episodes_per_position == 0:
            self.current_cube_pos = self._sample_random_pos()
            xs = self.current_cube_pos[:, 0]
            print(f"[Episode {self.episode_count}] Cube X-positions: {xs}")

        # build and return state
        self.build_env()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        obj = self.cube.get_pos()
        grip = (
                       self.franka.get_link("left_finger").get_pos() +
                       self.franka.get_link("right_finger").get_pos()
               ) / 2
        state = torch.concat([obj, grip], dim=1)
        self.prev_dist = torch.norm(obj - grip, dim=1)
        return state

    def step(self, actions):
        # apply discrete moves
        pos = self.pos.clone()
        pos[actions == 0, 0] += 0.05
        pos[actions == 1, 0] -= 0.05
        pos[actions == 2, 1] += 0.05
        pos[actions == 3, 1] -= 0.05
        pos[actions == 4, 2] += 0.05
        pos[actions == 5, 2] -= 0.05

        # IK & control
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=pos, quat=self.quat
        )
        self.franka.control_dofs_position(
            qpos[:, :-2], torch.arange(7, device=self.device), self.envs_idx
        )
        self.franka.control_dofs_position(
            self.fixed_finger_pos, torch.arange(7, 9, device=self.device), self.envs_idx
        )
        self.scene.step()

        # observe & reward
        obj = self.cube.get_pos()
        grip = (
            self.franka.get_link("left_finger").get_pos() +
            self.franka.get_link("right_finger").get_pos()
        ) / 2
        state = torch.concat([obj, grip], dim=1)

        dist_new = torch.norm(obj - grip, dim=1)
        dist_old = self.prev_dist
        if self.shaping_type == "exp":
            delta = self.shaping_coef * (
                torch.exp(-self.k * (dist_new - self.dist_offset)) -
                torch.exp(-self.k * (dist_old - self.dist_offset))
            )
        else:
            delta = self.shaping_coef * (dist_old - dist_new)

        mask = dist_new < self.success_thresh
        bonus = mask.to(delta.dtype) * self.success_bonus
        reward = delta + bonus
        dones = mask

        # accumulate
        self.sum_delta += delta
        self.sum_success += bonus
        self.prev_dist = dist_new
        self.pos = pos
        return state, reward, dones

if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubePositionEnv(
        vis=True,
        device="cuda",
        episodes_per_position=5,
        window_size=4,
        reward_thresholds=2.5
    )
    s = env.reset()
    print("Initial state:", s)
