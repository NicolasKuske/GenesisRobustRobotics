import numpy as np
import genesis as gs
import torch
from collections import deque

class ReachCubeTorqueEnv:
    def __init__(
        self,
        vis,
        device,
        num_envs=1,
        episodes_per_position=1,
        window_size=4,
        reward_thresholds=[3, 3.5, 3.5, 3.5, 3.5, 3.5, 4]
    ):
        self.device = device
        self.num_envs = num_envs

        # observation & action dimensions
        self.state_dim = 6    # [cube_x, cube_y, cube_z, gripper_x, gripper_y, gripper_z]
        self.action_space = 7 # seven continuous torques

        # episodes per position before resampling cube
        self.episodes_per_position = episodes_per_position

        # Y and Z sampling ranges
        self.min_y, self.max_y = -0.6, 0.6
        self.min_z, self.max_z = 0.1, 0.7

        # curriculum running window size and thresholds
        self.window_size = window_size
        self.last_episode_rewards = deque(maxlen=self.window_size)
        self.reward_thresholds = reward_thresholds

        # X-axis curriculum variables (start at 0.6, like the position env)
        self.fixed_x = 0.6
        self.dynamic_x = False
        self.min_x_dynamic = self.fixed_x
        self.max_x_dynamic = self.fixed_x
        self.x_stage = 0
        self.max_stages = int((self.fixed_x - (-0.6)) / 0.2)
        self.completed = False

        # success & shaping parameters
        self.success_thresh = 0.30
        self.success_bonus = 0.1
        self.shaping_type = "exp"
        self.shaping_coef = 10.0
        self.k = 0.5
        self.dist_offset = 0.0

        # keeping the arm upright: positive-exponential height reward
        self.z_k = 1                   # exponential rate
        self.z_reward_max = 0.01       # desired reward at z=1
        # normalization so that r_z(1) = z_reward_max
        self.z_norm_factor = 1 - np.exp(-self.z_k * 1.0)

        # trackers
        self.episode_count = 0
        self.prev_dist = None
        self.sum_delta = None
        self.sum_success = None
        self.sum_z = None

        # torque limits
        self.max_torque = torch.tensor([87, 87, 87, 87, 12, 12, 12], device=self.device)

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
        # floor and walls
        self.scene.add_entity(
            gs.morphs.Plane(), surface=gs.surfaces.Aluminium(ior=10.0)
        )
        for pos, color, euler in [
            ((4, 0, 1),   (0.9,0.9,0.9), (0, -20, 0)),
            ((-3,0,1),    (0.7,0.7,0.7), (0,  20, 0)),
            ((0, -3,1),   (0.56,0.57,0.58),(0,  20,90)),
        ]:
            self.scene.add_entity(
                gs.morphs.Box(size=(0.1,8,4), pos=pos, euler=euler, collision=False),
                surface=gs.surfaces.Rough(color=color),
                material=gs.materials.Rigid(gravity_compensation=1.0)
            )

        # add Franka and cube
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml")
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06)),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # finalize and build N envs
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)

        # --- tune PD gains on all 9 DOFs (7 arm joints + 2 fingers) ---
        jnt_names = [
            'joint1','joint2','joint3','joint4',
            'joint5','joint6','joint7',
            'finger_joint1','finger_joint2',
        ]
        dofs_idx = [
            self.franka.get_joint(name).dof_idx_local
            for name in jnt_names
        ]
        # PD gains (you can adjust these values)
        kp = np.array([4500,4500,3500,3500,2000,2000,2000,100,100], dtype=np.float32)
        kv = np.array([ 450, 450, 350, 350, 200, 200, 200,  10,  10], dtype=np.float32)
        self.franka.set_dofs_kp(kp=kp, dofs_idx_local=dofs_idx)
        self.franka.set_dofs_kv(kv=kv, dofs_idx_local=dofs_idx)
        # clamp torque limits for safety
        self.franka.set_dofs_force_range(
            lower = np.array([-87,-87,-87,-87,-12,-12,-12,-100,-100], dtype=np.float32),
            upper = np.array([ 87,  87,  87,  87,  12,  12,  12,  100, 100], dtype=np.float32),
            dofs_idx_local = dofs_idx,
        )

        # initial placement: sample X=0.6, Y/Z random
        self.build_env()
        self.current_cube_pos = self._sample_random_pos()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

    def _sample_random_pos(self):
        if self.dynamic_x:
            x = np.random.uniform(self.min_x_dynamic, self.max_x_dynamic, (self.num_envs,1))
        else:
            x = np.full((self.num_envs,1), self.fixed_x)
        y_low  = np.random.uniform(self.min_y, -0.15, (self.num_envs,1))
        y_high = np.random.uniform(0.15, self.max_y, (self.num_envs,1))
        mask = np.random.rand(self.num_envs,1) < 0.5
        y = np.where(mask, y_low, y_high)
        z = np.random.uniform(self.min_z, self.max_z, (self.num_envs,1))
        return np.concatenate([x, y, z], axis=1)

    def build_env(self):
        # reset to ready joint pose
        q0 = torch.tensor(
            [-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.01, 0.01],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        # lock fingers only
        self.fixed_finger_pos = q0[:, 7:9].clone()
        self.franka.set_dofs_kp(
            kp=np.array([100,100]), dofs_idx_local=self.fixed_finger_pos.new_tensor([7,8]).cpu().numpy()
        )
        self.franka.set_dofs_kv(
            kv=np.array([10,10]),  dofs_idx_local=self.fixed_finger_pos.new_tensor([7,8]).cpu().numpy()
        )
        self.franka.set_dofs_force_range(
            lower=np.array([-100,-100]),
            upper=np.array([100,100]),
            dofs_idx_local=self.fixed_finger_pos.new_tensor([7,8]).cpu().numpy()
        )

    def reset(self):
        if self.episode_count > 0:
            shaping   = self.sum_delta.cpu().mean().item()
            bonus     = self.sum_success.cpu().mean().item()
            z_reward  = self.sum_z.cpu().mean().item()
            ep_reward = shaping + bonus + z_reward
            print(
                f"[Episode {self.episode_count}] "
                f"Shaping: {shaping:.4f}  "
                f"Bonus: {bonus:.4f}  "
                f"Height-term: {z_reward:.4f}  "
                f"Total: {ep_reward:.4f}"
            )

            self.last_episode_rewards.append(ep_reward)
            if len(self.last_episode_rewards) == self.window_size:
                mean_reward = np.mean(self.last_episode_rewards)
                if self.x_stage >= len(self.reward_thresholds):
                    print(f"[Curriculum] All {len(self.reward_thresholds)} stages complete ➞ saving checkpoint.")
                    self.completed = True
                else:
                    thresh = self.reward_thresholds[self.x_stage]
                    print(
                        f"[Curriculum] last {self.window_size}-ep mean: {mean_reward:.4f}  "
                        f"(need > {thresh:.4f} to advance from stage {self.x_stage})"
                    )
                    if mean_reward > thresh:
                        self.x_stage += 1
                        self.dynamic_x = (self.x_stage > 0)
                        self.min_x_dynamic = self.fixed_x - 0.2 * self.x_stage
                        self.last_episode_rewards.clear()
                        if self.x_stage == len(self.reward_thresholds):
                            print(f"[Env] Passed final threshold (>{thresh}). Curriculum complete.")
                            self.completed = True
                        else:
                            print(
                                f"[Env] Stage {self.x_stage}: X ∈ "
                                f"[{self.min_x_dynamic:.2f}, {self.max_x_dynamic:.2f}]"
                            )
                            if self.x_stage == len(self.reward_thresholds) - 1:
                                final_t = self.reward_thresholds[-1]
                                print(
                                    f"[Env] Now in final stage: mean > {final_t:.4f} (over {self.window_size}) to finish."
                                )

        self.episode_count += 1
        self.sum_delta   = torch.zeros(self.num_envs, device=self.device)
        self.sum_success = torch.zeros(self.num_envs, device=self.device)
        self.sum_z       = torch.zeros(self.num_envs, device=self.device)

        if (self.episode_count - 1) % self.episodes_per_position == 0:
            self.current_cube_pos = self._sample_random_pos()
            xs = self.current_cube_pos[:,0]
            print(f"[Episode {self.episode_count}] Cube X-positions: {xs}")

        self.build_env()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

        obj  = self.cube.get_pos()
        grip = (
            self.franka.get_link("left_finger").get_pos() +
            self.franka.get_link("right_finger").get_pos()
        ) * 0.5
        state = torch.concat([obj, grip], dim=1)

        self.prev_dist = torch.norm(obj - grip, dim=1)
        return state

    def step(self, actions):
        torque = actions.clamp(-1.0, 1.0) * self.max_torque
        forces = torque.cpu().numpy()
        self.franka.control_dofs_force(
            forces,
            dofs_idx_local=np.arange(7),
            envs_idx=self.envs_idx
        )
        self.franka.control_dofs_position(
            self.fixed_finger_pos,
            torch.arange(7, 9, device=self.device),
            self.envs_idx
        )

        self.scene.step()

        obj  = self.cube.get_pos()
        grip = (
            self.franka.get_link("left_finger").get_pos() +
            self.franka.get_link("right_finger").get_pos()
        ) * 0.5
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

        success = dist_new < self.success_thresh
        bonus   = success.to(delta.dtype) * self.success_bonus

        z_val = obj[:, 2]
        z_term = (1 - torch.exp(-self.z_k * z_val)) / self.z_norm_factor
        z_term *= self.z_reward_max

        reward = delta + bonus - z_term
        dones = success

        self.sum_delta   += delta
        self.sum_success += bonus
        self.sum_z       += z_term
        self.prev_dist    = dist_new

        return state, reward, dones

if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeTorqueEnv(vis=True, device="cuda")
    s = env.reset()
    print("Initial state:", s)
