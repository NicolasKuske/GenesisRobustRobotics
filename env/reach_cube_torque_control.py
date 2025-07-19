import numpy as np
import genesis as gs
import torch

class ReachCubeTorqueEnv:
    def __init__(
        self,
        vis,
        device,
        num_envs=1,
        episodes_per_position=1,
    ):
        self.device = device
        self.num_envs = num_envs

        # observation & action dimensions
        self.state_dim = 6    # [cube_x, cube_y, cube_z, gripper_x, gripper_y, gripper_z]
        self.action_space = 7 # seven continuous torques

        # how many episodes to run before sampling a new cube position
        self.episodes_per_position = episodes_per_position

        # sampling bounds (stage 6)
        self.min_x, self.max_x = -0.6, 0.6
        self.min_y, self.max_y = -0.6, 0.6
        self.min_z, self.max_z = 0.1, 0.7

        # success & shaping parameters
        self.success_thresh = 0.30
        self.success_bonus = 0.1
        self.shaping_type = "exp"
        self.shaping_coef = 10.0
        self.k = 0.5
        self.dist_offset = 0.0

        # upright‐arm exponential height reward
        self.z_k = 1                   # exponential rate
        self.z_reward_max = 0.01       # desired reward at z=1
        self.z_norm_factor = 1 - np.exp(-self.z_k * 1.0)

        # bookkeeping
        self.episode_count = 0
        self.prev_dist = None
        self.sum_delta = None
        self.sum_success = None
        self.sum_z = None

        # torque limits for each of the 7 joints
        self.max_torque = torch.tensor(
            [87, 87, 87, 87, 12, 12, 12], device=self.device
        )

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
            ((4, 0, 1),   (0.9,0.9,0.9),   (0, -20, 0)),
            ((-3,0,1),    (0.7,0.7,0.7),   (0,  20, 0)),
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

        # initial placement
        self.current_cube_pos = self._sample_random_pos()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

    def _sample_random_pos(self):
        # X-coordinate uniform across [-0.6, 0.6]
        x = np.random.uniform(self.min_x, self.max_x, (self.num_envs, 1))

        # Y from two sub-intervals: [-0.6, -0.15] or [0.15, 0.6]
        y_low  = np.random.uniform(self.min_y, -0.15, (self.num_envs, 1))
        y_high = np.random.uniform( 0.15, self.max_y, (self.num_envs, 1))
        mask   = np.random.rand(self.num_envs, 1) < 0.5
        y      = np.where(mask, y_low, y_high)

        # Z uniform across [0.1, 0.7]
        z = np.random.uniform(self.min_z, self.max_z, (self.num_envs, 1))

        return np.concatenate([x, y, z], axis=1)

    def build_env(self):
        # reset to ready joint pose
        q0 = torch.tensor(
            [-1.0, -0.3,  0.3, -1.0, -0.1,  1.7,  1.0,
              0.01, 0.01],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()

        # lock fingers
        self.fixed_finger_pos = q0[:, 7:9].clone()
        finger_idxs = self.fixed_finger_pos.new_tensor([7, 8]).cpu().numpy()
        self.franka.set_dofs_kp(kp=np.array([100,100]), dofs_idx_local=finger_idxs)
        self.franka.set_dofs_kv(kv=np.array([10,10]),  dofs_idx_local=finger_idxs)
        self.franka.set_dofs_force_range(
            lower=np.array([-100,-100]),
            upper=np.array([ 100, 100]),
            dofs_idx_local=finger_idxs
        )

    def reset(self):
        # print previous-episode stats
        if self.episode_count > 0:
            shaping  = self.sum_delta.cpu().mean().item()
            bonus    = self.sum_success.cpu().mean().item()
            z_reward = self.sum_z.cpu().mean().item()
            total_r  = shaping + bonus + z_reward
            print(
                f"[Episode {self.episode_count}] "
                f"Shaping: {shaping:.4f}  "
                f"Bonus: {bonus:.4f}  "
                f"Height: {z_reward:.4f}  "
                f"Total: {total_r:.4f}"
            )

        # increment episode and reset accumulators
        self.episode_count += 1
        self.sum_delta   = torch.zeros(self.num_envs, device=self.device)
        self.sum_success = torch.zeros(self.num_envs, device=self.device)
        self.sum_z       = torch.zeros(self.num_envs, device=self.device)

        # sample a new cube position every `episodes_per_position` episodes
        if (self.episode_count - 1) % self.episodes_per_position == 0:
            self.current_cube_pos = self._sample_random_pos()
            xs = self.current_cube_pos[:, 0]
            print(f"[Episode {self.episode_count}] Cube X-positions: {xs}")

        # reset robot & place cube
        self.build_env()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

        # get initial state
        obj  = self.cube.get_pos()
        grip = (
            self.franka.get_link("left_finger").get_pos() +
            self.franka.get_link("right_finger").get_pos()
        ) * 0.5
        state = torch.concat([obj, grip], dim=1)
        self.prev_dist = torch.norm(obj - grip, dim=1)

        return state

    def step(self, actions):
        # apply torques
        torque = actions.clamp(-1.0, 1.0) * self.max_torque
        forces = torque.cpu().numpy()
        self.franka.control_dofs_force(
            forces,
            dofs_idx_local=np.arange(7),
            envs_idx=self.envs_idx
        )
        # keep fingers locked
        self.franka.control_dofs_position(
            self.fixed_finger_pos,
            torch.arange(7, 9, device=self.device),
            self.envs_idx
        )

        # advance simulation
        self.scene.step()

        # observe cube & gripper
        obj  = self.cube.get_pos()
        grip = (
            self.franka.get_link("left_finger").get_pos() +
            self.franka.get_link("right_finger").get_pos()
        ) * 0.5
        state = torch.concat([obj, grip], dim=1)

        # distance‐based shaping
        dist_new = torch.norm(obj - grip, dim=1)
        dist_old = self.prev_dist
        if self.shaping_type == "exp":
            delta = self.shaping_coef * (
                torch.exp(-self.k * (dist_new - self.dist_offset)) -
                torch.exp(-self.k * (dist_old - self.dist_offset))
            )
        else:
            delta = self.shaping_coef * (dist_old - dist_new)

        # success bonus
        success = dist_new < self.success_thresh
        bonus   = success.to(delta.dtype) * self.success_bonus

        # height‐based reward
        z_val  = obj[:, 2]
        z_term = (1 - torch.exp(-self.z_k * z_val)) / self.z_norm_factor
        z_term *= self.z_reward_max

        # total reward and done
        reward = delta + bonus - z_term
        dones  = success

        # accumulate for logging
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

