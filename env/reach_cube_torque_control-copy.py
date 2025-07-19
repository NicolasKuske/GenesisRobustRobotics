# reach_cube_torque_control.py

import numpy as np
import genesis as gs
import torch

class ReachCubeTorqueEnv:

    def __init__(self, vis, device, num_envs=1, randomize_every=100):
        self.device = device
        self.num_envs = num_envs
        self.randomize_every = randomize_every

        # Success criteria
        self.success_thresh = 0.3
        self.success_bonus = 50.0

        # Potential-based shaping
        self.prev_dist = None

        # Episode counting & cube placement
        self.episode_count = 0
        #self.initial_pos = np.array([0.65, 0.0, 0.1])[None, :]
        self.initial_pos = np.array([-0.5, 0.3, 0.7])[None, :] #new_position1
        #self.initial_pos = np.array([0.1, 0.5, 0.3])[None, :]  # new_position2
        self.current_cube_pos = None

        self.state_dim = 6
        self.action_space = 7  # 7 joint torques

        # Torque limits from provided data
        self.max_torque = torch.tensor([87, 87, 87, 87, 12, 12, 12], device=self.device)

        # Build Genesis scene
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
            vis_options=gs.options.VisOptions(plane_reflection=True),
            renderer=gs.renderers.Rasterizer(),
        )

        # Floor
        self.scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Aluminium(ior=10.0))

        # Robot arm (Franka Panda)
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml")
        )

        # Cube
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # Build multiple envs
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)

        self.build_env()
        self.cube.set_pos(
            self.initial_pos.repeat(self.num_envs, axis=0),
            envs_idx=self.envs_idx
        )

    def build_env(self):
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)

        q0 = torch.tensor(
            [-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.01, 0.01],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()

        self.fixed_finger_pos = q0[:, 7:9].clone()

        # Set PD gains for finger stability
        self.franka.set_dofs_kp(
            kp=np.array([100, 100]),
            dofs_idx_local=self.fingers_dof.cpu().numpy()
        )
        self.franka.set_dofs_kv(
            kv=np.array([10, 10]),
            dofs_idx_local=self.fingers_dof.cpu().numpy()
        )
        self.franka.set_dofs_force_range(
            lower=np.array([-100, -100]),
            upper=np.array([100, 100]),
            dofs_idx_local=self.fingers_dof.cpu().numpy()
        )

    def reset(self):
        self.episode_count += 1

        if self.episode_count == 1:
            one_pos = self.initial_pos
        elif self.episode_count % self.randomize_every == 0:
            abs_xy = np.random.uniform(0.2, 1.0, size=(1, 2))
            signs = np.random.choice([-1.0, 1.0], size=(1, 2))
            xy = abs_xy * signs
            z = np.random.uniform(0.05, 1.0, size=(1, 1))
            one_pos = np.concatenate([xy, z], axis=1)
        else:
            one_pos = self.current_cube_pos[:1]

        self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)

        self.build_env()
        self.cube.set_pos(
            self.current_cube_pos,
            envs_idx=self.envs_idx
        )

        object_pos = self.cube.get_pos()
        gripper_pos = (
            self.franka.get_link("left_finger").get_pos() +
            self.franka.get_link("right_finger").get_pos()
        ) / 2
        state = torch.concat([object_pos, gripper_pos], dim=1)

        self.prev_dist = torch.norm(object_pos - gripper_pos, dim=1)

        return state

    def step(self, actions):
        # Apply continuous torques as forces
        torque_actions = actions.clamp(-1.0, 1.0) * self.max_torque
        forces = torque_actions.cpu().numpy()
        self.franka.control_dofs_force(
            forces,
            dofs_idx_local=self.motors_dof.cpu().numpy(),
            envs_idx=self.envs_idx
        )

        # Fix fingers in position
        self.franka.control_dofs_position(
            self.fixed_finger_pos,
            self.fingers_dof,
            self.envs_idx
        )

        self.scene.step()

        # observe
        object_pos = self.cube.get_pos()
        gripper_pos = (self.franka.get_link("left_finger").get_pos() +
                       self.franka.get_link("right_finger").get_pos()) / 2
        state = torch.concat([object_pos, gripper_pos], dim=1)

        # distance‐based reward
        dist = torch.norm(object_pos - gripper_pos, dim=1)
        base_rew = torch.exp(-4 * (dist - 0.1))

        # success detection → bonus + done
        success = dist < self.success_thresh
        reward = base_rew  + success.to(base_rew.dtype) * self.success_bonus
        rewards = torch.clamp(reward, min=0.0, max=self.success_bonus + 1.0)
        dones = success

        return state, reward, dones

if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeTorqueEnv(vis=True, device="cuda")