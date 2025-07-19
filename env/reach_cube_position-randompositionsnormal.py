# env/reach_cube_position.py

import numpy as np
import genesis as gs
import torch

class ReachCubePositionEnv:

    # Build the scene & entities
    def __init__(self, vis, device, num_envs=1, randomize_every=2):
        self.device          = device
        self.num_envs        = num_envs
        self.randomize_every = randomize_every

        # success criteria
        self.success_thresh = 0.3    # 30 cm threshold for “at goal”
        self.success_bonus  = 50.0   # large bonus when goal reached

        # for potential‐based shaping
        self.prev_dist = None

        # episode counting & cube placement
        self.episode_count    = 0
        #self.initial_pos = np.array([0.65, 0.0, 0.1])[None, :]
        #self.initial_pos = np.array([-0.5, 0.3, 0.7])[None, :] #new_position1
        self.initial_pos = np.array([0.1, 0.5, 0.3])[None, :]  #new_position2
        #self.current_cube_pos = None

        self.state_dim    = 6  # [cube_xyz, gripper_xyz]
        self.action_space = 6  # six discrete ±x/±y/±z moves

        # build Genesis scene
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

        ##### define objects in scene #####

        # floor
        self.scene.add_entity(
            gs.morphs.Plane(),
            surface=gs.surfaces.Aluminium(ior=10.0),
        )

        # walls
        self.scene.add_entity(
            gs.morphs.Box(size=(0.1, 8, 4), pos=(4, 0, 1), euler=(0, -20, 0), collision=False),
            surface=gs.surfaces.Rough(color=(0.9, 0.9, 0.9)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )
        self.scene.add_entity(
            gs.morphs.Box(size=(0.1, 4, 4), pos=(-3, 0, 1), euler=(0, 20, 0), collision=False),
            surface=gs.surfaces.Rough(color=(0.7, 0.7, 0.7)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )
        self.scene.add_entity(
            gs.morphs.Box(size=(0.1, 8, 4), pos=(0, -3, 1), euler=(0, 20, 90), collision=False),
            surface=gs.surfaces.Rough(color=(0.56, 0.57, 0.58)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # robot arm
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"),
        )
        # cube
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # finalize multiple envs
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)

        # initial pose & cube placement
        self.build_env()
        self.cube.set_pos(
            self.initial_pos.repeat(self.num_envs, axis=0),
            envs_idx=self.envs_idx
        )

    def build_env(self):
        # joint DOFs
        self.motors_dof  = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)

        # “ready” joint configuration
        q0 = torch.tensor([
            -1.0, -0.3,  0.3, -1.0, -0.1,  1.7,  1.0,
             0.01, 0.01
        ], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()

        # fix fingers & identify end‐effector
        self.fixed_finger_pos = q0[:, 7:9].clone()
        self.end_effector     = self.franka.get_link("hand")

        # desired hand pose for IK
        pos  = torch.tensor([0.2720, -0.1683, 1.0164], dtype=torch.float32, device=self.device)
        quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], dtype=torch.float32, device=self.device)
        self.pos  = pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(
            qpos[:, :-2],
            self.motors_dof,
            self.envs_idx
        )

    def reset(self):
        # increment episode count
        self.episode_count += 1

        # randomize cube pos occasionally
        if self.episode_count == 1:
            one_pos = self.initial_pos
        elif self.episode_count % self.randomize_every == 0:
            abs_xy = np.random.uniform(0.2, 1.0, size=(1, 2))
            signs  = np.random.choice([-1.0, 1.0], size=(1, 2))
            xy     = abs_xy * signs
            z      = np.random.uniform(0.05, 1.0, size=(1, 1))
            one_pos = np.concatenate([xy, z], axis=1)
        else:
            one_pos = self.current_cube_pos[:1]

        self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)

        # rebuild & place cube
        self.build_env()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

        # initial observation
        object_pos  = self.cube.get_pos()
        gripper_pos = (
            self.franka.get_link("left_finger").get_pos() +
            self.franka.get_link("right_finger").get_pos()
        ) / 2
        state = torch.concat([object_pos, gripper_pos], dim=1)

        # initialize previous distance
        self.prev_dist = torch.norm(object_pos - gripper_pos, dim=1)

        return state

    def step(self, actions):
        # apply discrete ±x/±y/±z via IK
        pos = self.pos.clone()
        pos[actions == 0, 0] += 0.05
        pos[actions == 1, 0] -= 0.05
        pos[actions == 2, 1] += 0.05
        pos[actions == 3, 1] -= 0.05
        pos[actions == 4, 2] += 0.05
        pos[actions == 5, 2] -= 0.05

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        # observe new positions
        object_pos  = self.cube.get_pos()
        gripper_pos = (
            self.franka.get_link("left_finger").get_pos() +
            self.franka.get_link("right_finger").get_pos()
        ) / 2
        state = torch.concat([object_pos, gripper_pos], dim=1)

        # compute potential‐based reward = reduction in distance
        dist_new = torch.norm(object_pos - gripper_pos, dim=1)
        dist_old = self.prev_dist
        # exponential “value” at old and new distances
        base_old = torch.exp(-4 * (dist_old - 0.1))
        base_new = torch.exp(-4 * (dist_new - 0.1))

        # shaped reward = increase in exponential value
        delta = base_new - base_old

        # success detection → add bonus
        success_mask = dist_new < self.success_thresh  # boolean tensor
        # add bonus to reward
        reward = delta + success_mask.to(delta.dtype) * self.success_bonus
        # return dones as boolean
        dones = success_mask

        # update for next step
        self.prev_dist = dist_new
        self.pos       = pos

        return state, reward, dones


# main guard
if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubePositionEnv(vis=True, device="cuda")
