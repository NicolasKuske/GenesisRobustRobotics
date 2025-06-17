import numpy as np
import genesis as gs
import torch

class GraspFixedCubeEnv:

    # Build the scene & entities
    def __init__(self, vis, device, num_envs=1):
        self.device   = device
        self.num_envs = num_envs

        # state: [cube_xyz, gripper_xyz]
        self.state_dim    = 6
        # 6 joints × (±0.05 rad) → 12 discrete slots (multi-hot)
        self.action_space = 12

        # scene setup
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 2, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options   = gs.options.SimOptions(dt=0.01),
            rigid_options = gs.options.RigidOptions(box_box_detection=True),
            show_viewer   = vis,
        )
        self.plane  = self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml")
        )
        self.cube   = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos =(0.65, 0.0, 0.02),
            )
        )

        # finalize build
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx   = np.arange(self.num_envs)
        # control the first 6 joints
        self.motors_dof = torch.arange(6, device=self.device)

        # track end-effector for observations
        self.end_effector = self.franka.get_link("hand")
        # set robot & cube to initial pose
        self.reset()


    def reset(self):
        # neutral “home” qpos for all 9 dims
        home_qpos = torch.tensor([
            -1.0,  # joint 0 (foot yaw)
            -0.3,  # joint 1 (lower arm pitch)
             0.3,  # joint 2 (upper arm yaw)
            -1.0,  # joint 3 (upper arm pitch)
            -0.1,  # joint 4 (head yaw)
             1.7,  # joint 5 (head pitch)
             1.0,  # joint 6 (hand roll)
             0.02, # joint 7 (gripper left)
             0.02, # joint 8 (gripper right)
        ], dtype=torch.float32, device=self.device)
        home = home_qpos.unsqueeze(0).repeat(self.num_envs,1)
        self.franka.set_qpos(home, envs_idx=self.envs_idx)
        self.scene.step()

        # reset cube position
        cube_pos = np.array([0.65, 0.0, 0.02])
        cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)

        # return initial [cube_xyz, gripper_xyz]
        obs1 = self.cube.get_pos()
        obs2 = self.end_effector.get_pos()
        return torch.concat([obs1, obs2], dim=1)


    def step(self, actions):
        """
        actions: LongTensor of shape (num_envs,12), multi-hot {0,1}
          cols 0/1 → joint0 ±0.05 rad
          cols 2/3 → joint1 ±0.05 rad
          …
          cols 10/11 → joint5 ±0.05 rad
        """

        # current joint angles (num_envs,9) → keep first 6
        qpos_full = self.franka.get_qpos(envs_idx=self.envs_idx)
        joints     = qpos_full[:, :6].clone()

        # apply every flagged delta
        mask = actions.bool()  # (num_envs,12)
        # joint0 +
        joints[mask[:,0], 0] += 0.05
        # joint0 –
        joints[mask[:,1], 0] -= 0.05
        # joint1 +
        joints[mask[:,2], 1] += 0.05
        # joint1 –
        joints[mask[:,3], 1] -= 0.05
        # joint2 +
        joints[mask[:,4], 2] += 0.05
        # joint2 –
        joints[mask[:,5], 2] -= 0.05
        # joint3 +
        joints[mask[:,6], 3] += 0.05
        # joint3 –
        joints[mask[:,7], 3] -= 0.05
        # joint4 +
        joints[mask[:,8], 4] += 0.05
        # joint4 –
        joints[mask[:,9], 4] -= 0.05
        # joint5 +
        joints[mask[:,10],5] += 0.05
        # joint5 –
        joints[mask[:,11],5] -= 0.05

        # (optional) clamp to real joint limits here if you want

        # command & step
        self.franka.control_dofs_position(joints, self.motors_dof, self.envs_idx)
        self.scene.step()

        # observe & reward
        obj_pos  = self.cube.get_pos()
        grip_pos = self.end_effector.get_pos()
        state    = torch.concat([obj_pos, grip_pos], dim=1)
        rewards  = -torch.norm(obj_pos - grip_pos, dim=1)
        dones    = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return state, rewards, dones


if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = GraspFixedCubeEnv(vis=True, device=torch.device('cuda'), num_envs=1)
