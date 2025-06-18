#grasp_fixed_cube_vis.py

import numpy as np
import genesis as gs
import torch
import math

class GraspFixedCubeVisEnv:

    # Build the scene & entities
    def __init__(self, vis, device, num_envs=1):
        self.device     = device
        self.num_envs   = num_envs

        # now our “state” is the camera image:
        # we’re using a 120×120 RGB camera
        self.obs_shape = (3, 120, 120)
        self.action_space = 6

        self.scene = gs.Scene(
            #show_FPS=False,  # Don't show simulation speed
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 2, 1.5),        # x, y, z
                camera_lookat=(0.0, 0.0, 0.5), # focus point
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
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(0.65, 0.0, 0.02),
            )
        )

        ##### cam #### attaching cams does not work for parallelized scenes yet, so we manually link cam to arm position below
        self.cams = []
        env_space = 5.0  # must match your scene.build(env_spacing=(5.0,5.0))

        M = int(math.sqrt(self.num_envs))
        assert M * M == self.num_envs, "num_envs must be a perfect square for an M×M grid"

        for idx in range(self.num_envs):
            # compute row, col in M×M
            row = idx // M
            col = idx % M

            # center the grid at (0,0)
            x_off = (col - (M - 1) / 2) * env_space
            y_off = (row - (M - 1) / 2) * env_space

            cam = self.scene.add_camera(
                res=(120, 120),
                pos=(2.5 + x_off, 0.5 + y_off, 2.5),
                lookat=(x_off, y_off, 0.2),
                fov=30,
                GUI=True,
            )
            self.cams.append(cam)

        #T = np.eye(4)
        # Define Rx 180° rotation matrix - rotates around the x-axis in the direction away from the robot
        #Rx_180 = np.array([
        #    [1, 0, 0],
        #    [0, -1, 0],
        #    [0, 0, -1]
        #])
        #T[:3, :3] = Rx_180
        #T[:3, 3] = np.array([0.05, 0.0, 0.0])  # each link has a different wire frame which can be observed in the scene view window (press l)
        # for the hand link the z axis goes out upward, and the x axis orthogonal to arm direction outward (foward in x direction)


        self.scene.build(n_envs=self.num_envs, env_spacing=(env_space, env_space)) #only space envs in x direction
        self.envs_idx = np.arange(self.num_envs)

        # (Re)position robot to a default “ready” pose
        self.build_env()

        # add camera and start recording

        for cam in self.cams:
            cam.start_recording()


    # (Re)position robot to a default “ready” pose
    def build_env(self):
        # only the 7 arm joints (ignore fingers now)
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        #Joint limits ≈ [-2.8973, 2.8973] radians (for most joints)
        franka_pos = torch.tensor([
            -1.0,  # foot yaw - rotation around z-axis of the foot. -1 is foot directed towards x-axis. Decreasing values with clock (0 is y-axis)
            -0.3,  # lower arm segment pitch - rotation around x-or y-axis depending on foot position. I.e., lowest link deciding if arm is high or low. 0.3 is arm close to straight upward. 1.5 is parallel to ground.
             0.3,  # upper arm segment yaw - rotation around z-direction of lower arm segment. Given above values: 1.3 directed towards x. Decreasing values with clock (0 is negative y-axis, given above values)
            -1.0,  # upper arm segment pitch - Decreasing values with clock, delta 0.7 approx 45°
            -0.1,  # head yaw - Decreasing values with clock, delta 0.7 approx 45°
             1.7,  # head pitch - Increasing values with clock, delta 0.7 approx 45°
             1.0,  # hand roll - Decreasing values with clock
             0.02, # left and right gripper distance to middle
             0.02  # (ignored)
        ], dtype=torch.float32, device=self.device)
        franka_pos = franka_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(franka_pos, envs_idx=self.envs_idx)
        self.scene.step()

        # === STORE FIXED FINGER TARGETS ===
        # We read back the same joint values we just set,
        # and save them so we can re-command every step
        self.fixed_finger_pos = franka_pos[:, 7:9].clone()
        # Identify end‐effector link for IK calls
        self.end_effector = self.franka.get_link("hand")
        pos = self.end_effector.get_pos().clone()  # (num_envs×3)
        quat = self.end_effector.get_quat().clone()  # (num_envs×4)

        ## Define a fixed Cartesian target for the hand (just the above q values in approx. cartesian)
        ## NOT SURE if this is necessary but might get better control values for the robot
        pos = torch.tensor([0.2720, -0.1683, 1.0164], dtype=torch.float32, device=self.device)
        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], dtype=torch.float32, device=self.device)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=self.pos,
            quat=self.quat,
        )
        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)






    # Reset cube position + robot → return initial state
    def reset(self):
        self.build_env()

        # fixed cube position
        cube_pos = np.array([0.65, 0.0, 0.02])
        cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)


        states = []
        for cam in self.cams:
            frame = cam.render()[0]   # extract RGB image only  # shape (120,120,3), uint8
            img = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
            states.append(img)
        state = torch.stack(states, dim=0)  # (num_envs, 3, 120, 120)

        return state


    # Apply discrete actions → step physics → return (state, reward, done)
    def step(self, actions):
        action_mask_0 = actions == 0  # +x
        action_mask_1 = actions == 1  # –x
        action_mask_2 = actions == 2  # +y
        action_mask_3 = actions == 3  # –y
        action_mask_4 = actions == 4  # +z
        action_mask_5 = actions == 5  # –z

        # 1) start from last target
        pos = self.pos.clone()

        # 2) apply your δ moves
        pos[action_mask_0, 0] += 0.05
        pos[action_mask_1, 0] -= 0.05
        pos[action_mask_2, 1] += 0.05
        pos[action_mask_3, 1] -= 0.05
        pos[action_mask_4, 2] += 0.05
        pos[action_mask_5, 2] -= 0.05

        # 3) solve IK from that new pos + fixed quat
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos = pos,
            quat = self.quat, # keep the original quaternion value
        )


        # 4) command the arm joints, keep the gripper position constant, then step the sim
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof)# self.envs_idx)
        # 4.1) force fingers constant
        # Gravity might pull them down, so we re-send the same opening
        self.franka.control_dofs_position(
            self.fixed_finger_pos, self.fingers_dof, self.envs_idx
        )

        #link = self.franka.get_link("hand")
        #link_T = link.get_transform()  # torch.Tensor, shape: (num_envs, 4, 4)
        #print("link_T.shape=", link_T.shape)
        #print(link_T)  # will dump each 4×4 matrix
        ## then exit so you don’t spam your console forever…
        #import sys;
        #sys.exit(0)

        self.scene.step()


        # 5) observe & compute reward/done
        #self.cam.render() #view scene
        #frame = self.cam.render()[0]
        #img = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0

        object_position  = self.cube.get_pos()
        gripper_position = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link(
            "right_finger").get_pos()) / 2


        #states = img.repeat(self.num_envs, 1, 1, 1)  # shape: (num_envs, 3, 120, 120)  # unimodal states

        states = []
        for cam in self.cams:
            frame = cam.render()[0]
            img = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
            states.append(img)
        states = torch.stack(states, dim=0)  # (num_envs, 3, 120, 120)

        # --- CORRECTED EXPONENTIAL REWARD ---
        # reward = exp(-k * (dist - 0.1)), max=1.0 at dist=0.1, decays for larger distances
        dist = torch.norm(object_position - gripper_position, dim=1)
        reward = torch.exp(-4 * (dist - 0.1))
        rewards = torch.clamp(reward, min=0.0, max=1.0)

        # for a simple reach task you can set done=False always,
        # or drive resets in your training loop by episode length
        dones   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 6) save target for next step
        self.pos = pos

        return states, rewards, dones


# main guard, preventing this code from running on import (only run as executable if directly called as python ....py)
if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = GraspFixedCubeVisEnv(vis=True)
