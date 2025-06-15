import numpy as np
import genesis as gs
import torch

class GraspFixedCubeEnv:

    # Build the scene & entities
    def __init__(self, vis, device, num_envs=1):
        self.device = device

        self.state_dim = 6  # input dimension of the agent, here xyz position of object and gripper

        self.action_space = 8  #output dimension of the agent,



        self.scene = gs.Scene(
            show_FPS=False, # Don't show simulation speed
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 2, 1.5), #x, y , z
                camera_lookat=(0.0, 0.0, 0.5), #
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.01,
            ),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
            ),
            show_viewer=vis,

        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(0.65, 0.0, 0.02),
            )
        )
        self.num_envs = num_envs
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)
        self.build_env()


    # (Re)position robot to a default “ready” pose
    def build_env(self):
        self.motors_dof = torch.arange(7).to(self.device) #7 joints (for the Franka arm)
        self.fingers_dof = torch.arange(7, 9).to(self.device) #2 gripper positions

        #Joint limits ≈ [-2.8973, 2.8973] radians (for most joints)
        franka_pos = torch.tensor([-1.0, #foot yaw - rotation around z-axis of the foot. -1 is foot directed towards x-axis. Decreasing values with clock (0 is y-axis)
                                   -0.3, #lower arm segment pitch - rotation around x-or y-axis depending on foot position. I.e., lowest link deciding if arm is high or low. 0.3 is arm close to straight upward. 1.5 is parallel to ground.
                                   0.3, #upper arm segment yaw - rotation around z-direction of lower arm segment. Given above values: 1.3 directed towards x. Decreasing values with clock (0 is negative y-axis, given above values)
                                   -1.0, #upper arm segment pitch - Decreasing values with clock, delta 0.7 approx 45°
                                   -0.1, #head yaw - Decreasing values with clock, delta 0.7 approx 45°
                                   1.7, #head pitch - Increasing values with clock, delta 0.7 approx 45°
                                   1.0, #hand roll - Decreasing values with clock
                                   0.02, #left and right gripper distance to middle
                                   0.02]).to(self.device)
        franka_pos = franka_pos.unsqueeze(0).repeat(self.num_envs, 1) 
        self.franka.set_qpos(franka_pos, envs_idx=self.envs_idx)
        self.scene.step()

        self.end_effector = self.franka.get_link("hand")

        ## here self.pos and self.quat is target for the end effector; not the cube. cube position is set in reset()
        #pos = torch.tensor([0.3, 0.0, 0.7], dtype=torch.float32, device=self.device)
        #self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        #quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
        #self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        #self.qpos = self.franka.inverse_kinematics(
        #    link=self.end_effector,
        #    pos = self.pos,
        #    quat = self.quat,
        #)
        #self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)


    # Reset cube position + robot → return initial state
    def reset(self):
        self.build_env()
        # fixed cube position
        cube_pos = np.array([0.65, 0.0, 0.02])
        cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)

        obs1 = self.cube.get_pos()
        obs2 = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2 
        state = torch.concat([obs1, obs2], dim=1)
        return state


    # Apply discrete actions → step physics → return (state, reward, done)
    def step(self, actions):
        action_mask_0 = actions == 0 # Open gripper
        action_mask_1 = actions == 1 # Close gripper
        action_mask_2 = actions == 2 # Lift gripper
        action_mask_3 = actions == 3 # Lower gripper
        action_mask_4 = actions == 4 # Move left
        action_mask_5 = actions == 5 # Move right
        action_mask_6 = actions == 6 # Move forward
        action_mask_7 = actions == 7 # Move backward

        finger_pos = torch.full((self.num_envs, 2), 0.04, dtype=torch.float32, device=self.device)
        finger_pos[action_mask_1] = 0
        finger_pos[action_mask_2] = 0
        
        pos = self.pos.clone()
        pos[action_mask_2, 2] = 0.4
        pos[action_mask_3, 2] = 0
        pos[action_mask_4, 0] -= 0.05
        pos[action_mask_5, 0] += 0.05
        pos[action_mask_6, 1] -= 0.05
        pos[action_mask_7, 1] += 0.05

        self.pos = pos
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=pos,
            quat=self.quat,
        )

        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        object_position = self.cube.get_pos()
        gripper_position = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2
        states = torch.concat([object_position, gripper_position], dim=1)

        rewards = -torch.norm(object_position - gripper_position, dim=1) + torch.maximum(torch.tensor(0.02), object_position[:, 2]) * 10
        dones = object_position[:, 2] > 0.35
        return states, rewards, dones



#main guard, preventing the init and env to be called when the module is imported as module instead of run alone (as "main" script)
if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = GraspFixedCubeEnv(vis=True)