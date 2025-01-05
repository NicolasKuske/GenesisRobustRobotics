from .env import Env
import numpy as np
import torch
class WrappedEnv(Env):
    def __init__(self, vis, num_envs):
        super().__init__(vis, num_envs)
        self.action_space = 8  
        self.state_dim = 6  

    def reset(self):
        self.build_env()
        cube_pos = np.array([0.65, 0.0, 0.02])
        cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)
        obs1 = self.cube.get_pos()
        obs2 = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2 
        state = torch.concat([obs1, obs2], dim=1)
        return state

    def step(self, actions):
        action_mask_0 = actions == 0 # Open gripper
        action_mask_1 = actions == 1 # Close gripper
        action_mask_2 = actions == 2 # Lift gripper
        action_mask_3 = actions == 3 # Lower gripper
        action_mask_4 = actions == 4 # Move left
        action_mask_5 = actions == 5 # Move right
        action_mask_6 = actions == 6 # Move forward
        action_mask_7 = actions == 7 # Move backward

        finger_pos = torch.full((self.num_envs, 2), 0.04, dtype=torch.float32, device='cuda:0')
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

        block_position = self.cube.get_pos()
        gripper_position = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2
        states = torch.concat([block_position, gripper_position], dim=1)

        rewards = -torch.norm(block_position - gripper_position, dim=1) + torch.maximum(torch.tensor(0.02), block_position[:, 2]) * 10
        dones = block_position[:, 2] > 0.35
        return states, rewards, dones
