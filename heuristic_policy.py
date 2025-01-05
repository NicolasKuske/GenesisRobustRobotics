import argparse
import numpy as np
import time
import genesis as gs
import torch
from env.env import Env

gs.init(backend=gs.gpu, precision="32")

class CheckSuccess:
    def __init__(self):
        self.time = 0

    def is_block_in_gripper(self, block_position, gripper_position, threshold=0.02):
        distance = torch.norm(block_position - gripper_position)
        return distance < threshold

    def check_success(self, block_position, gripper_position):
        if block_position[2] > 0.15 and self.is_block_in_gripper(block_position, gripper_position, 0.5):
            self.time += 1
            print("Block successfully lifted!")
        return self.time > 10

class Rollout:
    def __init__(self, args):
        self.finger_pos = 0.0
        self.env = Env(args.vis)
        self.checker = CheckSuccess()
    
    def rollout(self):
        for i in range(20):
            print("grasp", i)
            self.env.franka.control_dofs_position(self.env.qpos[:-2], self.env.motors_dof)
            self.env.franka.control_dofs_position(np.array([self.finger_pos, self.finger_pos]), self.env.fingers_dof)
            self.env.scene.step()

            block_position = self.env.cube.get_pos() 
            gripper_position = (self.env.franka.get_link("left_finger").get_pos() \
                                + self.env.franka.get_link("right_finger").get_pos()) / 2
            
            if self.checker.is_block_in_gripper(block_position, gripper_position, 0.051):
                print("Block successfully picked up!")
                break
        else:
            print("Failed to pick up the block within 20 attempts.")

        # lift
        qpos_lift = self.env.franka.inverse_kinematics(
            link=self.env.end_effector,
            pos=np.array([0.65, 0.0, 0.3]),
            quat=np.array([0, 1, 0, 0]),
        )

        for i in range(100):
            print("lift", i)
            self.env.franka.control_dofs_position(qpos_lift[:-2], self.env.motors_dof)
            self.env.franka.control_dofs_position(np.array([self.finger_pos, self.finger_pos]), self.env.fingers_dof)
            self.env.scene.step()

            block_position_after_lift = self.env.cube.get_pos()  
            gripper_position = (self.env.franka.get_link("left_finger").get_pos() + self.env.franka.get_link("right_finger").get_pos()) / 2
            if self.checker.check_success(block_position_after_lift, gripper_position):
                break
        else:
            print("Failed to lift the block within 100 steps.")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    rollout = Rollout(args)
    rollout.rollout()
