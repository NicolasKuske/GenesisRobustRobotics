# reach_random_cube_pos.py

import numpy as np
import genesis as gs
import torch

class ReachRandomCubePosEnv:

    # Build the scene & entities
    def __init__(self, vis, device, num_envs=1):
        self.device     = device
        self.num_envs   = num_envs

        # count how many episodes (resets) have occurred
        self.episode_count    = 0
        # will hold the cube position for the current batch of 10 episodes
        self.current_cube_pos = None

        self.state_dim    = 6  # input dimension: [cube_xyz, gripper_xyz]
        self.action_space = 6  # output dimension: ±x, ±y, ±z

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
                size=(0.1, 0.1, 0.1),  # x,y,z dimensions in meters
                pos=(0.35, 0.0, 1.18),  # initial position in world coords
            ),
            #this one is blind so colors dont matter
            #surface=gs.surfaces.Rough(
            #    color=(0.7, 0, 0),
            #),
            material=gs.materials.Rigid(gravity_compensation=1.0)  # ,
            # surface=gs.surfaces.Metal(double_sided=False, metal_type='copper', metallic=100, color=[206,112,43])
        )

        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)

        # (Re)position robot to a default “ready” pose
        self.build_env()


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
             0.01, # left and right gripper distance to middle
             0.01  # (ignored)
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
        pos = torch.tensor([ 0.2720, -0.1683,  1.0164], dtype=torch.float32, device=self.device)
        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        quat = torch.tensor([ 0.1992,  0.7857, -0.3897,  0.4371], dtype=torch.float32, device=self.device)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=self.pos,
            quat=self.quat,
        )
        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)


    # Reset cube position + robot → return initial state
    def reset(self):
        self.episode_count += 1

        # every new batch of 10 episodes:
        if (self.episode_count - 1) % 10 == 0:
            # sample a single random pos
            abs_xy = np.random.uniform(0.2, 1.0, size=(1, 2))
            signs = np.random.choice([-1.0, 1.0], size=(1, 2))
            xy = abs_xy * signs  # shape (1,2)
            z = np.random.uniform(0.05, 1.0, size=(1, 1))
            one_pos = np.concatenate([xy, z], axis=1)  # shape (1,3)

            # broadcast to all envs
            self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)  # (num_envs,3)

        # rebuild robot, and same cube for all envs.
        self.build_env()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

        obs1 = self.cube.get_pos()
        obs2 = (self.franka.get_link("left_finger").get_pos() +
                self.franka.get_link("right_finger").get_pos()) / 2
        state = torch.concat([obs1, obs2], dim=1)
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
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)

        # 4.1) FORCE FINGERS CONSTANT ===
        # Gravity might pull them down, so we re-send the same opening
        self.franka.control_dofs_position(
            self.fixed_finger_pos, self.fingers_dof, self.envs_idx
        )
        self.scene.step()

        # 5) observe & compute reward/done
        object_position  = self.cube.get_pos()
        gripper_position = (
            self.franka.get_link("left_finger").get_pos() +
            self.franka.get_link("right_finger").get_pos()
        ) / 2
        states = torch.concat([object_position, gripper_position], dim=1)

        # --- CORRECTED EXPONENTIAL REWARD ---
        # reward = exp(-k * (dist - 0.1)), max=1.0 at dist=0.1, decays for larger distances
        dist    = torch.norm(object_position - gripper_position, dim=1)
        reward  = torch.exp(-4 * (dist - 0.1))
        rewards = torch.clamp(reward, min=0.0, max=1.0)

        # for a simple reach task you can set done=False always,
        # or drive resets in your training loop by episode length
        dones   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 6) save target for next step
        self.pos = pos

        return states, rewards, dones


# main guard, preventing this code from running on import
if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachRandomCubePosEnv(vis=True, device="cuda")
