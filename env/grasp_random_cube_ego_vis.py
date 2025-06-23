# grasp_random_cube_ego_vis.py


import numpy as np
import genesis as gs
import torch
import math

class GraspRandomCubeEgoVisEnv:

    # Build the scene & entities
    def __init__(self, vis, device, num_envs=1):
        self.device     = device
        self.num_envs   = num_envs

        # count how many episodes (resets) have occurred
        self.episode_count = 0
        # will hold the cube position & color for the current batch of 10 episodes
        self.current_cube_pos = None
        self.current_color = None

        # now our “state” is the camera image:
        # we’re using a 120×120 RGB camera
        self.obs_shape = (3, 120, 120)
        self.action_space = 6

        self.scene = gs.Scene(
            #show_FPS=False,  # Don't show simulation speed
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 2, 1.5),        # x, y, z
                camera_lookat=(0.0, 0.0, 0.2), # focus point
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
                pos=(0.35, 0.0, 1.18),  # initial position in world coords (will be overwritten)
            ),
            # this one can see, so colors matter
            surface=gs.surfaces.Rough(color=(0.2, 0.2, 0.8)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )


        ##### cam #### attaching cams does not work for parallelized scenes yet, so we manually link cam to arm position below
        self.cams = []
        env_space = 5.0  # must match your scene.build(env_spacing=(5.0,5.0))

        M = int(math.sqrt(self.num_envs))
        assert M * M == self.num_envs, "num_envs must be a perfect square for an M×M grid"

        for idx in range(self.num_envs):

            cam = self.scene.add_camera(
                res=(120, 120),
                fov=90,
                GUI=True,
            )
            self.cams.append(cam)

        self.scene.build(n_envs=self.num_envs, env_spacing=(env_space, env_space)) #only space envs in x direction
        self.envs_idx = np.arange(self.num_envs)

        # (Re)position robot to a default “ready” pose
        self.build_env()

        # start recording
        for cam in self.cams:
            cam.start_recording()


    # (Re)position robot to a default “ready” pose
    def build_env(self):
        # only the 7 arm joints (ignore fingers now)
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        #Joint limits ≈ [-2.8973, 2.8973] radians (for most joints)
        franka_pos = torch.tensor([
            -1.0,  # foot yaw - rotation around z-axis of the foot. -1 is foot directed towards x-axis.
            -0.3,  # lower arm segment pitch - arm close to straight upward.
             0.3,  # upper arm segment yaw - directed towards x-axis.
            -1.0,  # upper arm segment pitch.
            -0.1,  # head yaw.
             1.7,  # head pitch.
             1.0,  # hand roll.
             0.02, # left and right gripper distance.
             0.02  # (ignored)
        ], dtype=torch.float32, device=self.device)
        franka_pos = franka_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(franka_pos, envs_idx=self.envs_idx)
        self.scene.step()

        # === STORE FIXED FINGER TARGETS ===
        self.fixed_finger_pos = franka_pos[:, 7:9].clone()
        # Identify end‐effector link for IK calls
        self.end_effector = self.franka.get_link("hand")

        # Initialize pos & quat for IK reset targets
        pos = torch.tensor([0.2720, -0.1683, 1.0164], dtype=torch.float32, device=self.device)
        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], dtype=torch.float32, device=self.device)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)

        # Solve IK to initialize
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=self.pos,
            quat=self.quat,
        )
        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)


    def _collect_states(self):
        """Re-position all wrist cameras, render RGB, and stack into a tensor."""
        states = []
        M = int(math.sqrt(self.num_envs))
        assert M * M == self.num_envs, "num_envs must be a perfect square for an M×M grid"
        env_space = 5.0
        link = self.franka.get_link("hand")

        for idx, cam in enumerate(self.cams):
            # a) Get world-space hand pos & quat
            pos_i = link.get_pos(envs_idx=[idx])[0]
            qx, qy, qz, qw = link.get_quat(envs_idx=[idx])[0].unbind(0)

            # b) Compute forward = local +Z via quaternion
            forward = torch.tensor([
                2 * (qx * qz + qw * qy),
                2 * (qy * qz - qw * qx),
                1 - 2 * (qx * qx + qy * qy)
            ], device=pos_i.device)

            # c) Compute grid offset
            col = idx // M
            row = idx % M
            x_off = (col - (M - 1) / 2) * env_space
            y_off = (row - (M - 1) / 2) * env_space
            offset = torch.tensor([x_off, y_off, 0.0], device=pos_i.device)

            # d) Place & aim camera
            cam_pos = pos_i + 0.1 * forward + offset
            lookat  = (pos_i + forward + offset).cpu().numpy()
            cam.set_pose(pos=cam_pos.cpu().numpy(), lookat=lookat)

            # e) Render and collect
            rgb = cam.render()[0]  # (120,120,3) uint8
            img = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
            states.append(img)

        return torch.stack(states, dim=0)  # (num_envs, 3, 120, 120)



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

        # …render & return state
        return self._collect_states()



    def step(self, actions):
        """
        Apply discrete ±x/±y/±z moves to the end-effector via IK,
        step the sim, re-position the wrist cameras (with corrected grid offsets),
        render RGB observations, compute rewards, and return (states, rewards, dones).
        """
        # 1) Decode actions into boolean masks for each of the 6 directions
        masks = [actions == i for i in range(6)]  # 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z

        # 2) Start from last target self.pos and apply ±5cm moves
        pos = self.pos.clone()
        pos[masks[0], 0] += 0.05
        pos[masks[1], 0] -= 0.05
        pos[masks[2], 1] += 0.05
        pos[masks[3], 1] -= 0.05
        pos[masks[4], 2] += 0.05
        pos[masks[5], 2] -= 0.05

        # 3) Solve IK for the new end-effector targets
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=pos,
            quat=self.quat,
        )

        # 4) Command arm joints + keep fingers fixed
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)

        # 5) Step the physics simulator
        self.scene.step()

        # 6) Re-compute camera poses & render new images & return state
        states =  self._collect_states()

        # 7) Compute reward: exponential distance between cube and gripper midpoint
        obj_pos = self.cube.get_pos()
        gp_l = self.franka.get_link("left_finger").get_pos()
        gp_r = self.franka.get_link("right_finger").get_pos()
        gripper_pos = (gp_l + gp_r) / 2
        dist = torch.norm(obj_pos - gripper_pos, dim=1)
        rewards = torch.clamp(torch.exp(-4 * (dist - 0.1)), 0.0, 1.0)

        # 8) Never done for this reach task
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 9) Store for next step
        self.pos = pos

        return states, rewards, dones


# main guard
if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = GraspRandomCubeEgoVisEnv(vis=True)
