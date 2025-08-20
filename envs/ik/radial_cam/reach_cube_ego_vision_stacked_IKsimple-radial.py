# envs/ik/reach_cube_ego_video_stacked_IKsimple.py

import numpy as np
import genesis as gs
import torch
import math
import matplotlib.pyplot as plt
from collections import deque
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat

class ReachCubeEgoVisionStackedEnv:
    def __init__(self, vis, device, num_envs=1, randomize_every=100):
        self.device = device
        self.num_envs = num_envs
        self.randomize_every = randomize_every
        self.episode_count = 0
        self.step_count = 0

        # history settings
        self.history_length = 25
        self.sample_offsets = [-21, -16, -11, -6, -1]
        self.image_history = deque(maxlen=self.history_length)
        self.render_every = 5
        self._step_count = 0

        # plotting of cameraframes
        self.fig, self.axes = plt.subplots(2, 3, figsize=(8, 8))
        plt.ion()  # Turn interactive mode on
        plt.show()

        self.initial_pos = np.array([-0.9, 0.6, 0.7])[None, :]
        self.current_cube_pos = None

        self.obs_shape = (3 * len(self.sample_offsets), 120, 120)
        self.action_space = 6

        # Genesis scene setup
        self.scene = gs.Scene(show_FPS=False,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 2, 1.5),
                camera_lookat=(0.0, 0.0, 0.2),
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

        # Add reflecting plane
        self.scene.add_entity(
            gs.morphs.Plane(),
            surface=gs.surfaces.Aluminium(ior=10.0)
        )

        # Add walls
        for pos, color, euler in [
            ((4,  0, 1),  (0.9, 0.9, 0.9),  (0, -20,  0)),
            ((-3, 0, 1),  (0.7, 0.7, 0.7),  (0,  20,  0)),
            ((0, -3, 1),  (0.56,0.57,0.58), (0,  20, 90)),
        ]:
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.1, 8, 4),
                    pos=pos,
                    euler=euler,
                    collision=False
                ),
                surface=gs.surfaces.Rough(color=color),
                material=gs.materials.Rigid(gravity_compensation=1.0)
            )

        # Robot and cube
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="assets/xml/franka_emika_panda/panda.xml")
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06,0.06,0.06)),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # Camera setup
        self.cams = []
        for _ in range(self.num_envs):
            cam = self.scene.add_camera(res=(120,120), fov=90, GUI=True)
            self.cams.append(cam)

        # Build parallel environments
        env_space = 100.0
        self.scene.build(n_envs=self.num_envs, env_spacing=(env_space, env_space))
        self.envs_idx = np.arange(self.num_envs)

        for cam in self.cams:
            cam.start_recording()

        self._init_robot()


    def _init_robot(self):
        self.motors_dof = torch.arange(7, device=self.device)
        self.fingers_dof = torch.arange(7,9, device=self.device)

        q0 = torch.tensor(
            [-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:,7:9].clone()
        self.end_effector = self.franka.get_link("hand")

        base_pos = torch.tensor([0.2720, -0.1683, 1.0164], device=self.device)
        self.pos = base_pos.unsqueeze(0).repeat(self.num_envs, 1)
        base_quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], device=self.device)
        self.quat = base_quat.unsqueeze(0).repeat(self.num_envs, 1)

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:,:-2], self.motors_dof, self.envs_idx)

    def _render(self):
        imgs = []
        e = 0.1  # outward offset distance
        M = int(math.sqrt(self.num_envs))
        env_space = 100.0

        for idx, cam in enumerate(self.cams):
            # Current end-effector position
            ee_pos = self.end_effector.get_pos(envs_idx=[idx])[0].cpu().numpy()

            # Multi-environment offset calculation
            col = idx // M
            row = idx % M
            x_off = (col - (M - 1) / 2) * env_space
            y_off = (row - (M - 1) / 2) * env_space
            ee_pos_offset = ee_pos + np.array([x_off, y_off, 0.0])

            # IMPORTANT FIX HERE:
            # Compute radial direction relative to local environment center (x_off, y_off)
            radial_xy = ee_pos_offset[:2] - np.array([x_off, y_off])
            norm = np.linalg.norm(radial_xy)

            if norm < 1e-6:
                radial_dir = np.array([1.0, 0.0])
            else:
                radial_dir = radial_xy / norm

            # Camera position outward along radial direction
            cam_xy = radial_xy + e * radial_dir + np.array([x_off, y_off])
            cam_z = ee_pos_offset[2]  # same height as end-effector
            cam_pos = np.array([cam_xy[0], cam_xy[1], cam_z])

            # Yaw to face outward along radial direction
            yaw = np.arctan2(radial_dir[1], radial_dir[0]) - np.pi / 2
            pitch = np.deg2rad(0)
            roll = 45.5  # Adjust roll as desired

            cam_quat = xyz_to_quat(np.rad2deg(np.array([roll, pitch, yaw])))
            cam_T = trans_quat_to_T(cam_pos, cam_quat)

            # Set camera pose
            cam.set_pose(transform=cam_T)

            # Render frame
            rgb = cam.render()[0]
            img = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
            imgs.append(img)

        return torch.stack(imgs, dim=0)

    def _build_observation(self):
        samples = [self.image_history[i] for i in self.sample_offsets]
        return torch.cat(samples, dim=1)

    def reset(self):
        self.episode_count += 1
        # determine cube pos
        if self.episode_count == 1:
            one_pos = self.initial_pos
        elif self.episode_count % self.randomize_every == 0:
            abs_xy = np.random.uniform(0.2, 1.0, (1,2))
            signs = np.random.choice([-1.0,1.0], (1,2))
            xy = abs_xy * signs
            z = np.random.uniform(0.1, 1.0, (1,1))
            one_pos = np.concatenate([xy, z], axis=1)
        else:
            one_pos = self.current_cube_pos[:1]

        self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

        # prime history
        self.image_history.clear()
        first = self._render()
        for _ in range(self.history_length):
            self.image_history.append(first.clone())

        # reset frame-skip counter
        self._step_count = 0

        return self._build_observation()

    def step(self, actions):
        d_radial = 0.05  # radial step distance
        d_angle = np.deg2rad(5)  # angular step size
        d_z = 0.05  # vertical step size
        min_radius = 0.05  # Minimum allowed radius from local center

        pos = self.pos.clone()

        M = int(math.sqrt(self.num_envs))
        env_space = 100.0

        for idx in range(self.num_envs):
            action = actions[idx].item()

            ee_pos = pos[idx].cpu().numpy()

            col = idx // M
            row = idx % M
            x_off = (col - (M - 1) / 2) * env_space
            y_off = (row - (M - 1) / 2) * env_space
            local_origin = np.array([x_off, y_off])

            radial_xy = ee_pos[:2] - local_origin
            radial_norm = np.linalg.norm(radial_xy)

            # Handle edge-case (near-origin)
            if radial_norm < 1e-6:
                radial_dir = np.array([1.0, 0.0])
            else:
                radial_dir = radial_xy / radial_norm

            tangent_dir = np.array([-radial_dir[1], radial_dir[0]])

            # Apply actions with radius check
            if action == 0:  # Forward along radial
                ee_pos[:2] += radial_dir * d_radial
            elif action == 1:  # Backward along radial, but prevent crossing local center
                new_radial_norm = radial_norm - d_radial
                if new_radial_norm < min_radius:
                    new_radial_norm = min_radius  # stop at minimum radius
                ee_pos[:2] = local_origin + radial_dir * new_radial_norm
            elif action == 2:  # Rotate left around center
                angle = d_angle
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                ee_pos[:2] = local_origin + rot_matrix @ radial_xy
            elif action == 3:  # Rotate right around center
                angle = -d_angle
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                ee_pos[:2] = local_origin + rot_matrix @ radial_xy
            elif action == 4:  # Up
                ee_pos[2] += d_z
            elif action == 5:  # Down
                ee_pos[2] -= d_z

            pos[idx] = torch.from_numpy(ee_pos).to(self.device)

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        if self._step_count % self.render_every == 0:
            new_frame = self._render()
        else:
            new_frame = self.image_history[-1]

        self._step_count += 1
        self.image_history.append(new_frame)

        obs = self._build_observation()

        # plot the not_stacked frames
        if self.num_envs == 1 and self.step_count % 100 == 0:
            frames = obs[0].cpu().numpy().reshape(len(self.sample_offsets), 3, 120, 120)

            for i, ax in enumerate(self.axes.flatten()):
                if i < len(self.sample_offsets):
                    img = np.transpose(frames[i], (1, 2, 0))
                    ax.clear()
                    ax.imshow(img)
                    ax.axis('off')
                else:
                    ax.clear()
                    ax.axis('off')

            self.fig.suptitle(f"Stacked frames at step {self.step_count}")
            self.fig.canvas.draw_idle()  # Efficient redraw of canvas
            self.fig.canvas.flush_events()  # Ensure GUI responsiveness

        obj_pos = self.cube.get_pos()
        gp_l = self.franka.get_link("left_finger").get_pos()
        gp_r = self.franka.get_link("right_finger").get_pos()
        dist = torch.norm(obj_pos - (gp_l + gp_r) / 2, dim=1)
        rewards = torch.clamp(torch.exp(-4 * (dist - 0.1)), 0.0, 1.0)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.pos = pos
        return obs, rewards, dones


if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoVisionStackedEnv(vis=True, device=torch.device("cuda"))
    obs = env.reset()
    for _ in range(200):
        actions = torch.randint(0, 6, (env.num_envs,), device=env.device)
        obs, rewards, dones = env.step(actions)
        if dones.any():
            print("Done!", dones)
            break
