# reach_cube_vision.py

import numpy as np
import genesis as gs
import torch
#torch.backends.nnpack.enabled = False
import math
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat

class ReachCubeVisionEnv:

    def __init__(self, vis, device, num_envs=1, randomize_every=2):
        self.device = device
        self.num_envs = num_envs
        self.randomize_every = randomize_every
        self.episode_count = 0
        # initial fixed cube position
        self.initial_pos = np.array([0.65, 0.0, 0.1])[None, :]
        self.current_cube_pos = None

        # observation and action dimensions
        self.obs_shape = (3, 120, 120)
        self.action_space = 6

        # set up scene
        self.scene = gs.Scene(
            #show_FPS=False,
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
        )
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"),
        )
        # target cube with features from ego version
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06)),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # cameras: fixed overhead per env
        self.cams = []
        env_space = 5.0
        M = int(math.sqrt(self.num_envs))
        assert M * M == self.num_envs, "num_envs must be perfect square"
        for idx in range(self.num_envs):
            row, col = divmod(idx, M)
            x_off = (col - (M-1)/2) * env_space
            y_off = (row - (M-1)/2) * env_space
            cam = self.scene.add_camera(
                res=(120, 120),
                pos=(2.5 + x_off, 0.5 + y_off, 3.5),
                lookat=(x_off, y_off, 0.35),
                fov=30,
                GUI=True,
            )
            self.cams.append(cam)

        self.scene.build(n_envs=self.num_envs, env_spacing=(env_space, env_space))
        self.envs_idx = np.arange(self.num_envs)

        for cam in self.cams:
            cam.start_recording()

        # initialize robot and cube position
        self._init_robot()

    def _init_robot(self):
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        # default joint configuration
        q0 = torch.tensor(
            [-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:, 7:9].clone()
        self.end_effector = self.franka.get_link("hand")
        # fixed Cartesian target
        base_pos = torch.tensor([0.2720, -0.1683, 1.0164], device=self.device)
        base_quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], device=self.device)
        self.pos = base_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.quat = base_quat.unsqueeze(0).repeat(self.num_envs, 1)
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)

    def _collect_states(self):
        states = []
        for cam in self.cams:
            rgb = cam.render()[0]
            img = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
            states.append(img)
        return torch.stack(states, dim=0)

    def reset(self):
        self.episode_count += 1
        # determine cube pos
        if self.episode_count == 1:
            one_pos = self.initial_pos
        elif self.episode_count % self.randomize_every == 0:
            abs_xy = np.random.uniform(0.2, 1.0, size=(1, 2))
            signs = np.random.choice([-1.0, 1.0], size=(1, 2))
            xy = abs_xy * signs
            z = np.random.uniform(0.1, 1.0, size=(1, 1))
            one_pos = np.concatenate([xy, z], axis=1)
        else:
            one_pos = self.current_cube_pos[:1]
        self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        return self._collect_states()

    def step(self, actions):
        masks = [actions == i for i in range(6)]
        pos = self.pos.clone()
        pos[masks[0], 0] += 0.05; pos[masks[1], 0] -= 0.05
        pos[masks[2], 1] += 0.05; pos[masks[3], 1] -= 0.05
        pos[masks[4], 2] += 0.05; pos[masks[5], 2] -= 0.05
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()
        states = self._collect_states()
        obj_pos = self.cube.get_pos()
        gp_l = self.franka.get_link("left_finger").get_pos()
        gp_r = self.franka.get_link("right_finger").get_pos()
        dist = torch.norm(obj_pos - (gp_l + gp_r) / 2, dim=1)
        rewards = torch.clamp(torch.exp(-4 * (dist - 0.1)), 0.0, 1.0)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pos = pos
        return states, rewards, dones

if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeVisionEnv(vis=True, device=torch.device("cuda"))
