#env/reach_cube_position_stacked.py


import numpy as np
import genesis as gs
import torch
from collections import deque

class ReachCubePositionStackedEnv:

    # Build the scene & entities
    def __init__(self, vis, device, num_envs=1, randomize_every=100):
        self.device     = device
        self.num_envs   = num_envs
        self.randomize_every = randomize_every
        # count how many episodes (resets) have occurred
        self.episode_count    = 0
        # counter for simulation steps within an episode
        self.step_count = 0
        # initial fixed cube position for reset logic; [None, :] adds a new first axis so shape is (1,3)
        self.initial_pos = np.array([0.65, 0.0, 0.1])[None, :]
        # will hold the cube position for the current batch of episodes
        self.current_cube_pos = None

        # state now includes 5 past gripper positions sampled over last 500ms (5*3 dims) + cube pos (3 dims)
        self.state_dim    = 5*3 + 3  # 18 total: 5 gripper poses (3D) + cube (3D)
        self.action_space = 6        # output dimension: ±x, ±y, ±z

        # HISTORY SETTINGS for gripper trajectory
        # simulation dt=0.01s => 100 Hz, so 500ms = 50 steps
        self.history_length = 10
        # sample at offsets: -40, -30, -20, -10, -1 (i.e. t-400, t-300, t-200, t-100, t)
        self.sample_offsets = [-9, -7, -5, -3, -1]
        #self.sample_offsets = [-1]
        # deque holds last 50 gripper positions, each shape [num_envs,3]
        self.gripper_history = deque(maxlen=self.history_length)

        self.scene = gs.Scene(
            show_FPS=False,  # Don't show simulation speed
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
        # create cube without initial pos; we'll place it in reset
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)

        # (Re)position robot to a default “ready” pose
        self.build_env()
        # set cube to initial position
        self.cube.set_pos(self.initial_pos.repeat(self.num_envs, axis=0), envs_idx=self.envs_idx)

    # (Re)position robot to a default “ready” pose
    def build_env(self):
        # only the 7 arm joints (ignore fingers now)
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        franka_pos = torch.tensor([
            -1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.01, 0.01
        ], dtype=torch.float32, device=self.device)
        franka_pos = franka_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(franka_pos, envs_idx=self.envs_idx)
        self.scene.step()

        # store fixed finger targets and end effector
        self.fixed_finger_pos = franka_pos[:, 7:9].clone()
        self.end_effector = self.franka.get_link("hand")

        # define fixed Cartesian target
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

    # helper to build the 18-dim state: 5 sampled gripper poses + cube
    def _build_state(self, cube_pos, current_gripper_pos):
        samples = []
        for offs in self.sample_offsets:
            samples.append(self.gripper_history[offs])
        gripper_seq = torch.cat(samples, dim=1)
        return torch.cat([gripper_seq, cube_pos], dim=1)

    # Reset cube position + robot → return initial state
    def reset(self):
        self.episode_count += 1
        self.step_count = 0
        # cube randomization logic
        if self.episode_count == 1:
            one_pos = self.initial_pos
        elif self.episode_count % self.randomize_every == 0:
            abs_xy = np.random.uniform(0.2, 1.0, size=(1, 2))
            signs = np.random.choice([-1.0, 1.0], size=(1, 2))
            xy = abs_xy * signs
            z = np.random.uniform(0.05, 1.0, size=(1, 1))
            one_pos = np.concatenate([xy, z], axis=1)
        else:
            one_pos = self.current_cube_pos[:1]
        self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)
        self.build_env()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

        obs1 = self.cube.get_pos()
        obs2 = (self.franka.get_link("left_finger").get_pos() +
                self.franka.get_link("right_finger").get_pos()) / 2

        # initialize history
        self.gripper_history.clear()
        for _ in range(self.history_length):
            self.gripper_history.append(obs2.clone())

        state = self._build_state(obs1, obs2)
        return state

    # Apply discrete actions → step physics → return (state, reward, done)
    def step(self, actions):
        self.step_count += 1
        pos = self.pos.clone()
        # apply deltas
        masks = [(actions == i) for i in range(6)]
        deltas = [(0, 0.05), (0, -0.05), (1, 0.05), (1, -0.05), (2, 0.05), (2, -0.05)]
        for mask, (idx, delta) in zip(masks, deltas):
            pos[mask, idx] += delta
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=pos, quat=self.quat)
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        object_position  = self.cube.get_pos()
        gripper_position = (
            self.franka.get_link("left_finger").get_pos() +
            self.franka.get_link("right_finger").get_pos()
        ) / 2

        # debug print only if single env and every 50th step
        if self.num_envs == 1 and self.step_count % 50 == 0:
            for offs in self.sample_offsets:
                sample = self.gripper_history[offs][0]
                print(sample[0].item())
                print(sample[1].item())
                print(sample[2].item())
                print('---')  # separator between samples

        self.gripper_history.append(gripper_position.clone())
        state = self._build_state(object_position, gripper_position)

        dist    = torch.norm(object_position - gripper_position, dim=1)
        reward  = torch.exp(-4 * (dist - 0.1))
        rewards = torch.clamp(reward, min=0.0, max=1.0)
        dones   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.pos = pos
        return state, rewards, dones

# main guard
if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubePositionStackedEnv(vis=True, device="cuda")
