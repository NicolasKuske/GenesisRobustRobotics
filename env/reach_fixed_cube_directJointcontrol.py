import numpy as np
import genesis as gs
import torch

class ReachFixedCubeDirectJointControlEnv:

    # Build the scene & entities
    def __init__(self, vis, device, num_envs=1):
        self.device    = device
        self.num_envs  = num_envs

        # state: [cube_xyz, gripper_xyz]
        self.state_dim    = 6
        # discrete index actions: 7 joints × (±0.05 rad) → 14 choices
        self.action_space = 14

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
        self.envs_idx     = np.arange(self.num_envs)
        self.motors_dof   = torch.arange(7, device=self.device)
        self.end_effector = self.franka.get_link("hand")

        # placeholders for action-repeat logic
        self.current_mask       = torch.zeros((self.num_envs, self.action_space), dtype=torch.bool, device=self.device)
        self.remaining_repeat   = 0

        self.reset()

    def reset(self):
        home_qpos = torch.tensor([
            -1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02
        ], dtype=torch.float32, device=self.device)
        home = home_qpos.unsqueeze(0).repeat(self.num_envs,1)
        self.franka.set_qpos(home, envs_idx=self.envs_idx)
        self.scene.step()

        cube_pos = np.array([0.65, 0.0, 0.02])[None,:]
        cube_pos = np.repeat(cube_pos, self.num_envs, axis=0)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)

        # reset repeat
        self.remaining_repeat = 0
        self.current_mask.zero_()

        obs1 = self.cube.get_pos()
        obs2 = self.end_effector.get_pos()
        return torch.concat([obs1, obs2], dim=1)

    def step(self, actions):
        """
        actions: LongTensor of shape (num_envs,) or (1, num_envs), each in [0..13].
        Discrete choice selects joint (idx//2) and direction (even:+, odd:-).
        Repeats last valid multi-hot mask for 5 steps before accepting new input.
        """
        # accept action indices
        idx = actions.to(self.device).long()
        if idx.ndim == 2 and idx.shape == (1, self.num_envs):
            idx = idx.squeeze(0)
        if idx.ndim != 1 or idx.shape[0] != self.num_envs:
            raise ValueError(f"Expected actions of shape ({self.num_envs},), got {tuple(idx.shape)}")

        # build new mask from indices
        new_mask = torch.nn.functional.one_hot(idx, num_classes=self.action_space).bool()

        # action-repeat
        if self.remaining_repeat > 0:
            use_mask = self.current_mask
            self.remaining_repeat -= 1
        else:
            # enforce per-joint exclusivity
            use_mask = new_mask.clone()
            for j in range(7):
                pos, neg = 2*j, 2*j+1
                conflict = use_mask[:,pos] & use_mask[:,neg]
                use_mask[conflict,neg] = False
            self.current_mask = use_mask
            self.remaining_repeat = 4  # current + 4 more repeats = 5

        # apply deltas
        qpos_full = self.franka.get_qpos(envs_idx=self.envs_idx)
        joints     = qpos_full[:, :7].clone()
        for j in range(7):
            pos, neg = 2*j, 2*j+1
            joints[use_mask[:,pos], j] += 0.05
            joints[use_mask[:,neg], j] -= 0.05

        # command & step
        self.franka.control_dofs_position(joints, self.motors_dof, self.envs_idx)
        self.scene.step()

        # observe & reward
        obj_pos  = self.cube.get_pos()
        grip_pos = self.end_effector.get_pos()
        state    = torch.concat([obj_pos, grip_pos], dim=1)
        dist     = torch.norm(obj_pos - grip_pos, dim=1)
        reward   = torch.exp(-4 * (dist - 0.1))
        rewards  = torch.clamp(reward, 0.0, 1.0)
        dones    = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return state, rewards, dones

if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachFixedCubeDirectJointControlEnv(vis=True, device=torch.device('cuda'), num_envs=1)
