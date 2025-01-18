import argparse
import numpy as np
import genesis as gs
import torch

class ShadowHandBaseEnv:
    def __init__(self, vis, device, num_envs=1):
        self.device = device
        self.action_space = 30
        self.state_dim = 78

        ########################## create a scene ##########################
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.5, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=vis,
            rigid_options=gs.options.RigidOptions(
                gravity=(0, 0, 0),
                enable_collision=False,
                enable_joint_limit=False,
            ),
        )
        ########################## entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04), # block
                pos=(0.65, 0.0, 0.02),
            )
        )
        
        self.shadow_hand = self.scene.add_entity(
            morph=gs.morphs.URDF(
                scale=1.0,
                file="urdf/shadow_hand/shadow_hand.urdf",
            ),
            surface=gs.surfaces.Reflective(color=(0.4, 0.4, 0.4)),
        )

        ########################## build ##########################
        self.num_envs = num_envs
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)
        self.scene.reset()
        self.dofs_limit = self.shadow_hand.get_dofs_limit()
        '''
        (tensor([   -inf,    -inf,    -inf,    -inf,    -inf,    -inf, -0.5236, -0.6981,
        -1.0472, -0.3491, -0.3491, -0.3491,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000, -0.3491, -0.2094,  0.0000,  0.0000,  0.0000,  0.0000, -0.6981,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000], device='mps:0'), tensor([   inf,    inf,    inf,    inf,    inf,    inf, 0.1745, 0.4887, 1.0472,
        0.3491, 0.3491, 0.3491, 0.7854, 1.2217, 1.5708, 1.5708, 1.5708, 0.3491,
        0.2094, 1.5708, 1.5708, 1.5708, 1.5708, 0.6981, 1.5708, 1.5708, 1.5708,
        1.5708, 1.5708, 1.5708], device='mps:0'))
        '''
       
    def build_env(self):
        pass
    
    def reset(self):
        self.build_env()
        # fixed cube position
        cube_pos = np.array([0.65, 0.0, 0.02])
        cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)
        states = self.get_obs()
        return states
    
    def step(self, actions):
        current_pos = self.shadow_hand.get_dofs_position() # shape: [30]
        pos_target = current_pos + actions
        self.shadow_hand.control_dofs_position(pos_target, torch.arange(current_pos.shape[-1]).to(self.device), self.envs_idx)
        self.scene.step()
        states = self.get_obs()
        rewards = self.get_rewards()
        dones = self.get_dones()
        return states, rewards, dones
    
    def get_obs(self):
        obs1 = self.cube.get_pos()
        obs2 = self.shadow_hand.get_links_pos().view(self.num_envs, -1)
        states = torch.concat([obs1, obs2], dim=1)
        return states

    def get_dones(self):
        return torch.BoolTensor([self.num_envs]).to(self.device)

    def get_rewards(self):
        return torch.zeros([self.num_envs], device=self.device)

if __name__ == "__main__":
    gs.init(seed=0, precision="32", logging_level="debug")
    device = "cuda" if torch.cuda.is_available() else "mps"
    env = ShadowHandBaseEnv(vis=True, device=device)
    obs = env.reset()
    for i in range(1000):
        obs, rewards, dones = env.step(torch.zeros([1, 30], device=device))
