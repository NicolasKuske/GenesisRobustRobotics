import numpy as np
import genesis as gs
import torch

class Env:
    def __init__(self, vis, num_envs=1):
        ########################## create a scene ##########################
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
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

        ########################## entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        )

        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(0.65, 0.0, 0.02),
            )
        )
        ########################## build ##########################
        self.num_envs = num_envs
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)

        self.build_env()
    
    def build_env(self):
        self.motors_dof = torch.arange(7).to("cuda:0")
        self.fingers_dof = torch.arange(7, 9).to("cuda:0")
        franka_pos = torch.tensor([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]).to("cuda:0")
        franka_pos = franka_pos.unsqueeze(0).repeat(self.num_envs, 1) 
        self.franka.set_qpos(franka_pos, envs_idx=self.envs_idx)
        self.scene.step()

        self.end_effector = self.franka.get_link("hand")
        pos = torch.tensor([0.65, 0.0, 0.135], dtype=torch.float32, device='cuda:0')
        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device='cuda:0')
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos = self.pos,
            quat = self.quat,
        )

        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)

if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32")
    env = Env(vis=True)