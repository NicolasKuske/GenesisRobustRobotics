# envs/reach_cube_ego_vision_stacked_torque.py
import numpy as np
import genesis as gs
import torch
import math
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat

class ReachCubeEgoVisionStackedTorqueEnv:
    """
    Ego-centric, not_stacked-frame, continuous-torque environment with curriculum on cube X-axis sampling.
    Agent outputs a 7-dim torque vector, scaled by ±1.
    """

    def __init__(
        self,
        vis: bool,
        device: torch.device,
        num_envs: int = 1,
        episodes_per_position: int = 3,
        window_size: int = 4,
        reward_thresholds=None,
    ):
        # Device & parallelism
        self.device         = device
        self.num_envs       = num_envs

        # --- Curriculum parameters ---
        self.episodes_per_position = episodes_per_position
        self.window_size          = window_size
        self.reward_thresholds    = reward_thresholds or [3, 3, 3, 3, 3.5, 3.5, 3.5]
        self.last_rewards         = deque(maxlen=self.window_size)
        self.x_bounds             = [0.4, 0.2, 0.0, -0.2, -0.4, -0.6]
        self.fixed_x              = 0.6
        self.max_stages           = len(self.x_bounds)
        self.x_stage              = 0
        self.completed            = False

        # Reward shaping
        self.success_thresh = 0.30
        self.success_bonus  = 0.1
        self.shaping_type   = 'exp'
        self.shaping_coef   = 10.0
        self.k              = 0.5
        self.dist_offset    = 0.0

        # Episode trackers
        self.episode_count = 0
        self.prev_dist     = None
        self.episode_reward= 0.0
        self.sum_delta     = None
        self.sum_success   = None

        # History settings & frame-skip
        self.history_length = 25
        #self.sample_offsets = [-1, -6, -11, -16, -21]
        self.sample_offsets = [-21, -16, -11, -6, -1]
        self.image_history  = deque(maxlen=self.history_length)
        self.render_every   = 5
        self._step_count    = 0

        # Observation & action dims
        self.obs_shape    = (3 * len(self.sample_offsets), 120, 120)
        self.action_space = 7
        self.max_torque   = torch.tensor([87,87,87,87,12,12,12],
                                         dtype=torch.float32,
                                         device=self.device)

        # —— build Genesis scene ——
        self.scene = gs.Scene(
            #show_FPS=False,
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


        # Floor and walls (replicated across envs grid)
        self.scene.add_entity(
            gs.morphs.Plane(),
            surface=gs.surfaces.Aluminium(ior=10.0)
        )
        for pos, color, euler in [
            ((4,  0, 1),  (0.9, 0.9, 0.9),  (0, -20,  0)),
            ((-3, 0, 1),  (0.7, 0.7, 0.7),  (0,  20,  0)),
            ((0, -3, 1),  (0.56,0.57,0.58),(0,  20, 90)),
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
        self.franka = self.scene.add_entity(gs.morphs.MJCF(
            file="assets/xml/franka_emika_panda/panda.xml"
        ))
        self.cube   = self.scene.add_entity(
            gs.morphs.Box(size=(0.06,0.06,0.06)),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # Camera setup: ego-centric transform
        self.cams = []
        self.cam_transform = trans_quat_to_T(
            np.array([0.03, 0, 0.03]),
            xyz_to_quat(np.array([185, 0, 90]))
        )
        for _ in range(self.num_envs):
            cam = self.scene.add_camera(res=(120,120), fov=90, GUI=True)
            self.cams.append(cam)

        # Build parallel envs & start recording
        env_space = 100.0
        self.scene.build(n_envs=self.num_envs,
                         env_spacing=(env_space, env_space))
        self.envs_idx = np.arange(self.num_envs)
        for cam in self.cams:
            cam.start_recording()

        # Initialize robot pose
        self._init_robot()


    def _init_robot(self):
        self.motors_dof  = torch.arange(7, device=self.device)
        self.fingers_dof = torch.arange(7,9, device=self.device)

        q0 = torch.tensor(
            [-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:,7:9].clone()
        self.end_effector     = self.franka.get_link("hand")

    def _render(self):
        imgs = []
        M    = int(math.sqrt(self.num_envs))
        env_space = 100.0
        for idx, cam in enumerate(self.cams):
            ee_pos  = self.end_effector.get_pos(envs_idx=[idx])[0].cpu().numpy()
            ee_quat = self.end_effector.get_quat(envs_idx=[idx])[0].cpu().numpy()
            col,row = divmod(idx, M)
            x_off   = (col - (M-1)/2) * env_space
            y_off   = (row - (M-1)/2) * env_space
            ee_pos_offset = ee_pos + np.array([x_off, y_off, 0.0])
            ee_T    = trans_quat_to_T(ee_pos_offset, ee_quat)
            cam_T   = ee_T @ self.cam_transform
            cam.set_pose(transform=cam_T)
            rgb = cam.render()[0]
            img = torch.from_numpy(rgb.copy()).permute(2,0,1).float() / 255.0
            imgs.append(img)
        return torch.stack(imgs, dim=0)

    def _build_observation(self):
        samples = [self.image_history[i] for i in self.sample_offsets]
        return torch.cat(samples, dim=1)

    def _sample_cube_pos(self):
        idx   = min(self.x_stage, self.max_stages)
        lower = self.x_bounds[idx] if idx < self.max_stages else self.x_bounds[-1]
        x     = np.random.uniform(lower, self.fixed_x, (self.num_envs, 1))
        y     = np.random.uniform(-0.6, 0.6,       (self.num_envs, 1))
        z     = np.random.uniform(0.1,  1.0,       (self.num_envs, 1))
        return np.concatenate([x, y, z], axis=1)

    def reset(self):
        if self.episode_count > 0:
            self._process_episode_end()
        self.episode_count += 1

        self.sum_delta       = torch.zeros(self.num_envs, device=self.device)
        self.sum_success     = torch.zeros(self.num_envs, device=self.device)
        self.episode_reward  = 0.0

        if (self.episode_count - 1) % self.episodes_per_position == 0:
            self.current_cube_pos = self._sample_cube_pos()

        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

        obj    = self.cube.get_pos()
        gp_l   = self.franka.get_link("left_finger").get_pos()
        gp_r   = self.franka.get_link("right_finger").get_pos()
        self.prev_dist = torch.norm(obj - (gp_l + gp_r)/2, dim=1)

        # reset counter & prime history
        self._step_count = 0
        first = self._render()
        for _ in range(self.history_length):
            self.image_history.append(first.clone())

        return self._build_observation()

    def _plot_history(self):
        """Plot all 20 frames and highlight sampled ones."""
        frames = list(self.image_history)
        sample_idxs = [(self.history_length + offset) for offset in self.sample_offsets]
        # sample_offsets = [-21,-16,-11,-6,-1] → sample_idxs = [0,5,10,15,20] most recent frames

        fig, axes = plt.subplots(4, 5, figsize=(12, 10))
        axes = axes.flatten()
        for i, (ax, frame) in enumerate(zip(axes, frames)):
            img = frame[0].permute(1, 2, 0).cpu().numpy()  # envs 0
            ax.imshow(img)
            ax.axis('off')
            if i in sample_idxs:
                # draw a red border
                rect = patches.Rectangle((0,0), frame.shape[2], frame.shape[1],
                                         linewidth=4, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            ax.set_title(f"{i}")
        plt.suptitle(f"History frames at step {self._step_count}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def _process_episode_end(self):
        shaping = self.sum_delta.mean().item()
        bonus   = self.sum_success.mean().item()
        total   = self.episode_reward
        print(f"[Episode {self.episode_count}] Shaping: {shaping:.4f}, Bonus: {bonus:.4f}, Total: {total:.4f}")

        self.last_rewards.append(total)
        if len(self.last_rewards) == self.window_size:
            mean_r = sum(self.last_rewards) / self.window_size
            thr    = self.reward_thresholds[min(self.x_stage, len(self.reward_thresholds)-1)]
            print(f"[Curriculum] last {self.window_size}-ep mean: {mean_r:.4f}, threshold: {thr:.4f}")
            if mean_r > thr:
                self._advance_stage()

    def _advance_stage(self):
        self.x_stage += 1
        if self.x_stage < self.max_stages:
            lb = self.x_bounds[self.x_stage]
            print(f"Advanced to stage {self.x_stage}: X ∈ [{lb:.2f}, {self.fixed_x:.2f}]")
            self.last_rewards.clear()
        else:
            self.completed = True
            print("Curriculum complete!")

    def step(self, actions: torch.Tensor):
        torque = actions.clamp(-1.0, 1.0) * self.max_torque.unsqueeze(0)
        self.franka.control_dofs_force(
            torque.cpu().numpy(),
            dofs_idx_local=np.arange(7),
            envs_idx=self.envs_idx
        )
        self.franka.control_dofs_position(
            self.fixed_finger_pos,
            self.fingers_dof,
            envs_idx=self.envs_idx
        )
        self.scene.step()

        # render every Nth step, else reuse last frame
        if self._step_count % self.render_every == 0:
            new_frame = self._render()
        else:
            new_frame = self.image_history[-1]

        self._step_count += 1
        self.image_history.append(new_frame)

        # debug: after 50 steps, plot entire history
        #if self._step_count == 50:
        #    self._plot_history()

        obs = self._build_observation()

        obj_pos  = self.cube.get_pos()
        gp_l     = self.franka.get_link("left_finger").get_pos()
        gp_r     = self.franka.get_link("right_finger").get_pos()
        dist_new = torch.norm(obj_pos - (gp_l + gp_r)/2, dim=1)

        if self.shaping_type == 'exp':
            delta = self.shaping_coef * (
                torch.exp(-self.k * (dist_new - self.dist_offset))
                - torch.exp(-self.k * (self.prev_dist - self.dist_offset))
            )
        else:
            delta = self.shaping_coef * (self.prev_dist - dist_new)

        success  = (dist_new < self.success_thresh).float()
        bonus    = success * self.success_bonus
        rewards  = delta + bonus

        self.sum_delta      += delta
        self.sum_success    += bonus
        self.prev_dist       = dist_new
        self.episode_reward += rewards.mean().item()

        dones = success.bool()
        return obs, rewards, dones


if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoVisionStackedTorqueEnv(vis=True,
                                             device=torch.device("cuda"))
    obs = env.reset()
    for t in range(200):
        actions = torch.randn(env.num_envs,
                              env.action_space,
                              device=env.device)
        obs, reward, done = env.step(actions)
        if done.any():
            print("Done!", done)
            break

