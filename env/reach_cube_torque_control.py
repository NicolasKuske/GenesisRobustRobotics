import numpy as np
import genesis as gs
import torch
from collections import deque

class ReachCubeTorqueEnv:
    """
    Robotic environment where a Franka arm reaches for a cube with a curriculum on X-axis sampling.

    Curriculum stages (lower bound → 0.6):
      Stage 0: X ∈ [0.4, 0.6]
      Stage 1: X ∈ [0.2, 0.6]
      Stage 2: X ∈ [0.0, 0.6]
      Stage 3: X ∈ [-0.2, 0.6]
      Stage 4: X ∈ [-0.4, 0.6]
      Stage 5: X ∈ [-0.6, 0.6]

    Final stability round (Stage 6): same bounds as Stage 5 ([-0.6, 0.6]), but uses the same window size for evaluation.
    """

    def __init__(
        self,
        vis: bool,
        device: str,
        num_envs: int = 1,
        episodes_per_position: int = 3,
        window_size: int = 4,
        reward_thresholds: list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5]
    ):
        # Basic settings
        self.device = device
        self.num_envs = num_envs
        self.episodes_per_position = episodes_per_position

        # Constant window size for all stages (including stability)
        self.window_size = window_size
        self.last_rewards = deque(maxlen=self.window_size)

        # thresholds: one per stage 0–5 plus final stability = 7 entries
        # if you pass reward_thresholds, make sure it has length ≥ 7
        self.reward_thresholds = reward_thresholds

        # Observation/action dimensions
        self.state_dim = 6
        self.action_space = 7

        # Sampling bounds for Y, Z
        self.min_y, self.max_y = -0.6, 0.6
        self.min_z, self.max_z = 0.1, 0.7

        # Curriculum X-axis settings
        self.fixed_x = 0.6
        # six progressive lower bounds
        self.x_bounds = [0.4, 0.2, 0.0, -0.2, -0.4, -0.6]
        # stage indices: 0..5 are the six stages, 6 is final stability
        self.max_stages = len(self.x_bounds)
        self.x_stage = 0
        self.completed = False

        # Reward shaping parameters
        self.success_thresh = 0.30
        self.success_bonus = 0.1
        self.shaping_type = "exp"
        self.shaping_coef = 10.0
        self.k = 0.5
        self.dist_offset = 0.0

        # Episode trackers
        self.episode_count = 0
        self.prev_dist = None
        self.sum_delta = None
        self.sum_success = None

        # Torque limits
        self.max_torque = torch.tensor([87,87,87,87,12,12,12], device=self.device)

        # Build scene
        self._setup_scene(vis)
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)

        # Initial cube placement
        self.build_env()
        self.current_cube_pos = self._sample_cube_pos()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

    def _setup_scene(self, vis: bool):
        opts = gs.options.ViewerOptions(
            camera_pos=(3,2,1.5), camera_lookat=(0,0,0.5), camera_fov=30,
            res=(960,640), max_FPS=60
        )
        self.scene = gs.Scene(
            show_FPS=False,
            viewer_options=opts,
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=vis,
            vis_options=gs.options.VisOptions(plane_reflection=True),
            renderer=gs.renderers.Rasterizer(),
        )
        # Floor and walls
        self.scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Aluminium(ior=10.0))
        for pos,color,euler in [((4,0,1),(0.9,0.9,0.9),(0,-20,0)),
                                ((-3,0,1),(0.7,0.7,0.7),(0,20,0)),
                                ((0,-3,1),(0.56,0.57,0.58),(0,20,90))]:
            self.scene.add_entity(
                gs.morphs.Box(size=(0.1,8,4), pos=pos, euler=euler, collision=False),
                surface=gs.surfaces.Rough(color=color),
                material=gs.materials.Rigid(gravity_compensation=1.0)
            )
        # Robot and cube
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"))
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06,0.06,0.06)),
            surface=gs.surfaces.Rough(color=(0.99,0.82,0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

    def _sample_cube_pos(self) -> np.ndarray:
        """Random cube position based on curriculum stage."""
        # pick lower bound from x_bounds, saturating at last entry for stability
        lower = self.x_bounds[min(self.x_stage, len(self.x_bounds)-1)]
        x = np.random.uniform(lower, self.fixed_x, (self.num_envs,1))

        y_low = np.random.uniform(self.min_y, -0.15, (self.num_envs,1))
        y_high = np.random.uniform(0.15, self.max_y, (self.num_envs,1))
        mask = np.random.rand(self.num_envs,1) < 0.5
        y = np.where(mask, y_low, y_high)

        z = np.random.uniform(self.min_z, self.max_z, (self.num_envs,1))
        return np.hstack((x,y,z))

    def build_env(self):
        """Reset to initial joint pose and lock fingers."""
        q0 = torch.tensor(
            [-1.0,-0.3,0.3,-1.0,-0.1,1.7,1.0,0.01,0.01],
            dtype=torch.float32, device=self.device
        ).repeat(self.num_envs,1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:,7:9].clone()
        dofs = self.fixed_finger_pos.new_tensor([7,8]).cpu().numpy()
        self.franka.set_dofs_kp(kp=np.array([100,100]), dofs_idx_local=dofs)
        self.franka.set_dofs_kv(kv=np.array([10,10]), dofs_idx_local=dofs)
        self.franka.set_dofs_force_range(lower=[-100,-100], upper=[100,100], dofs_idx_local=dofs)

    def reset(self) -> torch.Tensor:
        """Starts new episode, logs end-of-episode, and places cube."""
        if self.episode_count > 0:
            self._process_episode_end()
        self.episode_count += 1
        self.sum_delta = torch.zeros(self.num_envs, device=self.device)
        self.sum_success = torch.zeros(self.num_envs, device=self.device)

        # resample cube every few episodes
        if (self.episode_count-1) % self.episodes_per_position == 0:
            self.current_cube_pos = self._sample_cube_pos()

        self.build_env()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

        # initial observation
        obj = self.cube.get_pos()
        grip = 0.5*(self.franka.get_link("left_finger").get_pos() +
                    self.franka.get_link("right_finger").get_pos())
        self.prev_dist = torch.norm(obj - grip, dim=1)
        return torch.cat((obj, grip), dim=1)

    def _process_episode_end(self):
        shaping = self.sum_delta.mean().item()
        bonus = self.sum_success.mean().item()
        ep_reward = shaping + bonus
        print(f"[Episode {self.episode_count}] Shaping: {shaping:.4f}, Bonus: {bonus:.4f}, Total: {ep_reward:.4f}")
        self.last_rewards.append(ep_reward)

        if len(self.last_rewards) == self.window_size:
            mean_r = np.mean(self.last_rewards)
            # pick threshold, saturating at last entry
            thr_idx = min(self.x_stage, len(self.reward_thresholds)-1)
            thresh = self.reward_thresholds[thr_idx]
            print(f"[Curriculum] last {self.window_size}-ep mean: {mean_r:.4f}, threshold: {thresh:.4f}")
            if mean_r > thresh:
                self._advance_stage()

    def _advance_stage(self):
        """Progress through the six stages and then stability round."""
        self.x_stage += 1

        if self.x_stage < self.max_stages:
            lower = self.x_bounds[self.x_stage]
            print(f"[Env] Advanced to stage {self.x_stage}: X ∈ [{lower:.2f}, {self.fixed_x:.2f}]")
            self.last_rewards.clear()

        elif self.x_stage == self.max_stages:
            # final stability round
            lower = self.x_bounds[-1]
            print(f"[Env] Advanced to FINAL stability round (stage {self.x_stage}): X ∈ [{lower:.2f}, {self.fixed_x:.2f}]")
            self.last_rewards.clear()

        else:
            # beyond final stability → done
            self.completed = True
            print("[Env] Curriculum complete!")

    def step(self, actions: torch.Tensor):
        torque = actions.clamp(-1.0,1.0) * self.max_torque
        self.franka.control_dofs_force(torque.cpu().numpy(),
                                       dofs_idx_local=np.arange(7),
                                       envs_idx=self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos,
                                          torch.arange(7,9,device=self.device),
                                          envs_idx=self.envs_idx)
        self.scene.step()

        obj = self.cube.get_pos()
        grip = 0.5*(self.franka.get_link("left_finger").get_pos() +
                    self.franka.get_link("right_finger").get_pos())
        dist_new = torch.norm(obj - grip, dim=1)

        if self.shaping_type == "exp":
            delta = self.shaping_coef * (
                torch.exp(-self.k*(dist_new - self.dist_offset))
                - torch.exp(-self.k*(self.prev_dist - self.dist_offset))
            )
        else:
            delta = self.shaping_coef * (self.prev_dist - dist_new)
        self.prev_dist = dist_new

        success = (dist_new < self.success_thresh).float()
        bonus = success * self.success_bonus

        self.sum_delta += delta
        self.sum_success += bonus

        reward = delta + bonus
        done = success.bool()
        return torch.cat((obj, grip), dim=1), reward, done

if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeTorqueEnv(vis=True, device="cuda")
    s = env.reset()
    print("Initial state:", s)

