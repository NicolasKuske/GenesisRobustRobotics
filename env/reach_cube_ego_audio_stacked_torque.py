import numpy as np
import genesis as gs
import torch
import librosa
from scipy.signal import chirp
from collections import deque
import matplotlib.pyplot as plt
import sounddevice as sd

class ReachCubeEgoAudioStackedTorqueEnv:
    """
    Audio-only, stacked-frame, continuous-torque environment with curriculum on cube X-axis sampling.
    Agent outputs a 7-dim torque vector, scaled by ±1.
    For single-env mode, displays the stacked spectrogram and plays audio.
    """

    def __init__(
        self,
        vis: bool,
        device: torch.device,
        num_envs: int = 1,
        episodes_per_position: int = 3,
        history_length: int = 25,
        reward_thresholds=None,
        window_size: int = 4,
        success_thresh: float = 0.30,
        success_bonus: float = 1,
        shaping_coef: float = 10.0,
        k: float = 2,
        dist_offset: float = 0.0,
        show_every: int = 10,
    ):
        # Device & parallelism
        self.device = device
        self.num_envs = num_envs

        # Curriculum parameters
        self.episodes_per_position = episodes_per_position
        self.window_size = window_size
        self.reward_thresholds = reward_thresholds or [10, 10, 10, 10, 12, 12, 12]
        self.last_rewards = deque(maxlen=self.window_size)
        self.x_bounds = [0.4, 0.2, 0.0, -0.2, -0.4, -0.6]
        self.fixed_x = 0.6
        self.max_stages = len(self.x_bounds)
        self.x_stage = 0
        self.completed = False
        self.episode_count = 0

        # Reward shaping parameters
        self.success_thresh = success_thresh
        self.success_bonus = success_bonus
        self.shaping_coef = shaping_coef
        self.k = k
        self.dist_offset = dist_offset
        self.prev_dist = None
        self.sum_delta = None
        self.sum_success = None
        self.episode_reward = 0.0

        # History settings & sampling offsets
        self.history_length = history_length
        self.sample_offsets = [-21, -16, -11, -6, -1]
        self.audio_history = deque(maxlen=self.history_length)
        self.raw_audio_history = deque(maxlen=self.history_length)

        # Spectrogram dims and action space
        self.freq_bins = 257
        self.time_bins = len(self.sample_offsets)
        self.obs_shape = (1, self.freq_bins, self.time_bins)
        self.action_space = 7

        # Display settings for single-env
        self.show_every = show_every
        self._fig = plt.figure(num='Stacked Spectrogram') if self.num_envs == 1 else None

        # Build the scene and robot
        self._build_scene(vis)
        self._init_robot()

        # Internal step counter
        self.step_count = 0

    def _build_scene(self, show_viewer: bool):
        self.scene = gs.Scene(
        show_FPS=False,
            show_viewer=show_viewer,
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml")
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06)),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )
        self.scene.build(n_envs=self.num_envs, env_spacing=(5.0, 5.0))
        self.envs_idx = np.arange(self.num_envs)

    def _init_robot(self):
        neutral_q = torch.tensor(
            [-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(neutral_q, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = neutral_q[:, 7:9].clone()
        self.end_effector = self.franka.get_link("hand")

    def simulate_audio(self, dist: float) -> np.ndarray:
        sr, dur = 22050, 0.01
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        tone = chirp(t, f0=1000, f1=1000, t1=dur) / (dist**2 + 1e-6)
        noise = sum(
            np.random.rand() * chirp(
                t,
                np.random.randint(300, 3000),
                np.random.randint(300, 3000),
                dur
            )
            for _ in range(5)
        ) * 0.1
        return tone + noise

    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        S = librosa.stft(audio, n_fft=512, hop_length=256)
        return librosa.amplitude_to_db(np.abs(S), ref=1.0)[:self.freq_bins, :9]

    def _collect_spectrograms(self) -> torch.Tensor:
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube = self.cube.get_pos()
        dists = torch.norm((left + right) / 2 - cube, dim=1).cpu().numpy()

        specs = []
        for i, dist in enumerate(dists):
            audio = self.simulate_audio(dist)
            if i == 0:
                self.raw_audio_history.append(audio)
            S_db = self._compute_spectrogram(audio)
            specs.append(torch.from_numpy(S_db).float())

        self.scene.step()
        self.step_count += 1
        return torch.stack(specs, dim=0).to(self.device)

    def _build_observation(self) -> torch.Tensor:
        slices = [self.audio_history[offset] for offset in self.sample_offsets]
        stacked = torch.cat(slices, dim=2)
        return stacked.unsqueeze(1)

    def _plot_stacked(self, data: torch.Tensor):
        plt.clf()
        # X axis: 10 ms per frame × number of frames
        extent = [0, 10 * len(self.sample_offsets), 0, (22050 / 2) / 1000]
        plt.imshow(
            data.cpu().numpy(),
            origin='lower',
            aspect='auto',
            extent=extent,
            vmin=-40,      # match your first script
            vmax=100
        )
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (kHz)')
        plt.title(f'Step {self.step_count}')
        plt.pause(0.01)
        self._fig.canvas.flush_events()

    def _sample_cube_pos(self) -> np.ndarray:
        idx = min(self.x_stage, self.max_stages - 1)
        lower = self.x_bounds[idx]
        x = np.random.uniform(lower, self.fixed_x, (self.num_envs, 1))
        y = np.random.uniform(-0.6, 0.6, (self.num_envs, 1))
        z = np.random.uniform(0.1, 1.0, (self.num_envs, 1))
        return np.concatenate([x, y, z], axis=1)

    def _process_episode_end(self):
        shaping = self.sum_delta.mean().item()
        bonus = self.sum_success.mean().item()
        total = self.episode_reward
        print(f"[Episode {self.episode_count}] Shaping: {shaping:.4f}, Bonus: {bonus:.4f}, Total: {total:.4f}")

        self.last_rewards.append(total)
        if len(self.last_rewards) == self.window_size:
            mean_r = sum(self.last_rewards) / self.window_size
            thr = self.reward_thresholds[min(self.x_stage, len(self.reward_thresholds)-1)]
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

    def reset(self) -> torch.Tensor:
        # End–of–episode processing
        if self.episode_count > 0:
            self._process_episode_end()
        self.episode_count += 1

        # Possibly sample a new cube pose
        if (self.episode_count - 1) % self.episodes_per_position == 0:
            self.current_cube_pos = self._sample_cube_pos()

        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        # Reset reward trackers
        self.sum_delta = torch.zeros(self.num_envs, device=self.device)
        self.sum_success = torch.zeros(self.num_envs, device=self.device)
        self.episode_reward = 0.0

        # Prime history with first slice
        first_spec = self._collect_spectrograms()
        first_raw = self.raw_audio_history[-1].copy()
        self.audio_history.clear()
        self.raw_audio_history.clear()
        for _ in range(self.history_length):
            self.audio_history.append(first_spec.clone())
            self.raw_audio_history.append(first_raw.copy())

        # Initialize prev_dist so step() never sees None
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube = self.cube.get_pos()
        self.prev_dist = torch.norm((left + right) / 2 - cube, dim=1)

        obs = self._build_observation()
        if self.num_envs == 1:
            # Play & plot immediately on reset
            buf = np.concatenate([self.raw_audio_history[o] for o in self.sample_offsets])
            sd.play(buf, 22050); sd.wait()
            self._plot_stacked(obs[0,0])
        return obs

    def step(self, actions: torch.Tensor):
        """
        Apply torques, collect audio, update history, plot stacked spectrogram,
        and compute reward/done flags.
        """
        # 1) Apply torques
        torque = actions.clamp(-1.0, 1.0) * torch.tensor(
            [87, 87, 87, 87, 12, 12, 12], device=self.device
        )
        self.franka.control_dofs_force(
            torque.cpu().numpy(),
            dofs_idx_local=np.arange(7),
            envs_idx=self.envs_idx
        )
        # Keep fingers closed
        self.franka.control_dofs_position(
            self.fixed_finger_pos,
            torch.arange(7, 9, device=self.device),
            envs_idx=self.envs_idx
        )
        self.scene.step()

        # 2) Collect new spectrogram slice and append
        new_slice = self._collect_spectrograms()
        self.audio_history.append(new_slice)

        # 3) Build the full stacked observation
        obs = self._build_observation()

        # 4) For single-env, every `show_every` steps: play & plot the FULL stack
        if self.num_envs == 1 and (self.step_count % self.show_every == 0):
            buf = np.concatenate([self.raw_audio_history[o] for o in self.sample_offsets])
            sd.play(buf, 22050)
            sd.wait()
            self._plot_stacked(obs[0, 0])

        # 5) Reward computation: pure exponential decay + success bonus
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube = self.cube.get_pos()
        dist_new = torch.norm((left + right) / 2 - cube, dim=1)

        # shaped reward: e^{-k * distance}
        shaped = 0.1*torch.exp(-self.k * dist_new)
        # bonus for reaching the target threshold
        success = (dist_new < self.success_thresh).float()
        bonus = success * self.success_bonus
        rewards = shaped + bonus

        # Update episode trackers
        self.sum_success += bonus
        self.prev_dist = dist_new
        self.episode_reward += rewards.mean().item()

        dones = success.bool()
        return obs, rewards, dones




if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoAudioStackedTorqueEnv(vis=True, device=torch.device("cuda"), num_envs=1)
    obs = env.reset()
    for t in range(200):
        actions = torch.randn(env.num_envs, env.action_space, device=env.device)
        obs, reward, done = env.step(actions)
        if done.any():
            print("Done!", done)
            break
