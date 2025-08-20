#envs/ik/reach_cube_ego_audio_stacked_IKsimple.py


import numpy as np
import genesis as gs
import torch
import librosa
import matplotlib.pyplot as plt
from scipy.signal import chirp
from collections import deque
import sounddevice as sd


class ReachCubeEgoAudioStackedEnv:
    """
    Genesis environment with audio-only observations and configurable cube repositioning schedule.
    Observations are stacked spectrogram frames over a short history.

    Each step returns an observation tensor of shape (num_envs, 1, F, T),
    where F is the number of frequency bins and T is the number of stacked time frames.
    Optionally plays back the full stacked audio window for the designated listener index.

    The cube alternates between two fixed positions: [0.2, -0.8, 0.2] (easy) and [0.2, 0.8, 0.2] (hard).
    It stays at the hard position for `hard_episodes` episodes, then at the easy position for
    `easy_episodes` episodes, repeating this cycle indefinitely. This ensures the very first
    episode starts with the hard position.
    """

    def __init__(
        self,
        vis: bool,
        device: torch.device,
        num_envs: int = 1,
        listen_idx: int = 0,
        show_every: int = 10,
        history_length: int = 25,
        sample_offsets=None,
        noise_config: dict = None,
        inference_mode: bool = False,
    ):
        # --- Configuration ---
        self.device = device
        self.num_envs = num_envs
        self.listen_idx = listen_idx
        self.show_every = show_every
        # inside __init__
        self.success_thresh = 0.3001  # meters
        self.report_success_as_done = True
        self.inference_mode = inference_mode
        # inside __init__
        self.pos_jitter = 0.1  # max absolute noise per axis (meters)
        self._rng = np.random.default_rng()
        self._pos_noise_cache = {}  # remember per-position noise until it advances

        # Number of episodes in easy and hard segments
        # New cube positions and durations
        self.cube_positions = [
            np.array([0.6, -0.3, 0.6]),  # left up

            np.array([0.6, -0.3, 0.4]),  # left middle

            np.array([0.6, -0.3, 0.2]),  # left down

            np.array([0.6, 0.0, 0.6]),  # middle up

            np.array([0.6, 0.0, 0.4]),  # middle middle

            np.array([0.6, 0.0, 0.2]),  # middle down

            np.array([0.6, 0.3, 0.6]),  # right up

            np.array([0.6, 0.3, 0.4]),  # right middle

            np.array([0.6, 0.3, 0.2])  # right down

        ]
        # --- Curriculum-by-performance state ---
        self.pos_count = len(self.cube_positions)
        self.current_idx = 0  # which position we're currently training
        # Track last episode return per position (not best-ever)
        self.last_return = np.full(self.pos_count, -np.inf, dtype=np.float32)
        self.advance_margin = 0.0  # set >0 to allow small gap, e.g. 0.5

        self.repeat_counter = 0

        # Episode accounting
        self._episode_active = False
        self._episode_return = 0.0

        self.cube_durations = [1, 1, 1, 1]  # Number of episodes each position lasts
        self.cube_cycle_length = sum(self.cube_durations)

        # History for spectrograms and raw audio
        self.history_length = history_length
        self.sample_offsets = sample_offsets or [-21, -16, -11, -6, -1]
        self.audio_history = deque(maxlen=self.history_length)
        self.raw_audio_history = deque(maxlen=self.history_length)

        # Store noise config
        self.noise_config = noise_config if noise_config else {"audio_noise_level": 0.0}

        # Spectrogram dimensions: freq bins and stacked time frames
        self.freq_bins = 257
        self.time_bins = len(self.sample_offsets)
        self.obs_shape = (1, self.freq_bins, self.time_bins)
        self.action_space = 6

        # Matplotlib figure for live preview
        self._fig = plt.figure("Stacked Spectrogram Preview")

        # Build the simulation scene and initialize the robot
        self._build_scene(vis)
        self._init_robot()

        # Internal counters and state
        self.step_count = 0
        self.episode_count = 0
        self.current_cube_pos = None


    def _build_scene(self, show_viewer: bool):
        """Set up the Genesis scene, ground plane, robot, and cube."""
        self.scene = gs.Scene(
            show_FPS=False,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 2, 1.5),
                camera_lookat=(0, 0, 0.2),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=show_viewer,
        )

        # Add environment entities
        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="assets/xml/franka_emika_panda/panda.xml")
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06)),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # Build multiple copies if needed
        self.scene.build(n_envs=self.num_envs, env_spacing=(5.0, 5.0))
        self.envs_idx = np.arange(self.num_envs)

    def _init_robot(self):
        """Reset the Franka robot to a neutral pose in all environments."""
        self.motors_dof = torch.arange(7, device=self.device)
        self.fingers_dof = torch.arange(7, 9, device=self.device)

        # Neutral joint configuration
        neutral_q = torch.tensor(
            [-.2, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self.franka.set_qpos(neutral_q, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = neutral_q[:, 7:9].clone()
        self.end_effector = self.franka.get_link("hand")

        # Target end-effector pose via IK
        base_pos = torch.tensor([0.2720, -0.1683, 1.0164], device=self.device)
        base_quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], device=self.device)
        self.pos = base_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.quat = base_quat.unsqueeze(0).repeat(self.num_envs, 1)

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)

    def _get_noisy_pos(self, base_pos: np.ndarray) -> np.ndarray:
        """
        Add per-episode uniform noise in [-pos_jitter, +pos_jitter] to x,y,z.
        Returns shape (3,) ndarray.
        """
        noise = self._rng.uniform(-self.pos_jitter, self.pos_jitter, size=3)
        return (base_pos + noise).astype(np.float32)

    def simulate_audio(self, dist: float) -> np.ndarray:
        sr, dur = 22050, 0.01
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        tone = chirp(t, f0=1000, f1=1000, t1=dur) / (dist ** 2 + 1e-6)

        # Base random noise
        noise = sum(
            np.random.rand() * chirp(
                t,
                np.random.randint(300, 3000),
                np.random.randint(300, 3000),
                dur
            )
            for _ in range(5)
        ) * 0.1

        # Additional noise based on config
        audio_noise_level = self.noise_config.get("audio_noise_level", 0.0)
        additional_noise = np.random.normal(0, audio_noise_level, tone.shape)

        return tone + noise + additional_noise

    def _compute_spectrogram(self, audio: np.ndarray) -> torch.Tensor:
        S = librosa.stft(audio, n_fft=512, hop_length=256)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=1.0)[:self.freq_bins, :self.time_bins]

        # Explicit normalization: [-20 dB, 120 dB] → [0,1]
        S_db_normalized = (S_db + 20) / 140
        S_db_normalized = np.clip(S_db_normalized, 0.0, 1.0)

        return torch.from_numpy(S_db_normalized).float()

    def _collect_spectrograms(self, play_audio: bool = False) -> torch.Tensor:
        """
        For each envs: simulate audio, optionally play only the most recent slice,
        convert to spectrogram, and record raw audio for the listener index.
        Returns a tensor of spectrograms shaped (num_envs, freq_bins, time_bins_per_slice).
        """
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube_pos = self.cube.get_pos()
        dists = torch.norm((left + right) / 2 - cube_pos, dim=1).cpu().numpy()

        specs = []
        for i, dist in enumerate(dists):
            audio = self.simulate_audio(dist)
            # Store raw audio only for the listener index
            if i == self.listen_idx:
                self.raw_audio_history.append(audio)

            # Compute spectrogram slice
            S_db = self._compute_spectrogram(audio)
            # Optional immediate playback of just this slice
            if play_audio and i == self.listen_idx and self.num_envs == 1:
                sd.play(audio, 22050)
                sd.wait()

            specs.append(S_db)

        # Count this step
        self.step_count += 1
        return torch.stack(specs, dim=0).to(self.device)

    def _build_observation(self) -> torch.Tensor:
        """
        Stack past spectrogram slices at the defined offsets,
        yielding a tensor (num_envs, 1, freq_bins, time_bins).
        """
        slices = [self.audio_history[offset] for offset in self.sample_offsets]
        stacked = torch.cat(slices, dim=2)
        return stacked.unsqueeze(1)

    def reset(self) -> torch.Tensor:
        """
        Reset environment and start a new episode.

        Modes:
          - INFERENCE (self.inference_mode=True): cycle deterministically through
            all cube positions (0 -> 1 -> 2 -> 3 -> 0 ...), ignoring curriculum.
          - TRAINING (default): performance-based scheduler (NORMAL / SWEEP).

        Adds per-episode uniform jitter in [-self.pos_jitter, +self.pos_jitter] (per axis)
        to the chosen base position, shared across all envs this episode.
        """
        # ====================================================
        # Inference mode: cycle positions deterministically
        # ====================================================
        if getattr(self, "inference_mode", False):
            self.episode_count += 1
            self._episode_return = 0.0
            self._episode_active = True

            # Cycle deterministically
            if not hasattr(self, "_infer_cycle_idx"):
                self._infer_cycle_idx = 0
            else:
                self._infer_cycle_idx = (self._infer_cycle_idx + 1) % self.pos_count

            self.current_idx = self._infer_cycle_idx

            # --- Jittered per-episode position (shared across envs) ---
            base = self.cube_positions[self.current_idx]
            noisy = self._get_noisy_pos(base)  # uniform [-pos_jitter, +pos_jitter] per axis
            one_pos = noisy.reshape(1, -1)
            self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)

            # Reset robot & place cube
            self._init_robot()
            self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
            self.scene.step()

            print(f"[Inference] Episode {self.episode_count}: pos idx={self.current_idx} "
                  f"pos={self.current_cube_pos[0]}")

            # Reset histories + prime with first slice to fill history_length
            self.audio_history.clear()
            self.raw_audio_history.clear()

            first_spec = self._collect_spectrograms(play_audio=False)
            first_raw = self.raw_audio_history[-1].copy()

            self.audio_history.clear()
            self.raw_audio_history.clear()
            for _ in range(self.history_length):
                self.audio_history.append(first_spec.clone())
                self.raw_audio_history.append(first_raw.copy())

            obs = self._build_observation()
            if self.num_envs == 1 and self._fig is not None:
                self._plot_stacked(obs[0, 0])

            done_array = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return obs, done_array

        # ====================================================
        # Training mode: curriculum scheduler
        # ====================================================

        # Lazy init (safe if already defined in __init__)
        if not hasattr(self, "pos_count"):
            self.pos_count = len(self.cube_positions)
        if not hasattr(self, "current_idx"):
            self.current_idx = 0
        if not hasattr(self, "last_return"):
            self.last_return = np.full(self.pos_count, -np.inf, dtype=np.float32)
        if not hasattr(self, "advance_margin"):
            self.advance_margin = 0.0  # require strictly greater than lowest-other
        if not hasattr(self, "repeat_streak"):
            self.repeat_streak = 0
        if not hasattr(self, "_mode"):
            self._mode = "NORMAL"  # or "SWEEP"
        if not hasattr(self, "_sweep_targets"):
            self._sweep_targets = []
        if not hasattr(self, "_sweep_i"):
            self._sweep_i = 0
        if not hasattr(self, "_sweep_target_idx"):
            self._sweep_target_idx = None
        if not hasattr(self, "_episode_active"):
            self._episode_active = False
        if not hasattr(self, "_episode_return"):
            self._episode_return = 0.0

        # Close previous episode (if active): update scheduler state
        if self._episode_active:
            prev_idx = self.current_idx
            prev_ret = float(self._episode_return)
            self.last_return[prev_idx] = prev_ret

            if self._mode == "SWEEP":
                if self._sweep_i < len(self._sweep_targets):
                    self.current_idx = self._sweep_targets[self._sweep_i]
                    self._sweep_i += 1
                    decision = f"sweep→idx={self.current_idx} ({self._sweep_i}/{len(self._sweep_targets)})"
                else:
                    self.current_idx = self._sweep_target_idx
                    self._mode = "NORMAL"
                    self._sweep_targets = []
                    self._sweep_i = 0
                    self._sweep_target_idx = None
                    self.repeat_streak = 0
                    decision = "return_to_target_after_sweep"
            else:
                idxs = np.arange(self.pos_count)
                others_mask = idxs != prev_idx
                seen_others = np.isfinite(self.last_return) & others_mask
                other_min = np.min(self.last_return[seen_others]) if np.any(seen_others) else -np.inf

                if prev_ret > (other_min + self.advance_margin):
                    self.current_idx = (prev_idx + 1) % self.pos_count
                    self.repeat_streak = 0
                    decision = "advance"
                else:
                    self.current_idx = prev_idx
                    self.repeat_streak += 1
                    if self.repeat_streak >= 5:
                        self._mode = "SWEEP"
                        self._sweep_target_idx = prev_idx
                        self._sweep_targets = [int(i) for i in idxs if i != prev_idx]
                        self._sweep_i = 0
                        self.current_idx = self._sweep_targets[self._sweep_i]
                        self._sweep_i += 1
                        decision = f"start_sweep→idx={self.current_idx} (1/{len(self._sweep_targets)})"
                    else:
                        decision = f"repeat ({self.repeat_streak})"

            idxs = np.arange(self.pos_count)
            others_mask = idxs != prev_idx
            seen_others = np.isfinite(self.last_return) & others_mask
            others_min_val = np.min(self.last_return[seen_others]) if np.any(seen_others) else -np.inf
            lr = ", ".join([f"{i}:{'%.3f' % v if np.isfinite(v) else '-'}"
                            for i, v in enumerate(self.last_return)])
            print(f"[EP end] pos={prev_idx} return={prev_ret:.3f} | last={lr} "
                  f"| others_min={'%.3f' % others_min_val if np.isfinite(others_min_val) else '-'} "
                  f"| decision={decision}")

        # Start new episode (training mode)
        self.episode_count += 1
        self._episode_return = 0.0
        self._episode_active = True

        # --- Jittered per-episode position (shared across envs) ---
        base = self.cube_positions[self.current_idx]

        if self.current_idx not in self._pos_noise_cache:
            # only sample when we switch to a new base position
            self._pos_noise_cache[self.current_idx] = self._get_noisy_pos(base)

        noisy = self._pos_noise_cache[self.current_idx]
        one_pos = noisy.reshape(1, -1)
        self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)

        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        print(f"Episode {self.episode_count}: training position idx={self.current_idx} "
              f"pos={self.current_cube_pos[0]}")

        # Histories
        self.audio_history.clear()
        self.raw_audio_history.clear()

        first_spec = self._collect_spectrograms(play_audio=False)
        first_raw = self.raw_audio_history[-1].copy()

        self.audio_history.clear()
        self.raw_audio_history.clear()
        for _ in range(self.history_length):
            self.audio_history.append(first_spec.clone())
            self.raw_audio_history.append(first_raw.copy())

        obs = self._build_observation()
        if self.num_envs == 1 and self._fig is not None:
            self._plot_stacked(obs[0, 0])

        done_array = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return obs, done_array

    def step(self, actions: torch.Tensor):
        """
        Discrete Cartesian moves, audio/spectrogram update, reward, and done flags.

        Changes:
          - success @ <= 0.20 m: +20 bonus and done=True (per-env)
          - base reward kept (exp(-4*dist))
        """
        # --- Move EE by discrete deltas ---
        deltas = torch.tensor([
            [0.05, 0.00, 0.00],  # +x
            [-0.05, 0.00, 0.00],  # -x
            [0.00, 0.05, 0.00],  # +y
            [0.00, -0.05, 0.00],  # -y
            [0.00, 0.00, 0.05],  # +z
            [0.00, 0.00, -0.05],  # -z
        ], device=self.device)

        masks = [actions == i for i in range(self.action_space)]
        self.pos += sum(deltas[i] * masks[i].unsqueeze(1) for i in range(self.action_space))

        # IK control to reach the new pose
        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=self.pos, quat=self.quat)
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        # Collect new spectrogram slice and append to history
        new_slice = self._collect_spectrograms(play_audio=False)
        self.audio_history.append(new_slice)

        # Build stacked observation (num_envs, 1, freq_bins, time_bins)
        obs = self._build_observation()
        if self.num_envs == 1 and self.step_count % self.show_every == 0:
            self._plot_stacked(obs[0, 0])

        # ---- Distance & reward ----
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube_pos = self.cube.get_pos()
        dist = torch.norm((left + right) / 2 - cube_pos, dim=1)  # [num_envs]

        base_reward = torch.clamp(torch.exp(-4 * dist), 0.0, 1.0)

        success_thresh = getattr(self, "success_thresh", 0.30)  # meters
        report_success = getattr(self, "report_success_as_done", True)

        success_mask = (dist <= success_thresh)
        bonus = success_mask.float() * 20.0  # +20 on success
        rewards = (base_reward + bonus).to(self.device)

        # accumulate per-episode return (mean over envs this step)
        self._episode_return += rewards.mean().item()

        if report_success:
            dones = success_mask.to(self.device)
        else:
            dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return obs, rewards, dones

    def _plot_stacked(self, data: torch.Tensor):
        """
        Render the stacked spectrogram in the live preview figure.
        """
        plt.clf()
        extent = [0, 10 * len(self.sample_offsets), 0, (22050 / 2) / 1000]

        # Explicit normalization [0, 1] scaled back to dB for visualization
        vmin, vmax = 0, 1
        plt.imshow(data.cpu().numpy(), origin='lower', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, cmap='magma')
        plt.colorbar(label='Amplitude (dB)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (kHz)')
        plt.title(f'Step {self.step_count} Stacked Spectrogram')
        plt.draw()
        plt.pause(0.01)
        self._fig.canvas.flush_events()


if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoAudioStackedEnv(vis=True, device=torch.device('cuda'), listen_idx=0)
