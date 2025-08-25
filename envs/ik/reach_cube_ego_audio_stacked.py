# envs/ik/reach_cube_ego_audio_stacked_IK.py

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

    === Key Features in this version ===
      - Two microphones: one on the left finger tip, one on the right finger tip.
      - Gripper kept fully open to maximize binaural baseline.
      - Stereo audio per step -> two spectrogram channels (Left, Right).
      - Observations are stacked spectrogram frames over short history:
            obs shape = (num_envs, 2, F, T)
        where F is the number of frequency bins and T = len(sample_offsets).
      - Very simple wide visualization: Left ear on the left, Right ear on the right.

    The cube cycles across multiple positions. Each episode picks one of four source sounds.
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

        # Success shaping
        self.success_thresh = 0.3001  # meters
        self.success_bonus = 20.0     # positive -> terminal bonus on success
        self.report_success_as_done = True

        # Sound choices per episode
        self.n_sounds = 4
        self.current_sound_id = 0  # set on reset()

        self.inference_mode = inference_mode

        # Audio playback
        self.enable_playback = True  # master switch; auto-disables on failure
        self._audio_warned = False
        self._sd_rate = 22050  # playback (and synthesis) sample rate

        # Speed of sound in air (m/s)
        self._c_sound = 3.0
        self._sr = 22050
        self._slice_dur = 0.01  # 10 ms

        # Curriculum positions for the cube
        self.cube_positions = [
            np.array([0.6, -0.3, 0.6]),  # left up
            np.array([0.6, -0.3, 0.4]),  # left middle
            np.array([0.6, -0.3, 0.2]),  # left down
            np.array([0.6,  0.0, 0.6]),  # middle up
            np.array([0.6,  0.0, 0.4]),  # middle middle
            np.array([0.6,  0.0, 0.2]),  # middle down
            np.array([0.6,  0.3, 0.6]),  # right up
            np.array([0.6,  0.3, 0.4]),  # right middle
            np.array([0.6,  0.3, 0.2]),  # right down
        ]

        # Anti-collapse scheduler bookkeeping
        self.pos_count = len(self.cube_positions)
        self.current_idx = 0
        self.last_return = np.full(self.pos_count, -np.inf, dtype=np.float32)
        self.advance_margin = 0.0
        self.repeat_counter = 0

        # Episode accounting
        self._episode_active = False
        self._episode_return = 0.0

        # History buffers
        self.history_length = history_length
        # default offsets: 5 recent slices spaced by ~5 steps
        self.sample_offsets = sample_offsets or [-21, -16, -11, -6, -1]
        self.audio_history = deque(maxlen=self.history_length)     # stores spectrogram slices: (N, 2, F, 1)
        self.raw_audio_history = deque(maxlen=self.history_length) # stores raw stereo audio: (Nsamples, 2)

        # Noise config
        self.noise_config = noise_config if noise_config else {"audio_noise_level": 0.0}

        # Spectrogram dimensions
        # We compute one spectrogram "column" per 10 ms audio slice -> Tslice=1.
        # Final stacked time dimension equals len(sample_offsets).
        self.freq_bins = 257
        self.time_bins = len(self.sample_offsets)
        self.obs_shape = (2, self.freq_bins, self.time_bins)  # 2 channels: Left, Right

        # Action space: 6 discrete Cartesian moves
        self.action_space = 6

        # Mic marker config (for visualization only)
        self.mic_radius: float = 0.02
        self.mic_color_left = (1.0, 0.0, 0.0)   # red
        self.mic_color_right = (1.0, 0.0, 0.0)  #
        self.mic_offset_z: float = 0.05         # small lift for visibility

        # Matplotlib figure for live preview
        self._fig = plt.figure("Stacked Spectrogram Preview")

        # Build scene and robot
        self._build_scene(vis)
        self._init_robot()

        # Counters
        self.step_count = 0
        self.episode_count = 0
        self.current_cube_pos = None

        # Position sampling & scheduler state
        self.pos_probs = np.ones(self.pos_count, dtype=np.float32) / self.pos_count
        self.env_pos_idx = np.zeros(self.num_envs, dtype=np.int64)

        # Per-episode return per env (for scheduler)
        self._episode_return_per_env = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._done_mask_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # -----------------------------
    # Scene / Robot initialization
    # -----------------------------
    def _build_scene(self, show_viewer: bool):
        """Set up Genesis scene, ground plane, robot, cube, and mic markers."""
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

        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="assets/xml/franka_emika_panda/panda.xml")
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06)),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # Two mic markers per env (left/right)
        self.mic_markers_left = []
        self.mic_markers_right = []
        for _ in range(self.num_envs):
            ml = self.scene.add_entity(
                gs.morphs.Sphere(radius=self.mic_radius, collision=False),
                surface=gs.surfaces.Rough(color=self.mic_color_left),
            )
            mr = self.scene.add_entity(
                gs.morphs.Sphere(radius=self.mic_radius, collision=False),
                surface=gs.surfaces.Rough(color=self.mic_color_right),
            )
            self.mic_markers_left.append(ml)
            self.mic_markers_right.append(mr)

        self.scene.build(n_envs=self.num_envs, env_spacing=(5.0, 5.0))
        self.envs_idx = np.arange(self.num_envs)

    def _init_robot(self):
        """Reset the Franka robot to a neutral pose (fingers fully open)."""
        self.motors_dof = torch.arange(7, device=self.device)
        self.fingers_dof = torch.arange(7, 9, device=self.device)

        # Fingers fully open; adjust if your MJCF differs
        neutral_q = torch.tensor(
            [-.2, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.04, 0.04],
            device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self.franka.set_qpos(neutral_q, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = neutral_q[:, 7:9].clone()  # keep max-open during control

        self.end_effector = self.franka.get_link("hand")

        # Target EE pose via IK
        base_pos = torch.tensor([0.2720, -0.1683, 1.0164], device=self.device)
        base_quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], device=self.device)
        self.pos = base_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.quat = base_quat.unsqueeze(0).repeat(self.num_envs, 1)

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

    def _update_mic_markers(self):
        """Glue visual mic markers to the actual finger tip poses."""
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()

        left_np = left.cpu().numpy()
        right_np = right.cpu().numpy()
        left_np[:, 2] += self.mic_offset_z
        right_np[:, 2] += self.mic_offset_z

        left_pos_t = torch.from_numpy(left_np).float().to(self.device)
        right_pos_t = torch.from_numpy(right_np).float().to(self.device)
        for idx in range(self.num_envs):
            self.mic_markers_left[idx].set_pos(left_pos_t[idx].unsqueeze(0), envs_idx=[idx])
            self.mic_markers_right[idx].set_pos(right_pos_t[idx].unsqueeze(0), envs_idx=[idx])

    # -----------------------------
    # Audio: playback & synthesis
    # -----------------------------
    def _warn_once(self, msg: str):
        if not getattr(self, "_audio_warned", False):
            print(f"[Audio] {msg}")
            self._audio_warned = True

    def _play_stacked_buffer(self):
        """
        Concatenate the raw stereo audio snippets at self.sample_offsets and play them.
        Auto-disables playback (with one-time warning) if sounddevice isn't usable.
        """
        if not self.enable_playback or self.num_envs != 1:
            return
        try:
            # Build stacked window from stereo history (listener env only)
            snippets = [self.raw_audio_history[offset] for offset in self.sample_offsets]  # each (Nsamples, 2)
            full_buffer = np.concatenate(snippets, axis=0)  # (Nsamples_total, 2)
            sd.play(full_buffer, self._sd_rate)
            sd.wait()
        except Exception as e:
            self._warn_once(f"Playback disabled ({type(e).__name__}: {e})")
            self.enable_playback = False

    def _generate_source_buffer(self) -> np.ndarray:
        """
        Generate a 10 ms mono 'scene' signal: carrier + background noise.
        One shared source per step so ITD (time shift) is meaningful.
        """
        sr, dur = self._sr, self._slice_dur
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)

        # ---- Carrier by current_sound_id (same as before, but mono) ----
        sid = self.current_sound_id
        if sid == 0:
            carrier = chirp(t, f0=1000, f1=1000, t1=dur, method="linear")
        elif sid == 1:
            carrier = chirp(t, f0=600, f1=2400, t1=dur, method="quadratic")
        elif sid == 2:
            base = chirp(t, f0=1500, f1=1500, t1=dur, method="linear")
            mod = 0.5 * (1.0 + np.sin(2 * np.pi * 8 * t))  # tremolo @ 8 Hz
            carrier = base * mod
        else:
            carrier = chirp(t, f0=200, f1=4000, t1=dur, method="linear")

        # ---- Background scene noise (shared across ears) ----
        noise = sum(
            np.random.rand() * chirp(
                t,
                np.random.randint(300, 3000),
                np.random.randint(300, 3000),
                dur
            )
            for _ in range(5)
        ) * 0.1

        scene = carrier + noise
        return scene

    def _compute_spectrogram(self, audio: np.ndarray) -> torch.Tensor:
        """
        Return dB spectrogram slice with shape (F, 1).
        (We keep one time column per 10 ms snippet for stacking later.)
        """
        S = librosa.stft(audio, n_fft=512, hop_length=256)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=1.0)[:self.freq_bins, :1]
        # Explicit normalization: [-20 dB, 120 dB] → [0,1]
        S_db_normalized = (S_db + 20) / 140
        S_db_normalized = np.clip(S_db_normalized, 0.0, 1.0)
        return torch.from_numpy(S_db).float()  # (F, 1)

    def _collect_spectrograms(self, play_audio: bool = False) -> torch.Tensor:
        """
        For each env:
          - build one mono source buffer (scene),
          - compute ear delays TL, TR = d/c,
          - realign by subtracting min(TL, TR) so at least one ear has 0 delay,
          - apply fractional delay + inverse-square attenuation per ear,
          - (optional) add small independent mic noise,
          - compute spectrograms and return (N, 2, F, 1).
        """
        left_pos = self.franka.get_link("left_finger").get_pos()
        right_pos = self.franka.get_link("right_finger").get_pos()
        cube_pos = self.cube.get_pos()

        dL = torch.norm(left_pos - cube_pos, dim=1).cpu().numpy()
        dR = torch.norm(right_pos - cube_pos, dim=1).cpu().numpy()

        specs = []
        for i in range(self.num_envs):
            # One coherent source for this env
            scene = self._generate_source_buffer()

            # Propagation delays (absolute)
            TL = dL[i] / self._c_sound
            TR = dR[i] / self._c_sound

            # Re-align so that the earliest arrival is time 0
            T0 = min(TL, TR)
            dL_rel = TL - T0
            dR_rel = TR - T0

            # Apply fractional delays
            xL = self._fractional_delay(scene, dL_rel, self._sr)
            xR = self._fractional_delay(scene, dR_rel, self._sr)

            # Inverse-square attenuation (ILD)
            eps = 1e-6
            xL = xL / (dL[i] ** 2 + eps)
            xR = xR / (dR[i] ** 2 + eps)

            # Optional additional mic noise (independent per ear)
            sigma = float(self.noise_config.get("audio_noise_level", 0.0))
            if sigma > 0:
                xL = xL + np.random.normal(0, sigma, size=xL.shape)
                xR = xR + np.random.normal(0, sigma, size=xR.shape)

            # Optional immediate playback (listener env only)
            if play_audio and i == self.listen_idx and self.num_envs == 1:
                sd.play(np.column_stack([xL, xR]), self._sr)
                sd.wait()

            # Record stereo raw audio only for the listener index
            if i == self.listen_idx:
                self.raw_audio_history.append(np.column_stack([xL, xR]))  # (Nsamples, 2)

            # Spectrograms per channel -> stack into (2, F, 1)
            S_L = self._compute_spectrogram(xL)
            S_R = self._compute_spectrogram(xR)
            S_pair = torch.stack([S_L, S_R], dim=0)  # (2, F, 1)
            specs.append(S_pair)

        self.step_count += 1
        return torch.stack(specs, dim=0).to(self.device)  # (N, 2, F, 1)

    # -----------------------------
    # Observation building & plots
    # -----------------------------
    def _build_observation(self) -> torch.Tensor:
        """
        Stack past spectrogram slices at defined offsets -> (N, 2, F, T),
        where T = len(sample_offsets).
        """
        slices = [self.audio_history[offset] for offset in self.sample_offsets]  # each: (N, 2, F, 1)
        stacked = torch.cat(slices, dim=3)  # -> (N, 2, F, T)
        return stacked

    def _plot_stacked(self, data: torch.Tensor):
        """
        Very simple wide plot: Left ear (L) on the left, Right ear (R) on the right.
        `data` is (2, F, Tstack) for the listener env.
        """
        plt.clf()
        dL = data[0].cpu().numpy()  # (F, T)
        dR = data[1].cpu().numpy()  # (F, T)

        # Create a wide image by concatenating L and R along time (width)
        combo = np.concatenate([dL, dR], axis=1)  # (F, 2*T)

        # Normalize to [0,1] for display only (training uses raw dB values)
        combo_norm = (combo + 20.0) / 140.0
        combo_norm = np.clip(combo_norm, 0.0, 1.0)

        # Time extent: ~10 ms per column; doubled width due to [L|R]
        T = data.shape[2]
        extent = [0, 2 * 10 * T, 0, (self._sd_rate / 2) / 1000]  # ms, kHz

        plt.imshow(combo_norm, origin='lower', aspect='auto', extent=extent, vmin=0, vmax=1, cmap='magma')
        # Thin center divider between L and R halves
        plt.axvline(x=10 * T, linewidth=1)

        plt.xlabel('Time (ms)   [Left | Right]')
        plt.ylabel('Frequency (kHz)')
        plt.title(f'Step {self.step_count}  Binaural Spectrogram (L → R)')
        plt.colorbar(label='Normalized amplitude')
        plt.draw()
        plt.pause(0.01)
        self._fig.canvas.flush_events()

    # -----------------------------
    # Reset / Step
    # -----------------------------
    def reset(self) -> torch.Tensor:
        """
        Reset environment and start a new episode.

        Modes:
          - INFERENCE (self.inference_mode=True): cycle deterministically through positions.
          - TRAINING (default): parallel sampling per env + anti-collapse probability update.
        Returns: obs (N, 2, F, T), done_flags (N,)
        """
        # =======================
        # Inference mode (sequential positions)
        # =======================
        if getattr(self, "inference_mode", False):
            self.episode_count += 1
            self._episode_return = 0.0
            self._episode_active = True

            if not hasattr(self, "_infer_cycle_idx"):
                self._infer_cycle_idx = 0
            else:
                self._infer_cycle_idx = (self._infer_cycle_idx + 1) % self.pos_count

            self.current_idx = self._infer_cycle_idx
            one_pos = self.cube_positions[self.current_idx].reshape(1, -1)
            self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)

            # Choose sound for this episode
            self.current_sound_id = np.random.randint(0, self.n_sounds)

            # Reset robot & place cube
            self._init_robot()
            self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
            self.scene.step()

            # Update mic markers on the fingers
            self._update_mic_markers()

            print(f"[Inference] Episode {self.episode_count}: pos idx={self.current_idx} "
                  f"pos={self.current_cube_pos[0]} | sound_id={self.current_sound_id}")

            # Prime histories with the first slice
            self.audio_history.clear()
            self.raw_audio_history.clear()
            first_spec = self._collect_spectrograms(play_audio=False)  # (N, 2, F, 1)
            # Store the last raw stereo buffer we just created
            first_raw = self.raw_audio_history[-1].copy() if len(self.raw_audio_history) else np.zeros((int(self._sd_rate*0.01), 2))

            self.audio_history.clear()
            self.raw_audio_history.clear()
            for _ in range(self.history_length):
                self.audio_history.append(first_spec.clone())
                self.raw_audio_history.append(first_raw.copy())

            obs = self._build_observation()
            if self.num_envs == 1 and getattr(self, "_fig", None) is not None:
                self._plot_stacked(obs[0])  # (2, F, T)

            done_array = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return obs, done_array

        # =======================
        # Training mode (parallel sampling + anti-collapse)
        # =======================
        if getattr(self, "_episode_active", False):
            # Aggregate per-position returns from last episode
            per_env = self._episode_return_per_env.detach().cpu().numpy()
            per_pos_returns = np.zeros(self.pos_count, dtype=np.float64)
            for i in range(self.pos_count):
                per_pos_returns[i] = per_env[self.env_pos_idx == i].sum()

            # Compute next distribution (reverse-rank)
            self.pos_probs = self._anti_collapse_probs(per_pos_returns)

            counts = np.bincount(self.env_pos_idx, minlength=self.pos_count)
            pr = np.round(per_pos_returns, 3).tolist()
            pp = np.round(self.pos_probs, 3).tolist()
            print(f"[EP end] per-pos return={pr} | assignment_counts={counts.tolist()} | next p={pp}")

        # Start new episode
        self.episode_count += 1
        self._episode_active = True
        self._episode_return = 0.0
        self._episode_return_per_env.zero_()
        self._done_mask_episode.zero_()

        self.env_pos_idx = np.random.choice(self.pos_count, size=self.num_envs, p=self.pos_probs)
        self.current_cube_pos = np.stack([self.cube_positions[i] for i in self.env_pos_idx], axis=0)

        # Choose sound for this episode
        self.current_sound_id = np.random.randint(0, self.n_sounds)

        # Reset robot & place cube per env
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        # Update mic markers on the fingers
        self._update_mic_markers()

        print(f"Episode {self.episode_count}: parallel sampling — "
              f"pos_probs={np.round(self.pos_probs, 3).tolist()} | sound_id={self.current_sound_id}")

        # Prime histories with the first slice
        self.audio_history.clear()
        self.raw_audio_history.clear()

        first_spec = self._collect_spectrograms(play_audio=False)  # (N, 2, F, 1)
        first_raw = self.raw_audio_history[-1].copy() if len(self.raw_audio_history) else np.zeros((int(self._sd_rate*0.01), 2))

        self.audio_history.clear()
        self.raw_audio_history.clear()
        for _ in range(self.history_length):
            self.audio_history.append(first_spec.clone())
            self.raw_audio_history.append(first_raw.copy())

        obs = self._build_observation()
        if self.num_envs == 1 and getattr(self, "_fig", None) is not None:
            self._plot_stacked(obs[0])  # (2, F, T)

        done_array = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return obs, done_array

    def step(self, actions: torch.Tensor):
        """
        Discrete Cartesian moves, audio/spectrogram update, reward, and done flags.

        - success @ <= self.success_thresh meters -> +self.success_bonus and done=True (per-env)
        - base reward: exp(-4*dist), clamped to [0, 1]
        - per-env episode return is accumulated only until that env's first success
        - plays back the stacked stereo audio every `show_every` steps for num_envs==1
        - updates two mic markers (on the two finger tips) every step
        """
        # Ensure actions are 1D int tensor on the right device
        actions = actions.long().to(self.device).view(-1)  # [num_envs]

        # Discrete deltas
        deltas = torch.tensor([
            [0.05, 0.00, 0.00],   # +x
            [-0.05, 0.00, 0.00],  # -x
            [0.00, 0.05, 0.00],   # +y
            [0.00, -0.05, 0.00],  # -y
            [0.00, 0.00, 0.05],   # +z
            [0.00, 0.00, -0.05],  # -z
        ], device=self.device, dtype=torch.float32)

        step_delta = deltas[actions]  # [num_envs, 3]
        self.pos = self.pos + step_delta  # update EE targets

        # IK control to reach the new pose; keep fingers open
        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=self.pos, quat=self.quat)
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        # Update mic markers on the fingers
        self._update_mic_markers()

        # Collect new spectrogram slice and append to history
        new_slice = self._collect_spectrograms(play_audio=False)  # (N, 2, F, 1)
        self.audio_history.append(new_slice)

        # Optional: play the full (stacked) stereo buffer at intervals (listener env only)
        if self.num_envs == 1 and (self.step_count % self.show_every == 0):
            self._play_stacked_buffer()

        # Build stacked observation (N, 2, F, T)
        obs = self._build_observation()
        if self.num_envs == 1 and self.step_count % self.show_every == 0 and getattr(self, "_fig", None) is not None:
            self._plot_stacked(obs[0])  # (2, F, T)

        # ---- Distance & reward (use midpoint of fingers) ----
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube_pos = self.cube.get_pos()
        dist = torch.norm((left + right) / 2 - cube_pos, dim=1)  # [num_envs]

        base_reward = torch.clamp(torch.exp(-4 * dist), 0.0, 1.0)

        success_mask = (dist <= self.success_thresh)  # [num_envs] bool
        if self.success_bonus < 0:
            bonus = torch.zeros_like(success_mask, dtype=torch.float32, device=self.device)
        else:
            bonus = success_mask.float() * self.success_bonus

        rewards = (base_reward + bonus).to(self.device)

        # Accumulate per-episode return *per env* until first success for that env
        active_mask = (~self._done_mask_episode).float()
        self._episode_return_per_env += rewards * active_mask
        self._done_mask_episode |= success_mask
        self._episode_return += rewards.mean().item()

        # Done flags (per env)
        dones = success_mask.to(self.device) if self.report_success_as_done \
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return obs, rewards, dones

    # -----------------------------
    # Anti-collapse scheduler
    # -----------------------------
    def _anti_collapse_probs(self, per_pos_returns: np.ndarray, min_prob: float = 0.05) -> np.ndarray:
        R = np.maximum(per_pos_returns.astype(np.float64), 0.0)
        S = R.sum()
        if not np.isfinite(S) or S <= 0.0:
            q = np.ones(self.pos_count, dtype=np.float64) / self.pos_count
            return q.astype(np.float32)

        p = R / S
        order_low_to_high = np.argsort(R, kind='mergesort')
        p_sorted_high_to_low = np.sort(p)[::-1]

        q = np.empty_like(p)
        q[order_low_to_high] = p_sorted_high_to_low
        q = q / q.sum()

        if min_prob and min_prob > 0.0:
            q = np.maximum(q, min_prob)
            q = q / q.sum()
        return q.astype(np.float32)

    def _fractional_delay(self, x: np.ndarray, delay_sec: float, sr: int) -> np.ndarray:
        """
        Delay mono signal x by delay_sec (can be fractional). Output length == input length.
        Simple linear interpolation with zero padding for pre-delay.
        """
        if abs(delay_sec) < 1e-9:
            return x.copy()
        d_samp = delay_sec * sr
        n = np.arange(len(x))
        # fractional index source positions
        src_idx = n - d_samp
        # integer neighbors
        i0 = np.floor(src_idx).astype(int)
        i1 = i0 + 1
        frac = src_idx - i0

        y = np.zeros_like(x)
        valid0 = (i0 >= 0) & (i0 < len(x))
        valid1 = (i1 >= 0) & (i1 < len(x))

        # y = (1-frac)*x[i0] + frac*x[i1]
        y_part0 = np.zeros_like(x)
        y_part1 = np.zeros_like(x)
        y_part0[valid0] = (1.0 - frac[valid0]) * x[i0[valid0]]
        y_part1[valid1] = frac[valid1] * x[i1[valid1]]
        y = y_part0 + y_part1
        return y


if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoAudioStackedEnv(vis=True, device=torch.device('cuda'), listen_idx=0)
