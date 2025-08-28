#envs/ik/reach_cube_ego_audio_stacked_IK.py


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
    Observations are not_stacked spectrogram frames over a short history.

    Each step returns an observation tensor of shape (num_envs, 1, F, T),
    where F is the number of frequency bins and T is the number of not_stacked time frames.
    Optionally plays back the full not_stacked audio window for the designated listener index.

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
        self.success_bonus = 20.0  # default

        # --- Sound selection (2 sounds, chosen per episode) ---
        self.n_sounds = 4
        self.current_sound_id = 0  # set on reset()

        self.report_success_as_done = True
        self.inference_mode = inference_mode

        self.enable_playback = True  # master switch; auto-disables on failure
        self._audio_warned = False  # print audio warning only once
        self._sd_rate = 22050  # playback sample rate

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

        # Spectrogram dimensions: freq bins and not_stacked time frames
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

        # === Parallel position sampling + anti-collapse scheduler ===
        self.pos_probs = np.ones(self.pos_count, dtype=np.float32) / self.pos_count  # start uniform
        self.env_pos_idx = np.zeros(self.num_envs, dtype=np.int64)  # filled on reset()

        # Per-episode accounting per env (used only for probability update)
        self._episode_return_per_env = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._done_mask_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)



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

        # floor + renormalize
        if min_prob and min_prob > 0.0:
            q = np.maximum(q, min_prob)
            q = q / q.sum()
        return q.astype(np.float32)

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


    def _warn_once(self, msg: str):
        if not getattr(self, "_audio_warned", False):
            print(f"[Audio] {msg}")
            self._audio_warned = True

    def _play_stacked_buffer(self):
        """
        Concatenate the raw audio snippets at self.sample_offsets and play them.
        Auto-disables playback (with one-time warning) if sounddevice isn't usable.
        """
        if not self.enable_playback or self.num_envs != 1:
            return
        try:
            # Build not_stacked window from history (listener env only)
            snippets = [self.raw_audio_history[offset] for offset in self.sample_offsets]
            full_buffer = np.concatenate(snippets, axis=0)
            sd.play(full_buffer, self._sd_rate)
            sd.wait()
        except Exception as e:
            # Disable after first failure so training keeps going without spam
            self._warn_once(f"Playback disabled ({type(e).__name__}: {e})")
            self.enable_playback = False

    def simulate_audio(self, dist: float) -> np.ndarray:
        sr, dur = 22050, 0.01
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)

        if self.current_sound_id == 0:
            carrier = chirp(t, f0=1000, f1=1000, t1=dur, method="linear")
        elif self.current_sound_id == 1:
            carrier = chirp(t, f0=600, f1=2400, t1=dur, method="quadratic")
        elif self.current_sound_id == 2:
            base = chirp(t, f0=1500, f1=1500, t1=dur, method="linear")
            mod = 0.5 * (1.0 + np.sin(2 * np.pi * 8 * t))
            carrier = base * mod
        else:
            carrier = chirp(t, f0=200, f1=4000, t1=dur, method="linear")

        # Distance attenuation (inverse-square law)
        tone = carrier / (dist ** 2 + 1e-6)

        # Base random background “scene” noise (keep as before)
        noise = sum(
            np.random.rand() * chirp(
                t,
                np.random.randint(300, 3000),
                np.random.randint(300, 3000),
                dur
            )
            for _ in range(5)
        ) * 0.1

        # NOTE: no Gaussian "additional noise level" here anymore; we add it after spectrogram normalization
        return tone + noise

    def _compute_spectrogram(self, audio: np.ndarray) -> torch.Tensor:
        S = librosa.stft(audio, n_fft=512, hop_length=256)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=1.0)[:self.freq_bins, :self.time_bins]

        # Normalize [-20, 120] dB -> [0,1]
        S_db_normalized = (S_db + 20.0) / 140.0
        S_db_normalized = np.clip(S_db_normalized, 0.0, 1.0)

        spec = torch.from_numpy(S_db_normalized).float()

        # --- NEW: Gaussian noise in normalized [0,1] spectrogram domain ---
        audio_noise_level = float(self.noise_config.get("audio_noise_level", 0.0))
        if audio_noise_level > 0.0:
            spec = torch.clamp(spec + torch.randn_like(spec) * audio_noise_level, 0.0, 1.0)

        return spec

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
          - INFERENCE (self.inference_mode=True): cycle deterministically through positions.
          - TRAINING (default): parallel sampling per env + anti-collapse probability update.
        """
        # ====================================================
        # Inference mode: cycle positions deterministically
        # ====================================================
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

            # --- choose sound for this episode ---
            self.current_sound_id = np.random.randint(0, self.n_sounds)

            # Reset robot & place cube
            self._init_robot()
            self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
            self.scene.step()

            print(f"[Inference] Episode {self.episode_count}: pos idx={self.current_idx} "
                  f"pos={self.current_cube_pos[0]} | sound_id={self.current_sound_id}")

            # Reset histories + prime
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
        # Training mode: PARALLEL sampling + anti-collapse
        # ====================================================

        # If we are closing a previous episode, update next-episode probabilities
        if getattr(self, "_episode_active", False):
            # Aggregate per-position returns from last episode
            per_env = self._episode_return_per_env.detach().cpu().numpy()
            per_pos_returns = np.zeros(self.pos_count, dtype=np.float64)
            for i in range(self.pos_count):
                per_pos_returns[i] = per_env[self.env_pos_idx == i].sum()

            # Compute next distribution (reverse-rank)
            self.pos_probs = self._anti_collapse_probs(per_pos_returns)

            # Debug info
            counts = np.bincount(self.env_pos_idx, minlength=self.pos_count)
            pr = np.round(per_pos_returns, 3).tolist()
            pp = np.round(self.pos_probs, 3).tolist()
            print(f"[EP end] per-pos return={pr} | assignment_counts={counts.tolist()} | next p={pp}")

        # Start a new episode
        self.episode_count += 1
        self._episode_active = True
        self._episode_return = 0.0
        self._episode_return_per_env.zero_()
        self._done_mask_episode.zero_()

        # First episode: uniform (already set). Thereafter: anti-collapse p from above.
        self.env_pos_idx = np.random.choice(self.pos_count, size=self.num_envs, p=self.pos_probs)
        self.current_cube_pos = np.stack([self.cube_positions[i] for i in self.env_pos_idx], axis=0)

        # --- choose sound for this episode ---
        self.current_sound_id = np.random.randint(0, self.n_sounds)

        # Reset robot & place cube per env
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        print(f"Episode {self.episode_count}: parallel sampling — "
              f"pos_probs={np.round(self.pos_probs, 3).tolist()} | sound_id={self.current_sound_id}")

        # Reset histories + prime
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

        Notes:
          - success @ <= self.success_thresh meters -> +self.success_bonus and done=True (per-env)
          - base reward: exp(-4*dist), clamped to [0, 1]
          - per-env episode return is accumulated only until that env's first success
          - plays back the *not_stacked* audio window every `show_every` steps for num_envs==1
        """
        # Ensure actions are 1D int tensor on the right device
        actions = actions.long().to(self.device).view(-1)  # [num_envs]

        # --- Move end-effector by discrete deltas (vectorized via indexing) ---
        deltas = torch.tensor([
            [0.05, 0.00, 0.00],  # +x
            [-0.05, 0.00, 0.00],  # -x
            [0.00, 0.05, 0.00],  # +y
            [0.00, -0.05, 0.00],  # -y
            [0.00, 0.00, 0.05],  # +z
            [0.00, 0.00, -0.05],  # -z
        ], device=self.device, dtype=self.pos.dtype)  # [6, 3]

        step_delta = deltas[actions]  # [num_envs, 3]
        self.pos = self.pos + step_delta

        # IK control to reach the new pose
        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=self.pos, quat=self.quat)
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        # Collect new spectrogram slice and append to history
        new_slice = self._collect_spectrograms(play_audio=False)  # [num_envs, F, Tslice]
        self.audio_history.append(new_slice)

        # Optional: play the full not_stacked buffer at intervals (listener env only)
        if self.num_envs == 1 and (self.step_count % self.show_every == 0):
            self._play_stacked_buffer()

        # Build not_stacked observation (num_envs, 1, F, Tstack)
        obs = self._build_observation()
        if self.num_envs == 1 and self.step_count % self.show_every == 0:
            self._plot_stacked(obs[0, 0])

        # ---- Distance & reward ----
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
        active_mask = (~self._done_mask_episode).float()  # [num_envs]
        self._episode_return_per_env += rewards * active_mask
        self._done_mask_episode |= success_mask  # latch per-env done

        # Keep existing mean bookkeeping if used elsewhere
        self._episode_return += rewards.mean().item()

        # Done flags (per env)
        if self.report_success_as_done:
            dones = success_mask.to(self.device)
        else:
            dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return obs, rewards, dones


    def _plot_stacked(self, data: torch.Tensor):
        """
        Render the not_stacked spectrogram in the live preview figure.
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
