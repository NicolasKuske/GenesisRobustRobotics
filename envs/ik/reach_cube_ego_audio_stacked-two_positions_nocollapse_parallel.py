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
    Audio-only IK reaching environment with a Franka end-effector.

    Cube-only + two-position curriculum (hard/easy):
      - Training: anti-collapse per-episode update over per-position returns; sample positions per env.
      - Inference: deterministic alternating hard/easy shared across envs.

    Observations
      - Stacked spectrogram slices at indices `sample_offsets`, shape (N, 1, F, K),
        where each slice is the LAST STFT time bin (F x 1) and K=len(sample_offsets).

    Reward / done
      - base = clamp(exp(-4 * dist), 0, 1)
      - +success_bonus if dist <= success_thresh
      - optional done on success (per-env)
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
        noise_config: dict | None = None,
        # reward / done
        success_thresh: float = 0.3001,
        success_bonus: float = 20.0,
        report_success_as_done: bool = True,
        # positions (two-point curriculum)
        hard_pos=None,                 # defaults to [0.2,  0.8, 0.2] if None
        easy_pos=None,                 # defaults to [0.2, -0.8, 0.2] if None
        # curriculum controls
        inference_mode: bool = False,  # deterministic alternating eval
        min_prob: float = 0.05,        # probability floor in anti-collapse
        ensure_each_position: bool = True,  # try to include both positions when num_envs >= 2
    ):
        # --- Configuration ---
        self.device = device
        self.num_envs = num_envs
        self.listen_idx = int(listen_idx)
        self.show_every = int(show_every)

        # Reward / done
        self.success_thresh = float(success_thresh)
        self.success_bonus = float(success_bonus)
        self.report_success_as_done = bool(report_success_as_done)

        # Two cube positions
        hp = np.array(hard_pos if hard_pos is not None else [0.2,  0.8, 0.2], dtype=float).reshape(3,)
        ep = np.array(easy_pos if easy_pos is not None else [0.2, -0.8, 0.2], dtype=float).reshape(3,)
        self.cube_positions = [hp, ep]
        self.pos_names = ["hard", "easy"]
        self.pos_count = 2

        # Curriculum state
        self.pos_probs = np.ones(self.pos_count, dtype=np.float32) / self.pos_count  # start uniform
        self.env_pos_idx = np.zeros(self.num_envs, dtype=np.int64)  # filled on reset()
        self.min_prob = float(min_prob)
        self.ensure_each_position = bool(ensure_each_position)
        self.inference_mode = bool(inference_mode)
        self._infer_cycle_idx = 0  # 0 -> hard, 1 -> easy

        # Episode accounting
        self._episode_active = False
        self._episode_return = 0.0
        self._episode_return_per_env = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._done_mask_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # History for spectrograms and raw audio
        self.history_length = int(history_length)
        self.sample_offsets = sample_offsets or [-21, -16, -11, -6, -1]
        self.audio_history = deque(maxlen=self.history_length)      # each: (N, F, 1)
        self.raw_audio_history = deque(maxlen=self.history_length)  # each: np.ndarray (10 ms)

        # Noise config
        self.noise_config = noise_config if noise_config else {"audio_noise_level": 0.0}

        # Spectrogram dimensions
        self.freq_bins = 257
        self.obs_shape = (1, self.freq_bins, len(self.sample_offsets))
        self.action_space = 6

        # Matplotlib figure for live preview
        try:
            self._fig = plt.figure("Stacked Spectrogram Preview")
        except Exception:
            self._fig = None

        # Playback control
        self.enable_playback = True
        self._audio_warned = False
        self._sd_rate = 22050

        # Build the simulation scene and initialize the robot
        self._build_scene(vis)
        self._init_robot()

        # Counters
        self.step_count = 0
        self.episode_count = 0
        self.current_cube_pos = None

    # -------------------------
    # Scene / robot
    # -------------------------
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

        neutral_q = torch.tensor(
            [-.2, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            device=self.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self.franka.set_qpos(neutral_q, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = neutral_q[:, 7:9].clone()
        self.end_effector = self.franka.get_link("hand")

        base_pos = torch.tensor([0.2720, -0.1683, 1.0164], device=self.device)
        self.pos = base_pos.unsqueeze(0).repeat(self.num_envs, 1)
        base_quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], device=self.device)
        self.quat = base_quat.unsqueeze(0).repeat(self.num_envs, 1)

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)

    # -------------------------
    # Curriculum utils
    # -------------------------
    def _anti_collapse_probs(self, per_pos_returns: np.ndarray, min_prob: float) -> np.ndarray:
        """
        Reverse-rank distribution:
          - Higher-return positions -> lower prob next episode;
          - Lower-return positions -> higher prob (bounded below by min_prob).
        """
        R = np.maximum(per_pos_returns.astype(np.float64), 0.0)
        S = R.sum()
        if not np.isfinite(S) or S <= 0.0:
            return (np.ones(self.pos_count, dtype=np.float64) / self.pos_count).astype(np.float32)

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

    def _ensure_each_position_present(self, idxs: np.ndarray) -> np.ndarray:
        """If possible, ensure at least one env is assigned to each position."""
        if self.num_envs < self.pos_count:
            return idxs
        missing = [i for i in range(self.pos_count) if i not in idxs]
        if not missing:
            return idxs
        ptr = 0
        for m in missing:
            if ptr >= self.num_envs:
                break
            idxs[ptr] = m
            ptr += 1
        return idxs

    # -------------------------
    # Audio & observation
    # -------------------------
    def _warn_once(self, msg: str):
        if not getattr(self, "_audio_warned", False):
            print(f"[Audio] {msg}")
            self._audio_warned = True

    def _play_stacked_buffer(self):
        """Concatenate the raw audio snippets at self.sample_offsets and play them (single-env only)."""
        if not self.enable_playback or self.num_envs != 1:
            return
        try:
            snippets = [self.raw_audio_history[offset] for offset in self.sample_offsets]
            full_buffer = np.concatenate(snippets, axis=0)
            sd.play(full_buffer, self._sd_rate)
            sd.wait()
        except Exception as e:
            self._warn_once(f"Playback disabled ({type(e).__name__}: {e})")
            self.enable_playback = False

    def simulate_audio(self, dist: float) -> np.ndarray:
        """10 ms audio slice; inverse-square amplitude; background chirp noise + Gaussian noise."""
        sr, dur = 22050, 0.01
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)

        # Prototype 1 kHz carrier
        carrier = chirp(t, f0=1000, f1=1000, t1=dur, method="linear")
        tone = carrier / (dist ** 2 + 1e-6)

        # Background mixture of small chirps
        noise = sum(
            np.random.rand() * chirp(
                t,
                np.random.randint(300, 3000),
                np.random.randint(300, 3000),
                dur
            )
            for _ in range(5)
        ) * 0.1

        # Additional configurable Gaussian noise
        audio_noise_level = float(self.noise_config.get("audio_noise_level", 0.0))
        additional_noise = np.random.normal(0, audio_noise_level, tone.shape)

        return tone + noise + additional_noise

    def _compute_spectrogram_slice(self, audio: np.ndarray) -> torch.Tensor:
        """
        Return LAST time-bin only: shape (F, 1).
        Normalized from [-20 dB, +120 dB] -> [0, 1].
        """
        S = librosa.stft(audio, n_fft=512, hop_length=256)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=1.0)
        frame = S_db[: self.freq_bins, -1:]  # (F, 1)
        S_norm = (frame + 20.0) / 140.0
        S_norm = np.clip(S_norm, 0.0, 1.0)
        return torch.from_numpy(S_norm).float()  # (F, 1)

    def _collect_spectrograms(self, play_audio: bool = False) -> torch.Tensor:
        """
        Simulate audio per env, optionally play the slice for `listen_idx`,
        convert to (F,1) spectrogram, and stack to (N, F, 1) on device.
        """
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube_pos = self.cube.get_pos()
        dists = torch.norm((left + right) / 2 - cube_pos, dim=1).cpu().numpy()

        specs = []
        for i, dist in enumerate(dists):
            audio = self.simulate_audio(float(dist))
            if i == self.listen_idx:
                self.raw_audio_history.append(audio)

            if play_audio and i == self.listen_idx and self.num_envs == 1:
                try:
                    sd.play(audio, self._sd_rate)
                    sd.wait()
                except Exception as e:
                    self._warn_once(f"Playback disabled ({type(e).__name__}: {e})")
                    self.enable_playback = False

            specs.append(self._compute_spectrogram_slice(audio))

        self.step_count += 1
        return torch.stack(specs, dim=0).to(self.device)  # (N, F, 1)

    def _build_observation(self) -> torch.Tensor:
        """
        Stack past (F,1) slices at offsets -> (N, 1, F, K).
        """
        slices = [self.audio_history[offset] for offset in self.sample_offsets]  # each (N, F, 1)
        stacked = torch.cat(slices, dim=2)  # (N, F, K)
        return stacked.unsqueeze(1)         # (N, 1, F, K)

    # -------------------------
    # Reset
    # -------------------------
    def reset(self) -> torch.Tensor:
        """
        Inference:
          - All envs take the same position, alternating hard/easy deterministically.
        Training:
          - Update anti-collapse probs from last episode's returns, then sample per-env positions
            (optionally guaranteeing both positions appear).
        Returns: obs (N, 1, F, K), done_array (N,)
        """
        # ---- Inference mode ----
        if self.inference_mode:
            self.episode_count += 1
            self._episode_return = 0.0
            self._episode_active = True

            idx = self._infer_cycle_idx % self.pos_count  # 0: hard, 1: easy
            self._infer_cycle_idx = (idx + 1) % self.pos_count

            one_pos = self.cube_positions[idx].reshape(1, 3)
            self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)
            self.env_pos_idx = np.full(self.num_envs, idx, dtype=np.int64)

            # Reset robot & place cube
            self._init_robot()
            self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
            self.scene.step()

            print(f"[Inference] Episode {self.episode_count}: position='{self.pos_names[idx]}' "
                  f"| pos={self.current_cube_pos[0].tolist()}")

            # Prime histories
            self.audio_history.clear()
            self.raw_audio_history.clear()
            first_spec = self._collect_spectrograms(play_audio=False)  # (N, F, 1)
            first_raw = self.raw_audio_history[-1].copy() if len(self.raw_audio_history) else np.zeros(int(self._sd_rate * 0.01), dtype=np.float32)

            self.audio_history.clear()
            self.raw_audio_history.clear()
            for _ in range(self.history_length):
                self.audio_history.append(first_spec.clone())
                self.raw_audio_history.append(first_raw.copy())

            self._episode_return_per_env.zero_()
            self._done_mask_episode.zero_()

            obs = self._build_observation()
            done_array = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if self.num_envs == 1 and self._fig is not None:
                self._plot_stacked(obs[0, 0])
            return obs, done_array

        # ---- Training mode ----
        if self._episode_active:
            # Update next-episode probabilities from last episode's returns
            per_env = self._episode_return_per_env.detach().cpu().numpy()
            per_pos_returns = np.zeros(self.pos_count, dtype=np.float64)
            for i in range(self.pos_count):
                per_pos_returns[i] = per_env[self.env_pos_idx == i].sum()
            self.pos_probs = self._anti_collapse_probs(per_pos_returns, min_prob=self.min_prob)

            counts = np.bincount(self.env_pos_idx, minlength=self.pos_count)
            print(f"[EP end] per-pos return={np.round(per_pos_returns, 3).tolist()} "
                  f"| counts={counts.tolist()} "
                  f"| next p={np.round(self.pos_probs, 3).tolist()}")

        # Start a new episode
        self.episode_count += 1
        self._episode_active = True
        self._episode_return = 0.0
        self._episode_return_per_env.zero_()
        self._done_mask_episode.zero_()

        # Sample per-env positions
        self.env_pos_idx = np.random.choice(self.pos_count, size=self.num_envs, p=self.pos_probs)
        if self.ensure_each_position:
            self.env_pos_idx = self._ensure_each_position_present(self.env_pos_idx)
        self.current_cube_pos = np.stack([self.cube_positions[i] for i in self.env_pos_idx], axis=0)

        # Reset robot & place cube
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        unique, counts = np.unique(self.env_pos_idx, return_counts=True)
        asg = {self.pos_names[int(u)]: int(c) for u, c in zip(unique, counts)}
        print(f"Episode {self.episode_count}: pos_probs={np.round(self.pos_probs, 3).tolist()} | assigned={asg}")

        # Prime histories
        self.audio_history.clear()
        self.raw_audio_history.clear()
        first_spec = self._collect_spectrograms(play_audio=False)  # (N, F, 1)
        first_raw = self.raw_audio_history[-1].copy() if len(self.raw_audio_history) else np.zeros(int(self._sd_rate * 0.01), dtype=np.float32)

        self.audio_history.clear()
        self.raw_audio_history.clear()
        for _ in range(self.history_length):
            self.audio_history.append(first_spec.clone())
            self.raw_audio_history.append(first_raw.copy())

        obs = self._build_observation()
        done_array = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.num_envs == 1 and self._fig is not None:
            self._plot_stacked(obs[0, 0])
        return obs, done_array

    # -------------------------
    # Step
    # -------------------------
    def step(self, actions: torch.Tensor):
        """
        Discrete Cartesian moves, audio/spectrogram update, reward, and done flags.

        Notes:
          - success @ <= self.success_thresh meters -> +self.success_bonus and done=True (per-env)
          - base reward: exp(-4*dist), clamped to [0, 1]
          - per-env episode return is accumulated only until that env's first success
          - plays back the stacked audio window every `show_every` steps for num_envs==1
        """
        actions = actions.long().to(self.device).view(-1)  # [num_envs]

        # Move end-effector by discrete deltas (vectorized)
        deltas = torch.tensor([
            [0.05, 0.00, 0.00],  # +x
            [-0.05, 0.00, 0.00],  # -x
            [0.00, 0.05, 0.00],  # +y
            [0.00, -0.05, 0.00],  # -y
            [0.00, 0.00, 0.05],  # +z
            [0.00, 0.00, -0.05],  # -z
        ], device=self.device, dtype=self.pos.dtype)  # [6, 3]
        self.pos = self.pos + deltas[actions]

        # IK control
        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=self.pos, quat=self.quat)
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        # New spectrogram slice and history update
        new_slice = self._collect_spectrograms(play_audio=False)  # (N, F, 1)
        self.audio_history.append(new_slice)

        # Optional playback of full stacked window at intervals
        if self.num_envs == 1 and (self.step_count % self.show_every == 0):
            self._play_stacked_buffer()

        # Build observation and optional live plot
        obs = self._build_observation()
        if self.num_envs == 1 and self._fig is not None and (self.step_count % self.show_every == 0):
            self._plot_stacked(obs[0, 0])

        # Distance & rewards
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube_pos = self.cube.get_pos()
        dist = torch.norm((left + right) / 2 - cube_pos, dim=1)  # [num_envs]

        base_reward = torch.clamp(torch.exp(-4 * dist), 0.0, 1.0)
        success_mask = (dist <= self.success_thresh)

        if self.success_bonus < 0:
            bonus = torch.zeros_like(dist, dtype=torch.float32, device=self.device)
        else:
            bonus = success_mask.float() * self.success_bonus

        rewards = (base_reward + bonus).to(self.device)

        # Accumulate per-episode return *per env* until first success
        active_mask = (~self._done_mask_episode).float()
        self._episode_return_per_env += rewards * active_mask
        self._done_mask_episode |= success_mask

        # Done flags (per env)
        if self.report_success_as_done:
            dones = success_mask.to(self.device)
        else:
            dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return obs, rewards, dones

    # -------------------------
    # Plot (optional)
    # -------------------------
    def _plot_stacked(self, data: torch.Tensor):
        """Render the stacked spectrogram in the live preview figure."""
        if self._fig is None:
            return
        plt.figure(self._fig.number)
        plt.clf()
        extent = [0, 10 * len(self.sample_offsets), 0, (self._sd_rate / 2) / 1000]  # ms vs kHz
        vmin, vmax = 0, 1
        plt.imshow(data.cpu().numpy(), origin='lower', aspect='auto',
                   extent=extent, vmin=vmin, vmax=vmax, cmap='magma')
        plt.colorbar(label='Amplitude (normalized dB)')
        plt.xlabel('Time (ms, stacked offsets)')
        plt.ylabel('Frequency (kHz)')
        plt.title(f'Step {self.step_count} Stacked Spectrogram')
        plt.draw()
        plt.pause(0.01)
        self._fig.canvas.flush_events()


if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoAudioStackedEnv(
        vis=True,
        device=torch.device('cuda'),
        num_envs=8,                   # >1 learns both positions in parallel
        listen_idx=0,
        show_every=25,
        inference_mode=False,         # True -> deterministic alternating evaluation
        hard_pos=[0.2,  0.8, 0.2],
        easy_pos=[0.2, -0.8, 0.2],
        min_prob=0.05,
        ensure_each_position=True,
        noise_config={"audio_noise_level": 0.00},
    )
    obs, _ = env.reset()
    for _ in range(200):
        actions = torch.randint(0, 6, (env.num_envs,), device=env.device)
        obs, rewards, dones = env.step(actions)
        if dones.any():
            print("Done!", dones.nonzero(as_tuple=False).flatten().tolist())
            break
