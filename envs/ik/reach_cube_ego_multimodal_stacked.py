import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import genesis as gs
import librosa
from scipy.signal import chirp
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat
import sounddevice as sd


class ReachCubeEgoMultimodalStackedEnv:
    """
    Multimodal (vision + audio) IK reaching environment with stacked RGB frames and stacked spectrogram slices.

    Curriculum (two positions: hard/easy):
      - Training: anti-collapse per-episode update over per-position returns; sample positions per-env.
      - Inference: deterministic alternating hard/easy shared across envs.

    Observations
      - Vision: stacked RGB frames at indices `sample_offsets`, shape N×(3k)×H×W
      - Audio: stacked spectrogram slices at the same offsets, shape N×1×F×k  (F=257; one STFT time slice per offset)

    Reward / done
      - base = clamp(exp(-4 * dist), 0, 1)
      - +success_bonus if dist <= success_thresh
      - optional done on success (per-env)
    """

    # -------------------------
    # Construction
    # -------------------------
    def __init__(
        self,
        vis: bool,
        device: torch.device,
        num_envs: int = 1,
        listen_idx: int = 0,
        show_every: int = 10,          # audio playback / plotting cadence (steps)
        render_every: int = 5,         # frame skip for vision/audio generation
        randomize_every: int = 100,    # kept for compatibility; not used
        noise_config: dict | None = None,

        # reward/done parity
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

        # history / stacking
        history_length: int = 25,
        sample_offsets=None,           # e.g. [-21, -16, -11, -6, -1]

        # --- legacy schedule params (ignored, kept for compatibility) ---
        easy_episodes: int = 1,
        hard_episodes: int = 1,
        cube_positions=None,
    ):
        # --- core params ---
        self.device = device
        self.num_envs = num_envs
        self.listen_idx = int(listen_idx)
        self.show_every = int(show_every)
        self.render_every = int(render_every)
        self.randomize_every = int(randomize_every)  # not used

        # reward / done config
        self.success_thresh = float(success_thresh)
        self.success_bonus = float(success_bonus)
        self.report_success_as_done = bool(report_success_as_done)

        # noise config
        self.noise_config = noise_config if noise_config else {}
        self._visual_noise_level = float(self.noise_config.get("visual_noise_level",
                                         self.noise_config.get("visual", 0.0)))
        self._audio_noise_level = float(self.noise_config.get("audio_noise_level",
                                        self.noise_config.get("audio", 0.0)))

        # positions / curriculum setup
        hp = np.array(hard_pos if hard_pos is not None else [0.2,  0.8, 0.2], dtype=float).reshape(3,)
        ep = np.array(easy_pos if easy_pos is not None else [0.2, -0.8, 0.2], dtype=float).reshape(3,)
        self.cube_positions = [hp, ep]
        self.pos_names = ["hard", "easy"]
        self.pos_count = 2

        self.pos_probs = np.ones(self.pos_count, dtype=np.float32) / self.pos_count
        self.env_pos_idx = np.zeros(self.num_envs, dtype=np.int64)
        self.current_cube_pos = None  # (N, 3)
        self.min_prob = float(min_prob)
        self.ensure_each_position = bool(ensure_each_position)

        # accounting
        self.episode_count = 0
        self.step_count = 0
        self._step_count = 0  # for frame-skip rendering/audio

        # playback controls
        self.enable_playback = True  # auto-disables if sounddevice fails
        self._audio_warned = False
        self._sd_rate = 22050

        # --- history config ---
        self.history_length = int(history_length)
        self.sample_offsets = sample_offsets or [-21, -16, -11, -6, -1]  # time indices into deques

        # action shape
        self.action_space = 6

        # Vision history/shape
        self.image_history = deque(maxlen=self.history_length)
        self.obs_shape_vision = (3 * len(self.sample_offsets), 120, 120)

        # Audio history/shape
        self.audio_history = deque(maxlen=self.history_length)       # each: [N, F, 1]
        self.raw_audio_history = deque(maxlen=self.history_length)   # each: 1D np array (10 ms)
        self.freq_bins = 257
        self.obs_shape_audio = (1, self.freq_bins, len(self.sample_offsets))  # one time slice per offset

        # plotting
        self.enable_plotting = (num_envs == 1)
        try:
            self.fig_multimodal = plt.figure("Multimodal Observation", figsize=(14, 6))
        except Exception:
            self.enable_plotting = False

        # per-episode accounting (success bookkeeping)
        self._episode_active = False
        self._episode_return = 0.0
        self._episode_return_per_env = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._done_mask_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # curriculum mode
        self.inference_mode = bool(inference_mode)
        self._infer_cycle_idx = 0  # 0 -> hard, 1 -> easy

        # --- build scene ---
        self.scene = gs.Scene(
            show_FPS=False,
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
            vis_options=gs.options.VisOptions(show_world_frame=False, plane_reflection=True),
            renderer=gs.renderers.Rasterizer(),
        )

        # plane + walls
        self.scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Aluminium(ior=10.0))
        for pos, color, euler in [
            ((4,  0, 1),  (0.9, 0.9, 0.9),  (0, -20,  0)),
            ((-3, 0, 1),  (0.7, 0.7, 0.7),  (0,  20,  0)),
            ((0, -3, 1),  (0.56, 0.57, 0.58), (0,  20, 90)),
        ]:
            self.scene.add_entity(
                gs.morphs.Box(size=(0.1, 8, 4), pos=pos, euler=euler, collision=False),
                surface=gs.surfaces.Rough(color=color),
                material=gs.materials.Rigid(gravity_compensation=1.0),
            )

        # robot
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="assets/xml/franka_emika_panda/panda.xml"))

        # ----- ONLY CUBE -----
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06), collision=False),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        # cameras mounted to EE
        self.cams = []
        self.cam_transform = trans_quat_to_T(np.array([0.03, 0, 0.03]), xyz_to_quat(np.array([185, 0, 90])))
        for _ in range(self.num_envs):
            self.cams.append(self.scene.add_camera(res=(120, 120), fov=90, GUI=True))

        # parallelize scene
        self._env_space = 100.0
        self.scene.build(n_envs=self.num_envs, env_spacing=(self._env_space, self._env_space))
        self.envs_idx = np.arange(self.num_envs)

        for cam in self.cams:
            cam.start_recording()

        self._init_robot()

    # -------------------------
    # Robot init
    # -------------------------
    def _init_robot(self):
        self.motors_dof = torch.arange(7, device=self.device)
        self.fingers_dof = torch.arange(7, 9, device=self.device)

        q0 = torch.tensor(
            [-0.2, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:, 7:9].clone()
        self.end_effector = self.franka.get_link("hand")

        base_pos = torch.tensor([0.2720, -0.1683, 1.0164], device=self.device)
        self.pos = base_pos.unsqueeze(0).repeat(self.num_envs, 1)
        base_quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], device=self.device)
        self.quat = base_quat.unsqueeze(0).repeat(self.num_envs, 1)

        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=self.pos, quat=self.quat)
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)

    # -------------------------
    # Rendering / observation (vision)
    # -------------------------
    def _render(self) -> torch.Tensor:
        imgs = []
        N = self.num_envs
        cols = int(math.ceil(math.sqrt(N)))
        rows = int(math.ceil(N / cols))
        env_space = self._env_space
        noise_level = float(self._visual_noise_level)

        for idx, cam in enumerate(self.cams):
            ee_pos = self.end_effector.get_pos(envs_idx=[idx])[0].cpu().numpy()
            ee_quat = self.end_effector.get_quat(envs_idx=[idx])[0].cpu().numpy()

            r = idx // cols
            c = idx % cols
            x_off = (c - (cols - 1) / 2) * env_space
            y_off = (r - (rows - 1) / 2) * env_space

            ee_pos_offset = ee_pos + np.array([x_off, y_off, 0.0])
            ee_T = trans_quat_to_T(ee_pos_offset, ee_quat)
            cam_T = ee_T @ self.cam_transform
            cam.set_pose(transform=cam_T)

            rgb = cam.render()[0]  # H×W×3, uint8

            img = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
            if noise_level > 0.0:
                noise = torch.randn_like(img) * (noise_level / 10.0)
                img = torch.clamp(img + noise, 0.0, 1.0)

            imgs.append(img)

        return torch.stack(imgs, dim=0)  # N×3×H×W

    # -------------------------
    # Audio synthesis & spectrogram (cube-only)
    # -------------------------
    def simulate_audio(self, dist: float) -> np.ndarray:
        """
        10 ms audio slice whose amplitude attenuates ~ inverse square with distance.
        Single prototype sound (matches audio env style).
        """
        sr, dur = 22050, 0.01
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        carrier = chirp(t, f0=1000, f1=1000, t1=dur, method="linear")  # steady 1 kHz tone
        tone = carrier / (dist ** 2 + 1e-6)

        # base random background noise (sum of small chirps)
        noise = sum(
            np.random.rand() * chirp(
                t,
                np.random.randint(300, 3000),
                np.random.randint(300, 3000),
                dur
            )
            for _ in range(5)
        ) * 0.1

        # additional configurable Gaussian noise
        additional = np.random.normal(0, float(self._audio_noise_level), tone.shape)

        return tone + noise + additional

    def _compute_spectrogram_slice(self, audio: np.ndarray) -> torch.Tensor:
        """
        Return LAST time-bin only: shape (F, 1).
        (Keeps final stacked observation width = len(sample_offsets).)
        """
        S = librosa.stft(audio, n_fft=512, hop_length=256)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=1.0)
        frame = S_db[: self.freq_bins, -1:]  # (F, 1)

        # Normalize from [-20, +120] dB -> [0, 1]
        S_norm = (frame + 20.0) / 140.0
        S_norm = np.clip(S_norm, 0.0, 1.0)
        return torch.from_numpy(S_norm).float()  # (F, 1)

    def _collect_spectrograms(self, play_audio_slice: bool = False) -> torch.Tensor:
        """
        Render one audio slice per env using distance from gripper midpoint to the cube.
        Returns: [N, F, 1]
        """
        cube_pos = self.cube.get_pos()  # N×3
        gp_l = self.franka.get_link("left_finger").get_pos()
        gp_r = self.franka.get_link("right_finger").get_pos()
        dists = torch.norm(cube_pos - (gp_l + gp_r) / 2, dim=1).cpu().numpy()

        specs = []
        for i, dist in enumerate(dists):
            audio = self.simulate_audio(float(dist))
            if i == self.listen_idx:
                self.raw_audio_history.append(audio)

            if play_audio_slice and i == self.listen_idx and self.num_envs == 1 and self.enable_playback:
                try:
                    sd.play(audio, self._sd_rate); sd.wait()
                except Exception as e:
                    if not self._audio_warned:
                        print(f"[Audio] Playback disabled ({type(e).__name__}: {e})")
                        self._audio_warned = True
                    self.enable_playback = False

            spec = self._compute_spectrogram_slice(audio)
            specs.append(spec)

        return torch.stack(specs, dim=0).to(self.device)  # N×F×1

    # -------------------------
    # Observation builders & plotting
    # -------------------------
    def _build_observation(self):
        # Vision: concat along channel (3 * k)
        vis_obs = torch.cat([self.image_history[i] for i in self.sample_offsets], dim=1)  # N×(3k)×H×W

        # Audio: concat along time for the stacked window (F×k), then add channel dim
        aud_slices = torch.cat([self.audio_history[i] for i in self.sample_offsets], dim=2)  # N×F×k
        return vis_obs, aud_slices.unsqueeze(1)  # vision: N×(3k)×H×W, audio: N×1×F×k

    def _plot_multimodal_obs(self, vis_obs, aud_obs):
        plt.figure(self.fig_multimodal.number)
        plt.clf()

        # Spectrogram (top)
        ax_spec = plt.subplot(2, 1, 1)
        extent = [0, 10 * len(self.sample_offsets), 0, (self._sd_rate / 2) / 1000]
        im = ax_spec.imshow(
            aud_obs[0, 0].cpu().numpy(),
            origin='lower',
            aspect='auto',
            extent=extent,
            vmin=0.0,
            vmax=1.0,
            cmap='magma'
        )
        ax_spec.set_xlabel('Time (ms)')
        ax_spec.set_ylabel('Frequency (kHz)')
        ax_spec.set_title('Stacked Spectrogram')
        cbar = plt.colorbar(im, ax=ax_spec, label='Amplitude (dB)')
        ticks = np.linspace(0, 1, 8)
        labels = np.linspace(-20, 120, 8).astype(int)
        cbar.set_ticks(ticks); cbar.set_ticklabels(labels)

        # Vision frames (bottom)
        k = len(self.sample_offsets)
        frames = vis_obs[0].cpu().numpy().reshape(k, 3, 120, 120)
        for i in range(k):
            ax_img = plt.subplot(2, k, k + i + 1)
            img = np.transpose(frames[i], (1, 2, 0))
            ax_img.imshow(img)
            ax_img.axis('off')
            ax_img.set_title(f'Frame {i + 1}')

        plt.suptitle(f"Step {self.step_count} Multimodal Observation")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.draw()
        plt.pause(0.01)
        self.fig_multimodal.canvas.flush_events()

        # Playback stacked buffer (10 ms × k)
        if self.enable_playback and self.num_envs == 1:
            try:
                snippets = [self.raw_audio_history[i] for i in self.sample_offsets]
                sd.play(np.concatenate(snippets, axis=0), self._sd_rate); sd.wait()
            except Exception as e:
                if not self._audio_warned:
                    print(f"[Audio] Playback disabled ({type(e).__name__}: {e})")
                    self._audio_warned = True
                self.enable_playback = False

    # -------------------------
    # Curriculum utils
    # -------------------------
    def _anti_collapse_probs(self, per_pos_returns: np.ndarray, min_prob: float) -> np.ndarray:
        """
        Reverse-rank distribution:
          - Positions with higher return -> lower prob next episode;
          - Positions with lower return -> higher prob (bounded below by min_prob).
        """
        R = np.maximum(per_pos_returns.astype(np.float64), 0.0)
        S = R.sum()
        if not np.isfinite(S) or S <= 0.0:
            return (np.ones(self.pos_count, dtype=np.float64) / self.pos_count).astype(np.float32)

        p = R / S
        order_low_to_high = np.argsort(R, kind="mergesort")  # stable
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
    # Reset (anti-collapse curriculum, cube only)
    # -------------------------
    def reset(self):
        """
        Inference:
          - All envs take the same position, alternating hard/easy deterministically.
        Training:
          - Update anti-collapse probs from last episode's returns, then sample per-env positions,
            optionally guaranteeing both positions appear.
        Returns: (vision_obs, audio_obs)
        """
        # ---- Inference mode ----
        if self.inference_mode:
            self.episode_count += 1
            self._episode_active = True
            self._episode_return = 0.0

            idx = self._infer_cycle_idx % self.pos_count  # 0: hard, 1: easy
            self._infer_cycle_idx = (idx + 1) % self.pos_count

            # shared position across all envs
            self.env_pos_idx = np.full(self.num_envs, idx, dtype=np.int64)
            pos1 = self.cube_positions[idx].reshape(1, 3)
            self.current_cube_pos = np.repeat(pos1, self.num_envs, axis=0)

            # reset robot & place cube
            self._init_robot()
            self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
            self.scene.step()

            print(f"[Inference] Episode {self.episode_count}: position='{self.pos_names[idx]}' "
                  f"| pos={self.current_cube_pos[0].tolist()}")

            # prime histories
            self.image_history.clear(); self.audio_history.clear(); self.raw_audio_history.clear()
            first_img = self._render()
            first_spec = self._collect_spectrograms(play_audio_slice=False)
            first_raw = self.raw_audio_history[-1].copy() if len(self.raw_audio_history) else np.zeros(int(self._sd_rate*0.01), dtype=np.float32)

            self.image_history.clear(); self.audio_history.clear(); self.raw_audio_history.clear()
            for _ in range(self.history_length):
                self.image_history.append(first_img.clone())
                self.audio_history.append(first_spec.clone())
                self.raw_audio_history.append(first_raw.copy())

            self._episode_return_per_env.zero_()
            self._done_mask_episode.zero_()
            self._step_count = 0
            self.step_count = 0
            return self._build_observation()

        # ---- Training mode ----
        if self._episode_active:
            # update probs from last episode
            per_env = self._episode_return_per_env.detach().cpu().numpy()
            per_pos_returns = np.zeros(self.pos_count, dtype=np.float64)
            for i in range(self.pos_count):
                per_pos_returns[i] = per_env[self.env_pos_idx == i].sum()
            self.pos_probs = self._anti_collapse_probs(per_pos_returns, min_prob=self.min_prob)

            counts = np.bincount(self.env_pos_idx, minlength=self.pos_count)
            print(f"[EP end] per-pos return={np.round(per_pos_returns, 3).tolist()} "
                  f"| counts={counts.tolist()} "
                  f"| next p={np.round(self.pos_probs, 3).tolist()}")

        # start new episode
        self.episode_count += 1
        self._episode_active = True
        self._episode_return = 0.0
        self._episode_return_per_env.zero_()
        self._done_mask_episode.zero_()

        # sample per-env positions
        self.env_pos_idx = np.random.choice(self.pos_count, size=self.num_envs, p=self.pos_probs)
        if self.ensure_each_position:
            self.env_pos_idx = self._ensure_each_position_present(self.env_pos_idx)
        self.current_cube_pos = np.stack([self.cube_positions[i] for i in self.env_pos_idx], axis=0)

        # reset robot & place cube
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        unique, counts = np.unique(self.env_pos_idx, return_counts=True)
        asg = {self.pos_names[int(u)]: int(c) for u, c in zip(unique, counts)}
        print(f"Episode {self.episode_count}: pos_probs={np.round(self.pos_probs, 3).tolist()} | assigned={asg}")

        # prime histories with first multimodal slice
        self.image_history.clear(); self.audio_history.clear(); self.raw_audio_history.clear()
        first_img = self._render()
        first_spec = self._collect_spectrograms(play_audio_slice=False)
        first_raw = self.raw_audio_history[-1].copy() if len(self.raw_audio_history) else np.zeros(int(self._sd_rate*0.01), dtype=np.float32)

        self.image_history.clear(); self.audio_history.clear(); self.raw_audio_history.clear()
        for _ in range(self.history_length):
            self.image_history.append(first_img.clone())
            self.audio_history.append(first_spec.clone())
            self.raw_audio_history.append(first_raw.copy())

        self._step_count = 0
        self.step_count = 0
        return self._build_observation()

    # -------------------------
    # Step
    # -------------------------
    def step(self, actions: torch.Tensor):
        """
        actions: int tensor [N] with values in {0..5} for ±x, ±y, ±z discrete moves.
        Returns: (vision_obs, audio_obs), rewards [N], dones [N]
        """
        actions = actions.long().to(self.device).view(-1)  # [N]

        # discrete Cartesian moves (vectorized by indexing)
        deltas = torch.tensor([
            [0.05, 0.00, 0.00],   # +x
            [-0.05, 0.00, 0.00],  # -x
            [0.00, 0.05, 0.00],   # +y
            [0.00, -0.05, 0.00],  # -y
            [0.00, 0.00, 0.05],   # +z
            [0.00, 0.00, -0.05],  # -z
        ], device=self.device, dtype=self.pos.dtype)
        pos = self.pos + deltas[actions]

        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=pos, quat=self.quat)
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        # frame-skip render & audio
        self._step_count += 1
        if self._step_count % self.render_every == 0:
            new_img = self._render()
            new_spec = self._collect_spectrograms(play_audio_slice=False)  # [N, F, 1]; appends raw for listen_idx
            last_raw = self.raw_audio_history[-1].copy() if len(self.raw_audio_history) else np.zeros(int(self._sd_rate*0.01), dtype=np.float32)
        else:
            new_img = self.image_history[-1]
            new_spec = self.audio_history[-1]
            last_raw = self.raw_audio_history[-1] if len(self.raw_audio_history) else np.zeros(int(self._sd_rate*0.01), dtype=np.float32)

        # update histories
        self.image_history.append(new_img)
        self.audio_history.append(new_spec)
        self.raw_audio_history.append(last_raw.copy())
        self.step_count += 1

        # build obs
        vis_obs, aud_obs = self._build_observation()

        # optional debug plotting & audio playback
        if self.num_envs == 1 and (self.step_count % self.show_every == 0) and self.enable_plotting:
            self._plot_multimodal_obs(vis_obs, aud_obs)

        # distance to cube (fingers midpoint)
        cube_pos = self.cube.get_pos()  # N×3
        gp_l = self.franka.get_link("left_finger").get_pos()
        gp_r = self.franka.get_link("right_finger").get_pos()
        dist = torch.norm(cube_pos - (gp_l + gp_r) / 2, dim=1)  # [N]

        # reward
        base_reward = torch.clamp(torch.exp(-4 * dist), 0.0, 1.0)
        success_mask = (dist <= self.success_thresh)
        if self.success_bonus < 0:
            bonus = torch.zeros_like(dist, dtype=torch.float32, device=self.device)
        else:
            bonus = success_mask.float() * self.success_bonus
        rewards = (base_reward + bonus).to(self.device)

        # per-env episode accounting until first success
        active_mask = (~self._done_mask_episode).float()
        self._episode_return_per_env += rewards * active_mask
        self._done_mask_episode |= success_mask

        # dones
        if self.report_success_as_done:
            dones = success_mask.to(self.device)
        else:
            dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # keep pos for next step
        self.pos = pos
        return (vis_obs, aud_obs), rewards, dones


if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoMultimodalStackedEnv(
        vis=True,
        device=torch.device("cuda"),
        num_envs=8,                   # >1 learns both positions in parallel
        listen_idx=0,
        show_every=25,
        render_every=5,
        inference_mode=False,         # True -> deterministic alternating evaluation
        hard_pos=[0.2, 0.8, 0.2],
        easy_pos=[0.2, -0.8, 0.2],
        min_prob=0.05,
        ensure_each_position=True,
        noise_config={"visual_noise_level": 0.00, "audio_noise_level": 0.00},
        # legacy args kept for compatibility; ignored:
        hard_episodes=1,
        easy_episodes=1,
        cube_positions=None,
    )
    vis_obs, aud_obs = env.reset()
    for _ in range(200):
        actions = torch.randint(0, 6, (env.num_envs,), device=env.device)
        (vis_obs, aud_obs), rewards, dones = env.step(actions)
        if dones.any():
            print("Done!", dones.nonzero(as_tuple=False).flatten().tolist())
            break
