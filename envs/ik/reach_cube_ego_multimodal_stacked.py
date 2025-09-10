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

    Key features:
      - Reward parity with audio/vision envs:
          base_reward = clamp(exp(-2 * dist), 0, 1)   # (doc updated to match code)
          success @ dist <= success_thresh -> +success_bonus and (optionally) done=True
          per-env episode return is accumulated only until first success
      - Parallel position curriculum with anti-collapse probability updates (training)
        and deterministic cycling (inference).
      - Four objects bound to four sounds:
          cube↔0, sphere↔1, bunny↔2, dragon↔3 (one object kind per episode, shared across envs).
      - Vision: stacked RGB with frame skipping.
      - Audio: stacked spectrogram slices; optional playback of the stacked raw buffer.
      - NEW: `train_object_count` (1..4) restricts which object kinds can be sampled per episode:
          1=cube; 2=cube+sphere; 3=cube+sphere+bunny; 4=all.
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
        noise_config: dict | None = None,
        # reward/done parity
        success_thresh: float = 0.3001,
        success_bonus: float = 20.0,
        report_success_as_done: bool = True,
        # curriculum controls
        cube_positions=None,           # list of (x, y, z); if None, use 3×3 grid
        inference_mode: bool = False,  # when True, cycle positions deterministically
        ### NEW
        train_object_count: int = 4,   # 1..4: restrict eligible object kinds for sampling
    ):
        # --- core params ---
        self.device = device
        self.num_envs = num_envs
        self.listen_idx = listen_idx
        self.show_every = show_every
        self.render_every = render_every
        self.inference_mode = inference_mode

        # reward / done config
        self.success_thresh = float(success_thresh)
        self.success_bonus = float(success_bonus)
        self.report_success_as_done = bool(report_success_as_done)

        # noise config (support both old and new keys)
        self.noise_config = noise_config or {}
        self._audio_noise_level = float(self.noise_config.get("audio_noise_level", 0.0))
        self._visual_noise_level = float(self.noise_config.get("visual_noise_level", 0.0))

        # accounting
        self.episode_count = 0
        self.step_count = 0
        self._step_count = 0  # for frame-skip rendering/audio

        # playback controls
        self.enable_playback = True  # auto-disables if sounddevice fails
        self._audio_warned = False
        self._sd_rate = 22050

        # --- history config ---
        self.history_length = 25
        self.sample_offsets = [-21, -16, -11, -6, -1]  # time indices into deques

        # Vision history/shape
        self.image_history = deque(maxlen=self.history_length)
        self.obs_shape_vision = (3 * len(self.sample_offsets), 120, 120)

        # Audio history/shape
        self.audio_history = deque(maxlen=self.history_length)       # each element: [N, F, 1]
        self.raw_audio_history = deque(maxlen=self.history_length)   # each element: 1D np array (10 ms)
        self.freq_bins = 257
        self.time_bins_per_slice = 1  # 10 ms -> single STFT time step with n_fft=512, hop=256
        self.obs_shape_audio = (1, self.freq_bins, self.time_bins_per_slice * len(self.sample_offsets))

        # --- objects & shape↔sound pairing ---
        self._object_kinds = ["cube", "sphere", "bunny", "dragon"]
        self._shape_to_sound = {0: 0, 1: 1, 2: 2, 3: 3}  # index-aligned

        ### NEW — restrict eligible objects to the first K (1..4)
        self.train_object_count = int(max(1, min(int(train_object_count), len(self._object_kinds))))
        self.allowed_object_ids = list(range(self.train_object_count))
        # Build probs: allowed get 1/K, others 0
        self.shape_probs = np.zeros(len(self._object_kinds), dtype=np.float32)
        self.shape_probs[self.allowed_object_ids] = 1.0 / self.train_object_count

        self._current_object_id = 0
        self._current_object_kind = self._object_kinds[self._current_object_id]
        self.current_sound_id = self._shape_to_sound[self._current_object_id]

        # Discrete action count (±x, ±y, ±z)
        self.action_space = 6

        # --- positions / curriculum ---
        if cube_positions is None:
            self.cube_positions = [
                np.array([0.6, -0.3, 0.6]),
                np.array([0.6, -0.3, 0.4]),
                np.array([0.6, -0.3, 0.2]),
                np.array([0.6,  0.0, 0.6]),
                np.array([0.6,  0.0, 0.4]),
                np.array([0.6,  0.0, 0.2]),
                np.array([0.6,  0.3, 0.6]),
                np.array([0.6,  0.3, 0.4]),
                np.array([0.6,  0.3, 0.2]),
            ]
        else:
            self.cube_positions = [np.array(p, dtype=float).reshape(3,) for p in cube_positions]
        self.pos_count = len(self.cube_positions)
        self.pos_probs = np.ones(self.pos_count, dtype=np.float32) / self.pos_count
        self.env_pos_idx = np.zeros(self.num_envs, dtype=np.int64)
        self.current_cube_pos = None  # will be (N,3) after reset

        # per-episode accounting (for curriculum)
        self._episode_active = False
        self._episode_return = 0.0
        self._episode_return_per_env = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._done_mask_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # inference cycle idx
        self._infer_cycle_idx = 0

        # --- plotting ---
        self.fig_multimodal = plt.figure("Multimodal Observation", figsize=(14, 6))

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

        # objects (collision off; they're targets, not physics obstacles)
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06), collision=False),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        self.sphere = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.03, pos=(0.0, 0.0, -1.0), collision=False),
            surface=gs.surfaces.Rough(color=(0.0, 0.7, 0.0)),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        self.bunny = self.scene.add_entity(
            gs.morphs.Mesh(file="meshes/bunny.obj", scale=0.2, pos=(0.0, 0.0, -1.0), collision=False),
            surface=gs.surfaces.Rough(color=(0.9, 0.9, 0.9)),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        self.dragon = self.scene.add_entity(
            gs.morphs.Mesh(file="meshes/dragon.obj", scale=0.3, pos=(0.0, 2.0, -1.0), euler=(90, 0, 115), collision=False),
            surface=gs.surfaces.Rough(color=(1.0, 0.6, 0.0)),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        self._object_entities = [self.cube, self.sphere, self.bunny, self.dragon]

        # cameras mounted to EE
        self.cams = []
        self.cam_transform = trans_quat_to_T(np.array([0.03, 0, 0.03]), xyz_to_quat(np.array([185, 0, 90])))
        for _ in range(self.num_envs):
            self.cams.append(self.scene.add_camera(res=(120, 120), fov=90, GUI=True))

        # parallelize scene
        self._env_space = 100.0
        self.scene.build(n_envs=self.num_envs, env_spacing=(self._env_space, self._env_space))
        self.envs_idx = np.arange(self.num_envs)

        # precompute far position (batched) for parking offscreen
        self._far_pos = np.tile(np.array([0.0, 0.0, -100.0], dtype=float), (self.num_envs, 1))

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
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)

    ### NEW
    def set_train_object_count(self, count: int):
        """
        Change the number of eligible object kinds during runtime (1..4).
        Rebuilds shape_probs accordingly for future episodes.
        """
        k = int(max(1, min(int(count), len(self._object_kinds))))
        self.train_object_count = k
        self.allowed_object_ids = list(range(k))
        self.shape_probs = np.zeros(len(self._object_kinds), dtype=np.float32)
        self.shape_probs[self.allowed_object_ids] = 1.0 / k

    # -------------------------
    # Curriculum utils
    # -------------------------
    def _anti_collapse_probs(self, per_pos_returns: np.ndarray, min_prob: float = 0.05) -> np.ndarray:
        """
        Reverse-rank distribution:
        - Higher-return positions get lower probability next episode;
        - Lower-return positions get higher probability (but never below min_prob).
        """
        R = np.maximum(per_pos_returns.astype(np.float64), 0.0)
        S = R.sum()
        if not np.isfinite(S) or S <= 0.0:
            return (np.ones(self.pos_count, dtype=np.float64) / self.pos_count).astype(np.float32)

        p = R / S
        order_low_to_high = np.argsort(R, kind="mergesort")
        p_sorted_high_to_low = np.sort(p)[::-1]

        q = np.empty_like(p)
        q[order_low_to_high] = p_sorted_high_to_low
        q = q / q.sum()

        if min_prob and min_prob > 0.0:
            q = np.maximum(q, min_prob)
            q = q / q.sum()
        return q.astype(np.float32)

    def _place_objects_for_current_episode(self):
        """
        Park all in FAR except the active one at current_cube_pos (batched).
        """
        active_pos = self.current_cube_pos  # shape: (N, 3)
        for obj_id, ent in enumerate(self._object_entities):
            ent.set_pos(active_pos if obj_id == self._current_object_id else self._far_pos,
                        envs_idx=self.envs_idx)

    # -------------------------
    # Rendering / observation
    # -------------------------
    def _render(self) -> torch.Tensor:
        imgs = []
        N = self.num_envs
        cols = int(math.ceil(math.sqrt(N)))
        rows = int(math.ceil(N / cols))
        env_space = self._env_space
        noise_level = float(self._visual_noise_level)  # σ in [0,1]

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
                img = torch.clamp(img + torch.randn_like(img) * noise_level, 0.0, 1.0)

            #img = torch.clamp(torch.randn(3, 120, 120) * noise_level, 0.0, 1.0)

            imgs.append(img)

        return torch.stack(imgs, dim=0)  # N×3×H×W

    # -------------------------
    # Audio synthesis & spectrogram
    # -------------------------
    def simulate_audio(self, dist: float, sound_id: int) -> np.ndarray:
        sr, dur = 22050, 0.01
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)

        if sound_id == 0:
            carrier = chirp(t, f0=1000, f1=1000, t1=dur, method="linear")
        elif sound_id == 1:
            carrier = chirp(t, f0=600, f1=2400, t1=dur, method="quadratic")
        elif sound_id == 2:
            base = chirp(t, f0=1500, f1=1500, t1=dur, method="linear")
            mod = 0.5 * (1.0 + np.sin(2 * np.pi * 8 * t))
            carrier = base * mod
        else:
            carrier = chirp(t, f0=200, f1=4000, t1=dur, method="linear")

        tone = carrier / (dist ** 2 + 1e-6)

        noise = sum(
            np.random.rand() * chirp(
                t,
                np.random.randint(300, 3000),
                np.random.randint(300, 3000),
                dur
            )
            for _ in range(5)
        ) * 0.1

        # NOTE: no Gaussian "additional" here anymore; we add it after spectrogram normalization
        #return tone + noise

        return tone + noise


    def _compute_spectrogram(self, audio: np.ndarray) -> torch.Tensor:
        S = librosa.stft(audio, n_fft=512, hop_length=256)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=1.0)[:self.freq_bins, :self.time_bins_per_slice]

        # Normalize [-20, 120] dB -> [0,1]
        S_db_norm = (S_db + 20.0) / 140.0
        S_db_norm = np.clip(S_db_norm, 0.0, 1.0)

        spec = torch.from_numpy(S_db_norm).float()

        # --- NEW: Gaussian noise in normalized [0,1] spectrogram domain ---
        audio_noise_level = float(self._audio_noise_level)  # pulled from noise_config at __init__
        if audio_noise_level > 0.0:
            spec = torch.clamp(spec + torch.randn_like(spec) * audio_noise_level, 0.0, 1.0)

        return spec

    def _collect_spectrograms(self, play_audio_slice: bool = False) -> torch.Tensor:
        """
        Render one audio slice per env using distance from gripper midpoint to the ACTIVE object.
        Returns: [N, F, 1]
        """
        obj_pos = self._object_entities[self._current_object_id].get_pos()  # N×3
        gp_l = self.franka.get_link("left_finger").get_pos()
        gp_r = self.franka.get_link("right_finger").get_pos()
        dists = torch.norm(obj_pos - (gp_l + gp_r) / 2, dim=1).cpu().numpy()

        specs = []
        for i, dist in enumerate(dists):
            audio = self.simulate_audio(dist, self.current_sound_id)
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

            spec = self._compute_spectrogram(audio)
            specs.append(spec)

        return torch.stack(specs, dim=0).to(self.device)  # N×F×1

    # -------------------------
    # Observation builders & plotting
    # -------------------------
    def _build_observation(self):
        # Vision: concat along channel (3 * k)
        vis_obs = torch.cat([self.image_history[i] for i in self.sample_offsets], dim=1)  # N×(3k)×H×W

        # Audio: concat along time for the stacked window
        aud_slices = torch.cat([self.audio_history[i] for i in self.sample_offsets], dim=2)  # N×F×(k*1)
        return vis_obs, aud_slices.unsqueeze(1)  # vision: N×(3k)×H×W, audio: N×1×F×T

    def _plot_multimodal_obs(self, vis_obs, aud_obs):
        plt.figure(self.fig_multimodal.number)
        plt.clf()

        # Spectrogram (top)
        ax_spec = plt.subplot(2, 1, 1)
        extent = [0, 10 * len(self.sample_offsets), 0, (22050 / 2) / 1000]
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
    # Reset
    # -------------------------
    def reset(self):
        """
        Inference:
          - cycle positions deterministically (shared across envs).
          - sample exactly ONE object kind for the whole episode (restricted by train_object_count).
        Training:
          - update anti-collapse position probs from last episode; sample per-env positions.
          - sample exactly ONE object kind for the whole episode (restricted by train_object_count).
        Always:
          - place only the active object at current positions; others parked far away;
          - prime stacked histories with the first frame/spec repeated.
        Returns: (vision_obs, audio_obs)
        """
        # ----- Inference mode -----
        if self.inference_mode:
            self.episode_count += 1
            self._episode_active = True
            self._episode_return = 0.0

            idx = self._infer_cycle_idx % self.pos_count
            self._infer_cycle_idx = (idx + 1) % self.pos_count

            self.env_pos_idx = np.full(self.num_envs, idx, dtype=np.int64)
            pos1 = self.cube_positions[idx].reshape(1, 3)
            self.current_cube_pos = np.repeat(pos1, self.num_envs, axis=0)

            self._init_robot()

            # choose one object kind for the episode (restricted by shape_probs)
            obj_id = int(np.random.choice(len(self._object_entities), p=self.shape_probs))
            self._current_object_id = obj_id
            self._current_object_kind = self._object_kinds[self._current_object_id]
            self.current_sound_id = self._shape_to_sound[self._current_object_id]

            self._place_objects_for_current_episode()
            self.scene.step()

            print(f"[Inference] Episode {self.episode_count}: pos idx={idx} "
                  f"pos={self.current_cube_pos[0].tolist()} | object={self._current_object_kind} "
                  f"| sound_id={self.current_sound_id}")

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

        # ----- Training mode -----
        # If closing a previous episode, update position probabilities
        if self._episode_active:
            per_env = self._episode_return_per_env.detach().cpu().numpy()
            per_pos_returns = np.zeros(self.pos_count, dtype=np.float64)
            for i in range(self.pos_count):
                per_pos_returns[i] = per_env[self.env_pos_idx == i].sum()
            self.pos_probs = self._anti_collapse_probs(per_pos_returns, min_prob=0.05)

            counts = np.bincount(self.env_pos_idx, minlength=self.pos_count)
            print(f"[EP end] per-pos return={np.round(per_pos_returns,3).tolist()} "
                  f"| assignment_counts={counts.tolist()} "
                  f"| next p={np.round(self.pos_probs,3).tolist()}")

        # start new training episode
        self.episode_count += 1
        self._episode_active = True
        self._episode_return = 0.0
        self._episode_return_per_env.zero_()
        self._done_mask_episode.zero_()

        # sample per-env positions from learned distribution
        self.env_pos_idx = np.random.choice(self.pos_count, size=self.num_envs, p=self.pos_probs)
        self.current_cube_pos = np.stack([self.cube_positions[i] for i in self.env_pos_idx], axis=0)

        # reset robot
        self._init_robot()

        # sample one object kind for all envs (restricted by shape_probs)
        obj_id = int(np.random.choice(len(self._object_entities), p=self.shape_probs))
        self._current_object_id = obj_id
        self._current_object_kind = self._object_kinds[self._current_object_id]
        self.current_sound_id = self._shape_to_sound[self._current_object_id]

        # place objects
        self._place_objects_for_current_episode()
        self.scene.step()

        print(f"Episode {self.episode_count}: pos_probs={np.round(self.pos_probs, 3).tolist()} "
              f"| assigned idx={self.env_pos_idx.tolist()} "
              f"| object={self._current_object_kind} | sound_id={self.current_sound_id}")

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
        # Ensure actions are 1D int tensor on the right device
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
            new_spec = self._collect_spectrograms(play_audio_slice=False)  # [N, F, 1], also appends raw audio for listener env
            # new raw slice already appended by _collect_spectrograms for listen_idx; replicate for others so history lengths match
            if len(self.raw_audio_history) > 0:
                last_raw = self.raw_audio_history[-1].copy()
            else:
                last_raw = np.zeros(int(self._sd_rate*0.01), dtype=np.float32)
        else:
            # reuse previous frame/spec for skipped steps
            new_img = self.image_history[-1]
            new_spec = self.audio_history[-1]
            # keep raw history in sync for playback windowing
            last_raw = self.raw_audio_history[-1] if len(self.raw_audio_history) else np.zeros(int(self._sd_rate*0.01), dtype=np.float32)

        # update histories
        self.image_history.append(new_img)
        self.audio_history.append(new_spec)
        self.raw_audio_history.append(last_raw.copy())
        self.step_count += 1

        # build obs
        vis_obs, aud_obs = self._build_observation()

        # optional debug plotting & audio playback
        if self.num_envs == 1 and (self.step_count % self.show_every == 0):
            self._plot_multimodal_obs(vis_obs, aud_obs)

        # distance to ACTIVE object (fingers midpoint)
        obj_pos = self._object_entities[self._current_object_id].get_pos()  # N×3
        gp_l = self.franka.get_link("left_finger").get_pos()
        gp_r = self.franka.get_link("right_finger").get_pos()
        dist = torch.norm(obj_pos - (gp_l + gp_r) / 2, dim=1)  # [N]

        # reward parity
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
