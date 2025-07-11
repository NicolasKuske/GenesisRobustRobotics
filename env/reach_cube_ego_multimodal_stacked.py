import numpy as np
import genesis as gs
import torch
import librosa
import matplotlib.pyplot as plt
from scipy.signal import chirp
from collections import deque
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat
import sounddevice as sd


class ReachCubeEgoMultimodalStackedEnv:
    def __init__(self, vis, device, num_envs=1, randomize_every=100, listen_idx=0, show_every=10):
        self.device = device
        self.num_envs = num_envs
        self.randomize_every = randomize_every
        self.listen_idx = listen_idx
        self.show_every = show_every
        self.episode_count = 0
        self.step_count = 0

        # History length and sample offsets for stacking
        self.history_length = 20
        self.sample_offsets = [-20, -15, -10, -5, -1]
        self.action_space = 6

        # Vision dims
        self.image_history = deque(maxlen=self.history_length)
        self.obs_shape_vision = (3 * len(self.sample_offsets), 120, 120)

        # Audio dims
        self.audio_history = deque(maxlen=self.history_length)
        self.raw_audio_history = deque(maxlen=self.history_length)
        self.freq_bins = 257
        self.obs_shape_audio = (1, self.freq_bins, len(self.sample_offsets))
        self._fig = plt.figure("Stacked Spectrogram Preview")

        # Genesis Scene
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 2, 1.5), camera_lookat=(0, 0, 0.2), camera_fov=30,
                res=(960, 640), max_FPS=60),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=vis)

        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"))
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06,0.06,0.06)),
            surface=gs.surfaces.Rough(color=(0.99,0.82,0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # Setup cameras
        self.cams = []
        cam_transform = trans_quat_to_T(np.array([0.03, 0, 0.03]), xyz_to_quat(np.array([185,0,90])))
        for _ in range(num_envs):
            cam = self.scene.add_camera(res=(120,120), fov=90, GUI=True)
            self.cams.append(cam)
        self.cam_transform = cam_transform

        self.scene.build(n_envs=num_envs, env_spacing=(5,5))
        self.envs_idx = np.arange(num_envs)

        for cam in self.cams:
            cam.start_recording()

        # Initialize robot
        self._init_robot()

    def _init_robot(self):
        q0 = torch.tensor(
            [[-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02]],
            device=self.device
        ).repeat(self.num_envs, 1)

        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:,7:9].clone()
        self.end_effector = self.franka.get_link("hand")
        self.pos = torch.tensor([0.2720,-0.1683,1.0164],device=self.device).repeat(self.num_envs,1)
        self.quat = torch.tensor([0.1992,0.7857,-0.3897,0.4371],device=self.device).repeat(self.num_envs,1)

    def _render(self):
        imgs = []
        for idx, cam in enumerate(self.cams):
            ee_pos = self.end_effector.get_pos(envs_idx=[idx])[0].cpu().numpy()
            ee_quat = self.end_effector.get_quat(envs_idx=[idx])[0].cpu().numpy()
            ee_T = trans_quat_to_T(ee_pos, ee_quat)
            cam_T = ee_T @ self.cam_transform
            cam.set_pose(cam_T)
            rgb = cam.render()[0]
            img = torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255.0
            imgs.append(img)
        return torch.stack(imgs,dim=0)

    def simulate_audio(self, dist):
        sr, dur = 22050, 0.01
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        # constant 1 kHz tone, distance-attenuated
        tone = chirp(t, f0=1000, f1=1000, t1=dur) / (dist ** 2 + 1e-6)
        # sum of five random chirps, scaled down
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

    def _compute_spectrogram(self, audio):
        # Always use n_fft=512 as per original audio-only script
        S = librosa.stft(audio, n_fft=512, hop_length=256)
        return librosa.amplitude_to_db(np.abs(S), ref=1.0)[:self.freq_bins, :9]

    def _collect_spectrograms(self, play_audio=False):
        dists = torch.norm(self.franka.get_link("hand").get_pos()-self.cube.get_pos(),dim=1).cpu().numpy()
        specs = []
        for i, dist in enumerate(dists):
            audio = self.simulate_audio(dist)
            if i == self.listen_idx:
                self.raw_audio_history.append(audio)
            S_db = self._compute_spectrogram(audio)
            if play_audio and i == self.listen_idx:
                sd.play(audio, 22050)
                sd.wait()
            specs.append(torch.from_numpy(S_db).float())
        return torch.stack(specs, dim=0).to(self.device)

    def _plot_stacked(self, data: torch.Tensor):
        plt.clf()
        extent = [0, 10 * len(self.sample_offsets), 0, (22050 / 2) / 1000]
        plt.imshow(data.cpu().numpy(), origin='lower', aspect='auto', extent=extent, vmin=-40, vmax=100)
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (kHz)')
        plt.title(f'Step {self.step_count} Stacked Spec')
        plt.draw()
        plt.pause(0.01)
        self._fig.canvas.flush_events()

    def _build_observation(self):
        vis_obs = torch.cat([self.image_history[i] for i in self.sample_offsets], dim=1)
        aud_obs = torch.cat([self.audio_history[i] for i in self.sample_offsets], dim=2).unsqueeze(1)
        return vis_obs, aud_obs

    def reset(self):
        self.episode_count += 1
        cube_pos = (np.array([[0.1,0.5,0.3]])
                    if self.episode_count == 1
                    else np.random.uniform(-1,1,(1,3))*[1,1,0.5]+[0,0,0.5])
        self.current_cube_pos = np.repeat(cube_pos, self.num_envs, axis=0)
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        # Clear and initialize histories
        self.image_history.clear()
        self.audio_history.clear()
        self.raw_audio_history.clear()

        # First render and spectrogram collection
        frame = self._render()
        spec = self._collect_spectrograms(play_audio=False)
        first_raw = self.raw_audio_history[-1].copy()

        # Populate full histories
        for _ in range(self.history_length):
            self.image_history.append(frame)
            self.audio_history.append(spec)
            self.raw_audio_history.append(first_raw)

        vis_obs, aud_obs = self._build_observation()
        # Show and play on single-env reset
        if self.num_envs == 1:
            self._plot_stacked(aud_obs[0,0])
            snippets = [self.raw_audio_history[offset] for offset in self.sample_offsets]
            full_buffer = np.concatenate(snippets, axis=0)
            sd.play(full_buffer, 22050)
            sd.wait()

        return vis_obs, aud_obs

    def step(self, actions):
        # Movement deltas for 6 actions
        deltas = torch.tensor([
            [0.05, 0.0, 0.0],
            [-0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, -0.05, 0.0],
            [0.0, 0.0, 0.05],
            [0.0, 0.0, -0.05]
        ], device=self.device)
        self.pos += deltas[actions]
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:,:-2], torch.arange(7, device=self.device), self.envs_idx)
        self.scene.step()

        # Render vision and collect audio
        frame = self._render()
        self.image_history.append(frame)
        spec = self._collect_spectrograms(play_audio=False)
        self.audio_history.append(spec)
        self.step_count += 1

        vis_obs, aud_obs = self._build_observation()
        # Periodic display and playback
        if self.num_envs == 1 and (self.step_count % self.show_every == 0):
            self._plot_stacked(aud_obs[0,0])
            snippets = [self.raw_audio_history[offset] for offset in self.sample_offsets]
            full_buffer = np.concatenate(snippets, axis=0)
            sd.play(full_buffer, 22050)
            sd.wait()

        # Compute reward and done
        dist = torch.norm(
            self.franka.get_link("hand").get_pos() - self.cube.get_pos(),
            dim=1
        )
        rewards = torch.clamp(torch.exp(-4*(dist-0.1)), 0.0, 1.0)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return (vis_obs, aud_obs), rewards, dones


if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoMultimodalStackedEnv(vis=True, device=torch.device('cuda'), listen_idx=0)
