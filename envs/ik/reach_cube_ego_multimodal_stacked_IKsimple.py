import numpy as np
import genesis as gs
import torch
import math
import librosa
import matplotlib.pyplot as plt
from scipy.signal import chirp
from collections import deque
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat
import sounddevice as sd

class ReachCubeEgoMultimodalStackedEnv:
    """
    Multimodal environment with stacked vision and audio inputs.
    Uses frame-skipping for efficiency and aligned history of length 25.
    """
    def __init__(
        self,
        vis: bool,
        device: torch.device,
        num_envs: int = 1,
        randomize_every: int = 100,
        listen_idx: int = 0,
        show_every: int = 10,
        render_every: int = 5
    ):

        # Discrete action count
        self.action_space = 6


        # Device and env parameters
        self.device = device
        self.num_envs = num_envs
        self.randomize_every = randomize_every
        self.listen_idx = listen_idx
        self.show_every = show_every
        self.render_every = render_every
        self.episode_count = 0
        self.step_count = 0
        self._step_count = 0  # for frame-skip



        # History and sampling offsets
        self.history_length = 25
        self.sample_offsets = [-21, -16, -11, -6, -1]

        # Vision history and shape
        self.image_history = deque(maxlen=self.history_length)
        self.obs_shape_vision = (3 * len(self.sample_offsets), 120, 120)

        # Audio history and shape
        self.audio_history = deque(maxlen=self.history_length)
        self.raw_audio_history = deque(maxlen=self.history_length)
        self.freq_bins = 257
        # for 10ms simulation windows, STFT produces 1 time bin per slice (n_fft=512, hop_length=256)
        self.time_bins_per_slice = 1
        self.obs_shape_audio = (
            1,
            self.freq_bins,
            self.time_bins_per_slice * len(self.sample_offsets)
        )

        # Set up plotting
        self._fig = plt.figure("Stacked Spectrogram Preview")

        # Build the Genesis scene
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 2, 1.5),
                camera_lookat=(0, 0, 0.2),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=vis,
            vis_options=gs.options.VisOptions(plane_reflection=True),
            renderer=gs.renderers.Rasterizer(),
        )

        # Add entities
        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="assets/xml/franka_emika_panda/panda.xml")
        )

        # Add walls
        for pos, color, euler in [
            ((4,  0, 1),  (0.9, 0.9, 0.9),  (0, -20,  0)),
            ((-3, 0, 1),  (0.7, 0.7, 0.7),  (0,  20,  0)),
            ((0, -3, 1),  (0.56,0.57,0.58), (0,  20, 90)),
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

        # Add cube
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.06)),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # Cameras attached to end-effector
        self.cams = []
        cam_transform = trans_quat_to_T(
            np.array([0.03, 0, 0.03]),
            xyz_to_quat(np.array([185, 0, 90]))
        )
        for _ in range(self.num_envs):
            cam = self.scene.add_camera(res=(120, 120), fov=90, GUI=True)
            self.cams.append(cam)
        self.cam_transform = cam_transform

        # Build parallel environments
        env_space = 100.0
        self.scene.build(n_envs=self.num_envs, env_spacing=(env_space, env_space))
        self.envs_idx = np.arange(self.num_envs)

        # start the cameras
        for cam in self.cams:
            cam.start_recording()

        # Initialize robot
        self._init_robot()

    def _init_robot(self):
        # Neutral joint positions
        q0 = torch.tensor(
            [[-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02]],
            device=self.device
        ).repeat(self.num_envs, 1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:, 7:9].clone()
        self.end_effector = self.franka.get_link("hand")
        # Default EE pose
        self.pos = torch.tensor(
            [0.2720, -0.1683, 1.0164],
            device=self.device
        ).repeat(self.num_envs, 1)
        self.quat = torch.tensor(
            [0.1992, 0.7857, -0.3897, 0.4371],
            device=self.device
        ).repeat(self.num_envs, 1)

    def _render(self):
        imgs = []
        M = int(math.sqrt(self.num_envs))
        env_space = 100.0
        for idx, cam in enumerate(self.cams):
            ee_pos = self.end_effector.get_pos(envs_idx=[idx])[0].cpu().numpy()
            ee_quat = self.end_effector.get_quat(envs_idx=[idx])[0].cpu().numpy()

            # Add offset for parallel env positioning
            col = idx // M;
            row = idx % M
            x_off = (col - (M - 1) / 2) * env_space
            y_off = (row - (M - 1) / 2) * env_space
            ee_pos_offset = ee_pos + np.array([x_off, y_off, 0.0])

            ee_T = trans_quat_to_T(ee_pos_offset, ee_quat)
            cam_T = ee_T @ self.cam_transform
            cam.set_pose(transform=cam_T)

            rgb = cam.render()[0]
            img = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
            imgs.append(img)
        return torch.stack(imgs, dim=0)

    def simulate_audio(self, dist):
        # 10ms audio burst
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

    def _compute_spectrogram(self, audio):
        S = librosa.stft(audio, n_fft=512, hop_length=256)
        # first 257 freq bins, first 9 time bins
        return librosa.amplitude_to_db(np.abs(S), ref=1.0)[:self.freq_bins, :self.time_bins_per_slice]

    def _collect_spectrograms(self, play_audio=False):
        # Compute distance from EE to cube
        dists = torch.norm(
            self.franka.get_link("hand").get_pos() - self.cube.get_pos(),
            dim=1
        ).cpu().numpy()
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
        # Stack by offsets
        vis_obs = torch.cat([self.image_history[i] for i in self.sample_offsets], dim=1)
        aud_slices = torch.cat([self.audio_history[i] for i in self.sample_offsets], dim=2)
        return vis_obs, aud_slices.unsqueeze(1)

    def reset(self):
        self.episode_count += 1
        # Randomize cube
        if self.episode_count == 1:
            base = np.array([[0.6, 0.6, 0.7]])
        elif self.episode_count % self.randomize_every == 0:
            xy = np.random.uniform(-1, 1, (1, 2)) * [1, 1]
            z = np.random.uniform(0.1, 0.5, (1, 1))
            base = np.concatenate([xy, z], axis=1)
        else:
            base = self.current_cube_pos[:1]
        self.current_cube_pos = np.repeat(base, self.num_envs, axis=0)
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        # Clear histories
        self.image_history.clear()
        self.audio_history.clear()
        self.raw_audio_history.clear()

        # First full render + spectrogram
        frame = self._render()
        spec = self._collect_spectrograms(play_audio=False)
        first_raw = self.raw_audio_history[-1].copy()

        # Populate deque
        for _ in range(self.history_length):
            self.image_history.append(frame)
            self.audio_history.append(spec)
            self.raw_audio_history.append(first_raw)

        return self._build_observation()

    def step(self, actions):
        # Move end-effector
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
        self.franka.control_dofs_position(qpos[:, :-2], torch.arange(7, device=self.device), self.envs_idx)
        self.scene.step()

        # Frame-skip logic
        self._step_count += 1
        if self._step_count % self.render_every == 0:
            frame = self._render()
            spec = self._collect_spectrograms(play_audio=False)
        else:
            frame = self.image_history[-1]
            spec = self.audio_history[-1]

        # Update histories
        self.image_history.append(frame)
        self.audio_history.append(spec)
        self.step_count += 1

        # Build obs
        vis_obs, aud_obs = self._build_observation()
        if self.num_envs == 1 and (self.step_count % self.show_every == 0):
            self._plot_stacked(aud_obs[0, 0])
            snippets = [self.raw_audio_history[i] for i in self.sample_offsets]
            sd.play(np.concatenate(snippets, axis=0), 22050)
            sd.wait()

        # Compute reward and done
        dist = torch.norm(
            self.franka.get_link("hand").get_pos() - self.cube.get_pos(), dim=1
        )
        rewards = torch.clamp(torch.exp(-4 * (dist - 0.1)), 0.0, 1.0)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return (vis_obs, aud_obs), rewards, dones

if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoMultimodalStackedEnv(vis=True, device=torch.device('cuda'), listen_idx=0)
