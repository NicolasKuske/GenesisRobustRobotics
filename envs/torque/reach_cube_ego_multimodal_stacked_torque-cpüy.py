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

class ReachCubeEgoMultimodalStackedTorqueEnv:
    """
    Multimodal environment with not_stacked vision and audio inputs.
    Uses direct torque control (continuous actions).
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
        # Continuous torque actions (7 DoF)
        self.action_space = 7
        self.max_torque = torch.tensor([87,87,87,87,12,12,12], device=device)

        # Device and env parameters
        self.device = device
        self.num_envs = num_envs
        self.randomize_every = randomize_every
        self.listen_idx = listen_idx
        self.show_every = show_every
        self.render_every = render_every
        self.episode_count = 0
        self.step_count = 0
        self._step_count = 0

        # History and sampling offsets
        self.history_length = 25
        self.sample_offsets = [-21, -16, -11, -6, -1]

        # Observation shapes
        self.obs_shape_vision = (3 * len(self.sample_offsets), 120, 120)
        self.freq_bins = 257
        self.time_bins_per_slice = 1
        self.obs_shape_audio = (1, self.freq_bins, len(self.sample_offsets))

        # Histories
        self.image_history = deque(maxlen=self.history_length)
        self.audio_history = deque(maxlen=self.history_length)
        self.raw_audio_history = deque(maxlen=self.history_length)

        # Set up plotting
        self._fig = plt.figure("Stacked Spectrogram Preview")

        # Build Genesis scene
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

        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="assets/xml/franka_emika_panda/panda.xml")
        )

        # Add walls
        for pos, color, euler in [
            ((4, 0, 1), (0.9, 0.9, 0.9), (0, -20, 0)),
            ((-3, 0, 1), (0.7, 0.7, 0.7), (0, 20, 0)),
            ((0, -3, 1), (0.56, 0.57, 0.58), (0, 20, 90)),
        ]:
            self.scene.add_entity(
                morph=gs.morphs.Box(size=(0.1, 8, 4), pos=pos, euler=euler, collision=False),
                material=gs.materials.Rigid(gravity_compensation=1.0),
                surface=gs.surfaces.Rough(color=color)
            )

        # Cube
        self.cube = self.scene.add_entity(
            morph=gs.morphs.Box(size=(0.06, 0.06, 0.06)),
            surface=gs.surfaces.Rough(color=(0.99, 0.82, 0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # Camera setup
        self.cams = []
        cam_transform = trans_quat_to_T(np.array([0.03,0,0.03]), xyz_to_quat(np.array([185,0,90])))
        for _ in range(self.num_envs):
            cam = self.scene.add_camera(res=(120,120), fov=90, GUI=True)
            self.cams.append(cam)
        self.cam_transform = cam_transform

        env_space = 100.0
        self.scene.build(n_envs=self.num_envs, env_spacing=(env_space, env_space))
        self.envs_idx = np.arange(self.num_envs)


        # start the cameras
        for cam in self.cams:
            cam.start_recording()


        # Initialize robot
        self._init_robot()

    def _init_robot(self):
        q0 = torch.tensor([[-1.0,-0.3,0.3,-1.0,-0.1,1.7,1.0,0.02,0.02]], device=self.device).repeat(self.num_envs,1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:,7:9].clone()
        self.end_effector = self.franka.get_link("hand")

    def _render(self):
        imgs = []
        M = int(math.sqrt(self.num_envs))
        env_space = 100.0
        for idx, cam in enumerate(self.cams):
            ee_pos = self.end_effector.get_pos(envs_idx=[idx])[0].cpu().numpy()
            ee_quat = self.end_effector.get_quat(envs_idx=[idx])[0].cpu().numpy()
            col,row = divmod(idx,M)
            offset = np.array([(col-(M-1)/2)*env_space, (row-(M-1)/2)*env_space, 0])
            cam.set_pose(transform=trans_quat_to_T(ee_pos+offset, ee_quat) @ self.cam_transform)
            rgb = cam.render()[0]
            img = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
            imgs.append(img)
        return torch.stack(imgs)

    def simulate_audio(self, dist):
        # 10ms audio burst with noise
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
        S = librosa.stft(audio,n_fft=512,hop_length=256)
        return librosa.amplitude_to_db(np.abs(S), ref=1.0)[:self.freq_bins,:1]

    def _collect_spectrograms(self, play_audio=False):
        dists = torch.norm(
            self.end_effector.get_pos() - self.cube.get_pos(),
            dim=1
        ).cpu().numpy()
        specs = []
        for i, dist in enumerate(dists):
            audio = self.simulate_audio(dist)
            if i == self.listen_idx:
                self.raw_audio_history.append(audio)  # This was missing
            S_db = self._compute_spectrogram(audio)
            if play_audio and i == self.listen_idx:
                sd.play(audio, 22050)
                sd.wait()
            specs.append(torch.from_numpy(S_db).float())
        return torch.stack(specs, dim=0).to(self.device)

    def _build_observation(self):
        vis_obs = torch.cat([self.image_history[i] for i in self.sample_offsets],dim=1)
        aud_obs = torch.cat([self.audio_history[i] for i in self.sample_offsets],dim=2).unsqueeze(1)
        return vis_obs,aud_obs

    def reset(self):
        self.episode_count += 1
        base = np.array([[0.3, 0.3, 1.2]]) if self.episode_count == 1 else self.current_cube_pos[:1]
        self.current_cube_pos = np.repeat(base, self.num_envs, axis=0)
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)

        frame = self._render()
        spec = self._collect_spectrograms()

        first_raw_audio = self.raw_audio_history[-1]
        self.image_history = deque([frame] * self.history_length, maxlen=self.history_length)
        self.audio_history = deque([spec] * self.history_length, maxlen=self.history_length)
        self.raw_audio_history = deque([first_raw_audio] * self.history_length, maxlen=self.history_length)

        return self._build_observation()  # <-- This was missing previously!

    def step(self,actions):
        torque = actions.clamp(-1,1)*self.max_torque
        self.franka.control_dofs_force(torque.cpu().numpy(),dofs_idx_local=np.arange(7),envs_idx=self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos,torch.arange(7,9),self.envs_idx)
        self.scene.step()
        frame = self._render() if self._step_count%self.render_every==0 else self.image_history[-1]
        spec = self._collect_spectrograms()
        self.image_history.append(frame)
        self.audio_history.append(spec)
        self._step_count += 1
        self.step_count += 1

        # Build observations
        vis_obs, aud_obs = self._build_observation()

        # Optionally plot and play audio if single environment
        if self.num_envs == 1 and (self.step_count % self.show_every == 0):
            self._plot_stacked(aud_obs[0, 0])
            snippets = [self.raw_audio_history[i] for i in self.sample_offsets]
            sd.play(np.concatenate(snippets, axis=0), 22050)
            sd.wait()

        # Compute reward based on distance between end-effector and cube
        dist = torch.norm(self.end_effector.get_pos() - self.cube.get_pos(), dim=1)
        rewards = torch.clamp(torch.exp(-4 * (dist - 0.1)), 0.0, 1.0)

        # Set done flag (no termination condition here, but placeholder provided)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return (vis_obs, aud_obs), rewards, dones

    def _plot_stacked(self, data: torch.Tensor):
        plt.clf()
        extent = [0, 10 * len(self.sample_offsets), 0, (22050 / 2) / 1000]
        plt.imshow(data.cpu().numpy(), origin='lower', aspect='auto', extent=extent, vmin=-40, vmax=100)
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (kHz)')
        plt.title(f'Step {self.step_count} Stacked Spectrogram')
        plt.draw()
        plt.pause(0.01)
        self._fig.canvas.flush_events()

if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoMultimodalStackedTorqueEnv(vis=True, device=torch.device('cuda'), listen_idx=0)
    obs = env.reset()
    for _ in range(200):
        actions = torch.randn(env.num_envs, env.action_space, device=env.device)
        obs, reward, done = env.step(actions)
        if done.any():
            print("Done:", done)
            break
