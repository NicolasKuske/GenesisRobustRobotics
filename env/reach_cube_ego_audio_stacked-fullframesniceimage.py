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
    Genesis env with audio-only observations and random cube repositioning.
    Observations: a stack of the last N spectrogram frames (200ms window).
    Each step returns obs of shape (num_envs, C, F, T), where:
      C = history_length (20), F = n_fft/2+1 freq bins, T = fixed time bins per slice (9).
    Plays back the full 200ms audio window for the designated listener at intervals.
    """

    def __init__(
        self,
        vis: bool,
        device: torch.device,
        num_envs: int = 1,
        listen_idx: int = 0,
        show_every: int = 10,
        randomize_every: int = 100,
    ):
        # --- Config ---
        self.device = device
        self.num_envs = num_envs
        self.listen_idx = listen_idx
        self.show_every = show_every
        self.randomize_every = randomize_every

        # --- History settings (200ms @10ms per slice) ---
        self.history_length = 20
        # Offsets for past frames: [-20, -19, ..., -1]
        self.sample_offsets = list(range(-self.history_length, 0))
        # Deques to store past spectrograms and raw audio snippets
        self.audio_history = deque(maxlen=self.history_length)
        self.raw_audio_history = deque(maxlen=self.history_length)

        # --- STFT parameters ---
        self.n_fft = 256
        self.hop_length = self.n_fft // 2
        # Frequency bins and time bins per slice
        self.F = self.n_fft//2 + 1
        self.T = 9

        # Observation & action specs
        self.obs_shape = (self.history_length, self.F, self.T)
        self.action_space = 6

        # Matplotlib for live spectrogram
        self._fig = plt.figure("Stacked Spectrogram Preview")

        # Build scene and robot
        self._build_scene(vis)
        self._init_robot()

        # Counters
        self.step_count = 0
        self.episode_count = 0
        self.current_cube_pos = None

    def _build_scene(self, show_viewer: bool):
        """Set up Genesis scene, plane, robot, and cube."""
        self.scene = gs.Scene(
            show_FPS=False,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3,2,1.5), camera_lookat=(0,0,0.2),
                camera_fov=30, res=(960,640), max_FPS=60
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml")
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06,0.06,0.06)),
            surface=gs.surfaces.Rough(color=(0.99,0.82,0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )
        self.scene.build(n_envs=self.num_envs, env_spacing=(5.0,5.0))
        self.envs_idx = np.arange(self.num_envs)

    def _init_robot(self):
        """Reset Franka to neutral pose in all envs."""
        self.motors_dof = torch.arange(7, device=self.device)
        self.fingers_dof = torch.arange(7,9, device=self.device)
        neutral_q = torch.tensor(
            [-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            device=self.device
        ).unsqueeze(0).repeat(self.num_envs,1)
        self.franka.set_qpos(neutral_q, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = neutral_q[:,7:9].clone()
        self.end_effector = self.franka.get_link("hand")

        # Desired EE pose
        base_pos = torch.tensor([0.2720, -0.1683, 1.0164], device=self.device)
        base_quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], device=self.device)
        self.pos = base_pos.unsqueeze(0).repeat(self.num_envs,1)
        self.quat = base_quat.unsqueeze(0).repeat(self.num_envs,1)
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:,:-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)

    def simulate_audio(self, dist: float) -> np.ndarray:
        """Return 10ms of audio: distance-attenuated tone + noise."""
        sr, dur = 22050, 0.01
        t = np.linspace(0, dur, int(sr*dur), endpoint=False)
        tone = chirp(t, f0=1000, f1=1000, t1=dur) / (dist**2 + 1e-6)
        noise = sum(
            np.random.rand() * chirp(
                t,
                np.random.randint(300,3000),
                np.random.randint(300,3000),
                dur
            ) for _ in range(5)
        ) * 0.1
        return tone + noise

    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute (F,T) spectrogram slice, padding/trimming to fixed size."""
        S = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=1.0)
        # pad/trim freq
        if S_db.shape[0] < self.F:
            S_db = np.pad(S_db, ((0,self.F-S_db.shape[0]),(0,0)), mode='constant', constant_values=S_db.min())
        else:
            S_db = S_db[:self.F,:]
        # pad/trim time
        if S_db.shape[1] < self.T:
            S_db = np.pad(S_db, ((0,0),(0,self.T-S_db.shape[1])), mode='constant', constant_values=S_db.min())
        else:
            S_db = S_db[:,:self.T]
        return S_db

    def _collect_spectrograms(self) -> torch.Tensor:
        """Simulate audio + spectrogram for each env; record raw audio for listener."""
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube_pos = self.cube.get_pos()
        dists = torch.norm((left+right)/2 - cube_pos, dim=1).cpu().numpy()
        specs = []
        for i, dist in enumerate(dists):
            audio = self.simulate_audio(dist)
            if i == self.listen_idx:
                self.raw_audio_history.append(audio)
            spec = self._compute_spectrogram(audio)
            specs.append(torch.from_numpy(spec.astype(np.float32)))
        self.step_count += 1
        return torch.stack(specs, dim=0).to(self.device)

    def _build_observation(self) -> torch.Tensor:
        """Stack last N spectrograms into tensor (num_envs, C, F, T)."""
        frames = [self.audio_history[offset] for offset in self.sample_offsets]
        return torch.stack(frames, dim=1)

    def reset(self) -> torch.Tensor:
        """Reset env, clear and seed histories, return initial obs."""
        self.episode_count += 1
        # cube pos logic
        if self.episode_count == 1:
            one_pos = np.array([[0.65,0.0,0.1]])
        elif self.episode_count % self.randomize_every == 0:
            xy = np.random.uniform(0.2,1.0,(1,2))*np.random.choice([-1,1],(1,2))
            z = np.random.uniform(0.1,1.0,(1,1))
            one_pos = np.concatenate([xy,z],axis=1)
        else:
            one_pos = self.current_cube_pos[:1]
        self.current_cube_pos = np.repeat(one_pos,self.num_envs,axis=0)
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        # clear histories
        self.audio_history.clear()
        self.raw_audio_history.clear()
        # collect first slice
        first = self._collect_spectrograms()
        # seed both deques
        for _ in range(self.history_length):
            self.audio_history.append(first.clone())
            # raw_audio_history gets same snippet each time
            self.raw_audio_history.append(self.raw_audio_history[-1])

        obs = self._build_observation()
        if self.num_envs==1:
            self._plot_stacked(obs[0])
        return obs

    def step(self, actions: torch.Tensor):
        """Step sim, update histories, play full 200ms, return obs, rewards, dones."""
        # apply actions
        deltas = torch.tensor([[0.05,0,0],[-0.05,0,0],[0,0.05,0],[0,-0.05,0],[0,0,0.05],[0,0,0.05]], device=self.device)
        masks = [actions==i for i in range(self.action_space)]
        self.pos += sum(deltas[i]*masks[i].unsqueeze(1) for i in range(self.action_space))
        qpos = self.franka.inverse_kinematics(link=self.end_effector, pos=self.pos, quat=self.quat)
        self.franka.control_dofs_position(qpos[:,:-2],self.motors_dof,self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos,self.fingers_dof,self.envs_idx)
        self.scene.step()

        # collect and append new slice
        new = self._collect_spectrograms()
        self.audio_history.append(new)

        # play full 200ms when due
        if self.listen_idx is not None and self.step_count % self.show_every ==0:
            snippets = [self.raw_audio_history[offset] for offset in self.sample_offsets]
            full = np.concatenate(snippets, axis=0)
            sd.play(full,22050)
            sd.wait()

        obs = self._build_observation()
        if self.num_envs==1 and self.step_count % self.show_every==0:
            self._plot_stacked(obs[0])

        # compute reward
        left = self.franka.get_link("left_finger").get_pos()
        right= self.franka.get_link("right_finger").get_pos()
        cube_pos=self.cube.get_pos()
        dist=torch.norm((left+right)/2-cube_pos,dim=1)
        rewards=torch.clamp(torch.exp(-4*(dist-0.1)),0.0,1.0)
        dones=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        return obs,rewards,dones

    def _plot_stacked(self, data: torch.Tensor):
        """Visualize a true spectrogram of the concatenated raw 200ms audio window."""
        # Concatenate raw audio snippets into one buffer
        full_audio = np.concatenate(list(self.raw_audio_history), axis=0)
        # Compute spectrogram over the full buffer
        S = librosa.stft(full_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=1.0)
        # Build time (ms) and frequency (kHz) axes
        sr = 22050
        times = np.arange(S_db.shape[1]) * (self.hop_length / sr) * 1000
        freqs = np.linspace(0, sr / 2, S_db.shape[0]) / 1000
        # Plot
        plt.clf()
        plt.imshow(
            S_db,
            origin='lower',
            aspect='auto',
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            vmin=-40,
            vmax=100
        )
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (kHz)')
        plt.title(f'Step {self.step_count} Full Spectrogram (200ms)')
        plt.draw()
        plt.pause(0.01)
        self._fig.canvas.flush_events()

if __name__=="__main__":
    gs.init(backend=gs.gpu)
    env=ReachCubeEgoAudioStackedEnv(vis=True,device=torch.device('cuda'),listen_idx=0)
