

import numpy as np
import genesis as gs
import torch
import librosa
import matplotlib.pyplot as plt
from scipy.signal import chirp
from collections import deque

class ReachCubeEgoAudioStackedEnv:
    """
    Genesis environment with audio-only observations and random cube repositioning.
    The agent receives a stack of the last N spectrograms each step.
    Observations have shape (num_envs, C, F, T), where:
      C = history_length (20), F = freq bins (n_fft/2+1), T = time bins per slice (9)
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
        self.step_count = 0
        self.episode_count = 0
        self.current_cube_pos = None

        # --- Plotting ---
        self._fig = plt.figure("Stacked Spectrogram Preview")

        # --- History settings ---
        self.history_length = 20  # N frames (200ms window)
        self.sample_offsets = list(range(-self.history_length, 0))
        self.audio_history = deque(maxlen=self.history_length)

        # --- STFT settings ---
        self.n_fft = 256
        self.hop_length = self.n_fft // 2
        # deduce F and T
        self.F = self.n_fft // 2 + 1
        self.T = 9  # we will pad/trim to 9 time bins

        # --- Observation & Action specs ---
        # C = history_length
        self.obs_shape = (self.history_length, self.F, self.T)
        self.action_space = 6

        # --- Build Genesis scene ---
        self.scene = gs.Scene(
            #show_FPS=False,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3,2,1.5),
                camera_lookat=(0,0,0.2),
                camera_fov=30,
                res=(960,640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=vis,
        )

        # Add entities
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml")
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.06,0.06,0.06)),
            surface=gs.surfaces.Rough(color=(0.99,0.82,0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        self.scene.build(n_envs=self.num_envs, env_spacing=(5.0,5.0))
        self.envs_idx = np.arange(self.num_envs)
        self._init_robot()

    def _init_robot(self):
        self.motors_dof = torch.arange(7, device=self.device)
        self.fingers_dof = torch.arange(7,9, device=self.device)
        q0 = torch.tensor(
            [-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            device=self.device,
        ).unsqueeze(0).repeat(self.num_envs,1)
        self.franka.set_qpos(q0, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = q0[:,7:9].clone()
        self.end_effector = self.franka.get_link("hand")

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
        sr = 22050
        dur = 0.01
        t = np.linspace(0, dur, int(sr*dur), endpoint=False)
        tone = chirp(t, f0=1000, f1=1000, t1=dur) / (dist**2 + 1e-6)
        noise = sum(
            np.random.rand() * chirp(
                t,
                np.random.randint(300,3000),
                np.random.randint(300,3000),
                dur,
            ) for _ in range(5)
        ) * 0.1
        return tone + noise

    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute a fixed-size spectrogram slice of shape (F, T) by padding/trimming frequency and time axes.
        """
        # dynamic FFT up to n_fft
        n_fft = min(self.n_fft, audio.shape[0])
        hop = n_fft // 2
        S = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=1.0)
        f_bins, t_bins = S_db.shape
        # pad or trim frequency axis to self.F
        if f_bins < self.F:
            pad_f = self.F - f_bins
            S_db = np.pad(S_db, ((0, pad_f), (0,0)), mode='constant', constant_values=S_db.min())
        else:
            S_db = S_db[:self.F, :]
        # pad or trim time axis to self.T
        f_bins, t_bins = S_db.shape
        if t_bins < self.T:
            pad_t = self.T - t_bins
            S_db = np.pad(S_db, ((0,0), (0, pad_t)), mode='constant', constant_values=S_db.min())
        else:
            S_db = S_db[:, :self.T]
        return S_db

    def _collect_spectrograms(self) -> torch.Tensor:
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube_pos = self.cube.get_pos()
        dists = torch.norm((left + right)/2 - cube_pos, dim=1).cpu().numpy()
        specs = []
        for dist in dists:
            spec = self._compute_spectrogram(self.simulate_audio(dist))
            specs.append(torch.from_numpy(spec.astype(np.float32)))
        # stack into (num_envs, F, T)
        return torch.stack(specs, dim=0).to(self.device)

    def _build_observation(self) -> torch.Tensor:
        # stack along channel axis: (num_envs, C, F, T)
        frames = [self.audio_history[offset] for offset in self.sample_offsets]
        obs = torch.stack(frames, dim=1)  # C = history_length
        return obs

    def reset(self) -> torch.Tensor:
        self.episode_count += 1
        if self.episode_count == 1:
            pos = np.array([[0.65, 0.0, 0.1]])
        elif self.episode_count % self.randomize_every == 0:
            xy = np.random.uniform(0.2,1.0,(1,2)) * np.random.choice([-1,1],(1,2))
            z = np.random.uniform(0.1,1.0,(1,1))
            pos = np.concatenate([xy,z], axis=1)
        else:
            pos = self.current_cube_pos[:1]
        self.current_cube_pos = np.repeat(pos, self.num_envs, axis=0)
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        # prime history
        self.audio_history.clear()
        first = self._collect_spectrograms()
        for _ in range(self.history_length):
            self.audio_history.append(first.clone())

        obs = self._build_observation()
        if self.num_envs == 1:
            self._plot_stacked(obs[0])
        return obs

    def step(self, actions: torch.Tensor):
        deltas = torch.tensor(
            [[0.05,0,0],[-0.05,0,0],[0,0.05,0],[0,-0.05,0],[0,0,0.05],[0,0,-0.05]],
            device=self.device,
        )
        masks = [actions == i for i in range(self.action_space)]
        self.pos += sum(deltas[i] * masks[i].unsqueeze(1) for i in range(self.action_space))

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:,:-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        current = self._collect_spectrograms()
        self.audio_history.append(current)
        obs = self._build_observation()

        self.step_count += 1
        if self.num_envs == 1 and self.step_count % self.show_every == 0:
            self._plot_stacked(obs[0])

        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube_pos = self.cube.get_pos()
        dist = torch.norm((left + right)/2 - cube_pos, dim=1)
        rewards = torch.clamp(torch.exp(-4*(dist-0.1)), 0.0, 1.0)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return obs, rewards, dones

    def _plot_stacked(self, data: torch.Tensor):
        # data shape: (C, F, T)
        arr = data.cpu().numpy()
        # reshape for visualization: concatenate time bins across channels
        vis = arr.reshape(self.F, self.T * self.history_length)
        plt.clf()
        extent = [0, 10 * self.history_length, 0, (22050/2)/1000]
        plt.imshow(vis, origin='lower', aspect='auto', extent=extent, vmin=-40, vmax=100)
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (kHz)')
        plt.title(f'Step {self.step_count} Stacked Spec')
        plt.draw()
        plt.pause(0.01)
        self._fig.canvas.flush_events()

if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoAudioStackedEnv(vis=True, device=torch.device('cuda'), listen_idx=0)