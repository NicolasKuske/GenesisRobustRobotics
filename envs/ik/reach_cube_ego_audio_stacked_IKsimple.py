#envs/ik/reach_cube_ego_audio_stacked_IKsimple.py


import numpy as np
import genesis as gs
import torch
import librosa
import matplotlib.pyplot as plt
from scipy.signal import chirp
from collections import deque
import sounddevice as sd
import math


class ReachCubeEgoAudioStackedEnv:
    """
    Genesis environment with audio-only observations and random cube repositioning.
    Observations are stacked spectrogram frames over a short history.

    Each step returns an observation tensor of shape (num_envs, 1, F, T),
    where F is the number of frequency bins and T is the number of stacked time frames.
    Optionally plays back the full stacked audio window for the designated listener index.
    """

    def __init__(
        self,
        vis: bool,
        device: torch.device,
        num_envs: int = 1,
        listen_idx: int = 0,
        show_every: int = 10,
        randomize_every: int = 100,
        history_length: int = 25,
        sample_offsets=None,
    ):
        # --- Configuration ---
        self.device = device
        self.num_envs = num_envs
        self.listen_idx = listen_idx
        self.show_every = show_every
        self.randomize_every = randomize_every

        # History for spectrograms and raw audio
        self.history_length = history_length
        self.sample_offsets = sample_offsets or [-21, -16, -11, -6, -1]
        self.audio_history = deque(maxlen=self.history_length)
        self.raw_audio_history = deque(maxlen=self.history_length)

        # Spectrogram dimensions: freq bins and stacked time frames
        self.freq_bins = 257
        self.time_bins = len(self.sample_offsets)
        self.obs_shape = (1, self.freq_bins, self.time_bins) #(1,257,5)
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

        # Add microphone marker entities
        self.mic_markers = []
        for _ in range(self.num_envs):
            mic_marker = self.scene.add_entity(
                gs.morphs.Sphere(radius=0.02, collision=False),
                surface=gs.surfaces.Rough(color=(1.0, 0.0, 0.0))
            )
            self.mic_markers.append(mic_marker)

        # Build multiple copies if needed
        self.scene.build(n_envs=self.num_envs, env_spacing=(5.0, 5.0))
        self.envs_idx = np.arange(self.num_envs)

    def _init_robot(self):
        """Reset the Franka robot to a neutral pose in all environments."""
        self.motors_dof = torch.arange(7, device=self.device)
        self.fingers_dof = torch.arange(7, 9, device=self.device)

        # Neutral joint configuration
        neutral_q = torch.tensor(
            [-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
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

    def simulate_audio(self, dist: float) -> np.ndarray:
        """
        Generate a short burst (10ms) of audio: a distance-attenuated tone plus random noise.
        """
        sr, dur = 22050, 0.01
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        tone = chirp(t, f0=1000, f1=1000, t1=dur) / (dist ** 2 + 1e-6)
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

    def _compute_spectrogram(self, audio: np.ndarray) -> torch.Tensor:
        S = librosa.stft(audio, n_fft=512, hop_length=256)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=1.0)[:self.freq_bins, :self.time_bins]

        # Explicit normalization: [-20 dB, 120 dB] → [0,1]
        S_db_normalized = (S_db + 20) / 140
        S_db_normalized = np.clip(S_db_normalized, 0.0, 1.0)

        return torch.from_numpy(S_db_normalized).float()


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
            if play_audio and i == self.listen_idx:
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
        Reset the envs: randomize the cube, re-init the robot, clear history,
        populate with the first slice, and return the initial stacked obs.
        """
        self.episode_count += 1
        # Decide new cube position
        if self.episode_count == 1:
            one_pos = np.array([[-0.9, 0.6, 0.7]]).reshape(1, 3)   #default_position
       	    #one_pos = np.array([[-0.5, 0.3, 0.7]]).reshape(1, 3)  #new_position1
            #one_pos = np.array([[0.1, 0.5, 0.3]]).reshape(1, 3)   #new_position2
        elif self.episode_count % self.randomize_every == 0:
            xy = np.random.uniform(0.2, 1.0, size=(1, 2)) * np.random.choice([-1, 1], size=(1, 2))
            z = np.random.uniform(0.1, 1.0, size=(1, 1))
            one_pos = np.concatenate([xy, z], axis=1)
        else:
            one_pos = self.current_cube_pos[:1]

        self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)
        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        # Update mic markers on reset
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()

        mic_positions = ((left + right) / 2).cpu().numpy()
        mic_positions[:, 2] += 0.05

        mic_pos_tensor = torch.from_numpy(mic_positions).float().to(self.device)
        for idx, mic_marker in enumerate(self.mic_markers):
            mic_marker.set_pos(mic_pos_tensor[idx].unsqueeze(0), envs_idx=[idx])

        # Reset histories
        self.audio_history.clear()
        self.raw_audio_history.clear()

        # Collect the first slice (fills raw_audio_history once)
        first_spec = self._collect_spectrograms(play_audio=False)
        # Extract that first raw audio snippet
        first_raw = self.raw_audio_history[-1].copy()

        # Populate full history with clones
        self.audio_history.clear()
        self.raw_audio_history.clear()
        for _ in range(self.history_length):
            self.audio_history.append(first_spec.clone())
            self.raw_audio_history.append(first_raw.copy())

        # Build and optionally plot initial obs
        obs = self._build_observation()
        if self.num_envs == 1:
            self._plot_stacked(obs[0, 0])
        return obs

    def step(self, actions: torch.Tensor):
        # Define action step sizes
        d_radial = 0.05  # radial step size
        d_angle = np.deg2rad(10)  # rotation step size
        d_z = 0.05  # vertical step size
        min_radius = 0.05  # minimum radius from center

        # Clone current positions to modify safely
        pos = self.pos.clone()

        # Environment grid parameters for multiple environments
        M = int(math.sqrt(self.num_envs))
        env_space = 5.0

        # Loop through each environment to apply actions individually
        for idx in range(self.num_envs):
            action = actions[idx].item()

            # Current end-effector position
            ee_pos = pos[idx].cpu().numpy()

            # Calculate local environment center offset
            col = idx // M
            row = idx % M
            x_off = (col - (M - 1) / 2) * env_space
            y_off = (row - (M - 1) / 2) * env_space
            local_origin = np.array([x_off, y_off])

            # Calculate radial vector relative to local origin
            radial_xy = ee_pos[:2] - local_origin
            radial_norm = np.linalg.norm(radial_xy)

            # Handle near-origin case safely
            if radial_norm < 1e-6:
                radial_dir = np.array([1.0, 0.0])
            else:
                radial_dir = radial_xy / radial_norm

            # Apply new action logic
            if action == 0:  # Move forward along radial vector
                ee_pos[:2] += radial_dir * d_radial
            elif action == 1:  # Move backward along radial vector (avoiding origin crossing)
                new_radial_norm = max(radial_norm - d_radial, min_radius)
                ee_pos[:2] = local_origin + radial_dir * new_radial_norm
            elif action == 2:  # Rotate left around local origin
                angle = d_angle
                rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                       [np.sin(angle), np.cos(angle)]])
                ee_pos[:2] = local_origin + rot_matrix @ radial_xy
            elif action == 3:  # Rotate right around local origin
                angle = -d_angle
                rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                       [np.sin(angle), np.cos(angle)]])
                ee_pos[:2] = local_origin + rot_matrix @ radial_xy
            elif action == 4:  # Move upward (Z+)
                ee_pos[2] += d_z
            elif action == 5:  # Move downward (Z-)
                ee_pos[2] -= d_z

            # Update position back to tensor
            pos[idx] = torch.from_numpy(ee_pos).to(self.device)

        # Perform inverse kinematics to achieve desired positions
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        # === Mic position remains unchanged from original script ===
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        mic_positions = ((left + right) / 2).cpu().numpy()
        mic_positions[:, 2] += 0.05  # slight vertical offset for visualization

        mic_pos_tensor = torch.from_numpy(mic_positions).float().to(self.device)
        for idx, mic_marker in enumerate(self.mic_markers):
            mic_marker.set_pos(mic_pos_tensor[idx].unsqueeze(0), envs_idx=[idx])

        # Simulate audio using unchanged logic
        new_slice = self._collect_spectrograms(play_audio=False)
        self.audio_history.append(new_slice)

        # Occasionally play stacked audio for verification
        if self.listen_idx is not None and (self.step_count % self.show_every == 0):
            snippets = [self.raw_audio_history[offset] for offset in self.sample_offsets]
            full_buffer = np.concatenate(snippets, axis=0)
            sd.play(full_buffer, 22050)
            sd.wait()

        # Build observation tensor
        obs = self._build_observation()

        # Optional visualization of spectrogram
        if self.num_envs == 1 and self.step_count % self.show_every == 0:
            self._plot_stacked(obs[0, 0])

        # Compute rewards based on unchanged distance logic
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube_pos = self.cube.get_pos()
        dist = torch.norm((left + right) / 2 - cube_pos, dim=1)
        rewards = torch.clamp(torch.exp(-4 * (dist - 0.1)), 0.0, 1.0)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Update current positions and step count
        self.pos = pos
        self.step_count += 1

        return obs, rewards, dones

    def _plot_stacked(self, data: torch.Tensor):
        """
        Render the stacked spectrogram in the live preview figure.
        """
        plt.clf()
        extent = [0, 10 * len(self.sample_offsets), 0, (22050 / 2) / 1000]
        plt.imshow(data.cpu().numpy(), origin='lower', aspect='auto', extent=extent, vmin=-40, vmax=100)
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (kHz)')
        plt.title(f'Step {self.step_count} Stacked Spec')
        plt.draw()
        plt.pause(0.01)
        self._fig.canvas.flush_events()


if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoAudioStackedEnv(vis=True, device=torch.device('cuda'), listen_idx=0)
