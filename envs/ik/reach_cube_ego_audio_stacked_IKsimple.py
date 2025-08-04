#envs/ik/reach_cube_ego_audio_stacked_IKsimple.py


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
            episodes_per_position: int = 2,
            history_length: int = 25,
            reward_thresholds=None,
            window_size: int = 9,
            sample_offsets=None,
            noise_config: dict = None,
            initial_cube_pos=None, #[-0.9, 0.6, 0.7],

    ):
        self.initial_cube_pos = initial_cube_pos
        self.device = device
        self.num_envs = num_envs
        self.listen_idx = listen_idx
        self.show_every = show_every
        self.vis = vis  # <-- Add this line

        self.history_length = history_length
        self.sample_offsets = sample_offsets or [-21, -16, -11, -6, -1]
        self.audio_history = deque(maxlen=self.history_length)
        self.raw_audio_history = deque(maxlen=self.history_length)
        self.noise_config = noise_config if noise_config else {"audio_noise_level": 0.0}

        self.freq_bins = 257
        self.time_bins = len(self.sample_offsets)
        self.obs_shape = (1, self.freq_bins, self.time_bins)
        self.action_space = 6

        self._fig = plt.figure("Stacked Spectrogram Preview")
        self._build_scene(vis)
        self._init_robot()

        self.step_count = 0
        self.episode_count = 0

        # Curriculum parameters (exactly from torque script)
        self.episodes_per_position = episodes_per_position
        self.window_size = window_size
        self.reward_thresholds = reward_thresholds or [25, 25, 25, 25, 25,25, 30]  # Adjust based on desired difficulty progression

        self.last_rewards = deque(maxlen=self.window_size)
        self.x_bounds = [0.4, 0.2, 0.0, -0.2, -0.4, -0.6]
        self.fixed_x = 0.6
        self.max_stages = len(self.x_bounds)
        self.x_stage = 0
        self.completed = False


        # Episode tracking
        self.episode_reward = 0.0



    def _build_scene(self, vis: bool):
        """Set up the Genesis scene, ground plane, robot, and cube."""
        self.scene = gs.Scene(
            #show_FPS=False,
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
        sr, dur = 22050, 0.01
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        tone = chirp(t, f0=1000, f1=1000, t1=dur) / (dist ** 2 + 1e-6)

        # Base random noise
        noise = sum(
            np.random.rand() * chirp(
                t,
                np.random.randint(300, 3000),
                np.random.randint(300, 3000),
                dur
            )
            for _ in range(5)
        ) * 0.1

        # Additional noise based on config
        audio_noise_level = self.noise_config.get("audio_noise_level", 0.0)
        additional_noise = np.random.normal(0, audio_noise_level, tone.shape)

        return tone + noise + additional_noise

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
            # Store raw audio only for debugging/playback purposes (listener index)
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

    def reset(self):
        # Call _process_episode_end ONLY if this is not the very first reset call
        if self.episode_count >= 1:
            self._process_episode_end()

        self.episode_count += 1

        if self.completed:
            dones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            return None, dones

        # Sample new cube position only every episodes_per_position episodes
        if (self.episode_count - 1) % self.episodes_per_position == 0:
            if self.episode_count == 1 and self.initial_cube_pos is not None:
                self.current_cube_pos = np.repeat(
                    np.array([self.initial_cube_pos]), self.num_envs, axis=0
                )
            else:
                self.current_cube_pos = self._sample_cube_pos_shared()

            # Print cube positions clearly at reset
            print(f"[Episode {self.episode_count}] Cube X positions: {self.current_cube_pos[:, 0]}")

        self._init_robot()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        self.scene.step()

        self.episode_reward = 0.0

        first_spec = self._collect_spectrograms(play_audio=False)
        first_raw = self.raw_audio_history[-1].copy()
        self.audio_history.clear()
        self.raw_audio_history.clear()
        for _ in range(self.history_length):
            self.audio_history.append(first_spec.clone())
            self.raw_audio_history.append(first_raw.copy())

        obs = self._build_observation()
        if self.num_envs == 1 and self.vis:
            self._plot_stacked(obs[0, 0])

        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return obs, dones


    def step(self, actions: torch.Tensor):
        if self.completed:
            dones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            return None, None, dones

        # Move end-effector by fixed deltas
        deltas = torch.tensor([
            [0.05, 0, 0], [-0.05, 0, 0], [0, 0.05, 0],
            [0, -0.05, 0], [0, 0, 0.05], [0, 0, -0.05]
        ], device=self.device)
        masks = [actions == i for i in range(self.action_space)]
        self.pos += sum(deltas[i] * masks[i].unsqueeze(1) for i in range(self.action_space))

        # IK control
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        # Collect new slice
        new_slice = self._collect_spectrograms(play_audio=False)
        self.audio_history.append(new_slice)

        # Audio playback periodically
        if self.listen_idx is not None and (self.step_count % self.show_every == 0):
            snippets = [self.raw_audio_history[offset] for offset in self.sample_offsets]
            full_buffer = np.concatenate(snippets, axis=0)
            sd.play(full_buffer, 22050)
            sd.wait()

        # Build observation
        obs = self._build_observation()
        if self.num_envs == 1 and self.step_count % self.show_every == 0 and self.vis:

            self._plot_stacked(obs[0, 0])

        # Simplified exponential reward function
        left = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube_pos = self.cube.get_pos()
        dist = torch.norm((left + right) / 2 - cube_pos, dim=1)

        rewards = torch.clamp(torch.exp(-4 * dist), 0.0, 1.0)

        # Update trackers to keep compatibility with existing curriculum logging
        self.episode_reward += rewards.mean().item()


        # Dones logic: keep as zeros since no success threshold is used
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
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

    def _sample_cube_pos_shared(self) -> np.ndarray:
        """
        Sample a single cube position, then duplicate it for all environments.
        """
        idx = min(self.x_stage, self.max_stages - 1)
        lower = self.x_bounds[idx]
        x = np.random.uniform(lower, self.fixed_x)
        y = np.random.uniform(-0.6, 0.6)
        z = np.random.uniform(0.1, 1.0)
        single_pos = np.array([[x, y, z]])
        return np.repeat(single_pos, self.num_envs, axis=0)

    def _process_episode_end(self):
        # Only process rewards if at least one episode has completed
        if self.episode_count >= 1:
            print(f"[Episode {self.episode_count}] Total: {self.episode_reward:.4f}")

            self.last_rewards.append(self.episode_reward)
            if len(self.last_rewards) == self.window_size:
                mean_r = sum(self.last_rewards) / self.window_size
                thr = self.reward_thresholds[min(self.x_stage, len(self.reward_thresholds) - 1)]
                print(f"[Curriculum] last {self.window_size}-ep mean: {mean_r:.4f}, threshold: {thr:.4f}")
                if mean_r > thr:
                    self._advance_stage()

    def _advance_stage(self):
        self.x_stage += 1
        if self.x_stage < self.max_stages:
            lb = self.x_bounds[self.x_stage]
            print(f"Advanced to stage {self.x_stage}: X ∈ [{lb:.2f}, {self.fixed_x:.2f}]")
            self.last_rewards.clear()
        else:
            self.completed = True
            print("Curriculum complete!")






if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoAudioStackedEnv(vis=True, device=torch.device('cuda'), listen_idx=0)
