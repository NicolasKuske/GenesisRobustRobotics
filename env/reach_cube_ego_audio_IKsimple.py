import numpy as np
import genesis as gs
import torch
import librosa
import matplotlib.pyplot as plt
from scipy.signal import chirp

class ReachCubeEgoAudioEnv:
    """
    Genesis environment with audio-only observations and random cube repositioning.
    The agent receives a (1×257×9) spectrogram each step, optionally plays and plots one env's audio every `show_every` steps.
    Cube position is fixed initially, then randomized every `randomize_every` episodes.
    """
    def __init__(
        self, vis: bool, device: torch.device,
        num_envs: int = 1, listen_idx: int = None,
        show_every: int = 10, randomize_every: int = 100
    ):
        # --- Config ---
        self.device          = device
        self.num_envs        = num_envs
        self.listen_idx      = listen_idx
        self.show_every      = show_every
        self.randomize_every = randomize_every
        self.step_count      = 0
        self.episode_count   = 0
        self.current_cube_pos = None

        # --- Plot setup ---
        self._fig = plt.figure("Spectrogram Preview")

        # --- Observation & Action specs ---
        self.obs_shape    = (1, 257, 9)
        self.action_space = 6

        # --- Build Genesis scene ---
        self.scene = gs.Scene(
            show_FPS=False,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3,2,1.5), camera_lookat=(0,0,0.2),
                camera_fov=30, res=(960,640), max_FPS=60
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=vis,
            )

        # Add ground plane, robot, and cube (initial pos placeholder)
        self.plane  = self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml")
        )
        # initial cube; will be set on first reset
        initial_pos = (0.65, 0.0, 0.1)
        self.cube   = self.scene.add_entity(
            gs.morphs.Box(size=(0.06,0.06,0.06), pos=initial_pos),
            surface=gs.surfaces.Rough(color=(0.99,0.82,0.09)),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        # Build multiple environments
        self.scene.build(n_envs=self.num_envs, env_spacing=(5.0,5.0))
        self.envs_idx = np.arange(self.num_envs)

        # Initialize robot (cube pos will be set in reset)
        self.build_env()

    def build_env(self):
        """Reset robot to a neutral pose in all environments."""
        self.motors_dof  = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7,9).to(self.device)

        neutral_q = torch.tensor(
            [-1.0, -0.3, 0.3, -1.0, -0.1, 1.7, 1.0, 0.02, 0.02],
            device=self.device
        ).unsqueeze(0).repeat(self.num_envs,1)

        self.franka.set_qpos(neutral_q, envs_idx=self.envs_idx)
        self.scene.step()
        self.fixed_finger_pos = neutral_q[:,7:9].clone()
        self.end_effector      = self.franka.get_link("hand")

        pos  = torch.tensor([0.2720, -0.1683, 1.0164], device=self.device)
        quat = torch.tensor([0.1992, 0.7857, -0.3897, 0.4371], device=self.device)
        self.pos  = pos.unsqueeze(0).repeat(self.num_envs,1)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs,1)

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:,:-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)

    def simulate_audio(self, dist: float) -> np.ndarray:
        """Generate 100 ms of audio: fixed tone + random noise."""
        sr, dur = 22050, 0.1
        t = np.linspace(0, dur, int(sr*dur), endpoint=False)
        tone = chirp(t, f0=1000, f1=1000, t1=dur) / (dist**2 + 1e-6)
        noise = sum(
            np.random.rand() * chirp(
                t, np.random.randint(300,3000), np.random.randint(300,3000), dur
            ) for _ in range(5)
        ) * 0.1
        audio = tone + noise
        #print(f"[Audio] dist={dist:.2f}m, waveform max={audio.max():.3f}")
        return audio

    def _compute_spectrogram(self, audio: np.ndarray, play: bool=False) -> np.ndarray:
        """Convert audio to a spectrogram; play if requested."""
        if play and self.listen_idx is not None:
            import sounddevice as sd
            sd.play(audio, 22050)
            sd.wait()
        S = librosa.stft(audio, n_fft=512, hop_length=256)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=1.0)[:,:9]
        #print(f"[Spec] min={S_db.min():.1f}dB, max={S_db.max():.1f}dB")
        return S_db

    def _show_spec(self, S_db: np.ndarray):
        """Plot the spectrogram in the existing figure with fixed color limits."""
        plt.clf()
        plt.imshow(S_db, origin='lower', aspect='auto', extent=[0,100,0,11.025], vmin=-40, vmax=100)
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (kHz)')
        plt.title(f'Step {self.step_count} (max={S_db.max():.1f}dB)')
        plt.draw()
        plt.pause(0.01)
        self._fig.canvas.flush_events()

    def _collect_states(self, play_audio: bool=False) -> torch.Tensor:
        """
        1. Positions → distances using average finger pos.
        2. Simulate audio per distance + print max.
        3. Compute spectrogram with fixed ref + print dB range.
        4. Optionally play/plot every show_every steps.
        """
        left  = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        cube_pos = self.cube.get_pos()
        dists = torch.norm((left + right)/2 - cube_pos, dim=1).cpu().numpy()

        obs_list = []
        for i, dist in enumerate(dists):
            audio = self.simulate_audio(dist)
            play  = play_audio and (i == self.listen_idx)
            plot  = play and (self.step_count % self.show_every == 0)

            S_db = self._compute_spectrogram(audio, play=play)
            if plot:
                self._show_spec(S_db)

            obs_list.append(torch.from_numpy(S_db).float().unsqueeze(0))

        self.step_count += 1
        return torch.stack(obs_list).to(self.device)

    def reset(self) -> torch.Tensor:
        """Reset env, use fixed initial pos for episode 1, then randomize every randomize_every episodes."""
        self.episode_count += 1
        if self.episode_count == 1:
            # fixed start
            one_pos = np.array([0.65, 0.0, 0.1]).reshape(1,3)
            self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)
        elif self.episode_count % self.randomize_every == 0:
            # subsequent randomizations
            xy = np.random.uniform(0.2, 1.0, size=(1, 2)) * np.random.choice([-1, 1], size=(1, 2))
            z  = np.random.uniform(0.1, 1.0, size=(1, 1))
            one_pos = np.concatenate([xy, z], axis=1)
            self.current_cube_pos = np.repeat(one_pos, self.num_envs, axis=0)
        # else: keep previous position

        self.build_env()
        self.cube.set_pos(self.current_cube_pos, envs_idx=self.envs_idx)
        return self._collect_states(play_audio=(self.listen_idx is not None))

    def step(self, actions: torch.Tensor):
        """Apply actions, step sim, return (obs, rewards, dones)."""
        deltas = torch.tensor([
            [0.05,0,0],[-0.05,0,0],[0,0.05,0],
            [0,-0.05,0],[0,0,0.05],[0,0,-0.05]
        ], device=self.device)
        masks  = [actions == i for i in range(self.action_space)]
        self.pos += sum(deltas[i] * masks[i].unsqueeze(1) for i in range(self.action_space))

        qpos = self.franka.inverse_kinematics(
            link=self.end_effector, pos=self.pos, quat=self.quat
        )
        self.franka.control_dofs_position(qpos[:,:-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.fixed_finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        obs = self._collect_states(play_audio=(self.listen_idx is not None))
        cube_pos = self.cube.get_pos()
        left  = self.franka.get_link("left_finger").get_pos()
        right = self.franka.get_link("right_finger").get_pos()
        dist = torch.norm((left + right)/2 - cube_pos, dim=1)
        rewards = torch.exp(-4*(dist-0.1)).clamp(0,1)
        dones   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return obs, rewards, dones

# main guard
if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    env = ReachCubeEgoAudioEnv(vis=True, device=torch.device('cuda'), listen_idx=0)

