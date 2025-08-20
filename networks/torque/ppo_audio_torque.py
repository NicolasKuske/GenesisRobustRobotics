# networks/ppo_audio_torque.py
import torch
import torch.nn as nn

class PPOAudioTorque(nn.Module):
    """
    Convolutional Actor-Critic networks for PPO with continuous torque outputs.
    Processes not_stacked spectrogram inputs and produces a Gaussian policy (mean and std) over joint torques,
    along with a state-value estimate.
    """
    def __init__(self, obs_shape, action_dim, hidden_dim=256):
        super(PPOAudioTorque, self).__init__()
        # obs_shape: (C, F, T) where C=1, F=257, T=6 (not_stacked frames)
        C, F, T = obs_shape
        # 1) Convolutional trunk (adapted from simple audio networks)
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=(8, 3), stride=(2, 1), padding=(2, 1)), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 1), padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)), nn.ReLU(),
            nn.Flatten(),
        )
        # Determine conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, C, F, T)
            conv_out_size = self.conv(dummy).shape[1]
        # 2) Shared MLP
        self.shared = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # 3a) Actor head: outputs mean torques for each joint
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        # 3b) Actor head: learnable log-standard-deviation (one per joint)
        self.action_log_std = nn.Parameter(torch.full((action_dim,), -2.0))
        # 4) Critic head: outputs state-value estimate
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch_size, C, F, T]
        feats = self.conv(x)
        feats = self.shared(feats)
        mean = self.action_mean(feats)
        std = self.action_log_std.exp().expand_as(mean)
        value = self.value_head(feats).squeeze(-1)
        return mean, std, value

