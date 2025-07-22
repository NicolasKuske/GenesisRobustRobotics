# networks/ppo_vision_torque.py
# Continuous-torque PPO networks for vision-based control

import torch
import torch.nn as nn

class PPOVisionTorque(nn.Module):
    """
    Convolutional Actor-Critic networks for PPO with continuous torque outputs.
    Processes image inputs and produces a Gaussian policy (mean and std) over joint torques,
    along with a state-value estimate.
    """
    def __init__(self, obs_shape, action_dim, hidden_dim=512):
        super(PPOVisionTorque, self).__init__()

        # Unpack input image shape
        C, H, W = obs_shape

        # 1) Convolutional trunk
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        # Determine the size of the conv output
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
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
        """
        Forward pass.

        Args:
            x (torch.Tensor): Batch of images, shape (batch_size, C, H, W).

        Returns:
            action_mean (torch.Tensor): Mean torques, shape (batch_size, action_dim).
            action_std  (torch.Tensor): Standard deviations, same shape as mean.
            state_value (torch.Tensor): State-value estimates, shape (batch_size,).
        """
        # 1) Extract convolutional features
        feats = self.conv(x)

        # 2) Shared MLP
        feats = self.shared(feats)

        # 3) Actor outputs
        action_mean = self.action_mean(feats)
        action_std = self.action_log_std.exp().expand_as(action_mean)

        # 4) Critic output
        state_value = self.value_head(feats).squeeze(-1)

        return action_mean, action_std, state_value
