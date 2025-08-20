# networks/ppo_vision_IK.py


import torch
import torch.nn as nn


class PPOvision(nn.Module):
    def __init__(self, obs_shape, output_dim):
        """
        A convolutional Actor-Critic networks for PPO on vision inputs.

        Args:
            obs_shape (tuple): Input image shape (C, H, W).
            output_dim (int): Number of discrete actions.
        """
        super(PPOvision, self).__init__()

        C, H, W = obs_shape

        # convolutional trunk to extract features from the image
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )

        with torch.no_grad():
            dev = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device(
                'cpu')
            dummy = torch.zeros(1, C, H, W, device=dev)
            conv_out_size = self.conv(dummy).view(1, -1).shape[1]

        # policy head: produces action logits
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        # value head: produces state-value estimates
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Batch of images, shape (batch_size, C, H, W).

        Returns:
            logits (torch.Tensor): Unnormalized action scores, shape (batch_size, output_dim).
            value  (torch.Tensor): State-value estimates, shape (batch_size,).
        """
        features = self.conv(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value
