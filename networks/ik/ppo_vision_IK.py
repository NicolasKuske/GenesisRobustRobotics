# networks/ik/ppo_vision_IK.py


import torch
import torch.nn as nn


class PPOvision(nn.Module):
    def __init__(self, obs_shape, output_dim):
        super().__init__()
        C, H, W = obs_shape

        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )

        with torch.no_grad():
            dev = torch.device('cpu')
            dummy = torch.zeros(1, C, H, W, device=dev)
            conv_out_size = self.conv(dummy).view(1, -1).shape[1]

        # New: compact embedding used by the multimodal net
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
        )

        # Unimodal heads remain (useful for unimodal training / aux losses)
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def encode(self, x):
        return self.embed(self.conv(x))  # (B, 256)

    def forward(self, x):
        features = self.conv(x)
        logits = self.policy_head(features)
        value  = self.value_head(features).squeeze(-1)
        return logits, value
