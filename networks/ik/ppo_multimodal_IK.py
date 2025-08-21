# File: networks/ik/ppo_multimodal_IK.py

import torch
import torch.nn as nn


class _VisionStem(nn.Module):
    """
    Vision conv trunk (same spirit as PPOvision) for inputs (B, C_v, H, W).
    """
    def __init__(self, obs_shape_v):
        super().__init__()
        C, H, W = obs_shape_v
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        # probe output size
        with torch.no_grad():
            dev = torch.device('cpu')
            dummy = torch.zeros(1, C, H, W, device=dev)
            flat_dim = self.conv(dummy).view(1, -1).size(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.head(self.conv(x))  # (B, 256)


class _AudioStem(nn.Module):
    """
    Audio conv trunk (same spirit as PPOaudio) for inputs (B, C_a, F, T).
    Default C_a=1, F=257, T=5 (stacked 10ms slices at 5 offsets).
    """
    def __init__(self, obs_shape_a):
        super().__init__()
        C, F, T = obs_shape_a
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=(8, 3), stride=(2, 1), padding=(2, 1)), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 1), padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)), nn.ReLU(),
        )
        # probe output size
        with torch.no_grad():
            dev = torch.device('cpu')
            dummy = torch.zeros(1, C, F, T, device=dev)
            flat_dim = self.conv(dummy).view(1, -1).size(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.head(self.conv(x))  # (B, 256)


class PPOmultimodal(nn.Module):
    """
    Vision + Audio → fused features → policy logits + value.
    Mirrors your unimodal actor-critic heads.

    Inputs:
      vision: (B, C_v, H, W)
      audio:  (B, C_a, F, T)
    Outputs:
      logits: (B, action_dim), value: (B,)
    """
    def __init__(self, obs_shape_v, obs_shape_a, action_dim):
        super().__init__()
        self.vis = _VisionStem(obs_shape_v)
        self.aud = _AudioStem(obs_shape_a)

        # fuse (256 + 256) → 256
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.pi = nn.Linear(256, action_dim)
        self.v  = nn.Linear(256, 1)

    def forward(self, x_v, x_a):
        hv = self.vis(x_v)   # (B, 256)
        ha = self.aud(x_a)   # (B, 256)
        h  = self.fusion(torch.cat([hv, ha], dim=1))  # (B, 256)
        logits = self.pi(h)
        value  = self.v(h).squeeze(-1)
        return logits, value
