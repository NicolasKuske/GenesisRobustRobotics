# File: networks/ik/ppo_vision_IKsimple.py
import torch
import torch.nn as nn

class PPOvision(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super(PPOvision, self).__init__()
        C, H, W = obs_shape

        # Conv trunk only
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )

        # Infer flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            fmap = self.conv(dummy)
        flat_dim = fmap.view(1, -1).size(1)

        # Policy head (MLP)
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_shape),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        fmap = self.conv(x)                 # (B, C_v, H_v, W_v)
        logits = self.policy_head(fmap)    # flatten + MLP -> (B, action_shape/output_dim)
        return fmap, logits