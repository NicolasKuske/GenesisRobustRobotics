# File: networks/ik/ppo_audio_IKsimple.py
import torch
import torch.nn as nn

class PPOaudio(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super(PPOaudio, self).__init__()
        C, F, T = obs_shape

        # Conv trunk only
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=(8,3), stride=(2,1), padding=(2,1)), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5,3), stride=(2,1), padding=(1,1)), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(2,1), padding=(1,1)), nn.ReLU(),
        )

        # Infer flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, C, F, T)
            fmap = self.conv(dummy)
        flat_dim = fmap.view(1, -1).size(1)

        # Audio head (MLP)
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_shape),
        )

    def forward(self, x):
        # x: (B, C, F, T)
        fmap = self.conv(x)                 # (B, C_a, H_a, W_a)
        logits = self.policy_head(fmap)
        return fmap, logits