import torch
import torch.nn as nn

class PPOaudio(nn.Module):
    """
    Conv backbone -> shared features -> policy logits + state value.
    Input x: (B, C, F, T)
    """
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        C, F, T = obs_shape

        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=(8,3), stride=(2,1), padding=(2,1)), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5,3), stride=(2,1), padding=(1,1)), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(2,1), padding=(1,1)), nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, C, F, T)
            flat_dim = self.conv(dummy).view(1, -1).size(1)

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
        )
        self.pi = nn.Linear(256, action_dim)  # policy logits
        self.v  = nn.Linear(256, 1)           # state value

    def forward(self, x):
        h = self.shared(self.conv(x))
        logits = self.pi(h)
        value  = self.v(h).squeeze(-1)
        return logits, value
