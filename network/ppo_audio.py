# network/ppo_audio.py

import torch
import torch.nn as nn

class PPOaudio(nn.Module):
    def __init__(self, obs_shape, output_dim):
        super(PPOaudio, self).__init__()

        # Spectrogram shape: (1, 257, 9)
        C, F, T = obs_shape

        self.net = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=(8, 3), stride=(2, 1), padding=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        dummy_input = torch.zeros((1, C, F, T))
        flattened_size = self.net(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return self.fc(x)
