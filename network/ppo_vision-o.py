# network/ppo_vision-o.py

import torch
import torch.nn as nn


class PPOvision(nn.Module):
    def __init__(self, obs_shape, output_dim):
        # Python 3 allows either of these:
        # super().__init__()
        super(PPOvision, self).__init__()

        C, H, W = obs_shape

        # convolutional “trunk” to extract features from the image
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )

        # figure out the size of the conv output so we can
        # build the linear “policy head” dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out_size = self.conv(dummy).view(1, -1).shape[1]

        # a small MLP on top of the conv features to produce logits
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        """
        x: a batch of images, shape (batch_size, C, H, W)
        returns: action logits, shape (batch_size, output_dim)
        """
        features = self.conv(x)
        logits = self.policy_head(features)
        return logits
