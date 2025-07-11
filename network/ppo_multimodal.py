import torch
import torch.nn as nn
from network.ppo_vision import PPOvision
from network.ppo_audio import PPOaudio

class PPOmultimodal(nn.Module):
    def __init__(self, obs_shape_v, obs_shape_a, output_dim):
        super().__init__()
        # Instantiate vision and audio models to reuse their conv trunks
        dummy_v = torch.zeros((1, *obs_shape_v))
        vision_net = PPOvision(obs_shape_v, output_dim=1)
        self.vision_conv = vision_net.conv
        # Determine vision feature dimension
        with torch.no_grad():
            vision_feats = self.vision_conv(dummy_v)
        vision_feature_dim = vision_feats.view(1, -1).size(1)

        dummy_a = torch.zeros((1, *obs_shape_a))
        audio_net = PPOaudio(obs_shape_a, output_dim=1)
        self.audio_conv = audio_net.net
        # Determine audio feature dimension
        with torch.no_grad():
            audio_feats = self.audio_conv(dummy_a)
        audio_feature_dim = audio_feats.view(1, -1).size(1)

        # Combined MLP head
        self.fc = nn.Sequential(
            nn.Linear(vision_feature_dim + audio_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, vision_input, audio_input):
        # vision_input: (batch, C_v, H, W)
        v = self.vision_conv(vision_input)
        v = v.view(v.size(0), -1)
        # audio_input: (batch, C_a, F, T)
        a = self.audio_conv(audio_input)
        a = a.view(a.size(0), -1)
        # concatenate features
        combined = torch.cat([v, a], dim=1)
        # head to logits
        logits = self.fc(combined)
        return logits
