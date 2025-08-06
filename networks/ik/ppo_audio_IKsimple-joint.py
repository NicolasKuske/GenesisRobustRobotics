# networks/ik/ppo_audio_IKsimple.py
import torch
import torch.nn as nn

class PPOaudioJoints(nn.Module):
    def __init__(self, obs_shape, joint_dim, action_shape):
        super(PPOaudioJoints, self).__init__()
        C, F, T = obs_shape

        # Audio CNN trunk
        self.audio_conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=(8, 3), stride=(2, 1), padding=(2, 1)), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 1), padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)), nn.ReLU(),
        )

        # Infer flattened audio features size
        with torch.no_grad():
            dummy_audio = torch.zeros(1, C, F, T)
            audio_feat = self.audio_conv(dummy_audio)
        audio_flat_dim = audio_feat.view(1, -1).size(1)

        # Project audio features to embedding
        self.audio_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(audio_flat_dim, 128),
            nn.ReLU(),
        )

        # Joint embedding MLP
        self.joint_proj = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.ReLU(),
        )

        # Combined modality policy head
        self.policy_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_shape),
        )

    def forward(self, audio_obs, joint_obs):
        # audio_obs: (B, C, F, T)
        audio_embed = self.audio_proj(self.audio_conv(audio_obs))  # (B, 128)
        joint_embed = self.joint_proj(joint_obs)                  # (B, 128)

        combined = torch.cat([audio_embed, joint_embed], dim=-1)  # (B, 256)

        logits = self.policy_head(combined)
        return logits
