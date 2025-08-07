# networks/ik/ppo_audio_IKsimple.py
import torch
import torch.nn as nn

class PPOaudioGripper(nn.Module):
    def __init__(self, obs_shape, gripper_dim, action_shape):
        super().__init__()
        C, F, T = obs_shape

        self.audio_conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=(8, 3), stride=(2, 1), padding=(2, 1)), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 1), padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)), nn.ReLU(),
        )

        with torch.no_grad():
            dummy_audio = torch.zeros(1, C, F, T)
            audio_feat = self.audio_conv(dummy_audio)
        audio_flat_dim = audio_feat.view(1, -1).size(1)

        self.audio_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(audio_flat_dim, 128),
            nn.ReLU(),
        )

        # Rename joint_proj -> gripper_proj
        self.gripper_proj = nn.Sequential(
            nn.Linear(gripper_dim, 128),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_shape),
        )

    def forward(self, audio_obs, gripper_obs):
        audio_embed = self.audio_proj(self.audio_conv(audio_obs))
        gripper_embed = self.gripper_proj(gripper_obs)

        combined = torch.cat([audio_embed, gripper_embed], dim=-1)
        logits = self.policy_head(combined)
        return logits
