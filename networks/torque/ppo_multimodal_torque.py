# File: networks/torque/ppo_multimodal_torque.py


import torch
import torch.nn as nn
from networks.torque.ppo_vision_torque import PPOVisionTorque
from networks.torque.ppo_audio_torque import PPOAudioTorque


class PPOMultimodalTorque(nn.Module):
    """
    PPO Multimodal Network with Continuous Gaussian Policy for Torque Control.
    Combines vision and audio branches, includes actor (mean, log_std) and critic (value).
    """

    def __init__(self, obs_shape_v, obs_shape_a, action_dim, hidden_dim=512, use_attention=False):
        super(PPOMultimodalTorque, self).__init__()

        # Pre-trained unimodal feature extractors
        self.vision_net = PPOVisionTorque(obs_shape_v, action_dim=1)
        self.audio_net = PPOAudioTorque(obs_shape_a, action_dim=1)

        # Infer output dimensions dynamically
        with torch.no_grad():
            dummy_v = torch.zeros((1, *obs_shape_v))
            dummy_a = torch.zeros((1, *obs_shape_a))
            v_feats = self.vision_net.conv(dummy_v)
            a_feats = self.audio_net.conv(dummy_a)

        v_feat_dim = v_feats.view(1, -1).size(1)
        a_feat_dim = a_feats.view(1, -1).size(1)

        # Feature normalization
        self.norm_v = nn.LayerNorm(v_feat_dim)
        self.norm_a = nn.LayerNorm(a_feat_dim)

        # Fusion dimension
        fusion_dim = v_feat_dim + a_feat_dim

        # Optional attention-based fusion
        self.use_attention = use_attention
        if use_attention:
            self.attn_fc = nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 2),
                nn.Softmax(dim=-1)
            )

        # Shared MLP after fusion
        self.shared_fc = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head: Mean and Log-Std for Gaussian policy
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.full((action_dim,), -2.0))

        # Critic head: State-value estimation
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, vision_input, audio_input):
        # Extract unimodal features
        v_map = self.vision_net.conv(vision_input)
        a_map = self.audio_net.conv(audio_input)

        v_flat = v_map.view(v_map.size(0), -1)
        a_flat = a_map.view(a_map.size(0), -1)

        # Normalize features
        v_norm = self.norm_v(v_flat)
        a_norm = self.norm_a(a_flat)

        # Fusion
        combined = torch.cat([v_norm, a_norm], dim=1)
        if self.use_attention:
            w = self.attn_fc(combined)
            w_v, w_a = w[:, :1], w[:, 1:2]
            fused = torch.cat([v_norm * w_v, a_norm * w_a], dim=1)
        else:
            fused = combined

        # Shared fully-connected layers
        feats = self.shared_fc(fused)

        # Actor: continuous torque policy
        action_mean = self.action_mean(feats)
        action_std = self.action_log_std.exp().expand_as(action_mean)

        # Critic: state-value
        state_value = self.value_head(feats).squeeze(-1)

        return action_mean, action_std, state_value
