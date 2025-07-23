# File: networks/ik/ppo_multimodal_IKsimple.py
import torch
import torch.nn as nn
from networks.ik.ppo_vision_IKsimple import PPOvision
from networks.ik.ppo_audio_IKsimple import PPOaudio

class PPOmultimodal(nn.Module):
    def __init__(
        self,
        obs_shape_v,
        obs_shape_a,
        action_shape,
        freeze_unimodal=False,
        use_attention=False
    ):
        super().__init__()

        # Vision branch
        self.vision = PPOvision(obs_shape_v, action_shape=1)
        # Audio branch
        self.audio  = PPOaudio(obs_shape_a, action_shape=1)

        # Optionally freeze
        if freeze_unimodal:
            for p in self.vision.conv.parameters(): p.requires_grad = False
            for p in self.audio.conv.parameters():  p.requires_grad = False

        # Infer conv‐map shapes
        with torch.no_grad():
            v_map, _ = self.vision(torch.zeros((1, *obs_shape_v)))
            a_map, _ = self.audio(torch.zeros((1, *obs_shape_a)))

        # LayerNorm over feature‐maps
        self.norm_v = nn.LayerNorm(v_map.shape[1:])
        self.norm_a = nn.LayerNorm(a_map.shape[1:])

        # Compute flat dims and fusion dim
        flat_v = v_map.view(1, -1).size(1)
        flat_a = a_map.view(1, -1).size(1)
        fusion_dim = flat_v + flat_a

        self.use_attention = use_attention
        if use_attention:
            self.attn_fc = nn.Sequential(
                nn.Linear(fusion_dim, 128), nn.Tanh(),
                nn.Linear(128, 2),     nn.Softmax(dim=-1)
            )

        # Final policy head
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_shape)
        )

    def forward(self, vision_input, audio_input):
        # vision_input: (B, C_v, H, W)
        v_map, _logits_v = self.vision(vision_input)
        v_norm = self.norm_v(v_map)
        v_flat = v_norm.view(v_norm.size(0), -1)

        # audio_input: (B, C_a, F, T)
        a_map, _logits_a = self.audio(audio_input)
        a_norm = self.norm_a(a_map)
        a_flat = a_norm.view(a_norm.size(0), -1)

        # fuse
        combined = torch.cat([v_flat, a_flat], dim=1)
        if self.use_attention:
            w = self.attn_fc(combined)
            w_v, w_a = w[:, 0:1], w[:, 1:2]
            fused = torch.cat([v_flat*w_v, a_flat*w_a], dim=1)
        else:
            fused = combined

        # policy logits
        return self.fc(fused)