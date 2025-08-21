# File: networks/ik/ppo_multimodal_IK.py
import torch
import torch.nn as nn
from networks.ik.ppo_vision_IK import PPOvision
from networks.ik.ppo_audio_IK  import PPOaudio

class PPOmultimodal(nn.Module):
    """
    Vision + Audio -> fused embedding -> policy logits + value.
    Uses unimodal backbones as feature extractors (encode()).
    """
    def __init__(
        self,
        obs_shape_v,
        obs_shape_a,
        action_dim,
        freeze_unimodal: bool = False,
        use_attention: bool = False,
        use_layernorm: bool = True
    ):
        super().__init__()
        self.vision = PPOvision(obs_shape_v, output_dim=action_dim)
        self.audio  = PPOaudio(obs_shape_a, action_dim)

        # Optionally freeze only the conv (and shared) parts
        if freeze_unimodal:
            for p in self.vision.conv.parameters():  p.requires_grad = False
            for p in self.vision.embed.parameters(): p.requires_grad = False
            for p in self.audio.conv.parameters():   p.requires_grad = False
            for p in self.audio.shared.parameters(): p.requires_grad = False

        self.use_attention = use_attention
        self.norm_v = nn.LayerNorm(256) if use_layernorm else nn.Identity()
        self.norm_a = nn.LayerNorm(256) if use_layernorm else nn.Identity()

        # Optional 2-way attention gate over {vision, audio}
        if use_attention:
            self.attn = nn.Sequential(
                nn.Linear(512, 128), nn.Tanh(),
                nn.Linear(128, 2), nn.Softmax(dim=-1)
            )
        else:
            self.attn = None

        # Fusion + heads
        self.fusion = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.pi = nn.Linear(256, action_dim)
        self.v  = nn.Linear(256, 1)

    def forward(self, x_v, x_a):
        hv = self.vision.encode(x_v)  # (B, 256)
        ha = self.audio.encode(x_a)   # (B, 256)

        hv = self.norm_v(hv)
        ha = self.norm_a(ha)

        if self.attn is not None:
            comb = torch.cat([hv, ha], dim=1)        # (B, 512)
            w = self.attn(comb)                      # (B, 2)
            hv = hv * w[:, 0:1]
            ha = ha * w[:, 1:1+1]

        h  = self.fusion(torch.cat([hv, ha], dim=1)) # (B, 256)
        logits = self.pi(h)
        value  = self.v(h).squeeze(-1)
        return logits, value
