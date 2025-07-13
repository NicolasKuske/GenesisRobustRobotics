import torch
import torch.nn as nn

class PPOposition(nn.Module):
    """
    Actor-Critic network for PPO: shares a backbone then splits into policy (logits) and value outputs.
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super(PPOposition, self).__init__()
        # shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # policy head: outputs action logits
        self.pi = nn.Linear(hidden_dim, action_dim)
        # value head: outputs state-value estimate
        self.v  = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: input state tensor of shape (batch_size, input_dim)

        Returns:
            logits: unnormalized action scores, shape (batch_size, action_dim)
            value: state-value estimates, shape (batch_size,)
        """
        # extract features
        h = self.backbone(x)
        # compute policy logits
        logits = self.pi(h)
        # compute value and remove last dim
        value = self.v(h).squeeze(-1)
        return logits, value
