# network/ppo_torque_control.py

import torch
import torch.nn as nn

class PPOContinuousTorque(nn.Module):
    """
    Actor-Critic PPO network for continuous torque control.
    Outputs mean and standard deviation for continuous Gaussian policy,
    and a scalar state-value estimate.
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PPOContinuousTorque, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head: outputs mean action values
        self.action_mean = nn.Linear(hidden_dim, action_dim)

        # Log-standard deviation for actions (learnable parameters)
        self.action_log_std = nn.Parameter(torch.full((action_dim,), -2.0))  # small initial std dev

        # Critic head: outputs state value
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            x: State tensor of shape (batch_size, input_dim)

        Returns:
            action_mean: Tensor of shape (batch_size, action_dim)
            action_std: Tensor of shape (batch_size, action_dim)
            state_value: Tensor of shape (batch_size)
        """
        features = self.feature_extractor(x)

        # Continuous action outputs
        action_mean = self.action_mean(features)
        action_std = self.action_log_std.exp().expand_as(action_mean)

        # State-value prediction
        state_value = self.value_head(features).squeeze(-1)

        return action_mean, action_std, state_value
