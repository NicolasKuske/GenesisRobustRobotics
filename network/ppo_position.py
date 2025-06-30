#network/ppo_position.py


import torch.nn as nn

class PPOposition(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PPOposition, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)
