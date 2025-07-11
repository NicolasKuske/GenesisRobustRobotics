#algo/ppo_agent_multimodal.py


import torch
import torch.optim as optim
from torch.distributions import Categorical
from network.ppo_multimodal import PPOmultimodal

class PPOAgentMultimodal:
    def __init__(self, obs_shape_vision, obs_shape_audio, output_dim, lr, gamma, clip_epsilon, device, load, num_envs, checkpoint_path):
        self.device = device
        self.num_envs = num_envs
        self.model = PPOmultimodal(obs_shape_vision, obs_shape_audio, output_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.checkpoint_path = checkpoint_path

        if load:
            self.load_checkpoint()

    def save_checkpoint(self):
        torch.save({'model_state_dict': self.model.state_dict()}, self.checkpoint_path)

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def select_action(self, state_v, state_a):
        state_v, state_a = state_v.to(self.device), state_a.to(self.device)
        logits = self.model(state_v, state_a)
        return Categorical(logits=logits).sample()

    def train(self, states_v, states_a, actions, rewards, dones):
        # Similar logic as provided PPO agents
        pass  # Implement PPO training loop similarly to your original agents
