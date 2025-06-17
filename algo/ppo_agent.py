# ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from network.ppo import PPO  # your conv-based PPO class

class PPOAgent:
    def __init__(
        self,
        obs_shape,            # tuple, e.g. (3, 120, 120)
        output_dim,           # number of discrete actions
        lr,
        gamma,
        clip_epsilon,
        device,
        load=False,
        num_envs=1,
        checkpoint_path=None
    ):
        self.device = device
        self.num_envs = num_envs
        self.checkpoint_path = checkpoint_path

        # instantiate our conv-based policy network
        self.model = PPO(obs_shape, output_dim).to(self.device)

        if load:
            self.load_checkpoint()
            print("Loaded model from checkpoint")

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved to {self.checkpoint_path}")

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Checkpoint loaded from {self.checkpoint_path}")

    def select_action(self, state):
        """
        state: tensor of shape (num_envs, C, H, W), float32 in [0,1]
        returns: tensor of shape (num_envs,) with discrete actions
        """
        state = state.to(self.device)
        with torch.no_grad():
            logits = self.model(state)
        probs = nn.functional.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action

    def train(self, states, actions, rewards, dones):
        states_tensor = torch.stack(states).to(self.device)
        actions_tensor = torch.stack(actions).to(self.device)
        rewards_list = rewards
        dones_list = dones

        #print(f"[DEBUG] states_tensor shape: {states_tensor.shape}")
        #print(f"[DEBUG] actions_tensor shape: {actions_tensor.shape}")

        discounted_rewards = []
        R = torch.zeros(self.num_envs, device=self.device)
        for reward, done in zip(reversed(rewards_list), reversed(dones_list)):
            R = reward.to(self.device) + self.gamma * R * (~done.to(self.device))
            discounted_rewards.insert(0, R)
        discounted_tensor = torch.stack(discounted_rewards)

        advantages = discounted_tensor - discounted_tensor.mean()

        #print(f"[DEBUG] discounted_tensor shape: {discounted_tensor.shape}")

        # Now check dimensions explicitly
        #print(f"[DEBUG] len(states_tensor.shape): {len(states_tensor.shape)}")

        T, N = states_tensor.shape[:2]
        if len(states_tensor.shape) == 5:
            C, H, W = states_tensor.shape[2:]
            states_flat = states_tensor.reshape(T * N, C, H, W)
        else:
            raise ValueError(f"Unexpected states_tensor shape: {states_tensor.shape}")

        with torch.no_grad():
            logits_old = self.model(states_flat)
            probs_old = nn.functional.softmax(logits_old, dim=-1)

        actions_flat = actions_tensor.view(-1)
        advantages_flat = advantages.view(-1)

        for _ in range(10):
            logits_new = self.model(states_flat)
            probs_new = nn.functional.softmax(logits_new, dim=-1)

            dist_old = Categorical(probs_old)
            dist_new = Categorical(probs_new)

            ratio = (dist_new.log_prob(actions_flat) - dist_old.log_prob(actions_flat)).exp()

            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_flat
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


