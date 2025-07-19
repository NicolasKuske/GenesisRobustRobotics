#algo/ppo_agent_position.py


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import NamedTuple, Optional

# A simple container for rollout data
class RolloutBatch(NamedTuple):
    states: torch.Tensor       # [T+1, N, state_dim]
    actions: torch.Tensor      # [T,   N]
    log_probs: torch.Tensor    # [T,   N]
    values: torch.Tensor       # [T+1, N]
    rewards: torch.Tensor      # [T,   N]
    dones: torch.Tensor        # [T,   N]

class PPOAgentPosition:
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: Optional[str] = 'cpu',
        checkpoint_path: Optional[str] = None,
        load: bool = False,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # policy + value network
        from network.ppo_position import PPOposition
        self.model = PPOposition(input_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # checkpoint
        self.checkpoint_path = checkpoint_path
        if load and checkpoint_path:
            self.load_checkpoint()

    def save_checkpoint(self):
        torch.save({'model_state_dict': self.model.state_dict()}, self.checkpoint_path)
        print(f"Saved checkpoint to {self.checkpoint_path}")

    def load_checkpoint(self):
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint from {self.checkpoint_path}")

    def select_action(self, state: torch.Tensor):
        """
        Given a state tensor [N, state_dim], returns action, its log prob, entropy, and value.
        """
        state = state.to(self.device)
        logits, value = self.model(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute GAE advantages and returns.

        rewards: [T, N]
        values:  [L, N]  (should be T+1, but may be off by 1)
        dones:   [T, N]
        next_value: [N]
        """
        T, N = rewards.shape

        # ensure values has shape [T+1, N]
        if values.shape[0] > T + 1:
            values = values[:T+1]
        elif values.shape[0] < T + 1:
            # append bootstrap value to make length T+1
            values = torch.cat([values, next_value.unsqueeze(0)], dim=0)

        advantages = torch.zeros_like(rewards).to(self.device)
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def train(self, batch: RolloutBatch):
        # get final-step value for bootstrap
        with torch.no_grad():
            next_value = batch.values[-1]

        # compute advantages & returns (both [T, N])
        advantages, returns = self.compute_gae(
            batch.rewards.to(self.device),
            batch.values.to(self.device),
            batch.dones.to(self.device),
            next_value.to(self.device)
        )

        # advantage normalization (insert exactly here!)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # flatten [T, N] -> [T*N]
        T, N = batch.rewards.shape
        states = batch.states[:-1].reshape(-1, batch.states.size(-1)).to(self.device)
        actions = batch.actions.reshape(-1).to(self.device)
        old_logp = batch.log_probs.reshape(-1).to(self.device)
        advs = advantages.reshape(-1)
        rets = returns.reshape(-1)

        # PPO epochs with minibatches (unchanged)
        for _ in range(self.epochs):
            idxs = torch.randperm(T * N, device=self.device)
            for start in range(0, T * N, self.batch_size):
                mb_idx = idxs[start:start + self.batch_size]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_advs = advs[mb_idx]
                mb_rets = rets[mb_idx]

                logits, values = self.model(mb_states)
                dist = Categorical(torch.softmax(logits, dim=-1))
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_logp - mb_old_logp).exp()
                s1 = ratio * mb_advs
                s2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advs
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = nn.functional.mse_loss(values, mb_rets)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
