# File: agents/torque/ppo_agent_multimodal_torque.py

import torch
import torch.optim as optim
from torch.distributions import Normal
from networks.torque.ppo_multimodal_torque import PPOMultimodalTorque
from typing import NamedTuple, Optional


class RolloutBatch(NamedTuple):
    states_v: torch.Tensor   # [T+1, N, C_v, H, W]
    states_a: torch.Tensor   # [T+1, N, C_a, F, T]
    actions: torch.Tensor    # [T, N, action_dim]
    log_probs: torch.Tensor  # [T, N]
    values: torch.Tensor     # [T+1, N]
    rewards: torch.Tensor    # [T, N]
    dones: torch.Tensor      # [T, N]


class PPOAgentMultimodalTorque:
    def __init__(
        self,
        obs_shape_vision: tuple,
        obs_shape_audio: tuple,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = 'cuda',
        load: bool = False,
        checkpoint_path: Optional[str] = None
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.checkpoint_path = checkpoint_path

        # Instantiate multimodal policy with Gaussian continuous actions
        self.model = PPOMultimodalTorque(
            obs_shape_v=obs_shape_vision,
            obs_shape_a=obs_shape_audio,
            action_dim=action_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if load and checkpoint_path:
            self.load_checkpoint()

    def save_checkpoint(self):
        torch.save({'model_state_dict': self.model.state_dict()}, self.checkpoint_path)
        print(f"Checkpoint saved to {self.checkpoint_path}")

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {self.checkpoint_path}")

    def select_action(self, state_v, state_a):
        state_v = state_v.to(self.device)
        state_a = state_a.to(self.device)

        with torch.no_grad():
            mean, std, value = self.model(state_v, state_a)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value

    def compute_gae(self, rewards, values, dones):
        T, N = rewards.shape
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
        advantages, returns = self.compute_gae(batch.rewards, batch.values, batch.dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T, N = batch.rewards.shape

        states_v = batch.states_v[:-1].reshape(-1, *batch.states_v.shape[2:])
        states_a = batch.states_a[:-1].reshape(-1, *batch.states_a.shape[2:])
        actions = batch.actions.reshape(-1, batch.actions.shape[-1])
        old_logp = batch.log_probs.reshape(-1)
        advs = advantages.reshape(-1)
        rets = returns.reshape(-1)

        # The loop below must be indented here, inside the train() method!
        for _ in range(self.epochs):
            idxs = torch.randperm(T * N, device='cpu')  # Ensure CPU indices
            for start in range(0, T * N, self.batch_size):
                mb_idx = idxs[start:start+self.batch_size]

                mb_sv = states_v[mb_idx].to(self.device)
                mb_sa = states_a[mb_idx].to(self.device)
                mb_actions = actions[mb_idx].to(self.device)
                mb_old_logp = old_logp[mb_idx].to(self.device)
                mb_advs = advs[mb_idx].to(self.device)
                mb_rets = rets[mb_idx].to(self.device)

                mean, std, values = self.model(mb_sv, mb_sa)
                dist = Normal(mean, std)
                new_logp = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = (new_logp - mb_old_logp).exp()
                s1 = ratio * mb_advs
                s2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advs

                policy_loss = -torch.min(s1, s2).mean()
                value_loss = (values - mb_rets).pow(2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

