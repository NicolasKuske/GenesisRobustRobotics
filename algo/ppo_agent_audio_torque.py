# algo/ppo_agent_audio_torque.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import NamedTuple

from network.ppo_audio_torque import PPOAudioTorque

class RolloutBatch(NamedTuple):
    states:    torch.Tensor  # [T+1, N, C, F, T]
    actions:   torch.Tensor  # [T,   N, action_dim]
    log_probs: torch.Tensor  # [T,   N]
    values:    torch.Tensor  # [T+1, N]
    rewards:   torch.Tensor  # [T,   N]
    dones:     torch.Tensor  # [T,   N]

class PPOAgentAudioTorque:
    def __init__(
        self,
        obs_shape: tuple,      # (C, F, T)
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
        num_envs: int = 1,
        checkpoint_path: str = None,
    ):
        self.device         = torch.device(device)
        self.num_envs       = num_envs
        self.gamma          = gamma
        self.lam            = lam
        self.clip_epsilon   = clip_epsilon
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.value_coef     = value_coef
        self.entropy_coef   = entropy_coef
        self.checkpoint_path = checkpoint_path

        # actor–critic network
        self.model = PPOAudioTorque(obs_shape, action_dim).to(self.device)
        if load and checkpoint_path:
            self.load_checkpoint()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def save_checkpoint(self):
        torch.save({'model_state_dict': self.model.state_dict()}, self.checkpoint_path)
        print(f"[PPOAgentAudioTorque] Saved checkpoint to {self.checkpoint_path}")

    def load_checkpoint(self):
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        print(f"[PPOAgentAudioTorque] Loaded checkpoint from {self.checkpoint_path}")

    def select_action(self, state: torch.Tensor):
        """
        state: [N, C, F, T]
        returns:
          action:   [N, action_dim]
          log_prob: [N]
          entropy:  [N]
          value:    [N]
        """
        state = state.to(self.device)
        with torch.no_grad():
            mean, std, value = self.model(state)
        dist     = Normal(mean, std)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def compute_gae(self, rewards, values, dones, next_value):
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards, device=self.device)
        gae        = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            mask  = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            gae   = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def train(self, batch: RolloutBatch):
        # Compute GAE advantages and returns
        advantages, returns = self.compute_gae(
            batch.rewards.to(self.device),
            batch.values.to(self.device),
            batch.dones.to(self.device),
            batch.values[-1].to(self.device),
        )
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten tensors for batching (exclude the last bootstrap state)
        C, F, Tt = batch.states.shape[2:]
        states_flat  = batch.states[:-1].reshape(-1, C, F, Tt).to(self.device)
        action_dim   = batch.actions.shape[-1]
        actions_flat = batch.actions.reshape(-1, action_dim).to(self.device)
        oldlp_flat   = batch.log_probs.reshape(-1).to(self.device)
        returns_flat = returns.reshape(-1).to(self.device)
        adv_flat     = advantages.reshape(-1).to(self.device)

        # PPO update loop
        total_samples = states_flat.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, self.batch_size):
                idx    = perm[start:start + self.batch_size]
                s_batch = states_flat[idx]
                a_batch = actions_flat[idx]
                olp     = oldlp_flat[idx]
                adv     = adv_flat[idx]
                ret     = returns_flat[idx]

                mean, std, vals = self.model(s_batch)
                dist    = Normal(mean, std)
                logp    = dist.log_prob(a_batch).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = (logp - olp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = nn.functional.mse_loss(vals, ret)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
