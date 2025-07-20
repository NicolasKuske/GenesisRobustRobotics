import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import NamedTuple

from network.ppo_vision import PPOvision  # your updated conv-based PPO class

class RolloutBatch(NamedTuple):
    states:    torch.Tensor   # [T+1, N, C, H, W]
    actions:   torch.Tensor   # [T,   N]
    log_probs: torch.Tensor   # [T,   N]
    values:    torch.Tensor   # [T+1, N]
    rewards:   torch.Tensor   # [T,   N]
    dones:     torch.Tensor   # [T,   N]

class PPOAgentVision:
    def __init__(
        self,
        obs_shape,            # tuple, e.g. (3, 120, 120)
        output_dim,           # number of discrete actions
        lr: float,
        gamma: float,
        lam: float,
        clip_epsilon: float,
        epochs: int,
        batch_size: int,
        value_coef: float,
        entropy_coef: float,
        device: str,
        load: bool = False,
        num_envs: int = 1,
        checkpoint_path: str = None,
    ):
        self.device        = torch.device(device)
        self.num_envs      = num_envs
        self.gamma         = gamma
        self.lam           = lam
        self.clip_epsilon  = clip_epsilon
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef
        self.checkpoint_path = checkpoint_path

        # actor–critic conv network
        self.model = PPOvision(obs_shape, output_dim).to(self.device)

        if load and checkpoint_path:
            self.load_checkpoint()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def save_checkpoint(self):
        torch.save({'model_state_dict': self.model.state_dict()},
                   self.checkpoint_path)
        print(f"[PPOAgentVision] Saved checkpoint to {self.checkpoint_path}")

    def load_checkpoint(self):
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        print(f"[PPOAgentVision] Loaded checkpoint from {self.checkpoint_path}")

    def select_action(self, state: torch.Tensor):
        """
        state: [N, C, H, W]
        returns:
          action:    [N]
          log_prob:  [N]
          entropy:   [N]
          value:     [N]
        """
        state = state.to(self.device)
        with torch.no_grad():
            logits, value = self.model(state)
        probs    = torch.softmax(logits, dim=-1)
        dist     = Categorical(probs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value

    def compute_gae(self, rewards, values, dones, next_value):
        """
        rewards: [T, N], values: [T+1, N], dones: [T, N], next_value: [N]
        returns:
          advantages: [T, N]
          returns:    [T, N]
        """
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = torch.zeros(N, device=self.device)

        for t in reversed(range(T)):
            mask  = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            gae   = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def train(self, batch: RolloutBatch):
        # compute advantages & returns
        advantages, returns = self.compute_gae(
            batch.rewards.to(self.device),
            batch.values.to(self.device),
            batch.dones.to(self.device),
            batch.values[-1].to(self.device),
        )

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # flatten [T, N] → [T*N]
        T, N = batch.rewards.shape
        states_flat    = batch.states[:-1].reshape(-1, *batch.states.shape[2:]).to(self.device)
        actions_flat   = batch.actions.reshape(-1).to(self.device)
        old_logp_flat  = batch.log_probs.reshape(-1).to(self.device)
        returns_flat   = returns.reshape(-1).to(self.device)
        advs_flat      = advantages.reshape(-1).to(self.device)

        # PPO update loop
        for _ in range(self.epochs):
            perm = torch.randperm(T * N, device=self.device)
            for start in range(0, T * N, self.batch_size):
                idx      = perm[start:start + self.batch_size]
                s_batch  = states_flat[idx]
                a_batch  = actions_flat[idx]
                ol_batch = old_logp_flat[idx]
                adv_batch= advs_flat[idx]
                ret_batch= returns_flat[idx]

                logits, values = self.model(s_batch)
                dist    = Categorical(torch.softmax(logits, dim=-1))
                logp    = dist.log_prob(a_batch)
                entropy = dist.entropy().mean()

                ratio = (logp - ol_batch).exp()
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio,
                                   1.0 - self.clip_epsilon,
                                   1.0 + self.clip_epsilon) * adv_batch

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = nn.functional.mse_loss(values, ret_batch)

                loss = policy_loss + self.value_coef * value_loss \
                       - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
