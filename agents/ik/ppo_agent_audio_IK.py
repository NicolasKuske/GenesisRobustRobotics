import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import NamedTuple, Optional

class RolloutBatch(NamedTuple):
    states: torch.Tensor    # [T+1, N, C, F, Tspec]
    actions: torch.Tensor   # [T,   N]
    log_probs: torch.Tensor # [T,   N]
    values: torch.Tensor    # [T+1, N]
    rewards: torch.Tensor   # [T,   N]
    dones: torch.Tensor     # [T,   N]

class PPOAgentAudio:
    def __init__(
        self,
        obs_shape,
        action_shape,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        epochs=10,
        batch_size=64,
        value_coef=0.5,
        entropy_coef=0.01,
        device='cpu',
        checkpoint_path: Optional[str] = None,
        load: bool = False,
        num_envs: int = 1,
    ):
        from networks.ik.ppo_audio_IK import PPOaudio
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.num_envs = num_envs

        self.model = PPOaudio(obs_shape, action_shape).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.checkpoint_path = checkpoint_path
        if load and checkpoint_path:
            self.load_checkpoint()

    def save_checkpoint(self):
        if self.checkpoint_path:
            torch.save({'model_state_dict': self.model.state_dict()}, self.checkpoint_path)
            print(f"Saved checkpoint to {self.checkpoint_path}")

    def load_checkpoint(self):
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint from {self.checkpoint_path}")

    @torch.no_grad()
    def select_action(self, state: torch.Tensor):
        """
        Stochastic action for TRAINING.
        Returns: action, log_prob, entropy, value
        """
        logits, value = self.model(state.to(self.device))
        dist = Categorical(logits=logits)  # numerically stable (no manual softmax)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False):
        """
        Inference helper: returns ONLY the action.
        - deterministic=True -> argmax
        - deterministic=False -> sample
        """
        logits, _ = self.model(state.to(self.device))
        if deterministic:
            return logits.argmax(dim=-1)
        return Categorical(logits=logits).sample()

    def eval_mode(self, enabled: bool = True):
        """Put the model into eval() or train() mode."""
        if enabled:
            self.model.eval()
        else:
            self.model.train()

    def _compute_gae(self, rewards, values, dones, next_value):
        T, N = rewards.shape
        if values.shape[0] < T + 1:
            values = torch.cat([values, next_value.unsqueeze(0)], dim=0)

        advantages = torch.zeros_like(rewards, device=self.device)
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def train(self, batch: RolloutBatch):
        with torch.no_grad():
            next_value = batch.values[-1].to(self.device)

        advantages, returns = self._compute_gae(
            batch.rewards.to(self.device),
            batch.values.to(self.device),
            batch.dones.to(self.device),
            next_value
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T, N = batch.rewards.shape
        # flatten rollout
        states = batch.states[:-1].reshape(T * N, *batch.states.size()[2:]).to(self.device)
        actions = batch.actions.reshape(-1).to(self.device)
        old_logp = batch.log_probs.reshape(-1).to(self.device)
        advs = advantages.reshape(-1)
        rets = returns.reshape(-1)

        for _ in range(self.epochs):
            idxs = torch.randperm(T * N, device=self.device)
            for start in range(0, T * N, self.batch_size):
                mb = idxs[start:start + self.batch_size]
                logits, values = self.model(states[mb])
                dist = Categorical(torch.softmax(logits, dim=-1))
                new_logp = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                ratio = (new_logp - old_logp[mb]).exp()
                s1 = ratio * advs[mb]
                s2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advs[mb]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = nn.functional.mse_loss(values, rets[mb])

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
