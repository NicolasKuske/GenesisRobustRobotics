# agents/ik/ppo_agent_audio_IKsimple.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from networks.ik.ppo_audio_IKsimple import PPOaudioJoints


class PPOAgentAudio:
    def __init__(
            self,
            obs_shape,  # audio observation shape (C, F, T)
            joint_dim,  # dimension of joint state vector
            action_shape,  # number of discrete actions
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

        self.model = PPOaudioJoints(obs_shape, joint_dim, action_shape).to(self.device)

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

    def select_action(self, audio_state, joint_state, inference=False):
        audio_state = audio_state.to(self.device)
        joint_state = joint_state.to(self.device)

        with torch.no_grad():
            logits = self.model(audio_state, joint_state)

        probs = nn.functional.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if inference:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        return action

    def train(self, audio_states, joint_states, actions, rewards, dones):
        audio_states_tensor = torch.stack(audio_states).to(self.device)
        joint_states_tensor = torch.stack(joint_states).to(self.device)
        actions_tensor = torch.stack(actions).to(self.device)

        rewards_list = rewards
        dones_list = dones

        discounted_rewards = []
        R = torch.zeros(self.num_envs, device=self.device)
        for reward, done in zip(reversed(rewards_list), reversed(dones_list)):
            R = reward.to(self.device) + self.gamma * R * (~done.to(self.device))
            discounted_rewards.insert(0, R)
        discounted_tensor = torch.stack(discounted_rewards)

        advantages = discounted_tensor - discounted_tensor.mean()

        T, N = audio_states_tensor.shape[:2]
        C, H, W = audio_states_tensor.shape[2:]

        audio_states_flat = audio_states_tensor.reshape(T * N, C, H, W)
        joint_states_flat = joint_states_tensor.reshape(T * N, -1)
        actions_flat = actions_tensor.view(-1)
        advantages_flat = advantages.view(-1)

        with torch.no_grad():
            logits_old = self.model(audio_states_flat, joint_states_flat)
            probs_old = nn.functional.softmax(logits_old, dim=-1)

        for _ in range(10):
            logits_new = self.model(audio_states_flat, joint_states_flat)
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
