# agents/ik/ppo_agent_multimodal_IKsimple.py


import torch
import torch.optim as optim
from torch.distributions import Categorical
from networks.ik.ppo_multimodal_IKsimple import PPOmultimodal

class PPOAgentMultimodal:
    """
    Minimal PPO-style agent (actor-only) using the multimodal policy network.
    """
    def __init__(
        self,
        obs_shape_vision,
        obs_shape_audio,
        action_shape,
        lr=1e-3,
        gamma=0.99,
        clip_epsilon=0.2,
        device='cpu',
        load=False,
        num_envs=1,
        checkpoint_path=None
    ):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.checkpoint_path = checkpoint_path

        # Instantiate multimodal policy without attention
        self.model = PPOmultimodal(
            obs_shape_v=obs_shape_vision,
            obs_shape_a=obs_shape_audio,
            action_shape=action_shape,
            freeze_unimodal=False,
            use_attention=False
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if load and checkpoint_path:
            self.load_checkpoint()

    def save_checkpoint(self):
        torch.save(
            {'model_state_dict': self.model.state_dict()},
            self.checkpoint_path
        )

    def load_checkpoint(self):
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def select_action(self, state_v, state_a):
        """
        Sample an action from the policy and return its log-prob.
        """
        state_v = state_v.to(self.device)
        state_a = state_a.to(self.device)

        logits = self.model(state_v, state_a)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def train(self, states_v, states_a, actions, rewards, dones):
    	# --- compute discounted returns ---
    	discounted = []
    	R = torch.zeros(self.num_envs, device=self.device)
    	for r, done in zip(reversed(rewards), reversed(dones)):
        	R = r.to(self.device) + self.gamma * R * (~done.to(self.device))
        	discounted.insert(0, R)
    	returns = torch.stack(discounted)  # (T, N)

    	# advantages: zero baseline (mean)
    	advantages = returns - returns.mean()

    	# flatten T and N dims
    	T, N = returns.shape
    	actions_flat = torch.stack(actions).view(-1).to(self.device)
    	adv_flat = advantages.view(-1).to(self.device)

    	# stack states for model input, ensure tensors on correct device
    	sv = torch.stack(states_v).view(T*N, *states_v[0].shape[1:]).to(self.device)
    	sa = torch.stack(states_a).view(T*N, *states_a[0].shape[1:]).to(self.device)

    	# old log probs
    	with torch.no_grad():
        	logits_old = self.model(sv, sa)
        	logp_old = Categorical(logits=logits_old).log_prob(actions_flat)

    	# PPO update
    	loss = None
    	for _ in range(4):  # epochs
        	logits_new = self.model(sv, sa)
        	logp_new = Categorical(logits=logits_new).log_prob(actions_flat)
        	ratio = (logp_new - logp_old).exp()
        	s1 = ratio * adv_flat
        	s2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * adv_flat
        	loss = -torch.min(s1, s2).mean()

        	self.optimizer.zero_grad()
        	loss.backward()
        	self.optimizer.step()

    	return loss.item()
