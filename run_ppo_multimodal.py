import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import argparse
import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter

from algo.ppo_agent_multimodal import PPOAgentMultimodal
from env import *

def train_ppo(args):
    env = ReachCubeEgoMultimodalStackedEnv(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"[INFO] Created multimodal environment: {env}")

    agent = PPOAgentMultimodal(
        obs_shape_vision=env.obs_shape_vision,
        obs_shape_audio=env.obs_shape_audio,
        output_dim=env.action_space,
        device=args.device,
        lr=1e-3,
        gamma=0.99,
        clip_epsilon=0.2,
        load=args.load,
        num_envs=args.num_envs,
        checkpoint_path=args.checkpoint_path
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}")

    for episode in range(1000000):
        # Unpack reset correctly
        (state_vision, state_audio) = env.reset()
        total_reward = torch.zeros(env.num_envs, device=args.device)
        done_array = torch.zeros(env.num_envs, dtype=torch.bool, device=args.device)

        states_v, states_a, actions, rewards, dones = [], [], [], [], []

        for step in range(200):
            action = agent.select_action(state_vision, state_audio)
            # Unpack step correctly
            (next_state_v, next_state_a), reward, done = env.step(action)

            states_v.append(state_vision)
            states_a.append(state_audio)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state_vision, state_audio = next_state_v, next_state_a
            total_reward += reward
            done_array |= done
            if done_array.all():
                break

        agent.train(states_v, states_a, actions, rewards, dones)

        if episode % 5 == 0:
            agent.save_checkpoint()

        mean_reward = total_reward.mean().item()
        writer.add_scalar('Reward/Mean', mean_reward, episode)
        print(f"[Episode {episode}] Mean Reward: {mean_reward}")

    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Enable visualization")
    parser.add_argument("-n", "--num_envs", type=int, default=1)
    parser.add_argument("-t", "--task", type=str, default="ReachCubeEgoMultimodal")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-l", "--load", action="store_true")
    args = parser.parse_args()

    args.checkpoint_path = f"logs/{args.task}_ppo_checkpoint.pth"
    os.makedirs("logs", exist_ok=True)

    backend = gs.cpu if args.device.startswith("cpu") else gs.gpu
    gs.init(backend=backend)

    train_ppo(args)

if __name__ == "__main__":
    main()
