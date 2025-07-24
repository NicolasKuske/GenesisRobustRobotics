import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import argparse

import torch
import genesis as gs
from torch.utils.tensorboard import SummaryWriter

from agents.ik.ppo_agent_multimodal_IKsimple import PPOAgentMultimodal
from envs.ik.reach_cube_ego_multimodal_stacked_IKsimple import ReachCubeEgoMultimodalStackedEnv


def train_ppo(args):
    env = ReachCubeEgoMultimodalStackedEnv(
        vis=args.vis,
        device=args.device,
        num_envs=args.num_envs,
        randomize_every=args.randomize_every
    )
    print("Created multimodal environment:", env)

    agent = PPOAgentMultimodal(
        obs_shape_vision=env.obs_shape_vision,
        obs_shape_audio=env.obs_shape_audio,
        action_shape=env.action_space,
        lr=1e-5,
        gamma=0.99,
        clip_epsilon=0.2,
        device=args.device,
        load=args.load,
        num_envs=args.num_envs,
        checkpoint_path=args.checkpoint
    )

    with SummaryWriter(log_dir=f"runs/{args.task}") as writer:
        for episode in range(args.max_episodes):
            state_v, state_a = env.reset()
            total_reward = torch.zeros(env.num_envs, device=args.device)
            done_array = torch.zeros(env.num_envs, dtype=torch.bool, device=args.device)

            # collect rollout
            states_v, states_a, actions, rewards, dones = [], [], [], [], []
            for _ in range(args.max_steps):
                action, _ = agent.select_action(state_v, state_a)
                (next_v, next_a), reward, done = env.step(action)

                states_v.append(state_v)
                states_a.append(state_a)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                state_v, state_a = next_v, next_a
                total_reward += reward
                done_array |= done
                if done_array.all():
                    break

            agent.train(states_v, states_a, actions, rewards, dones)

            if episode % args.save_every == 0:
                agent.save_checkpoint()
                print(f"[Episode {episode}] Checkpoint saved to {args.checkpoint}")

            mean_reward = total_reward.mean().item()
            writer.add_scalar('Reward/Mean', mean_reward, episode)
            print(f"[Episode {episode}] Mean Reward: {mean_reward:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-v', '--vis', action='store_true', help='Enable visualization')
    p.add_argument('-l', '--load', nargs='?', const='default',
                   help='Load checkpoint: -l for default, -l PATH for custom')
    p.add_argument('-n', '--num_envs', type=int, default=1, help='Number of envs')
    p.add_argument('-t', '--task', type=str,
                   default='ReachCubeEgoMultimodalStacked', help='Task name')
    p.add_argument('-d', '--device', type=str, default='cuda', help='cpu or cuda')
    p.add_argument('--randomize_every', type=int, default=100, help='Cube randomization interval')
    p.add_argument('--max_episodes', type=int, default=1_000_000, help='Max number of episodes')
    p.add_argument('--max_steps', type=int, default=200, help='Max steps per episode')
    p.add_argument('--save_every', type=int, default=5, help='Episodes between saves')
    return p.parse_args()


def main():
    args = parse_args()

    # unify device
    args.device = torch.device(args.device)

    # checkpoint resolution
    default = Path('logs') / f"{args.task}_ppo_checkpoint.pth"
    if args.load:
        args.checkpoint = default if args.load == 'default' else Path(args.load)
        if not args.checkpoint.is_file():
            print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
            sys.exit(1)
        print("Loading checkpoint from:", args.checkpoint)
    else:
        args.checkpoint = default
        print("No checkpoint provided; starting from scratch.")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    backend = gs.cpu if args.device.type == 'cpu' else gs.gpu
    gs.init(backend=backend)

    train_ppo(args)


if __name__ == '__main__':
    main()
