# File: runners/torque/run_ppo_multimodal_torque.py

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import sys
from pathlib import Path
# Adds the root directory (two levels up from this file) to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import torch
import genesis as gs
from torch.utils.tensorboard import SummaryWriter
from agents.torque.ppo_agent_multimodal_torque import PPOAgentMultimodalTorque, RolloutBatch
from envs.torque.reach_cube_ego_multimodal_stacked_torque import ReachCubeEgoMultimodalStackedTorqueEnv

# PPO parameters
HORIZON = 200
TOTAL_TIMESTEPS = 100_000_000
SAVE_INTERVAL = 5


def train_ppo(args):
    env = ReachCubeEgoMultimodalStackedTorqueEnv(
        vis=args.vis,
        device=args.device,
        num_envs=args.num_envs
    )
    print("Multimodal torque environment initialized.")

    agent = PPOAgentMultimodalTorque(
        obs_shape_vision=env.obs_shape_vision,
        obs_shape_audio=env.obs_shape_audio,
        action_dim=env.action_space,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        epochs=10,
        batch_size=64,
        device=args.device,
        checkpoint_path=args.checkpoint,
        load=args.load
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}")
    num_updates = TOTAL_TIMESTEPS // (HORIZON * args.num_envs)

    for update in range(1, num_updates + 1):
        state_v, state_a = env.reset()
        buffers = {k: [] for k in ['states_v', 'states_a', 'actions', 'logps', 'values', 'rewards', 'dones']}

        for _ in range(HORIZON):
            action, logp, _, value = agent.select_action(state_v, state_a)
            buffers['states_v'].append(state_v)
            buffers['states_a'].append(state_a)
            buffers['actions'].append(action)
            buffers['logps'].append(logp)
            buffers['values'].append(value)

            (state_v, state_a), reward, done = env.step(action)

            buffers['rewards'].append(reward)
            buffers['dones'].append(done.float())

            if done.all():
                state_v, state_a = env.reset()

        # bootstrap value
        with torch.no_grad():
            _, _, last_val = agent.model(state_v.to(agent.device), state_a.to(agent.device))
        buffers['values'].append(last_val)
        buffers['states_v'].append(state_v)
        buffers['states_a'].append(state_a)

        batch = RolloutBatch(
            states_v=torch.stack(buffers['states_v']),
            states_a=torch.stack(buffers['states_a']),
            actions=torch.stack(buffers['actions']),
            log_probs=torch.stack(buffers['logps']),
            values=torch.stack(buffers['values']),
            rewards=torch.stack(buffers['rewards']),
            dones=torch.stack(buffers['dones']),
        )

        agent.train(batch)
        mean_reward = batch.rewards.sum(dim=0).mean().item()
        writer.add_scalar('Reward/Mean', mean_reward, update)

        if update % SAVE_INTERVAL == 0:
            agent.save_checkpoint()

        print(f"[Update {update}/{num_updates}] Mean Reward: {mean_reward:.3f}")

    writer.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-v', '--vis', action='store_true', help='Enable visualization')
    p.add_argument('-l', '--load', action='store_true', help='Load checkpoint')
    p.add_argument('-n', '--num_envs', type=int, default=1, help='Parallel envs')
    p.add_argument('-t', '--task', type=str, default='ReachCubeEgoMultimodalTorque', help='Task name')
    p.add_argument('-d', '--device', type=str, default='cuda', help='Device: cpu/cuda')
    return p.parse_args()


def main():
    args = parse_args()
    args.device = torch.device(args.device)

    # Checkpoint handling
    default_checkpoint = Path("logs") / f"{args.task}_ppo_checkpoint.pth"
    args.checkpoint = default_checkpoint

    if args.load:
        if not args.checkpoint.exists():
            print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading checkpoint: {args.checkpoint}")
    else:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        print("Starting from scratch, no checkpoint loaded.")

    backend = gs.cpu if args.device.type == 'cpu' else gs.gpu
    gs.init(backend=backend)

    train_ppo(args)


if __name__ == '__main__':
    main()
