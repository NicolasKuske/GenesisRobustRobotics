# File: runners/torque/run_ppo_multimodal_torque.py

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import torch
import genesis as gs
from torch.utils.tensorboard import SummaryWriter
from agents.torque.ppo_agent_multimodal_torque import PPOAgentMultimodalTorque, RolloutBatch
from envs.torque.reach_cube_ego_multimodal_stacked_torque import ReachCubeEgoMultimodalStackedTorqueEnv

# PPO parameters
HORIZON = 100
TOTAL_TIMESTEPS = 100_000_000
SAVE_INTERVAL = 5


def train_ppo(args):
    env = ReachCubeEgoMultimodalStackedTorqueEnv(
        vis=args.vis,
        device=args.device,
        num_envs=args.num_envs
    )
    print("[INFO] Training multimodal torque environment initialized.")

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
        checkpoint_path=args.checkpoint_path,
        load=args.load
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_train")
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


def inference_ppo(args):
    # Setup inference environment explicitly
    env = ReachCubeEgoMultimodalStackedTorqueEnv(
        vis=args.vis,
        device=args.device,
        num_envs=args.num_envs,
        episodes_per_position=1
    )
    print(f"[INFO] Inference multimodal torque environment initialized.")

    # Force full-range cube placement for inference
    env.x_stage = env.max_stages
    lower = env.x_bounds[-1]
    upper = env.fixed_x
    print(f"[INFO] Inference X-range forced: [{lower:.2f}, {upper:.2f}]")

    agent = PPOAgentMultimodalTorque(
        obs_shape_vision=env.obs_shape_vision,
        obs_shape_audio=env.obs_shape_audio,
        action_dim=env.action_space,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        load=True
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_inference")
    for ep in range(1, args.num_episodes + 1):
        state_v, state_a = env.reset()
        done_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        steps = 0

        while steps < HORIZON and not done_mask.all():
            action, _, _, _ = agent.select_action(state_v, state_a)
            (state_v, state_a), _, done = env.step(action)
            done_mask |= done.to(args.device)
            steps += 1

        writer.add_scalar('Episode/Steps', steps, ep)
        print(f"[Inference {ep}/{args.num_episodes}] Steps: {steps}")

    writer.close()


def parse_args():
    p = argparse.ArgumentParser("PPO Multimodal Torque Runner")
    p.add_argument('-v', '--vis', action='store_true', help='Enable visualization')
    p.add_argument('-l', '--load', action='store_true', help='Load checkpoint')
    p.add_argument('--load_path', type=str, default=None, help='Checkpoint path')
    p.add_argument('-n', '--num_envs', type=int, default=1, help='Parallel environments')
    p.add_argument('-t', '--task', type=str, default='ReachCubeEgoMultimodalTorque', help='Task name')
    p.add_argument('-d', '--device', type=str, default='cuda', help='Device: cpu or cuda')
    p.add_argument('-m', '--mode', choices=['train', 'inference'], default='train', help='Mode: train or inference')
    p.add_argument('--num_episodes', type=int, default=100, help='Number of inference episodes')
    return p.parse_args()


def main():
    args = parse_args()
    args.device = torch.device(args.device)

    # Set checkpoint path
    default_ckpt = f"{args.task}_ppo_checkpoint.pth"
    if args.load_path:
        args.load = True
        args.checkpoint_path = args.load_path
    else:
        args.checkpoint_path = os.path.join("logs", default_ckpt)

    if args.load:
        print(f"[INFO] Loading checkpoint from {args.checkpoint_path}")
        if not os.path.isfile(args.checkpoint_path):
            sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint_path}")
    else:
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
        if args.mode == 'inference':
            sys.exit("[ERROR] Inference mode requires a checkpoint. Use -l or --load_path to specify one.")
        print("[INFO] No checkpoint; training from scratch.")

    backend = gs.cpu if args.device.type == 'cpu' else gs.gpu
    gs.init(backend=backend)

    if args.mode == 'train':
        train_ppo(args)
    else:
        inference_ppo(args)


if __name__ == '__main__':
    main()
