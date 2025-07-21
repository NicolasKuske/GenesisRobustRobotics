# run_ppo_vision_torque.py


import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'  # comment out for Windows or MacOS

import sys
import argparse
import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter

from algo.ppo_agent_vision_torque import PPOAgentVisionTorque, RolloutBatch
from env.reach_cube_vision_torque import ReachCubeVisionTorqueEnv

# ===== Params =====
HORIZON = 100
TOTAL_TIMESTEPS = 1_000_000
SAVE_INTERVAL = 5


def train_ppo(args):
    """
    Run PPO training loop for vision+torque task.
    """
    env = ReachCubeVisionTorqueEnv(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"[INFO] Training environment: {env}")

    # Agent with PPO hyperparameters
    agent = PPOAgentVisionTorque(
        obs_shape       = env.obs_shape,
        action_dim      = env.action_space,
        lr              = 3e-4,
        gamma           = 0.99,
        lam             = 0.95,
        clip_epsilon    = 0.2,
        epochs          = 10,
        batch_size      = 64,
        value_coef      = 0.5,
        entropy_coef    = 0.01,
        device          = args.device,
        checkpoint_path = args.checkpoint_path,
        load            = args.load,
        num_envs        = args.num_envs,
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_train")
    num_updates = TOTAL_TIMESTEPS // (HORIZON * args.num_envs)

    for update in range(1, num_updates + 1):
        state = env.reset()
        buffers = {k: [] for k in ['states', 'actions', 'logps', 'values', 'rewards', 'dones']}

        for t in range(HORIZON):
            action, logp, _, value = agent.select_action(state)
            buffers['states'].append(state.to(args.device))
            buffers['actions'].append(action)
            buffers['logps'].append(logp.detach())
            buffers['values'].append(value.detach())

            state, reward, done = env.step(action)
            buffers['rewards'].append(reward.to(args.device))
            buffers['dones'].append(done.to(args.device).float())

            if done.all():
                state = env.reset()

        # bootstrap last value
        buffers['states'].append(state.to(args.device))
        with torch.no_grad():
            _, _, last_val = agent.model(state)
        buffers['values'].append(last_val)

        # pack and train
        batch = RolloutBatch(
            states=torch.stack(buffers['states']),
            actions=torch.stack(buffers['actions']),
            log_probs=torch.stack(buffers['logps']),
            values=torch.stack(buffers['values']),
            rewards=torch.stack(buffers['rewards']),
            dones=torch.stack(buffers['dones']),
        )
        agent.train(batch)

        # logging & checkpoint
        mean_reward = batch.rewards.sum(dim=0).mean().item()
        writer.add_scalar('Reward/Mean', mean_reward, update)
        if update % SAVE_INTERVAL == 0:
            agent.save_checkpoint()
        print(f"[Update {update}/{num_updates}] Mean Reward: {mean_reward:.3f}")

    writer.close()


def inference_ppo(args):
    """
    Run PPO inference loop for vision+torque task.
    """
    env = ReachCubeVisionTorqueEnv(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"[INFO] Inference environment: {env}")

    # Load agent (hyperparameters loaded from checkpoint)
    agent = PPOAgentVisionTorque(
        obs_shape       = env.obs_shape,
        action_dim      = env.action_space,
        device          = args.device,
        checkpoint_path = args.checkpoint_path,
        load            = True,
        num_envs        = args.num_envs,
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_inference")
    num_episodes = args.num_episodes

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        steps = 0

        while steps < HORIZON and not done_mask.all():
            action, _, _, _ = agent.select_action(state)
            state, _, done = env.step(action)
            done_mask |= done
            steps += 1

        writer.add_scalar('Episode/Steps', steps, ep)
        print(f"[Inference {ep}/{num_episodes}] Steps: {steps}")

    writer.close()


def parse_args():
    p = argparse.ArgumentParser("PPO Vision+Torque Runner")
    p.add_argument('-v', '--vis', action='store_true', help='Enable visualization')
    p.add_argument('-l', '--load', action='store_true', help='Load checkpoint (default path)')
    p.add_argument('--load_path', type=str, default=None, help='Path to checkpoint (overrides default)')
    p.add_argument('-n', '--num_envs', type=int, default=1, help='Parallel environments')
    p.add_argument('-t', '--task', type=str, default='ReachCubeVisionTorque', help='Task name')
    p.add_argument('-d', '--device', type=str, default='cuda', help='cpu or cuda')
    p.add_argument('-m', '--mode', choices=['train', 'inference'], default='train', help='Run mode')
    p.add_argument('--num_episodes', type=int, default=100, help='Episodes to run in inference')
    return p.parse_args()


def main():
    args = parse_args()

    # checkpoint logic
    default_ckpt = f"{args.task}_ppo_checkpoint.pth"
    if args.load_path:
        args.load = True
        args.checkpoint_path = args.load_path
    else:
        args.checkpoint_path = os.path.join("logs", default_ckpt)

    if args.load:
        print(f"[INFO] Loading checkpoint from {args.checkpoint_path}\n")
        if not os.path.isfile(args.checkpoint_path):
            sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint_path}")
    else:
        # ensure save directory
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
        if args.mode == 'inference':
            sys.exit("[ERROR] Inference mode requires a checkpoint. Use -l to specify one.")
        else:
            print("[INFO] No checkpoint; training from scratch.")

    # initialize Genesis backend
    backend = gs.cpu if args.device.lower().startswith("cpu") else gs.gpu
    gs.init(backend=backend)

    if args.mode == 'train':
        train_ppo(args)
    else:
        inference_ppo(args)


if __name__ == '__main__':
    main()

