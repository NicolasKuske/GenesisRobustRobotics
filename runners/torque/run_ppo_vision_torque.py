# runners/torque/run_ppo_vision_IKsimple.py


import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'  # comment out for Windows or MacOS

import sys
import argparse
import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter

from agents.torque.ppo_agent_vision_torque import PPOAgentVisionTorque, RolloutBatch
from envs.torque.reach_cube_vision_torque import ReachCubeVisionTorqueEnv
from envs.torque.reach_cube_ego_vision_stacked_torque import ReachCubeEgoVisionStackedTorqueEnv

# Map task names to environment classes
task_to_env = {
    'ReachCubeVisionTorque': ReachCubeVisionTorqueEnv,
    'ReachCubeEgoVisionStackedTorque': ReachCubeEgoVisionStackedTorqueEnv,
}

# Universal PPO parameters
HORIZON = 100
TOTAL_TIMESTEPS = 100_000_000
SAVE_INTERVAL = 5


def create_environment(task_name):
    if task_name in task_to_env:
        return task_to_env[task_name]
    raise ValueError(f"Task '{task_name}' is not recognized.")


def train_ppo(args):
    # Build environment
    env_cls = create_environment(args.task)
    device = torch.device(args.device)
    env = env_cls(vis=args.vis, device=device, num_envs=args.num_envs)
    print(f"[INFO] Training environment: {env}")

    # Build agent
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
        device          = device,
        checkpoint_path = args.checkpoint_path,
        load            = args.load,
        num_envs        = args.num_envs,
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_train")

    num_updates = TOTAL_TIMESTEPS // (HORIZON * args.num_envs)

    for update in range(1, num_updates + 1):
        state = env.reset().to(device)
        buffers = {k: [] for k in ['states', 'actions', 'logps', 'values', 'rewards', 'dones']}

        for t in range(HORIZON):
            action, logp, _, value = agent.select_action(state)
            buffers['states'].append(state)
            buffers['actions'].append(action)
            buffers['logps'].append(logp.detach())
            buffers['values'].append(value.detach())

            state, reward, done = env.step(action)
            state = state.to(device)

            buffers['rewards'].append(reward.to(device))
            buffers['dones'].append(done.to(device).float())

            if done.all():
                state = env.reset().to(device)

        # Bootstrap value
        with torch.no_grad():
            _, _, last_val = agent.model(state)
        buffers['values'].append(last_val)
        # Record final state
        buffers['states'].append(state)

        # Pack into RolloutBatch and train
        batch = RolloutBatch(
            states=torch.stack(buffers['states']),
            actions=torch.stack(buffers['actions']),
            log_probs=torch.stack(buffers['logps']),
            values=torch.stack(buffers['values']),
            rewards=torch.stack(buffers['rewards']),
            dones=torch.stack(buffers['dones']),
        )
        agent.train(batch)

        # Logging & checkpoint
        mean_reward = batch.rewards.sum(dim=0).mean().item()
        writer.add_scalar('Reward/Mean', mean_reward, update)
        if update % SAVE_INTERVAL == 0:
            agent.save_checkpoint()
        print(f"[Update {update}/{num_updates}] Mean Reward: {mean_reward:.3f}")

    writer.close()


def inference_ppo(args):
    # Build environment for inference
    env_cls = create_environment(args.task)
    device = torch.device(args.device)
    env = env_cls(vis=args.vis, device=device, num_envs=args.num_envs, episodes_per_position=1)
    print(f"[INFO] Inference environment: {env}")

    # Force full-range sampling
    env.x_stage = env.max_stages
    lower = env.x_bounds[-1]
    upper = env.fixed_x
    print(f"[INFO] Inference X-range: [{lower:.2f}, {upper:.2f}]")

    # Load agent
    agent = PPOAgentVisionTorque(
        obs_shape       = env.obs_shape,
        action_dim      = env.action_space,
        device          = device,
        checkpoint_path = args.checkpoint_path,
        load            = True,
        num_envs        = args.num_envs,
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_inference")
    for ep in range(1, args.num_episodes + 1):
        state = env.reset().to(device)
        done_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        steps = 0

        while steps < HORIZON and not done_mask.all():
            action, _, _, _ = agent.select_action(state)
            state, _, done = env.step(action)
            state = state.to(device)
            done_mask |= done.to(device)
            steps += 1

        writer.add_scalar('Episode/Steps', steps, ep)
        print(f"[Inference {ep}/{args.num_episodes}] Steps: {steps}")

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

    # Determine checkpoint path
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
            sys.exit("[ERROR] Inference mode requires a checkpoint. Use -l to specify one.")
        print("[INFO] No checkpoint; training from scratch.")

    # Initialize Genesis backend
    backend = gs.cpu if args.device.lower().startswith("cpu") else gs.gpu
    gs.init(backend=backend)

    # Launch
    if args.mode == 'train':
        train_ppo(args)
    else:
        inference_ppo(args)


if __name__ == '__main__':
    main()
