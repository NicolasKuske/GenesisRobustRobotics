# run_ppo_position_torque.py


import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'  # comment out for Windows or MacOS

import sys
import argparse
import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter
from algo.ppo_agent_torque import PPOAgentTorque, RolloutBatch
from env.reach_cube_torque import ReachCubeTorqueEnv

# ===== Params =====
HORIZON = 100
TOTAL_TIMESTEPS = 1_000_000_000
SAVE_INTERVAL = 5


def train_ppo(args):
    """
    Run PPO training loop with optional curriculum on the X-axis.
    """
    env = ReachCubeTorqueEnv(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"[INFO] Training environment: {env}")

    # Agent with explicit PPO hyperparameters
    agent = PPOAgentTorque(
        input_dim=env.state_dim,
        action_dim=env.action_space,
        hidden_dim=128,
        lr=1e-4,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        epochs=10,
        batch_size=64,
        value_coef=0.5,
        entropy_coef=0.001,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        load=args.load
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_torque_train")
    num_updates = TOTAL_TIMESTEPS // (HORIZON * args.num_envs)

    for update in range(1, num_updates + 1):
        state = env.reset()
        buffers = {k: [] for k in ['states', 'actions', 'logps', 'values', 'rewards', 'dones']}

        for _ in range(HORIZON):
            action, logp, _, value = agent.select_action(state)
            buffers['states'].append(state.to(agent.device))
            buffers['actions'].append(action)
            buffers['logps'].append(logp.detach())
            buffers['values'].append(value.detach())

            state, reward, done = env.step(action)
            buffers['rewards'].append(reward.to(agent.device))
            buffers['dones'].append(done.to(agent.device).float())

            if done.all():
                state = env.reset()

        # bootstrap last value
        buffers['states'].append(state.to(agent.device))
        with torch.no_grad():
            _, _, last_val = agent.model(state.to(agent.device))
        buffers['values'].append(last_val)

        batch = RolloutBatch(
            states=torch.stack(buffers['states']),
            actions=torch.stack(buffers['actions']),
            log_probs=torch.stack(buffers['logps']),
            values=torch.stack(buffers['values']),
            rewards=torch.stack(buffers['rewards']),
            dones=torch.stack(buffers['dones']),
        )

        agent.train(batch)
        mean_reward = batch.rewards.sum(dim=0).mean().item()
        writer.add_scalar('Reward/Mean', mean_reward, update)
        print(f"[Update {update}/{num_updates}] Avg Reward: {mean_reward:.3f}")

        if update % SAVE_INTERVAL == 0:
            agent.save_checkpoint()

        if getattr(env, 'completed', False):
            print("[INFO] Curriculum complete — saving and exiting.")
            agent.save_checkpoint()
            break

    writer.close()


def inference_ppo(args):
    """
    Run PPO inference loop, sampling full X-range.
    """
    # set episodes_per_position=1 for inference
    env = ReachCubeTorqueEnv(
        vis=args.vis,
        device=args.device,
        num_envs=args.num_envs,
        episodes_per_position=1
    )
    print(f"[INFO] Inference environment: {env}")

    # force full-range sampling: go to the final curriculum stage
    env.x_stage = env.max_stages
    lower = env.x_bounds[-1]
    upper = env.fixed_x
    print(f"[INFO] Inference X-range: [{lower:.2f}, {upper:.2f}]")

    # Load agent for inference (hyperparams from checkpoint)
    agent = PPOAgentTorque(
        input_dim=env.state_dim,
        action_dim=env.action_space,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        load=True
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_torque_inference")
    for ep in range(1, args.num_episodes + 1):
        state = env.reset()
        done_mask = torch.zeros(args.num_envs, device=args.device, dtype=torch.bool)
        steps = 0

        while steps < HORIZON and not done_mask.all():
            action, _, _, _ = agent.select_action(state)
            state, _, done = env.step(action)
            done_mask |= done
            steps += 1

        writer.add_scalar('Episode/Steps', steps, ep)
        print(f"[Inference {ep}/{args.num_episodes}] Steps: {steps}")

    writer.close()




def parse_args():
    p = argparse.ArgumentParser("PPO Torque Runner")
    p.add_argument('-v', '--vis', action='store_true', help='Enable visualization')
    p.add_argument('-l', '--load', action='store_true', help='Load checkpoint (default path)')
    p.add_argument('--load_path', type=str, default=None, help='Path to checkpoint (overrides default)')
    p.add_argument('-n', '--num_envs', type=int, default=1, help='Parallel environments')
    p.add_argument('-t', '--task', type=str, default='ReachCubeTorque', help='Task name')
    p.add_argument('-d', '--device', type=str, default='cuda', help='cpu or cuda')
    p.add_argument('-m', '--mode', choices=['train','inference'], default='train', help='Run mode')
    p.add_argument('--num_episodes', type=int, default=1000,
                   help='Episodes to run in inference')
    return p.parse_args()


def main():
    args = parse_args()

    # determine checkpoint path (always save)
    default_ckpt = f"logs/{args.task}_ppo_torque_checkpoint.pth"
    if args.load_path:
        args.load = True
        args.checkpoint_path = args.load_path
    else:
        args.checkpoint_path = default_ckpt

    if args.load:
        print(f"[INFO] Loading checkpoint from {args.checkpoint_path} \n")

    # ensure checkpoint directory exists
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    # if loading, verify checkpoint file exists
    if args.load and not os.path.isfile(args.checkpoint_path):
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint_path}")

    # init Genesis backend
    backend = gs.cpu if args.device.lower().startswith('cpu') else gs.gpu
    gs.init(backend=backend)

    # run
    if args.mode == 'train':
        train_ppo(args)
    else:
        inference_ppo(args)


if __name__ == '__main__':
    main()

