#run_ppo_position.py


import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'  # comment out for Windows or MacOS

import sys
import argparse
import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter

from algo.ppo_agent_position import PPOAgentPosition, RolloutBatch
from env import *

# map task names to env classes
task_to_class = {
    'ReachCubePosition': ReachCubePositionEnv,
    'ReachCubePositionStacked': ReachCubePositionStackedEnv,
    'ReachFixedCubeDirectJointControl': ReachFixedCubeDirectJointControlEnv
}

def create_environment(task_name):
    if task_name in task_to_class:
        return task_to_class[task_name]
    raise ValueError(f"\n Task '{task_name}' is not recognized.\n ")

def train_ppo(args):
    env_cls = create_environment(args.task)
    env = env_cls(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"\n [INFO] Created environment: {env} \n")

    agent = PPOAgentPosition(
        input_dim=env.state_dim,
        action_dim=env.action_space,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_epsilon=args.clip_epsilon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        load=args.load
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_train")

    total_steps = args.total_timesteps
    T = args.horizon
    N = args.num_envs
    num_updates = total_steps // (T * N)

    for update in range(1, num_updates + 1):
        state = env.reset()
        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

        # collect rollout
        for t in range(T):
            action, logp, entropy, value = agent.select_action(state)
            states.append(state.to(agent.device))
            actions.append(action)
            log_probs.append(logp.detach())
            values.append(value.detach())

            next_state, reward, done = env.step(action)
            rewards.append(reward.to(agent.device))
            dones.append(done.to(agent.device).float())

            # if all envs done, reset
            if done.all():
                next_state = env.reset()
            state = next_state

        # append last state/value for bootstrap
        states.append(state.to(agent.device))
        with torch.no_grad():
            _, last_value = agent.model(state.to(agent.device))
        values.append(last_value)

        # stack tensors
        batch = RolloutBatch(
            states=torch.stack(states),           # [T+1, N, state_dim]
            actions=torch.stack(actions),         # [T,   N]
            log_probs=torch.stack(log_probs),     # [T,   N]
            values=torch.stack(values),           # [T+1, N]
            rewards=torch.stack(rewards),         # [T,   N]
            dones=torch.stack(dones)              # [T,   N]
        )

        # update policy
        agent.train(batch)

        # log & save
        ep_reward = batch.rewards.sum(dim=0).mean().item()
        writer.add_scalar('Reward/Mean', ep_reward, update)
        print(f"[Update {update}/{num_updates}] Avg Reward per Env: {ep_reward:.3f}")

        # periodic checkpoint
        if update % args.save_interval == 0:
            agent.save_checkpoint()

        # curriculum completion handled by env.completed flag
        if hasattr(env, 'completed') and env.completed:
            print("[INFO] Curriculum complete—saving final checkpoint and ending training.")
            agent.save_checkpoint()
            break

    writer.close()


def inference_ppo(args):
    env_cls = create_environment(args.task)
    env     = env_cls(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"\n [INFO] Created environment: {env} \n")

    # ——— force full x-range sampling in inference ———
    if hasattr(env, "dynamic_x"):
        env.dynamic_x = True
        full_min = env.fixed_x - env.max_stages * 0.2
        full_max = env.fixed_x
        env.min_x_dynamic = full_min
        env.max_x_dynamic = full_max
        print(f"[INFO] Inference: sampling x ∈ [{full_min:.2f}, {full_max:.2f}]")

    agent = PPOAgentPosition(
        input_dim=env.state_dim,
        action_dim=env.action_space,
        hidden_dim=args.hidden_dim,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        load=True
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_inference")

    num_episodes = args.num_episodes or 100
    for ep in range(1, num_episodes + 1):
        state      = env.reset()
        step_count = 0
        done_array = torch.zeros(env.num_envs, dtype=torch.bool, device=args.device)

        for _ in range(args.horizon):
            action = agent.select_action(state)[0]
            state, _, done = env.step(action)
            done_array |= done
            step_count += 1
            if done_array.all():
                break

        writer.add_scalar('Episode/Steps', step_count, ep)
        print(f"[Inference {ep}/{num_episodes}] Steps until done: {step_count}")

    writer.close()


def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-v', '--vis', action='store_true', help='Enable visualization')
    p.add_argument('-l', '--load_path', nargs='?', const='default', default=None,
                   help='`-l` loads default ckpt; `-l path.pth` loads that file')
    p.add_argument('-n', '--num_envs', type=int, default=1, help='Number of parallel envs')
    p.add_argument('-t', '--task', type=str, default='ReachCubePosition', help='Task')
    p.add_argument('-d', '--device', type=str, default='cuda', help='cpu, cuda[:X], or mps')

    # PPO hyperparams
    p.add_argument('--horizon', type=int, default=100, help='Rollout horizon (T)')
    p.add_argument('--total_timesteps', type=int, default=1000000,
                   help='Total env timesteps (train only)')
    p.add_argument('--epochs', type=int, default=10, help='PPO epochs per update')
    p.add_argument('--batch_size', type=int, default=64, help='Minibatch size')
    p.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer size')
    p.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    p.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    p.add_argument('--lam', type=float, default=0.95, help='GAE lambda')
    p.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    p.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
    p.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy bonus coefficient')
    p.add_argument('--save_interval', type=int, default=5, help='Updates between saves')

    p.add_argument('-m', '--mode', choices=['train', 'inference'], default='train',
                   help="Run mode: 'train' (default) or 'inference'")
    p.add_argument('--num_episodes', type=int, default=None,
                   help="Number of episodes (inference only)")
    return p.parse_args()


def main():
    args = arg_parser()

    # checkpoint logic
    default_ckpt = f"logs/{args.task}_ppo_checkpoint.pth"
    if args.load_path:
        args.load = True
        args.checkpoint_path = (default_ckpt
                                if args.load_path == 'default'
                                else args.load_path)
    else:
        args.load = False
        args.checkpoint_path = default_ckpt

    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    if args.load and not os.path.isfile(args.checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)

    # init Genesis
    backend = gs.cpu if args.device.lower().startswith('cpu') else gs.gpu
    gs.init(backend=backend)

    # dispatch
    if args.mode == 'train':
        train_ppo(args)
    else:
        inference_ppo(args)

if __name__ == '__main__':
    main()
