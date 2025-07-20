import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'  # comment out for Windows or MacOS

import sys
import argparse
import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter

from algo.ppo_agent_vision import PPOAgentVision, RolloutBatch
from env import *

# map task names to env classes
task_to_class = {
    'ReachCubeVision':              ReachCubeVisionEnv,
    'ReachCubeVisionStacked':       ReachCubeVisionStackedEnv,
    'ReachCubeEgoVision':           ReachCubeEgoVisionEnv,
    'ReachCubeEgoVisionStacked':    ReachCubeEgoVisionStackedEnv,
}

def create_environment(task_name):
    if task_name in task_to_class:
        return task_to_class[task_name]
    raise ValueError(f"\n Task '{task_name}' is not recognized.\n")

def train_ppo(args):
    # build environment
    env_cls = create_environment(args.task)
    env = env_cls(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"\n [INFO] Created environment: {env}\n")

    # build agent with all new PPO params
    agent = PPOAgentVision(
        obs_shape    = env.obs_shape,
        output_dim   = env.action_space,
        lr           = args.lr,
        gamma        = args.gamma,
        lam          = args.lam,
        clip_epsilon = args.clip_epsilon,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        value_coef   = args.value_coef,
        entropy_coef = args.entropy_coef,
        device       = args.device,
        load         = args.load,
        num_envs     = args.num_envs,
        checkpoint_path = args.checkpoint_path,
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}")

    # rollout parameters
    T = args.horizon
    N = args.num_envs
    total_steps = args.total_timesteps
    num_updates = total_steps // (T * N)

    for update in range(1, num_updates + 1):
        # collect rollout
        state = env.reset()  # [N, C, H, W]
        states, actions, logps, values, rewards, dones = [], [], [], [], [], []

        for t in range(T):
            a, lp, ent, v = agent.select_action(state)
            states.append(state)
            actions.append(a)
            logps.append(lp)
            values.append(v)

            next_state, r, d = env.step(a)
            rewards.append(r.to(agent.device))
            dones.append(d.to(agent.device).float())

            # if all envs done early, reset
            if d.all():
                next_state = env.reset()
            state = next_state

        # bootstrap value for last state
        states.append(state)
        with torch.no_grad():
            _, last_val = agent.model(state.to(agent.device))
        values.append(last_val)

        # pack batch and train
        batch = RolloutBatch(
            states    = torch.stack(states),    # [T+1, N, C, H, W]
            actions   = torch.stack(actions),   # [T,   N]
            log_probs = torch.stack(logps),     # [T,   N]
            values    = torch.stack(values),    # [T+1, N]
            rewards   = torch.stack(rewards),   # [T,   N]
            dones     = torch.stack(dones),     # [T,   N]
        )
        agent.train(batch)

        # logging & checkpoint
        avg_reward = batch.rewards.sum(dim=0).mean().item()
        writer.add_scalar('Reward/Mean', avg_reward, update)
        if update % args.save_interval == 0:
            agent.save_checkpoint()
        print(f"[Update {update}/{num_updates}] Mean Reward: {avg_reward:.3f}")

    writer.close()

def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--vis",   action="store_true", help="Enable visualization")
    p.add_argument(
        "-l", "--load_path",
        nargs="?", const="default", default=None,
        help="`-l` alone loads default ckpt; `-l path.pth` loads that file"
    )
    p.add_argument("-n", "--num_envs",       type=int,   default=1,           help="Number of envs")
    p.add_argument("-t", "--task",           type=str,   default="ReachCubeVision", help="Task")
    p.add_argument("-d", "--device",         type=str,   default="cuda",      help="cpu, cuda[:X], or mps")

    # new PPO hyperparameters
    p.add_argument("--horizon",         type=int,   default=128,     help="Rollout horizon T")
    p.add_argument("--total_timesteps", type=int,   default=1_000_000, help="Total env steps")
    p.add_argument("--epochs",          type=int,   default=10,      help="PPO epochs per update")
    p.add_argument("--batch_size",      type=int,   default=64,      help="Minibatch size")
    p.add_argument("--lr",              type=float, default=3e-4,    help="Learning rate")
    p.add_argument("--gamma",           type=float, default=0.99,    help="Discount factor")
    p.add_argument("--lam",             type=float, default=0.95,    help="GAE lambda")
    p.add_argument("--clip_epsilon",    type=float, default=0.2,     help="Clipping epsilon")
    p.add_argument("--value_coef",      type=float, default=0.5,     help="Value loss coefficient")
    p.add_argument("--entropy_coef",    type=float, default=0.01,    help="Entropy bonus coefficient")
    p.add_argument("--save_interval",   type=int,   default=10,      help="Updates between checkpoints")

    return p.parse_args()

def main():
    args = arg_parser()

    # checkpoint logic
    default_ckpt = f"{args.task}_ppo_checkpoint.pth"
    if args.load_path:
        args.load = True
        args.checkpoint_path = (
            os.path.join("logs", default_ckpt)
            if args.load_path == "default"
            else os.path.join("logs", args.load_path)
        )
        print(f"\n[INFO] Loading checkpoint from: {args.checkpoint_path}\n")
        if not os.path.isfile(args.checkpoint_path):
            print(f"\n[ERROR] Checkpoint not found: {args.checkpoint_path}\n")
            sys.exit(1)
    else:
        args.load = False
        args.checkpoint_path = os.path.join("logs", default_ckpt)
        print("\n[INFO] No checkpoint; training from scratch.\n")
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

    # initialize Genesis
    backend = gs.cpu if args.device.lower().startswith("cpu") else gs.gpu
    gs.init(backend=backend)

    train_ppo(args)

if __name__ == "__main__":
    main()
