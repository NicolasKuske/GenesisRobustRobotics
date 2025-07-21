# run_ppo_vision_IK.py


import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'  # comment out for Windows or MacOS

import sys
import argparse
import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter

from algo.ppo_agent_vision_IK import PPOAgentVision, RolloutBatch
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

    # build agent
    agent = PPOAgentVision(
        obs_shape       = env.obs_shape,
        output_dim      = env.action_space,
        lr              = args.lr,
        gamma           = args.gamma,
        lam             = args.lam,
        clip_epsilon    = args.clip_epsilon,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        value_coef      = args.value_coef,
        entropy_coef    = args.entropy_coef,
        device          = args.device,
        load            = args.load,
        num_envs        = args.num_envs,
        checkpoint_path = args.checkpoint_path,
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_train")

    # rollout parameters
    T = args.horizon
    N = args.num_envs
    total_steps = args.total_timesteps
    num_updates = total_steps // (T * N)

    # pre-allocate all rollout buffers on GPU
    C, H, W = env.obs_shape
    states_buf  = torch.empty((T+1, N, C, H, W), device=args.device)
    actions_buf = torch.empty((T,   N),         dtype=torch.long, device=args.device)
    logps_buf   = torch.empty((T,   N),         device=args.device)
    values_buf  = torch.empty((T+1, N),         device=args.device)
    rewards_buf = torch.empty((T,   N),         device=args.device)
    dones_buf   = torch.empty((T,   N),         device=args.device)

    for update in range(1, num_updates + 1):
        # ---- collect rollout ----
        state = env.reset()  
        states_buf[0].copy_(state)

        for t in range(T):
            a, lp, ent, v = agent.select_action(state)
            actions_buf[t].copy_(a)
            logps_buf[t].copy_(lp)
            values_buf[t].copy_(v)

            next_state, r, d = env.step(a)
            rewards_buf[t].copy_(r.to(args.device))
            dones_buf[t].copy_(d.to(args.device).float())

            if d.all():
                next_state = env.reset()
            state = next_state
            states_buf[t+1].copy_(state)

        # bootstrap last‐step value
        with torch.no_grad():
            _, last_val = agent.model(state)
        values_buf[T].copy_(last_val)

        # pack into RolloutBatch and train
        batch = RolloutBatch(
            states    = states_buf,   # [T+1, N, C, H, W]
            actions   = actions_buf,  # [T,   N]
            log_probs = logps_buf,    # [T,   N]
            values    = values_buf,   # [T+1, N]
            rewards   = rewards_buf,  # [T,   N]
            dones     = dones_buf,    # [T,   N]
        )
        agent.train(batch)

        # logging & checkpoint
        avg_reward = rewards_buf.sum(dim=0).mean().item()
        writer.add_scalar('Reward/Mean', avg_reward, update)
        if update % args.save_interval == 0:
            agent.save_checkpoint()
        print(f"[Update {update}/{num_updates}] Mean Reward: {avg_reward:.3f}")

    writer.close()


def inference_ppo(args):
    # build environment for inference
    env_cls = create_environment(args.task)
    env = env_cls(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"\n [INFO] Created environment: {env} \n")

    # build and load agent
    agent = PPOAgentVision(
        obs_shape       = env.obs_shape,
        output_dim      = env.action_space,
        lr              = args.lr,
        gamma           = args.gamma,
        lam             = args.lam,
        clip_epsilon    = args.clip_epsilon,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        value_coef      = args.value_coef,
        entropy_coef    = args.entropy_coef,
        device          = args.device,
        load            = True,
        num_envs        = args.num_envs,
        checkpoint_path = args.checkpoint_path,
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_inference")
    num_episodes = args.num_episodes or 100

    for ep in range(1, num_episodes + 1):
        state      = env.reset()
        done_mask  = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        step_count = 0

        for _ in range(args.horizon):
            action, _, _, _ = agent.select_action(state)
            state, _, done = env.step(action)
            done_mask |= done
            step_count += 1
            if done_mask.all():
                break

        writer.add_scalar('Episode/Steps', step_count, ep)
        print(f"[Inference {ep}/{num_episodes}] Steps until done: {step_count}")

    writer.close()


def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--vis",   action="store_true", help="Enable visualization")
    p.add_argument(
        "-l", "--load_path",
        nargs="?", const="default", default=None,
        help="`-l` alone loads default ckpt; `-l path.pth` loads that file"
    )
    p.add_argument("-n", "--num_envs",       type=int,   default=1,               help="Number of envs")
    p.add_argument("-t", "--task",           type=str,   default="ReachCubeVision", help="Task")
    p.add_argument("-d", "--device",         type=str,   default="cuda",          help="cpu, cuda[:X], or mps")

    # new PPO hyperparameters
    p.add_argument("--horizon",         type=int,   default=128,       help="Rollout horizon T")
    p.add_argument("--total_timesteps", type=int,   default=1_000_000, help="Total env steps")
    p.add_argument("--epochs",          type=int,   default=10,        help="PPO epochs per update")
    p.add_argument("--batch_size",      type=int,   default=64,        help="Minibatch size")
    p.add_argument("--lr",              type=float, default=3e-4,      help="Learning rate")
    p.add_argument("--gamma",           type=float, default=0.99,      help="Discount factor")
    p.add_argument("--lam",             type=float, default=0.95,      help="GAE lambda")
    p.add_argument("--clip_epsilon",    type=float, default=0.2,       help="Clipping epsilon")
    p.add_argument("--value_coef",      type=float, default=0.5,       help="Value loss coefficient")
    p.add_argument("--entropy_coef",    type=float, default=0.01,      help="Entropy bonus coefficient")
    p.add_argument("--save_interval",   type=int,   default=10,        help="Updates between checkpoints")

    # add inference mode and episodes
    p.add_argument('-m', '--mode', choices=['train', 'inference'], default='train',
                   help="Run mode: 'train' (default) or 'inference'")
    p.add_argument('--num_episodes', type=int, default=None,
                   help="Number of episodes (inference only)")
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
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
        if args.mode == 'train':
            print("\n[INFO] No checkpoint; training from scratch.\n")
        else:
            print(f"\n[ERROR] Inference mode requires a checkpoint. Use -l to specify one.\n")
            sys.exit(1)

    # initialize Genesis
    backend = gs.cpu if args.device.lower().startswith("cpu") else gs.gpu
    gs.init(backend=backend)

    # dispatch based on mode
    if args.mode == 'train':
        train_ppo(args)
    else:
        inference_ppo(args)


if __name__ == "__main__":
    main()

