# runners/ik/run_ppo_vision_IK.py

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'  # comment out for Windows or MacOS

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter

from agents.ik.ppo_agent_vision_IK import PPOAgentVision, RolloutBatch
from envs import *

# Map task names to env classes (ensure these exist)
task_to_class = {
    'ReachCubeVision': ReachCubeVisionEnv,
    'ReachCubeVisionStacked': ReachCubeVisionStackedEnv,
    'ReachCubeEgoVision': ReachCubeEgoVisionEnv,
    'ReachCubeEgoVisionStacked': ReachCubeEgoVisionStackedEnv,
}

def create_environment(task_name):
    if task_name in task_to_class:
        return task_to_class[task_name]
    raise ValueError(f"\n Task '{task_name}' is not recognized.\n")

def train_ppo(args):
    env_cls = create_environment(args.task)
    env = env_cls(
        vis=args.vis,
        device=args.device,
        num_envs=args.num_envs,
        noise_config={"visual_noise_level": args.visual_noise_level}
    )

    agent = PPOAgentVision(
        obs_shape=env.obs_shape,
        action_dim=env.action_space,     # <-- name matches agent signature
        lr=1e-4,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        epochs=10,
        batch_size=64,
        value_coef=0.5,
        entropy_coef=0.01,
        device=args.device,
        load=args.load,
        num_envs=args.num_envs,
        checkpoint_path=args.checkpoint_path
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}")
    run(env, agent, args, writer)
    writer.close()

def run(env, agent, args, writer):
    T = 100                # max horizon per update (like audio)
    N = args.num_envs
    total_steps = 1_000_000
    num_updates = total_steps // (T * N)

    for update in range(1, num_updates + 1):
        # vision reset returns just obs; standardize to (state, dummy_dones)
        state = env.reset()
        done_dummy = torch.zeros(N, dtype=torch.bool, device=args.device)

        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

        for t in range(T):
            states.append(state)  # [N, C, H, W]

            a, logp, ent, val = agent.select_action(state)
            actions.append(a)
            log_probs.append(logp.detach())
            values.append(val.detach())

            next_state, reward, done = env.step(a)

            rewards.append(reward.to(agent.device))
            dones.append(done.to(agent.device).float())

            state = next_state

            # Optional: early stop if all envs done (if your env ever sets dones True)
            if done.all():
                break

        # append trailing state and its value for bootstrap
        states.append(state)
        with torch.no_grad():
            _, last_value = agent.model(state.to(agent.device))
        values.append(last_value.detach())

        batch = RolloutBatch(
            states=torch.stack(states),           # [t_len+1, N, C, H, W]
            actions=torch.stack(actions),         # [t_len,   N]
            log_probs=torch.stack(log_probs),     # [t_len,   N]
            values=torch.stack(values),           # [t_len+1, N]
            rewards=torch.stack(rewards),         # [t_len,   N]
            dones=torch.stack(dones),             # [t_len,   N]
        )

        agent.train(batch)

        # logging & checkpointing
        ep_reward = batch.rewards.sum(dim=0).mean().item()
        writer.add_scalar('Reward/Mean', ep_reward, update)
        print(f"[Update {update}/{num_updates}] Avg Reward per Env: {ep_reward:.3f} | "
              f"Collected steps this update: {batch.rewards.shape[0]}")

        if update % 3 == 0:
            agent.save_checkpoint()

def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--vis", action="store_true", help="Enable visualization")
    p.add_argument("-l", "--load_path", nargs="?", const="default", default=None,
                   help="`-l` alone loads default checkpoint; `-l path.pth` loads that file")
    p.add_argument("-n", "--num_envs", type=int, default=1, help="Number of envs")
    p.add_argument("-t", "--task", type=str, default="ReachCubeEgoVisionStacked", help="Task")
    p.add_argument("-d", "--device", type=str, default="cuda", help="cpu, cuda[:X], or mps")
    p.add_argument("--visual_noise_level", type=float, default=0.0,
                   help="Gaussian noise std-dev added to RGB")
    return p.parse_args()

def main():
    args = arg_parser()

    default_ckpt = f"logs/{args.task}_ppo_checkpoint.pth"
    if args.load_path:
        args.load = True
        args.checkpoint_path = default_ckpt if args.load_path == "default" else args.load_path
    else:
        args.load = False
        args.checkpoint_path = default_ckpt

    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

    if args.load and not os.path.isfile(args.checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)

    backend = gs.cpu if args.device.lower().startswith("cpu") else gs.gpu
    gs.init(backend=backend)

    train_ppo(args)

if __name__ == "__main__":
    main()
