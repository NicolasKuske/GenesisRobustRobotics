# runners/ik/run_ppo_multimodal_IK.py

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'  # comment out for Windows or MacOS

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter

from envs import *  # ensure ReachCubeEgoMultimodalStackedEnv is exported in envs/__init__.py
from agents.ik.ppo_agent_multimodal_IK import PPOAgentMultimodal, RolloutBatchMM


# -------------------------
# Task registry (mirror audio/vision)
# -------------------------
task_to_class = {
    'ReachCubeEgoMultimodalStacked': ReachCubeEgoMultimodalStackedEnv,
    # add more variants here if you expose them via envs/__init__.py
}

def create_environment(task_name):
    if task_name in task_to_class:
        return task_to_class[task_name]
    raise ValueError(f"\n Task '{task_name}' is not recognized.\n")


# -------------------------
# Train
# -------------------------
def train_ppo(args):
    env_cls = create_environment(args.task)
    env = env_cls(
        vis=args.vis,
        device=args.device,
        num_envs=args.num_envs,
        noise_config={
            "visual_noise_level": args.visual_noise_level,
            "audio_noise_level": args.audio_noise_level
        },
        inference_mode=False,
    )

    agent = PPOAgentMultimodal(
        obs_shape_vision=env.obs_shape_vision,
        obs_shape_audio=env.obs_shape_audio,
        action_shape=env.action_space,
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
    T = 100
    N = args.num_envs
    total_steps = 1_000_000
    num_updates = total_steps // (T * N)

    for update in range(1, num_updates + 1):
        sv, sa = env.reset()

        states_v, states_a, actions, log_probs, values, rewards, dones = [], [], [], [], [], [], []

        for t in range(T):
            states_v.append(sv)
            states_a.append(sa)

            # STRICT actor–critic API (no shim)
            a, logp, ent, val = agent.select_action(sv, sa)

            actions.append(a)
            log_probs.append(logp.detach())
            values.append(val.detach())

            (sv_next, sa_next), reward, done = env.step(a)

            rewards.append(reward.to(agent.device))
            dones.append(done.to(agent.device).float())

            sv, sa = sv_next, sa_next
            if done.all():
                break

        # Bootstrap with trailing value
        states_v.append(sv)
        states_a.append(sa)
        with torch.no_grad():
            _, last_value = agent.model(sv.to(agent.device), sa.to(agent.device))
        values.append(last_value.detach())

        batch = RolloutBatchMM(
            states_v=torch.stack(states_v),     # [t_len+1, N, C_v, H, W]
            states_a=torch.stack(states_a),     # [t_len+1, N, C_a, F, T]
            actions=torch.stack(actions),       # [t_len,   N]
            log_probs=torch.stack(log_probs),   # [t_len,   N]
            values=torch.stack(values),         # [t_len+1, N]
            rewards=torch.stack(rewards),       # [t_len,   N]
            dones=torch.stack(dones),           # [t_len,   N]
        )

        agent.train(batch)

        ep_reward = batch.rewards.sum(dim=0).mean().item()
        writer.add_scalar('Reward/Mean', ep_reward, update)
        print(f"[Update {update}/{num_updates}] Avg Reward per Env: {ep_reward:.3f} | "
              f"Collected steps this update: {batch.rewards.shape[0]}")

        if update % 3 == 0:
            agent.save_checkpoint()

# -------------------------
# Inference (parity with audio/vision)
# -------------------------
@torch.no_grad()
def inference_ppo(args):
    env_cls = create_environment(args.task)
    env = env_cls(
        vis=args.vis,
        device=args.device,
        num_envs=args.num_envs,
        noise_config={
            "visual_noise_level": args.visual_noise_level,
            "audio_noise_level": args.audio_noise_level
        },
        inference_mode=True,
    )

    agent = PPOAgentMultimodal(
        obs_shape_vision=env.obs_shape_vision,
        obs_shape_audio=env.obs_shape_audio,
        action_shape=env.action_space,
        device=args.device,
        load=True,
        num_envs=args.num_envs,
        checkpoint_path=args.checkpoint_path
    )
    agent.eval_mode(True)

    writer = SummaryWriter(log_dir=f"runs/{args.task}_inference")

    deterministic = True
    N = args.num_envs
    MAX_STEPS = 100

    total_successes = torch.zeros(N, dtype=torch.long)
    total_steps_to_success = torch.zeros(N, dtype=torch.long)
    total_rewards = torch.zeros(N, dtype=torch.float32)

    for ep in range(args.num_episodes):
        sv, sa = env.reset()
        if sv is None:
            print("[INFO] Environment returned None on reset, ending inference.")
            break

        steps = 0
        ep_reward = torch.zeros(N, dtype=torch.float32)
        reached = torch.zeros(N, dtype=torch.bool)
        steps_to_success = torch.zeros(N, dtype=torch.long)

        while steps < MAX_STEPS:
            a = agent.act(sv, sa, deterministic=deterministic)
            (sv_next, sa_next), reward, done = env.step(a)
            if sv_next is None:
                print("[INFO] Environment returned None during step, stopping current episode.")
                break

            ep_reward += reward.cpu()
            steps += 1

            newly = (~reached) & done.cpu()
            if newly.any():
                steps_to_success[newly] = steps
            reached |= done.cpu()

            sv, sa = sv_next, sa_next
            if reached.all():
                break

        total_rewards += ep_reward
        total_successes += reached.long()
        total_steps_to_success += steps_to_success * reached.long()

        mean_reward = ep_reward.mean().item()
        writer.add_scalar('Inference/MeanReward', mean_reward, ep)
        writer.add_scalar('Inference/StepsExecuted', steps, ep)

        if reached.any():
            avg_steps_this_ep = steps_to_success[reached].float().mean().item()
            print(
                f"[Inference {ep+1}/{args.num_episodes}] "
                f"Steps executed: {steps}, "
                f"Mean Reward: {mean_reward:.3f}, "
                f"Success envs: {reached.sum().item()}/{N}, "
                f"Avg steps-to-30cm (successful envs): {avg_steps_this_ep:.1f}"
            )
        else:
            print(
                f"[Inference {ep+1}/{args.num_episodes}] "
                f"Steps executed: {steps}, "
                f"Mean Reward: {mean_reward:.3f}, "
                f"Success envs: 0/{N} (no reach within 30cm)"
            )

    writer.close()


# -------------------------
# CLI (mirror audio/vision)
# -------------------------
def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-v",  "--vis", action="store_true", help="Enable visualization")
    p.add_argument("-l",  "--load_path", nargs="?", const="default", default=None,
                   help="`-l` alone loads default checkpoint; `-l path.pth` loads that file")
    p.add_argument("-n",  "--num_envs", type=int, default=1, help="Number of envs")
    p.add_argument("-t",  "--task", type=str, default="ReachCubeEgoMultimodalStacked", help="Task")
    p.add_argument("-d",  "--device", type=str, default="cuda", help="cpu, cuda[:X], or mps")

    # noise flags (passed as noise_config to env)
    p.add_argument("--visual_noise_level", type=float, default=0.0, help="Gaussian noise std-dev added to RGB")
    p.add_argument("--audio_noise_level",  type=float, default=0.0, help="Audio Gaussian noise std-dev")

    p.add_argument("-m",  "--mode", choices=['train', 'inference'], default='train', help="Run mode")
    p.add_argument("--num_episodes", type=int, default=100, help="Episodes for inference mode")
    return p.parse_args()


def main():
    args = arg_parser()

    # checkpoint path (same pattern as audio/vision)
    default_ckpt = f"logs/{args.task}_ppo_checkpoint.pth"
    if args.load_path:
        args.load = True
        args.checkpoint_path = default_ckpt if args.load_path == "default" else args.load_path
    else:
        args.load = False
        args.checkpoint_path = default_ckpt

    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

    if args.mode == 'inference' and not os.path.isfile(args.checkpoint_path):
        print(f"[ERROR] Checkpoint not found for inference: {args.checkpoint_path}")
        sys.exit(1)

    backend = gs.cpu if args.device.lower().startswith("cpu") else gs.gpu
    gs.init(backend=backend)

    # unify device
    args.device = torch.device(args.device)

    if args.mode == 'train':
        train_ppo(args)
    else:
        inference_ppo(args)


if __name__ == "__main__":
    main()
