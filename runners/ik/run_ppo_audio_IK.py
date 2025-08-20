# runners/ik/run_ppo_audio_IK.py

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'  # comment out for Windows or MacOS

import sys
from pathlib import Path

# Adds the root directory (two levels up from this file) to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter

from agents.ik.ppo_agent_audio_IK import PPOAgentAudio, RolloutBatch
from envs import *

task_to_class = {
    'ReachCubeEgoAudio': ReachCubeEgoAudioEnv,
    'ReachCubeEgoAudioStacked': ReachCubeEgoAudioStackedEnv,
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
        noise_config={"audio_noise_level": args.audio_noise_level}
    )

    agent = PPOAgentAudio(
        obs_shape=env.obs_shape,
        action_shape=env.action_space,
        lr=1e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        device=args.device,
        load=args.load,
        num_envs=args.num_envs,
        checkpoint_path=args.checkpoint_path
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}")

    run(env, agent, args, writer)

    writer.close()


def run(env, agent, args, writer):
    T = 100                 # max horizon per update
    N = args.num_envs
    total_steps = 1_000_000
    num_updates = total_steps // (T * N)

    for update in range(1, num_updates + 1):
        state, _ = env.reset()

        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

        # always start with the current state (for [T+1] stacking later)
        # we append the trailing state after the loop as well
        for t in range(T):
            # stash current state for time t
            states.append(state)

            a, logp, ent, val = agent.select_action(state)
            actions.append(a)
            log_probs.append(logp.detach())
            values.append(val.detach())

            next_state, reward, done = env.step(a)

            rewards.append(reward.to(agent.device))
            dones.append(done.to(agent.device).float())

            state = next_state

            # ---- EARLY STOP WHEN ANY ENV HITS ----
            if done.any():
                # stop collecting immediately; train on the partial rollout
                break

        # append the final state and its value for bootstrap
        states.append(state)
        with torch.no_grad():
            _, last_value = agent.model(state.to(agent.device))
        values.append(last_value.detach())

        # stack into tensors of shape:
        # states: [t_len+1, N, 1, F, Tspec], actions/logp/rew/done: [t_len, N]
        batch = RolloutBatch(
            states=torch.stack(states),           # [t_len+1, N, 1, F, Tspec]
            actions=torch.stack(actions),         # [t_len,   N]
            log_probs=torch.stack(log_probs),     # [t_len,   N]
            values=torch.stack(values),           # [t_len+1, N]
            rewards=torch.stack(rewards),         # [t_len,   N]
            dones=torch.stack(dones),             # [t_len,   N]
        )

        agent.train(batch)

        # logging
        ep_reward = batch.rewards.sum(dim=0).mean().item()
        writer.add_scalar('Reward/Mean', ep_reward, update)
        print(f"[Update {update}/{num_updates}] Avg Reward per Env: {ep_reward:.3f} | "
              f"Collected steps this update: {batch.rewards.shape[0]}")

        if update % 5 == 0:
            agent.save_checkpoint()



@torch.no_grad()
def inference_ppo(args):
    env_cls = create_environment(args.task)
    env = env_cls(
        vis=args.vis,
        device=args.device,
        num_envs=args.num_envs,
        noise_config={"audio_noise_level": args.audio_noise_level},
        inference_mode=True,
    )

    # Optional: unlock any curriculum limits
    if hasattr(env, "max_stages"):
        env.x_stage = env.max_stages

    agent = PPOAgentAudio(
        obs_shape=env.obs_shape,
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

    # Accumulators across all episodes (per-env)
    total_successes = torch.zeros(N, dtype=torch.long)
    total_steps_to_success = torch.zeros(N, dtype=torch.long)
    total_rewards = torch.zeros(N, dtype=torch.float32)

    for ep in range(args.num_episodes):
        state, _ = env.reset()
        if state is None:
            print("[INFO] Environment returned None on reset, ending inference.")
            break

        steps = 0
        ep_reward = torch.zeros(N, dtype=torch.float32)
        reached = torch.zeros(N, dtype=torch.bool)       # whether each env reached 30cm
        steps_to_success = torch.zeros(N, dtype=torch.long)

        while steps < MAX_STEPS:
            action = agent.act(state, deterministic=deterministic)
            next_state, reward, done = env.step(action)

            if next_state is None:
                print("[INFO] Environment returned None during step, stopping current episode.")
                break

            ep_reward += reward.cpu()
            steps += 1

            # Newly reached envs record their steps-to-30cm at the first time they reach it
            newly = (~reached) & done.cpu()
            if newly.any():
                steps_to_success[newly] = steps
            reached |= done.cpu()

            state = next_state

            # Stop early if all envs reached 30cm
            if reached.all():
                break

        # Episode accounting
        total_rewards += ep_reward
        total_successes += reached.long()
        total_steps_to_success += steps_to_success * reached.long()  # only add for successes

        # Per-episode logging/printing
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

    # --- Final summary across all episodes ---
    successes_total = int(total_successes.sum().item())
    attempts_total = N * args.num_episodes
    success_rate = successes_total / attempts_total if attempts_total > 0 else 0.0

    if successes_total > 0:
        avg_steps_over_successes = (total_steps_to_success.sum().item() / successes_total)
    else:
        avg_steps_over_successes = float('nan')

    avg_reward_per_env = (total_rewards / max(args.num_episodes, 1)).mean().item()

    print("\n=== Inference Summary ===")
    print(f"Episodes: {args.num_episodes} | Envs: {N} | Attempts: {attempts_total}")
    print(f"Reached 30cm: {successes_total}/{attempts_total} ({success_rate*100:.1f}%)")
    print(f"Avg steps-to-30cm over all successes: {avg_steps_over_successes:.1f}")
    print(f"Mean reward per env (averaged over episodes): {avg_reward_per_env:.3f}")



def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--vis", action="store_true", help="Enable visualization")
    p.add_argument("-l", "--load_path", nargs="?", const="default", default=None,
                   help="`-l` alone loads default checkpoint; `-l path.pth` loads that file")
    p.add_argument("-n", "--num_envs", type=int, default=1, help="Number of envs")
    p.add_argument("-t", "--task", type=str, default="ReachCubeEgoAudioStacked", help="Task")
    p.add_argument("-d", "--device", type=str, default="cuda", help="cpu, cuda[:X], or mps")
    p.add_argument("--audio_noise_level", type=float, default=0.0, help="Level of audio noise")
    p.add_argument("-m", "--mode", choices=['train', 'inference'], default='train', help="Run mode")
    # --- CHANGED default from 10 -> 100 ---
    p.add_argument("--num_episodes", type=int, default=100, help="Episodes for inference mode")
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

    if args.mode == 'train':
        train_ppo(args)
    else:
        inference_ppo(args)

if __name__ == "__main__":
    main()
