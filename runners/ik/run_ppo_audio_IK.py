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
        lr=1e-3,
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
    T = 100                    # horizon per update (match position runner default)
    N = args.num_envs
    total_steps = 1_000_000    # or args.total_timesteps if you add it
    num_updates = total_steps // (T * N)

    for update in range(1, num_updates + 1):
        state, _ = env.reset()
        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

        for t in range(T):
            a, logp, ent, val = agent.select_action(state)
            states.append(state)                       # [N, C, F, Tspec]
            actions.append(a)
            log_probs.append(logp.detach())
            values.append(val.detach())

            next_state, reward, done = env.step(a)

            # if env signals curriculum complete (none), break this update cleanly
            if next_state is None:
                # bootstrap with zeros
                rewards.append(torch.zeros_like(reward))
                dones.append(torch.ones_like(done).float())
                states.append(state)  # duplicate last state as T+1
                values.append(val)    # approximate bootstrap
                break

            rewards.append(reward.to(agent.device))
            dones.append(done.to(agent.device).float())

            # optional: per-env early reset
            if done.all():
                next_state, _ = env.reset()

            state = next_state

        # append last state/value for bootstrap if not added above
        if len(states) == T:
            states.append(state)
            with torch.no_grad():
                _, last_value = agent.model(state.to(agent.device))
            values.append(last_value)
        else:
            last_value = values[-1]

        batch = RolloutBatch(
            states=torch.stack(states),           # [T+1, N, 1, F, Tspec]
            actions=torch.stack(actions),         # [T,   N]
            log_probs=torch.stack(log_probs),     # [T,   N]
            values=torch.stack(values),           # [T+1, N]
            rewards=torch.stack(rewards),         # [T,   N]
            dones=torch.stack(dones),             # [T,   N]
        )

        agent.train(batch)

        # logging like position runner
        ep_reward = batch.rewards.sum(dim=0).mean().item()
        writer.add_scalar('Reward/Mean', ep_reward, update)
        print(f"[Update {update}/{num_updates}] Avg Reward per Env: {ep_reward:.3f}")

        if update % 5 == 0:
            agent.save_checkpoint()


@torch.no_grad()
def inference_ppo(args):
    env_cls = create_environment(args.task)
    env = env_cls(
        vis=args.vis,
        device=args.device,
        num_envs=args.num_envs,
        episodes_per_position=1,  # sample new cube pos each episode
        noise_config={"audio_noise_level": args.audio_noise_level}
    )

    # Optional: allow full X-range during inference (if curriculum exists)
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
    agent.eval_mode(True)  # freeze dropout/bn and avoid training-time behavior

    writer = SummaryWriter(log_dir=f"runs/{args.task}_inference")

    # Greedy by default; change to False for stochastic eval
    deterministic = True

    for ep in range(args.num_episodes):
        state, _ = env.reset()
        if state is None:
            print("[INFO] Environment returned None on reset, ending inference.")
            break

        steps = 0
        total_reward = torch.zeros(args.num_envs, device=args.device)

        while steps < 200:
            action = agent.act(state, deterministic=deterministic)
            next_state, reward, done = env.step(action)

            if next_state is None:
                print("[INFO] Environment returned None during step, stopping current episode.")
                break

            total_reward += reward.to(args.device)
            state = next_state
            steps += 1

            # stop early if all envs solved
            if done.all():
                break

        mean_reward = total_reward.mean().item()
        writer.add_scalar('Inference/MeanReward', mean_reward, ep)
        writer.add_scalar('Inference/Steps', steps, ep)
        print(f"[Inference {ep+1}/{args.num_episodes}] Steps: {steps}, Mean Reward: {mean_reward:.3f}")

    writer.close()



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
    p.add_argument("--num_episodes", type=int, default=10, help="Episodes for inference mode")
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
