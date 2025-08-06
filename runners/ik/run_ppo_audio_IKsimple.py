# runners/ik/run_ppo_audio_IKsimple.py

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

from agents.ik.ppo_agent_audio_IKsimple import PPOAgentAudio
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
        joint_dim=7 * env.history_length,  
        #joint_dim=7 * len(env.sample_offsets),
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
    num_episodes = 1000000

    for episode in range(num_episodes):
        audio_state, joint_state, done_array = env.reset()

        total_reward = torch.zeros(env.num_envs).to(args.device)

        audio_states, joint_states, actions, rewards, dones = [], [], [], [], []

        for step in range(100):
            action = agent.select_action(audio_state, joint_state)
            next_audio_state, next_joint_state, reward, done = env.step(action)

            audio_states.append(audio_state)
            joint_states.append(joint_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            audio_state, joint_state = next_audio_state, next_joint_state
            total_reward += reward
            done_array = torch.logical_or(done_array, done)
            if done_array.all():
                break

        agent.train(audio_states, joint_states, actions, rewards, dones)

        if episode % 3 == 0:
            agent.save_checkpoint()
            print(f"\n Saved checkpoint to logs :)\n ")

        mean_reward = total_reward.mean().item()
        writer.add_scalar('Reward/Mean', mean_reward, episode)

        print(f"[Episode {episode}] Mean Reward: {mean_reward:.4f}, Total Reward: {total_reward}\n")



@torch.no_grad()
def inference_ppo(args):
    env_cls = create_environment(args.task)
    env = env_cls(
        vis=args.vis,
        device=args.device,
        num_envs=args.num_envs,
        episodes_per_position=1,
        noise_config={"audio_noise_level": args.audio_noise_level}
    )

    env.x_stage = env.max_stages
    print(f"[INFO] Inference environment (full curriculum range): {env}")

    agent = PPOAgentAudio(
        obs_shape=env.obs_shape,
        action_shape=env.action_space,
        lr=1e-3,
        gamma=0.99,
        clip_epsilon=0.2,
        device=args.device,
        load=True,
        num_envs=args.num_envs,
        checkpoint_path=args.checkpoint_path
    )

    print(f"[INFO] Loaded checkpoint from {args.checkpoint_path}")

    writer = SummaryWriter(log_dir=f"runs/{args.task}_inference")

    for ep in range(args.num_episodes):
        state, done_array = env.reset()

        if state is None:
            print("[INFO] Environment returned None on reset, ending inference.")
            break

        steps = 0
        total_reward = torch.zeros(args.num_envs).to(args.device)

        while steps < 100:
            action = agent.select_action(state, inference=True)
            next_state, reward, done = env.step(action)

            if next_state is None:
                print("[INFO] Environment returned None during step, stopping current episode.")
                break

            total_reward += reward
            state = next_state
            steps += 1

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
    p.add_argument("-t", "--task", type=str, default="ReachCubeEgoAudio", help="Task")
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
