# run_ppo_position.py

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'  # comment out for Windows or MacOS

import sys
import argparse
import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter

from algo.ppo_agent_position import PPOAgentPosition
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
        load=args.load,
        device=args.device,
        num_envs=args.num_envs,
        checkpoint_path=args.checkpoint_path,
        hidden_dim=64,
        input_dim=env.state_dim,
        output_dim=env.action_space,
        lr=1e-3,
        gamma=0.99,
        clip_epsilon=0.2
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_train")

    if args.device.lower() == "mps":  # for MacOS
        gs.tools.run_in_another_thread(fn=run_loop, args=(env, agent, args, writer, True))
        env.scene.viewer.start()
    else:
        run_loop(env, agent, args, writer, train_mode=True)

    writer.close()

def inference_ppo(args):
    env_cls = create_environment(args.task)
    env = env_cls(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"\n [INFO] Created environment: {env} \n")

    agent = PPOAgentPosition(
        load=True,
        device=args.device,
        num_envs=args.num_envs,
        checkpoint_path=args.checkpoint_path,
        hidden_dim=64,
        input_dim=env.state_dim,
        output_dim=env.action_space,
        lr = 1e-3,
        gamma = 0.99,
        clip_epsilon = 0.2
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}_inference")

    # run a fixed number of episodes (or make it configurable)
    run_loop(env, agent, args, writer, train_mode=False)

    writer.close()

def run_loop(env, agent, args, writer, train_mode: bool):
    num_episodes = args.num_episodes or 100  # default fewer for inference
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = torch.zeros(env.num_envs).to(args.device)
        done_array = torch.zeros(env.num_envs, dtype=torch.bool).to(args.device)

        states, actions, rewards, dones = [], [], [], []

        for _ in range(200):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            if train_mode:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

            state = next_state
            total_reward += reward
            done_array = torch.logical_or(done_array, done)
            if done_array.all():
                break

        if train_mode:
            agent.train(states, actions, rewards, dones)
            if episode % 5 == 0:
                agent.save_checkpoint()

        mean_reward = total_reward.mean().item()
        writer.add_scalar('Reward/Mean', mean_reward, episode)
        print(f"\n [{'Inference' if not train_mode else 'Episode'} {episode}] Mean Reward: {mean_reward}\n ")

def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--vis", action="store_true", help="Enable visualization")
    p.add_argument(
        "-l", "--load_path",
        nargs="?", const="default", default=None,
        help="`-l` alone loads the default checkpoint; `-l path.pth` loads that file"
    )
    p.add_argument("-n", "--num_envs", type=int, default=1, help="Number of envs")
    p.add_argument("--num_episodes", type=int, default=None,
                   help="Number of episodes to run (only for inference)")
    p.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size")
    p.add_argument("-t", "--task", type=str, default="ReachCubePosition", help="Task")
    p.add_argument("-d", "--device", type=str, default="cuda", help="cpu, cuda[:X], or mps")
    p.add_argument("--mode", choices=["train", "inference"], default="train",
                   help="Select 'train' (default) or 'inference' mode")
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
    if args.load:
        print(f"\n [INFO] Loading checkpoint from: {args.checkpoint_path}\n")
        if not os.path.isfile(args.checkpoint_path):
            print(f"\n [ERROR] Checkpoint not found: {args.checkpoint_path}\n")
            sys.exit(1)
    else:
        print("\n [INFO] No checkpoint provided; training from scratch.\n")

    backend = gs.cpu if args.device.lower().startswith("cpu") else gs.gpu
    gs.init(backend=backend)

    if args.mode == "train":
        train_ppo(args)
    else:
        inference_ppo(args)

if __name__ == "__main__":
    main()
