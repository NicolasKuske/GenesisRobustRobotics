#run_ppo_position.py


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
    'ReachCubePositionStacked': ReachCubePositionStackedEnv
}

def create_environment(task_name):
    if task_name in task_to_class:
        return task_to_class[task_name]
    raise ValueError(f"\n Task '{task_name}' is not recognized.\n ")


def train_ppo(args):
    # build environment
    env_cls = create_environment(args.task)
    env = env_cls(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"\n [INFO] Created environment: {env} \n")

    # build agent
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

    # set up TensorBoard logger
    writer = SummaryWriter(log_dir=f"runs/{args.task}")

    # start training
    if args.device.lower() == "mps":  # for MacOS
        gs.tools.run_in_another_thread(fn=run, args=(env, agent, args, writer))
        env.scene.viewer.start()
    else:
        run(env, agent, args, writer)

    # close the writer when done
    writer.close()


def run(env, agent, args, writer):
    num_episodes = 1000000

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = torch.zeros(env.num_envs).to(args.device)
        done_array = torch.zeros(env.num_envs, dtype=torch.bool).to(args.device)

        states, actions, rewards, dones = [], [], [], []

        for _ in range(200):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            total_reward += reward
            done_array = torch.logical_or(done_array, done)
            if done_array.all():
                break

        # update agent
        agent.train(states, actions, rewards, dones)

        # save checkpoint periodically
        if episode % 5 == 0:
            agent.save_checkpoint()

        # log to TensorBoard: mean reward across environments
        mean_reward = total_reward.mean().item()
        writer.add_scalar('Reward/Mean', mean_reward, episode)

        print(f"\n [Episode {episode}] Total Reward: {total_reward}  Mean Reward: {mean_reward}\n ")


def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--vis", action="store_true", help="Enable visualization")
    p.add_argument(
        "-l", "--load_path",
        nargs="?", const="default", default=None,
        help="`-l` alone loads the default checkpoint; `-l path.pth` loads that file"
    )
    p.add_argument("-n", "--num_envs", type=int, default=1, help="Number of envs")
    p.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size")
    p.add_argument("-t", "--task", type=str, default="ReachCubePosition", help="Task")
    p.add_argument("-d", "--device", type=str, default="cuda", help="cpu, cuda[:X], or mps")
    return p.parse_args()


def main():
    args = arg_parser()

    # central checkpoint logic
    default_ckpt = f"logs/{args.task}_ppo_checkpoint.pth"

    if args.load_path:
        args.load = True
        args.checkpoint_path = default_ckpt if args.load_path == "default" else args.load_path
    else:
        args.load = False
        args.checkpoint_path = default_ckpt

    # make sure directory exists
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

    # report load vs. scratch
    if args.load:
        print(f"\n [INFO] Loading checkpoint from: {args.checkpoint_path}\n \n ")
        if not os.path.isfile(args.checkpoint_path):
            print(f"\n [ERROR] Checkpoint not found: {args.checkpoint_path}\n \n ")
            sys.exit(1)
    else:
        print("\n [INFO] No checkpoint provided; training from scratch.\n \n ")

    # now safe to init Genesis
    if args.device.lower().startswith("cpu"):
        backend = gs.cpu
    else:
        backend = gs.gpu
    gs.init(backend=backend)

    train_ppo(args)


if __name__ == "__main__":
    main()
