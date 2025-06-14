import argparse
import genesis as gs
import torch
from algo.ppo_agent import PPOAgent
from env import *
import os

task_to_class = {
    'GraspFixedBlock': GraspFixedBlockEnv,
    'GraspFixedRod': GraspFixedRodEnv,
    'GraspRandomBlock': GraspRandomBlockEnv,
    'GraspRandomRod': GraspRandomRodEnv,
    'ShadowHandBase': ShadowHandBaseEnv
}

def create_environment(task_name):
    if task_name in task_to_class:
        return task_to_class[task_name]
    else:
        raise ValueError(f"Task '{task_name}' is not recognized.")

def train_ppo(args):
    # determine checkpoint loading
    if args.load_path == "default":
        load = True
        checkpoint_path = f"logs/{args.task}_ppo_checkpoint_released.pth"
    elif args.load_path:
        load = True
        checkpoint_path = args.load_path
    else:
        load = False
        checkpoint_path = f"logs/{args.task}_ppo_checkpoint.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # create environment
    env_cls = create_environment(args.task)
    env = env_cls(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"Created environment: {env}")

    # build agent
    agent = PPOAgent(
        input_dim=env.state_dim,
        output_dim=env.action_space,
        lr=1e-3,
        gamma=0.99,
        clip_epsilon=0.2,
        device=args.device,
        load=load,
        num_envs=args.num_envs,
        hidden_dim=args.hidden_dim,
        checkpoint_path=checkpoint_path
    )

    # run training loop, with special case for MPS
    if args.device.lower() == "mps":
        gs.tools.run_in_another_thread(fn=run, args=(env, agent, args))
        env.scene.viewer.start()
    else:
        run(env, agent, args)

def run(env, agent, args):
    num_episodes = 500
    batch_size = args.batch_size if args.batch_size else 64 * args.num_envs

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = torch.zeros(env.num_envs).to(args.device)
        done_array = torch.zeros(env.num_envs, dtype=torch.bool).to(args.device)

        states, actions, rewards, dones = [], [], [], []

        for step in range(50):
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

        agent.train(states, actions, rewards, dones)

        if episode % 10 == 0:
            agent.save_checkpoint()
        print(f"Episode {episode}, Total Reward: {total_reward}")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--vis",
        action="store_true", default=False,
        help="Enable visualization"
    )
    parser.add_argument(
        "-l", "--load_path",
        type=str, nargs='?', default=None,
        help="Path for loading model from checkpoint"
    )
    parser.add_argument(
        "-n", "--num_envs",
        type=int, default=1,
        help="Number of environments to create"
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int, default=None,
        help="Batch size for training"
    )
    parser.add_argument(
        "-hd", "--hidden_dim",
        type=int, default=64,
        help="Hidden dimension for the network"
    )
    parser.add_argument(
        "-t", "--task",
        type=str, default="GraspFixedBlock",
        help="Task to train on"
    )
    parser.add_argument(
        "-d", "--device",
        type=str, default="cuda",
        help="Device: 'cpu', 'cuda' (or 'cuda:0'), or 'mps'"
    )
    return parser.parse_args()

def main():
    args = arg_parser()

    # initialize Genesis *after* parsing device flag
    if args.device.lower().startswith("cpu"):
        backend = gs.cpu
    elif args.device.lower().startswith("mps"):
        # macOS Metal Performance Shaders backend, if supported
        backend = gs.gpu  # MPS still uses gs.gpu in Genesis
    else:
        backend = gs.gpu

    gs.init(backend=backend, precision="32")

    train_ppo(args)

if __name__ == "__main__":
    main()
