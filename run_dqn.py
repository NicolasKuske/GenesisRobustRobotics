import argparse
import genesis as gs
import torch
from algo.dqn_agent import DQNAgent
from env import *
import os

gs.init(backend=gs.gpu, precision="32")

task_to_class = {
    'GraspFixedBlock': GraspFixedBlockEnv,
    'GraspFixedRod': GraspFixedRodEnv,
    'GraspRandomBlock': GraspRandomBlockEnv,
    'GraspRandomRod': GraspRandomRodEnv
}

def create_environment(task_name):
    if task_name in task_to_class:
        return task_to_class[task_name]  
    else:
        raise ValueError(f"Task '{task_name}' is not recognized.")


def train_dqn(args):
    checkpoint_path = f"logs/{args.task}_dqn_checkpoint.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    try:
        env = create_environment(args.task)(vis=args.vis, device=args.device, num_envs=args.num_envs)
        print(f"Created environment: {env}")
    except ValueError as e:
        print(e)
    agent = DQNAgent(input_dim=6, output_dim=8, lr=1e-3, gamma=0.99, epsilon=0.5, epsilon_decay=0.995, epsilon_min=0.01, device=args.device, load=args.load, num_envs=args.num_envs, hidden_dim=args.hidden_dim)
    num_episodes = 500
    batch_size = args.batch_size if args.batch_size else 64 * args.num_envs
    target_update_interval = 10

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = torch.zeros(env.num_envs).to(args.device)
        done_array = torch.tensor([False] * env.num_envs).to(args.device)
        for step in range(50):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)
            agent.train(batch_size)

            state = next_state
            total_reward += reward
            done_array = torch.logical_or(done_array, done)
            if done_array.all():
                break

        if episode % target_update_interval == 0:
            agent.update_target_network()
            agent.save_checkpoint(checkpoint_path)
        print(f"Episode {episode}, Total Reward: {total_reward}")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Enable visualization") 
    parser.add_argument("-l", "--load", action="store_true", default=False, help="Load model from checkpoint") 
    parser.add_argument("-n", "--num_envs", type=int, default=1, help="Number of environments to create") 
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=64, help="Hidden dimension for the network")
    parser.add_argument("-t", "--task", type=str, default="GraspFixedBlock", help="Task to train on")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="device: cpu or cuda:x or mps for macos")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    train_dqn(args)
