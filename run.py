import argparse
import genesis as gs
import torch
from algo.dqn_agent import DQNAgent
from env.wrapped_env import WrappedEnv

gs.init(backend=gs.gpu, precision="32")

def train_dqn(args):
    env = WrappedEnv(vis=args.vis, num_envs=args.num_envs)
    agent = DQNAgent(input_dim=6, output_dim=8, lr=1e-3, gamma=0.99, epsilon=0.5, epsilon_decay=0.995, epsilon_min=0.01, load=args.load, num_envs=args.num_envs, hidden_dim=args.hidden_dim)
    num_episodes = 500
    batch_size = args.batch_size if args.batch_size else 64 * args.num_envs
    target_update_interval = 10

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = torch.zeros(env.num_envs).to("cuda:0")
        done_array = torch.tensor([False] * env.num_envs).to("cuda:0")
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
            agent.save_checkpoint("logs/dqn_checkpoint.pth")
        print(f"Episode {episode}, Total Reward: {total_reward}")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Enable visualization") 
    parser.add_argument("-l", "--load", action="store_true", default=False, help="Load model from checkpoint") 
    parser.add_argument("-n", "--num_envs", type=int, default=1, help="Number of environments to create") 
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=64, help="Hidden dimension for the network")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    train_dqn(args)
