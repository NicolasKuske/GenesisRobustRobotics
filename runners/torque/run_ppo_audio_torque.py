
# runners/torque/run_ppo_audio_stacked_torque.py


import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'  # comment out for Windows or MacOS
import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter
from agents.torque.ppo_agent_audio_torque import PPOAgentAudioTorque, RolloutBatch
from envs.torque.reach_cube_ego_audio_stacked_torque import ReachCubeEgoAudioStackedTorqueEnv

HORIZON = 100
TOTAL_TIMESTEPS = 50_000_000
SAVE_INTERVAL = 5


def train_ppo(args):
    env = ReachCubeEgoAudioStackedTorqueEnv(
        vis=args.vis,
        device=torch.device(args.device),
        num_envs=args.num_envs
    )
    agent = PPOAgentAudioTorque(
        obs_shape   = env.obs_shape,
        action_dim  = env.action_space,
        device      = args.device,
        checkpoint_path=args.checkpoint_path,
        load        = args.load,
        num_envs    = args.num_envs
    )
    writer = SummaryWriter(log_dir=f"runs/{args.task}_train")
    num_updates = TOTAL_TIMESTEPS // (HORIZON * args.num_envs)
    for update in range(1, num_updates+1):
        state = env.reset().to(agent.device)
        buffers = {k: [] for k in ['states','actions','logps','values','rewards','dones']}
        for t in range(HORIZON):
            action, logp, _, value = agent.select_action(state)
            buffers['states'].append(state)
            buffers['actions'].append(action)
            buffers['logps'].append(logp.detach())
            buffers['values'].append(value.detach())
            state, reward, done = env.step(action)
            state = state.to(agent.device)
            buffers['rewards'].append(reward.to(agent.device))
            buffers['dones'].append(done.to(agent.device).float())
            if done.all():
                state = env.reset().to(agent.device)
        # bootstrap
        with torch.no_grad(): mean, std, last_val = agent.model(state)
        buffers['values'].append(last_val)
        buffers['states'].append(state)
        batch = RolloutBatch(
            states=torch.stack(buffers['states']),
            actions=torch.stack(buffers['actions']),
            log_probs=torch.stack(buffers['logps']),
            values=torch.stack(buffers['values']),
            rewards=torch.stack(buffers['rewards']),
            dones=torch.stack(buffers['dones']),
        )
        agent.train(batch)
        if update % SAVE_INTERVAL == 0:
            agent.save_checkpoint()
        mean_reward = batch.rewards.sum(dim=0).mean().item()
        writer.add_scalar('Reward/Mean', mean_reward, update)
        print(f"[Update {update}/{num_updates}] Mean Reward: {mean_reward:.3f}")
    writer.close()


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-v','--vis', action='store_true')
    p.add_argument('-l','--load', action='store_true')
    p.add_argument('--load_path', type=str, default=None)
    p.add_argument('-n','--num_envs', type=int, default=1)
    p.add_argument('-t','--task', type=str, default='ReachCubeEgoAudioStackedTorque')
    p.add_argument('-d','--device', type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    default_ckpt = f"{args.task}_ppo_ckpt.pth"
    if args.load_path:
        args.load = True
        args.checkpoint_path = args.load_path
    else:
        args.checkpoint_path = os.path.join('logs', default_ckpt)
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    backend = gs.cpu if args.device.startswith('cpu') else gs.gpu
    gs.init(backend=backend)
    train_ppo(args)

if __name__ == '__main__':
    main()
