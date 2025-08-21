import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'  # comment out for Windows or MacOS

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter

from agents.ik.ppo_agent_multimodal_IKsimple import PPOAgentMultimodal
from envs.ik.reach_cube_ego_multimodal_stacked import ReachCubeEgoMultimodalStackedEnv


# -------------------------
# Helpers
# -------------------------
def build_noise_config(args):
    """
    Merge new-style split flags and old-style '--noise visual 3 audio 5' list.
    New flags win if both are provided.
    """
    cfg = {}

    # Old style: --noise visual 3 audio 5
    if args.noise:
        try:
            tmp = {args.noise[i]: float(args.noise[i + 1]) for i in range(0, len(args.noise), 2)}
            # env supports both legacy keys and *_noise_level keys
            if "visual" in tmp:
                cfg["visual"] = tmp["visual"]
            if "audio" in tmp:
                cfg["audio"] = tmp["audio"]
        except Exception as e:
            print(f"[WARN] Could not parse --noise list: {e}. Ignoring.")

    # New style (preferred)
    if args.visual_noise_level is not None:
        cfg["visual_noise_level"] = float(args.visual_noise_level)
    if args.audio_noise_level is not None:
        cfg["audio_noise_level"] = float(args.audio_noise_level)

    return cfg


def make_env(args):
    noise_config = build_noise_config(args)
    env = ReachCubeEgoMultimodalStackedEnv(
        vis=args.vis,
        device=args.device,
        num_envs=args.num_envs,
        # success/done parity
        success_thresh=args.success_thresh,
        success_bonus=args.success_bonus,
        report_success_as_done=not args.no_done_on_success,
        # curriculum
        inference_mode=(args.mode == "inference"),
        # debugging: force object rotation each episode if desired
        #deterministic_object_cycle=args.deterministic_object_cycle,
        # adjustable cadence
        render_every=args.render_every,
        show_every=args.show_every,
        # noise
        noise_config=noise_config,
    )
    print("Created multimodal environment:", env)
    return env


# -------------------------
# TRAIN
# -------------------------
def train_ppo(args):
    env = make_env(args)

    agent = PPOAgentMultimodal(
        obs_shape_vision=env.obs_shape_vision,
        obs_shape_audio=env.obs_shape_audio,
        action_shape=env.action_space,
        lr=1e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        device=args.device,
        load=args.load,
        num_envs=args.num_envs,
        checkpoint_path=args.checkpoint
    )

    writer = SummaryWriter(log_dir=f"runs/{args.task}")

    # You were using an episode-style loop; we’ll keep that (your agent.train expects lists).
    for episode in range(args.max_episodes):
        state_v, state_a = env.reset()
        total_reward = torch.zeros(env.num_envs, device=args.device)
        done_array = torch.zeros(env.num_envs, dtype=torch.bool, device=args.device)

        states_v, states_a, actions, rewards, dones = [], [], [], [], []

        for _ in range(args.max_steps):
            action, _ = agent.select_action(state_v, state_a)
            (next_v, next_a), reward, done = env.step(action)

            states_v.append(state_v)
            states_a.append(state_a)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state_v, state_a = next_v, next_a
            total_reward += reward
            done_array |= done
            if done_array.all():
                break

        agent.train(states_v, states_a, actions, rewards, dones)

        if episode % args.save_every == 0:
            agent.save_checkpoint()
            print(f"[Episode {episode}] Checkpoint saved to {args.checkpoint}")

        mean_reward = total_reward.mean().item()
        writer.add_scalar('Reward/Mean', mean_reward, episode)
        print(f"[Episode {episode}] Mean Reward: {mean_reward:.4f}")

    writer.close()


# -------------------------
# INFERENCE
# -------------------------
@torch.no_grad()
def inference_ppo(args):
    # Inference env cycles positions deterministically when inference_mode=True.
    env = make_env(args)

    agent = PPOAgentMultimodal(
        obs_shape_vision=env.obs_shape_vision,
        obs_shape_audio=env.obs_shape_audio,
        action_shape=env.action_space,
        device=args.device,
        load=True,  # force load (no training)
        num_envs=args.num_envs,
        checkpoint_path=args.checkpoint
    )
    if hasattr(agent, "eval_mode"):
        agent.eval_mode(True)

    writer = SummaryWriter(log_dir=f"runs/{args.task}_inference")

    deterministic = True
    N = args.num_envs
    MAX_STEPS = args.max_steps

    total_successes = torch.zeros(N, dtype=torch.long)
    total_steps_to_success = torch.zeros(N, dtype=torch.long)
    total_rewards = torch.zeros(N, dtype=torch.float32)

    def act(agent, sv, sa, deterministic=True):
        # Prefer agent.act if available; otherwise fall back to select_action
        if hasattr(agent, "act"):
            return agent.act(sv, sa, deterministic=deterministic)
        a, _ = agent.select_action(sv, sa)
        return a

    for ep in range(args.num_episodes):
        sv, sa = env.reset()
        steps = 0
        ep_reward = torch.zeros(N, dtype=torch.float32)
        reached = torch.zeros(N, dtype=torch.bool)
        steps_to_success = torch.zeros(N, dtype=torch.long)

        while steps < MAX_STEPS:
            a = act(agent, sv, sa, deterministic=deterministic)
            (sv_next, sa_next), reward, done = env.step(a)

            ep_reward += reward.cpu()
            steps += 1

            newly = (~reached) & done.cpu()
            if newly.any():
                steps_to_success[newly] = steps
            reached |= done.cpu()

            sv, sa = sv_next, sa_next

            if reached.all():
                break

        # episode accounting
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
                f"Steps: {steps}, Mean Reward: {mean_reward:.3f}, "
                f"Success envs: {reached.sum().item()}/{N}, "
                f"Avg steps-to-30cm: {avg_steps_this_ep:.1f}"
            )
        else:
            print(
                f"[Inference {ep+1}/{args.num_episodes}] "
                f"Steps: {steps}, Mean Reward: {mean_reward:.3f}, "
                f"Success envs: 0/{N} (no reach within 30cm)"
            )

    writer.close()

    # Final summary
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
    print(f"Avg steps-to-30cm over successes: {avg_steps_over_successes:.1f}")
    print(f"Mean reward per env (avg over episodes): {avg_reward_per_env:.3f}")


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-v', '--vis', action='store_true', help='Enable visualization')
    p.add_argument('-l', '--load', nargs='?', const='default',
                   help='Load checkpoint: -l for default, -l PATH for custom')
    p.add_argument('-n', '--num_envs', type=int, default=1, help='Number of envs')
    p.add_argument('-t', '--task', type=str,
                   default='ReachCubeEgoMultimodalStacked', help='Task name (for logs/checkpoint naming)')
    p.add_argument('-d', '--device', type=str, default='cuda', help='cpu or cuda')
 
    # Training loop
    p.add_argument('--max_episodes', type=int, default=1_000_000, help='Max episodes (train)')
    p.add_argument('--max_steps', type=int, default=100, help='Max steps per episode')
    p.add_argument('--save_every', type=int, default=5, help='Episodes between saves')

    # Noise (new flags; old --noise list also supported)
    p.add_argument('--visual_noise_level', type=float, default=0.0, help='RGB noise std-dev')
    p.add_argument('--audio_noise_level', type=float, default=0.0, help='Audio Gaussian noise std-dev')
    p.add_argument('--noise', nargs='+', default=[],
                   help="Legacy: --noise visual 3 audio 5 (new flags override these)")

    # Success / reward parity
    p.add_argument('--success_thresh', type=float, default=0.3001, help='Success distance [m]')
    p.add_argument('--success_bonus', type=float, default=20.0, help='Bonus on success')
    p.add_argument('--no_done_on_success', action='store_true',
                   help="If set, env won't terminate on success (report_success_as_done=False)")

    # Env cadence
    p.add_argument('--render_every', type=int, default=5, help='Frame skip for rendering/audio')
    p.add_argument('--show_every', type=int, default=10, help='Plot/playback cadence (steps)')

    # Debug: prove shapes change episode-by-episode
    p.add_argument('--deterministic_object_cycle', action='store_true',
                   help='Cycle cube→sphere→bunny→dragon deterministically each episode')

    # Mode
    p.add_argument('-m', '--mode', choices=['train', 'inference'], default='train', help='Run mode')
    p.add_argument('--num_episodes', type=int, default=100,
                   help='Episodes for inference mode')

    return p.parse_args()


def main():
    args = parse_args()

    # unify device
    args.device = torch.device(args.device)

    # checkpoint path
    default = Path('logs') / f"{args.task}_ppo_checkpoint.pth"
    if args.load:
        args.checkpoint = default if args.load == 'default' else Path(args.load)
        if not args.checkpoint.is_file():
            print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
            sys.exit(1)
        print("Loading checkpoint from:", args.checkpoint)
    else:
        args.checkpoint = default
        print("No checkpoint provided; starting from scratch.")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    backend = gs.cpu if args.device.type == 'cpu' else gs.gpu
    gs.init(backend=backend)

    if args.mode == 'train':
        train_ppo(args)
    else:
        inference_ppo(args)


if __name__ == '__main__':
    main()
