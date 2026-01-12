# train_ppo_scratch.py
import os
import argparse
import numpy as np
import torch

from wrappers import make_env
from stash.rollout_buffer import RolloutBuffer
from stash.ppo_agent import PPOAgent


def evaluate(agent, env_id, seed=0, n_episodes=10):
    env = make_env(env_id=env_id, seed=seed)
    rets = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            # deterministic eval: take argmax of policy
            x = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            x = x.to(agent.device)
            with torch.no_grad():
                logits, _ = agent.net(x)
                action = int(torch.argmax(logits, dim=1).item())
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += float(r)
        rets.append(ep_ret)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--save-dir", type=str, default="runs/ppo_scratch")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_env(env_id=args.env_id, seed=args.seed)
    obs_shape = env.observation_space.shape  # (84,84,4)
    n_actions = env.action_space.n

    agent = PPOAgent(
        in_channels=obs_shape[2],
        n_actions=n_actions,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
    )

    buffer = RolloutBuffer(obs_shape=obs_shape, size=args.rollout, gamma=args.gamma, gae_lambda=args.gae_lambda)

    obs, _ = env.reset(seed=args.seed)
    steps = 0
    next_eval = args.eval_every

    while steps < args.total_steps:
        buffer.reset()

        # collect rollout
        for _ in range(args.rollout):
            action, logp, value = agent.act(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            buffer.add(obs, action, reward, done, value, logp)

            obs = next_obs
            steps += 1

            if done:
                obs, _ = env.reset()

            if steps >= args.total_steps:
                break

        # bootstrap value
        last_v = agent.value(obs)
        buffer.full = True
        buffer.compute_returns_and_advantages(last_value=last_v)

        # update PPO
        agent.update(
            obs_uint8=buffer.obs,
            actions=buffer.actions,
            old_logps=buffer.logps,
            advantages=buffer.advantages,
            returns=buffer.returns,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # eval + save
        if steps >= next_eval:
            mean_r, std_r = evaluate(agent, args.env_id, seed=args.seed + 10_000, n_episodes=10)
            print(f"[step {steps}] eval mean={mean_r:.2f} std={std_r:.2f}")
            ckpt = os.path.join(args.save_dir, f"ppo_scratch_step_{steps}.pt")
            torch.save({"net": agent.net.state_dict()}, ckpt)
            next_eval += args.eval_every

    env.close()
    final_path = os.path.join(args.save_dir, "ppo_scratch_final.pt")
    torch.save({"net": agent.net.state_dict()}, final_path)
    print(f"Saved final PPO (scratch) to: {final_path}")


if __name__ == "__main__":
    main()
