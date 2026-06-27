"""
Optional: train CartPole with PPO (Stable-Baselines3).

Same environment, different algorithm. Policy-gradient methods like PPO are
what many teams reach for today when they need a solid baseline fast.

  pip install 'stable-baselines3[extra]>=2.3'
  python -m workshop.ppo_demo
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="CartPole PPO comparator")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium env id")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--out", type=Path, default=Path("output/cartpole_ppo"))
    args = parser.parse_args()

    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.evaluation import evaluate_policy
    except ImportError as exc:
        raise SystemExit(
            "Install optional deps: pip install 'stable-baselines3[extra]>=2.3'"
        ) from exc

    slug = args.env.replace("-", "_").lower()
    args.out = args.out if args.out.name != "cartpole_ppo" else Path(f"output/{slug}_ppo")
    args.out.mkdir(parents=True, exist_ok=True)
    env = gym.make(args.env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    model.save(args.out / f"ppo_{slug}")
    print(f"PPO on {args.env}: {mean_reward:.1f} +/- {std_reward:.1f} (20 eval episodes)")
    env.close()


if __name__ == "__main__":
    main()
