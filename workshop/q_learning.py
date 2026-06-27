"""Tabular Q-Learning and SARSA for discrete Gymnasium environments."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def train_tabular(
    env_id: str,
    algorithm: str,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
    out_dir: Path,
) -> dict[str, float]:
    env = gym.make(env_id)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q = np.zeros((n_states, n_actions))
    rewards_per_ep: list[float] = []
    eps = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if algorithm == "q_learning":
                best_next = np.max(q[next_state])
                q[state, action] += alpha * (reward + gamma * best_next - q[state, action])
            else:  # sarsa
                if random.random() < eps:
                    next_action = env.action_space.sample()
                else:
                    next_action = int(np.argmax(q[next_state]))
                q[state, action] += alpha * (
                    reward + gamma * q[next_state, next_action] - q[state, action]
                )
            state = next_state
            total += reward
        eps = max(0.01, eps * epsilon_decay)
        rewards_per_ep.append(total)

    env.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_per_ep, alpha=0.4, label="episode reward")
    window = min(50, len(rewards_per_ep))
    if window > 1:
        smooth = np.convolve(rewards_per_ep, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(rewards_per_ep)), smooth, label=f"{window}-ep avg")
    plt.title(f"{algorithm.upper()} on {env_id}")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.legend()
    plt.tight_layout()
    chart = out_dir / f"{algorithm}_{env_id.replace('-', '_').lower()}.png"
    plt.savefig(chart, dpi=120)
    plt.close()
    tail = rewards_per_ep[-min(100, len(rewards_per_ep)) :]
    return {"episodes": episodes, "mean_reward_last100": float(np.mean(tail)), "chart": str(chart)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular RL on FrozenLake or Taxi")
    parser.add_argument("--env", default="FrozenLake-v1")
    parser.add_argument("--algorithm", choices=["q_learning", "sarsa"], default="q_learning")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "tabular")
    args = parser.parse_args()
    stats = train_tabular(
        args.env,
        args.algorithm,
        args.episodes,
        args.alpha,
        args.gamma,
        args.epsilon,
        args.epsilon_decay,
        args.out,
    )
    print(stats)


if __name__ == "__main__":
    main()
