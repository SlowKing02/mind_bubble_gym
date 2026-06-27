"""
CartPole DQN — the algorithm we used at MindBubble (2019), updated for Gymnasium.

Train an agent to keep a pole upright. Same idea as the workshop: try actions,
get rewards, learn from mistakes.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from workshop.score_logger import ScoreLogger

ROOT = Path(__file__).resolve().parents[1]
ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 100_000
BATCH_SIZE = 32
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        self.action_dim = action_dim
        self.exploration = EXPLORATION_MAX
        self.memory: deque = deque(maxlen=MEMORY_SIZE)
        self.model = Sequential(
            [
                Dense(24, input_shape=(obs_dim,), activation="relu"),
                Dense(24, activation="relu"),
                Dense(action_dim, activation="linear"),
            ]
        )
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.exploration:
            return random.randrange(self.action_dim)
        q = self.model.predict(state, verbose=0)
        return int(np.argmax(q[0]))

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def replay(self) -> None:
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += GAMMA * float(np.max(self.model.predict(next_state, verbose=0)[0]))
            q = self.model.predict(state, verbose=0)
            q[0][action] = target
            self.model.fit(state, q, verbose=0, epochs=1)
        self.exploration = max(EXPLORATION_MIN, self.exploration * EXPLORATION_DECAY)


def train(max_runs: int, render: bool, out_dir: Path) -> None:
    env = gym.make(ENV_NAME, render_mode="human" if render else None)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(obs_dim, action_dim)
    logger = ScoreLogger(ENV_NAME, out_dir)

    for run in range(1, max_runs + 1):
        state, _ = env.reset()
        state = np.reshape(state, [1, obs_dim])
        steps = 0
        while True:
            steps += 1
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            shaped = reward if not done else -reward
            next_state = np.reshape(next_state, [1, obs_dim])
            agent.remember(state, action, shaped, next_state, done)
            state = next_state
            if done:
                if logger.add_score(steps, run):
                    env.close()
                    return
                break
            agent.replay()
    env.close()
    print(f"Stopped after {max_runs} runs without hitting solve threshold.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CartPole DQN workshop demo")
    parser.add_argument("--max-runs", type=int, default=400, help="Stop after this many episodes")
    parser.add_argument("--render", action="store_true", help="Show the cart and pole window")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "cartpole",
        help="Where to write score charts",
    )
    args = parser.parse_args()
    train(args.max_runs, args.render, args.out)


if __name__ == "__main__":
    main()
