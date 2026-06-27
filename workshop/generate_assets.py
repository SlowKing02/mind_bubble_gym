"""Generate preview images for docs and website."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"

PREVIEWS = [
    ("CartPole-v1", "cartpole.png", []),
    ("FrozenLake-v1", "frozen_lake.png", ["render_mode"]),
    ("MountainCar-v0", "mountain_car.png", ["box2d"]),
    ("Taxi-v3", "taxi.png", ["toy_text"]),
]


def _frame(env_id: str, extras: list[str]) -> np.ndarray:
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset()
    for _ in range(5):
        env.step(env.action_space.sample())
    frame = env.render()
    env.close()
    return frame


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for env_id, filename, extras in PREVIEWS:
        try:
            frame = _frame(env_id, extras)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {env_id}: {exc}")
            continue
        plt.imsave(ASSETS / filename, frame)
        print(f"wrote {ASSETS / filename}")


if __name__ == "__main__":
    main()
