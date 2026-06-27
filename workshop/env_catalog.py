"""Gymnasium environments used in the MindBubble Agent Academy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvCard:
    id: str
    gym_id: str
    nickname: str
    kid_pitch: str
    algorithm: str
    notebook: str
    extra: str = ""


ENV_CATALOG: list[EnvCard] = [
    EnvCard(
        id="frozen_lake",
        gym_id="FrozenLake-v1",
        nickname="The Icy Grid",
        kid_pitch="Cross a frozen pond without falling through holes. Pick safe squares.",
        algorithm="Q-Learning (table of best moves)",
        notebook="notebooks/02_frozen_lake_q_learning.ipynb",
    ),
    EnvCard(
        id="cartpole",
        gym_id="CartPole-v1",
        nickname="Balance Beam",
        kid_pitch="Keep a pole upright on a rolling cart. Small fixes, big consequences.",
        algorithm="DQN (neural network Q-values)",
        notebook="notebooks/03_cartpole_dqn.ipynb",
    ),
    EnvCard(
        id="mountain_car",
        gym_id="MountainCar-v0",
        nickname="The Hill Climb",
        kid_pitch="A tiny car must swing back and forth to build speed and reach the flag.",
        algorithm="PPO (policy gradient)",
        notebook="notebooks/04_mountain_car_ppo.ipynb",
        extra="box2d",
    ),
    EnvCard(
        id="taxi",
        gym_id="Taxi-v3",
        nickname="City Cab",
        kid_pitch="Pick up a passenger and drop them off. Shortest route wins.",
        algorithm="SARSA (learn while exploring)",
        notebook="notebooks/05_taxi_sarsa.ipynb",
    ),
]


def get_env(gym_id: str):
    import gymnasium as gym

    extras = {
        "MountainCar-v0": ["box2d"],
        "Taxi-v3": ["toy_text"],
    }
    return gym.make(gym_id, render_mode="rgb_array")
