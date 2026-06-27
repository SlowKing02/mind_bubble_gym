# RL notes — 2019 workshop vs today

Quick map for practitioners landing on this repo from my website.

## What we taught in 2019

| Piece | Workshop choice |
|-------|-----------------|
| Environment API | OpenAI **Gym** |
| Starter game | **CartPole-v1** with tabular/neural **DQN** |
| Stretch goal | **Double DQN (DDQN)** on Atari pixels |
| Alternative | **Genetic evolution** on network weights (`legacy/atari-master`) |

DQN learns an action-value function Q(state, action): "how good is moving left if the pole is leaning this way?" Experience replay shuffles past tries so learning does not forget early mistakes.

DDQN adds a **target network** so the agent does not chase its own noisy estimates (the trick DeepMind used for Atari).

## What people use now (2024–2026)

| Topic | Today | This repo |
|-------|-------|-----------|
| Gym API | **[Gymnasium](https://gymnasium.farama.org/)** (Gym maintained by Farama) | `workshop/cartpole_dqn.py` |
| Fast baseline | **PPO**, SAC, TD3 via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) | optional `workshop/ppo_demo.py` |
| Atari | Same games, often `ALE/` namespace in Gymnasium | `legacy/atari-master` (2019 code) |
| Scaling | Distributed RL, offline RL, world models | Out of scope for a middle school demo |

**PPO (Proximal Policy Optimization)** is a policy-gradient method: learn a policy directly instead of Q-values. It is often the first serious baseline on new environments because it is stable and needs less babysitting than vanilla policy gradients.

You do not need PPO to teach the core loop. It is here so you can show students "same game, newer tool" in one command.

## Production lesson (unchanged)

The workshop punchline still applies to enterprise ML:

- Define the **reward** carefully; agents optimize the metric you give them, not the outcome you meant.  
- Separate **training** metrics from **deployment** metrics.  
- Exploration vs exploitation shows up in A/B tests, bandits, and LLM eval harnesses—not just games.

## File map

| Path | Algorithm |
|------|-----------|
| `workshop/cartpole_dqn.py` | DQN + experience replay (2019 teaching script, Gymnasium port) |
| `workshop/ppo_demo.py` | PPO via Stable-Baselines3 |
| `legacy/CP_master/` | Original CartPole fork (Python 2 era) |
| `legacy/atari-master/` | DDQN + genetic evolution on Atari |

## References

- Mnih et al., DQN (2015)  
- van Hasselt et al., Double DQN (2015)  
- Schulman et al., PPO (2017)  
- [Gymnasium migration guide](https://gymnasium.farama.org/introduction/migration_guide/)  
