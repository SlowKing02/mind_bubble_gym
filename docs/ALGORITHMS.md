# Algorithms

Four algorithms in Agent Academy, ordered from simplest to most modern. Same loop every time; different way of remembering what worked.

![RL loop](assets/rl_loop.svg)

## At a glance

| Algorithm | Type | Best on | Memory | MindBubble chapter |
|-----------|------|---------|--------|-------------------|
| **Q-Learning** | Value-based, tabular | FrozenLake | Q-table `(states × actions)` | Ch. 1 |
| **SARSA** | Value-based, on-policy | Taxi | Same table shape | Ch. 4 |
| **DQN** | Value-based, deep | CartPole | Neural network + replay buffer | Ch. 2 |
| **PPO** | Policy gradient | MountainCar | Policy network (SB3) | Ch. 3 |

Legacy reference: **Double DQN** on Atari pixels in `legacy/atari-master/` (2019 stretch goal).

---

## Q-Learning

**Idea:** Keep a scoreboard `Q(state, action)` — "how good is this move from here?" After each step, nudge the score toward what actually happened.

**Update (plain English):** New estimate = old estimate + learning_rate × (reward + discount × best_future − old).

**Exploration:** Early on, try random moves (ε-greedy). Later, mostly use the best-known move.

**Kid line:** "Write down what you learned on a cheat sheet for each square."

```bash
python -m workshop.q_learning --env FrozenLake-v1 --algorithm q_learning
```

---

## SARSA

**Idea:** Like Q-Learning, but you learn from the move you *will actually take next*, not the theoretically best move.

**Difference:** Safer when exploration is noisy (on-policy).

**Kid line:** "Update your plan based on the next step you're really going to take, not the perfect step."

```bash
python -m workshop.q_learning --env Taxi-v3 --algorithm sarsa --episodes 10000
```

---

## DQN (Deep Q-Network)

**Idea:** Too many states for a table (CartPole uses continuous numbers). Use a small neural network to predict Q-values.

**Extras:** Experience replay (shuffle past tries) and target stability (2019 workshop used vanilla DQN; Atari used **Double DQN** in legacy code).

**Kid line:** "The cheat sheet lives inside a brain instead of on paper."

```bash
python -m workshop.cartpole_dqn --max-runs 400
```

---

## PPO (Proximal Policy Optimization)

**Idea:** Learn a policy directly: "given what I see, what move should I make?" PPO updates in small safe steps so training does not blow up.

**Why now:** Default strong baseline for many Gymnasium environments in the 2020s via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

**Kid line:** "Practice the whole dance move, not just one foot placement."

```bash
pip install "stable-baselines3[extra]>=2.3"
python -m workshop.ppo_demo --env MountainCar-v0 --timesteps 80000
```

---

## What changed since 2019

| Then | Now |
|------|-----|
| OpenAI Gym | Gymnasium (Farama Foundation) |
| Keras DQN in workshop | Same idea; Gymnasium API + notebooks |
| DDQN Atari overnight | Still in `legacy/`; not the default path |
| — | PPO one-liner via SB3 for MountainCar |

See also [RL_NOTES.md](RL_NOTES.md) for practitioner detail.
