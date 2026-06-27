# mind_bubble_gym

**Agent Academy** — a reinforcement learning teaching demo from the [MindBubble](https://mindbubble.org/) workshop (metro Atlanta, July 2019). Middle school students learn the same loop used in production ML: act, observe, reward, improve.

Website story: [Reinforcement learning at MindBubble](https://brysonbonham.com/posts/archive/mindbubble-rl-2019.html)

## Start here

| Audience | Path |
|----------|------|
| Students / teachers | [notebooks/](notebooks/) (Jupyter walkthrough) |
| Session plan | [docs/STORY.md](docs/STORY.md) |
| Four worlds | [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) |
| Four algorithms | [docs/ALGORITHMS.md](docs/ALGORITHMS.md) |

![RL loop](docs/assets/rl_loop.svg)

## Agent Academy curriculum

| Chapter | World | Algorithm | Notebook |
|---------|-------|-----------|----------|
| 0 | CartPole | Random baseline | [01_welcome_the_loop.ipynb](notebooks/01_welcome_the_loop.ipynb) |
| 1 | Frozen Lake | Q-Learning | [02_frozen_lake_q_learning.ipynb](notebooks/02_frozen_lake_q_learning.ipynb) |
| 2 | Cart Pole | DQN | [03_cartpole_dqn.ipynb](notebooks/03_cartpole_dqn.ipynb) |
| 3 | Mountain Car | PPO | [04_mountain_car_ppo.ipynb](notebooks/04_mountain_car_ppo.ipynb) |
| 4 | Taxi | SARSA + reward design | [05_taxi_sarsa.ipynb](notebooks/05_taxi_sarsa.ipynb) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m workshop.generate_assets   # preview PNGs for docs / site
jupyter lab notebooks/
```

## CLI (same lessons without notebooks)

```bash
python -m workshop.q_learning --env FrozenLake-v1 --algorithm q_learning
python -m workshop.cartpole_dqn --max-runs 400
python -m workshop.ppo_demo --env MountainCar-v0 --timesteps 80000
python -m workshop.q_learning --env Taxi-v3 --algorithm sarsa --episodes 10000
```

## Repo layout

```text
notebooks/     chapter walkthroughs (kid-friendly markdown + runnable cells)
workshop/      training scripts (tabular, DQN, PPO)
docs/          storyline, environments, algorithms, assets/
legacy/        2019 Atari DDQN + original forks
output/        charts (gitignored)
```

## Legacy

2019 **Double DQN** on Atari (Breakout, Space Invaders) lives in `legacy/atari-master/`. Reference only; Agent Academy uses the Gymnasium paths above.

## License

MIT. Teaching materials by Bryson Bonham; not an official MindBubble product.
