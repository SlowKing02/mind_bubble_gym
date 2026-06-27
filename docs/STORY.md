# Agent Academy — storyline

A four-chapter path from "random button masher" to "agent that plans ahead." Built for the **MindBubble** workshop (metro Atlanta, 2019), updated for **Gymnasium** and notebook walkthroughs.

## Cast

| Name | Role |
|------|------|
| **The Agent** | Your learner program. It picks actions. |
| **The Environment** | The game world (lake, cart, hill, city). |
| **The Reward** | Points the game gives. The agent chases these. |
| **You** | The coach. You shape rewards and pick algorithms. |

## Chapter map

| # | Notebook | World | Algorithm | Kid pitch |
|---|----------|-------|-----------|-----------|
| 0 | [01_welcome_the_loop.ipynb](../notebooks/01_welcome_the_loop.ipynb) | CartPole | Random | "What happens with zero training?" |
| 1 | [02_frozen_lake_q_learning.ipynb](../notebooks/02_frozen_lake_q_learning.ipynb) | FrozenLake | **Q-Learning** | Ice grid: memorize the safe path |
| 2 | [03_cartpole_dqn.ipynb](../notebooks/03_cartpole_dqn.ipynb) | CartPole | **DQN** | Too many states for a table — use a brain |
| 3 | [04_mountain_car_ppo.ipynb](../notebooks/04_mountain_car_ppo.ipynb) | MountainCar | **PPO** | Swing wide, then conquer the hill |
| 4 | [05_taxi_sarsa.ipynb](../notebooks/05_taxi_sarsa.ipynb) | Taxi | **SARSA** | City cab + reward design capstone |

![RL loop](assets/rl_loop.svg)

## Production lesson (same as 2019)

Middle schoolers grasp it fast: **the agent optimizes whatever you score**. Production teams forget that at scale.

- Reward passenger pickup only → agent drives in circles near the depot  
- Reward pole-up steps only → agent jiggles forever without finishing  
- Reward clicks only → model chases clicks, not helpful answers  

That thread runs from FrozenLake through Taxi and still applies to LLM eval design.

## Run order in a 90-minute session

1. Read the loop diagram aloud (2 min)  
2. Notebook 01 — watch random CartPole fail (10 min)  
3. Notebook 02 — Q-Learning table fills in on FrozenLake (20 min)  
4. Notebook 03 — DQN chart climbs on CartPole; optional `--render` (25 min)  
5. Notebook 04 or 05 — pick one stretch goal (20 min)  
6. Reward design discussion using Taxi scenarios (10 min)  

## CLI equivalents

```bash
python -m workshop.q_learning --env FrozenLake-v1 --algorithm q_learning
python -m workshop.cartpole_dqn --max-runs 400
python -m workshop.ppo_demo --env MountainCar-v0
python -m workshop.q_learning --env Taxi-v3 --algorithm sarsa --episodes 10000
```

## Website

Workshop context and photos: [Reinforcement learning at MindBubble](https://brysonbonham.com/posts/archive/mindbubble-rl-2019.html)
