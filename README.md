# mind_bubble_gym

Teaching demo from a **MindBubble** reinforcement learning workshop (metro Atlanta, July 2019). Middle school students trained agents in [OpenAI Gym](https://gymnasium.farama.org/) before "AI" was on every slide deck.

This repo is the cleaned-up version: a runnable **CartPole** demo, workshop notes for students, and the original 2019 Atari/DDQN code in `legacy/`.

Website write-up: [Reinforcement learning at MindBubble](https://brysonbonham.com/posts/archive/mindbubble-rl-2019.html)

## The idea (30 seconds)

A computer plays a game by trial and error:

1. **Agent** picks left or right  
2. **Environment** moves the cart and pole, returns a **reward**  
3. The agent updates its strategy and tries again  

CartPole is the classic starter game: keep a pole balanced on a moving cart. The lesson I still use in production work: **what you reward is what you get**. A badly shaped reward teaches the wrong behavior fast.

## Run the workshop demo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m workshop.cartpole_dqn --max-runs 400
python -m workshop.cartpole_dqn --render --max-runs 50   # watch it learn
```

Charts land in `output/cartpole/scores.png`.

### Optional: modern PPO baseline

Policy-gradient methods like **PPO** are the usual 2020s baseline when you want strong results without hand-tuning a Q-network:

```bash
pip install 'stable-baselines3[extra]>=2.3'
python -m workshop.ppo_demo --timesteps 50000
```

Same CartPole environment; different algorithm. See [docs/RL_NOTES.md](docs/RL_NOTES.md) for how this relates to what we taught in 2019.

## Docs

| File | Audience |
|------|----------|
| [docs/WORKSHOP.md](docs/WORKSHOP.md) | Students and teachers — vocabulary, lesson flow |
| [docs/RL_NOTES.md](docs/RL_NOTES.md) | Practitioners — DQN vs DDQN vs PPO, Gym → Gymnasium |

## Layout

```text
workshop/           CartPole DQN demo + optional PPO comparator
docs/               Workshop guide and RL notes
legacy/             2019 scripts (Atari DDQN, genetic evolution, original CartPole)
output/             Training charts (gitignored)
```

## Legacy Atari code

The 2019 workshop also walked through **Double DQN** on Atari (Breakout, Space Invaders). That code lives under `legacy/atari-master/`. It targets old `gym` + TensorFlow 1.x era dependencies; use it as reference, not the default install path.

## License

MIT. MindBubble is an independent nonprofit; this repo is my teaching materials, not an official MindBubble product.
