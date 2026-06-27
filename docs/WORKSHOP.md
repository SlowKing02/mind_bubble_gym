# Workshop guide

For middle school STEAM sessions. Adjust timing for your group; we had about 90 minutes with laptops.

## Vocabulary

| Word | Plain English |
|------|----------------|
| **Agent** | The program that chooses actions |
| **Environment** | The game world (CartPole, Breakout, etc.) |
| **State** | What the agent can "see" right now (cart position, pole angle, …) |
| **Action** | A choice the agent can make (push cart left or right) |
| **Reward** | Points for good behavior (+1 each step the pole stays up) |
| **Policy** | The agent's strategy: "when I see X, I do Y" |

## CartPole in one sentence

Keep the pole from falling by sliding the cart underneath it.

Gymnasium considers the game **solved** when the agent averages **195+ steps** for 100 games in a row (the pole stays up that long).

## Lesson flow (what we did at MindBubble)

1. **Play humans first** — Have someone balance a ruler on their hand. Same problem: small corrections, delayed consequences.

2. **Watch random actions** — Run the env with no learning. The pole drops in a few steps. Count the steps aloud.

3. **Introduce the loop** — Draw on the board:
   ```text
   state → action → reward → new state → …
   ```

4. **Train DQN** — Run `python -m workshop.cartpole_dqn`. Explain exploration (random tries early) vs exploitation (using what worked).

5. **Read the chart** — Open `output/cartpole/scores.png`. Scores should climb over runs. Celebrate when the average crosses the solve line.

6. **Reward design discussion** — Ask: "What if we gave +100 for moving right?" (Agent might ignore the pole.) Tie to real systems: recommender clicks vs long-term satisfaction, chatbot length vs helpfulness.

## Optional stretch goals

- `--render` so students see the cart move  
- Compare DQN steps-to-solve vs PPO (`workshop/ppo_demo.py`) on the same game  
- Show a Breakout gif from `legacy/atari-master/assets/` and explain pixels vs CartPole numbers  

## Setup tips

- CartPole runs on CPU; no GPU required  
- `--max-runs 400` usually enough to see improvement; full solve may take longer  
- If TensorFlow install is heavy on school machines, use Google Colab with the same commands  

## Further reading (practitioner)

- [docs/RL_NOTES.md](RL_NOTES.md) — what changed in RL tooling since 2019  
- [MindBubble](https://mindbubble.org/) — the nonprofit that hosted the session  
