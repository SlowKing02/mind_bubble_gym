# Environments

Four standard **Gymnasium** worlds used in Agent Academy. Each teaches a different skill before the next chapter raises difficulty.

## Comparison

| Environment | Gym ID | Actions | Observation | Why it is here |
|-------------|--------|---------|-------------|----------------|
| **Frozen Lake** | `FrozenLake-v1` | 4 (N/S/E/W) | Discrete cell 0–15 | Tiny grid → table fits in memory |
| **Cart Pole** | `CartPole-v1` | 2 (left/right) | 4 continuous numbers | Classic balance; needs function approximation |
| **Mountain Car** | `MountainCar-v0` | 3 (left/nothing/right) | 2 continuous numbers | Sparse reward; needs momentum strategy |
| **Taxi** | `Taxi-v3` | 6 (move + pickup/dropoff) | Discrete 500 states | Planning + longer horizons |

Preview images: `docs/assets/*.png` (regenerate with `python -m workshop.generate_assets`).

---

## Frozen Lake — The Icy Grid

A 4×4 grid. Start top-left, goal bottom-right, holes in between. Each step can slip (stochastic ice) unless you use `is_slippery=False` for teaching.

**Kid version:** "Hop on safe tiles. Fall in a hole and you restart."

**Why researchers use it:** Small enough to print the Q-table on one slide.

---

## Cart Pole — Balance Beam

A cart moves on a track; a pole hinges on top. Push left or right each timestep. +1 reward while the pole stays up; episode ends when it falls.

**Kid version:** "Balance a broomstick on a skateboard."

**Why researchers use it:** Continuous state but low-dimensional; DQN still works with a small network.

---

## Mountain Car — The Hill Climb

A car sits in a valley. Engine is weak: you must rock back and forth to build speed and reach the flag on the right hill.

**Kid version:** "Swing like a swing set until you can reach the top."

**Why researchers use it:** Reward is sparse (only near the flag). Random actions rarely win; policy methods help.

---

## Taxi — City Cab

A 5×5 grid city. Pick up a passenger (color) and drop at the correct destination. -1 per step unless you complete the job (+20).

**Kid version:** "Uber simulator with a grid map."

**Why researchers use it:** Larger discrete space; good capstone for tabular methods and reward-shaping discussions.

---

## Install extras

```bash
pip install "gymnasium[classic-control,box2d,toy-text]"
```

Mountain Car needs `box2d`; Taxi and FrozenLake use `toy_text`.
