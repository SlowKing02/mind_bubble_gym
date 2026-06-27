"""Plot CartPole scores during training."""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOLVE_AVERAGE = 195
SOLVE_WINDOW = 100


class ScoreLogger:
    def __init__(self, env_name: str, out_dir: Path) -> None:
        self.env_name = env_name
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.scores: deque[int] = deque(maxlen=SOLVE_WINDOW)
        self.scores_csv = out_dir / "scores.csv"
        self.scores_png = out_dir / "scores.png"
        self.solved_csv = out_dir / "solved.csv"
        self.solved_png = out_dir / "solved.png"
        for path in (self.scores_csv, self.scores_png, self.solved_csv, self.solved_png):
            if path.exists():
                path.unlink()

    def add_score(self, score: int, run: int) -> bool:
        self._append_csv(self.scores_csv, score)
        self._plot_scores()
        self.scores.append(score)
        avg = mean(self.scores)
        print(f"Run {run}: score={score}  last-{len(self.scores)} avg={avg:.1f}")
        if len(self.scores) >= SOLVE_WINDOW and avg >= SOLVE_AVERAGE:
            solved_at = run - SOLVE_WINDOW
            print(f"Solved CartPole-v1 (avg >= {SOLVE_AVERAGE} over {SOLVE_WINDOW} runs) at run {solved_at}.")
            self._append_csv(self.solved_csv, solved_at)
            self._plot_solved()
            return True
        return False

    def _append_csv(self, path: Path, value: int) -> None:
        with path.open("a", newline="") as fh:
            csv.writer(fh).writerow([value])

    def _plot_scores(self) -> None:
        x, y = self._read_csv(self.scores_csv)
        plt.figure()
        plt.plot(x, y, label="score per run")
        window = min(SOLVE_WINDOW, len(y))
        if window:
            plt.plot(
                x[-window:],
                [np.mean(y[-window:])] * window,
                linestyle="--",
                label=f"last {window} runs avg",
            )
        plt.axhline(SOLVE_AVERAGE, linestyle=":", label=f"{SOLVE_AVERAGE} solve line")
        plt.title(self.env_name)
        plt.xlabel("run")
        plt.ylabel("steps")
        plt.legend(loc="upper left")
        plt.savefig(self.scores_png, bbox_inches="tight")
        plt.close()

    def _plot_solved(self) -> None:
        x, y = self._read_csv(self.solved_csv)
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.title(f"{self.env_name} — runs to solve")
        plt.xlabel("trial")
        plt.ylabel("run index")
        plt.savefig(self.solved_png, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _read_csv(path: Path) -> tuple[list[int], list[int]]:
        with path.open() as fh:
            rows = list(csv.reader(fh))
        x = list(range(len(rows)))
        y = [int(r[0]) for r in rows]
        return x, y
