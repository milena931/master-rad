"""Pomoćne funkcije za merenje i čuvanje metrika eksperimenata."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingRun:
    framework: str          # "ray_rllib" ili "stable_baselines3"
    env_id: str             # npr. "CartPole-v1"
    num_workers: int        # broj paralelnih env runner-a / n_envs
    seed: int
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    iterations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    def add_iteration(
        self,
        iteration: int,
        reward: float,
        throughput: float = 0.0,
        extra: dict | None = None,
    ) -> None:
        """
        Dodaje zapis za jednu iteraciju treninga.

        throughput = broj env koraka u sekundi (koraci/sec).
        Ovo je KLJUČNA metrika za analizu paralelizacije u master radu:
        više workera → veći throughput → brže sakupljanje iskustva.
        """
        record: dict[str, Any] = {
            "iteration": iteration,
            "episode_return_mean": reward,
            "throughput_steps_per_sec": round(throughput, 1),
            "elapsed_sec": round(time.time() - self.start_time, 2),
        }
        if extra:
            record.update(extra)
        self.iterations.append(record)

    def finish(self) -> None:
        self.end_time = time.time()

    def avg_throughput(self) -> float:
        vals = [r["throughput_steps_per_sec"] for r in self.iterations if r["throughput_steps_per_sec"] > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration_sec"] = self.duration_sec
        if self.iterations:
            data["final_reward"] = self.iterations[-1]["episode_return_mean"]
            data["best_reward"] = max(r["episode_return_mean"] for r in self.iterations)
            data["avg_throughput_steps_per_sec"] = round(self.avg_throughput(), 1)
        return data


def save_run(run: TrainingRun, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Ime fajla enkodira sve bitne parametre — lako za parsiranje
    name = f"{run.framework}_{run.env_id.replace('/', '_')}_w{run.num_workers}_s{run.seed}.json"
    path = output_dir / name
    path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    return path


def load_runs(results_dir: Path) -> list[dict]:
    """Učitava sve JSON run-ove iz foldera (za analizu i grafike)."""
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("summary_"):
            continue
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs
