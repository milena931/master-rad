"""
Generiše grafike iz sačuvanih eksperimenata za master rad.

Grafici:
  Po okruženju:
    - Kriva učenja (reward vs iteracija)
    - Throughput (koraci/sec) vs broj workera
    - Speedup bar chart (sa idealnim linearnim speedupom)

  Ukupno poređenje:
    - Sva tri okruženja

Pokretanje:
  python plot_results.py                 # koristi results/ automatski
  python plot_results.py --env cartpole  # samo jedno okruženje
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).parent.parent
RESULTS_DEFAULT = ROOT / "results"

# Boje po envu za konzistentnost kroz grafike
ENV_COLORS = {
    "cartpole": "steelblue",
    "lunarlander": "darkorange",
    "bipedalwalker": "mediumseagreen",
}
RAY_COLOR = "steelblue"


def load_runs(results_dir: Path, framework: str | None = None, env_id: str | None = None) -> list[dict]:
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("summary_"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if framework and data.get("framework") != framework:
            continue
        if env_id and data.get("env_id") != env_id:
            continue
        runs.append(data)
    return runs


def plot_single_env(env_id: str, results_dir: Path, output_path: Path) -> None:
    """
    3-panelni grafik za jedno okruženje:
      1. Kriva učenja  2. Throughput  3. Speedup
    """
    ray_runs = sorted(
        load_runs(results_dir, framework="ray_rllib", env_id=env_id),
        key=lambda r: r["num_workers"],
    )

    if not ray_runs:
        print(f"  Nema rezultata za {env_id}, preskačem.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Skalabilnost — {env_id}", fontsize=13, fontweight="bold")

    blues = plt.cm.Blues([0.45, 0.60, 0.75, 0.90])

    # --- Panel 1: Kriva učenja ---
    ax = axes[0]
    ax.set_title("Kriva učenja (reward)")
    for i, run in enumerate(ray_runs):
        iters = [r["iteration"] for r in run["iterations"]]
        rewards = [r["episode_return_mean"] for r in run["iterations"]]
        ax.plot(iters, rewards, "o-", markersize=3, color=blues[i % len(blues)],
                label=f"Ray w={run['num_workers']}")
    ax.set_xlabel("Iteracija / rollout")
    ax.set_ylabel("Episode return mean")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Throughput ---
    ax = axes[1]
    ax.set_title("Throughput (koraci/sec)")
    if ray_runs:
        x_r = [r["num_workers"] for r in ray_runs]
        y_r = [r.get("avg_throughput_steps_per_sec", 0) for r in ray_runs]
        ax.plot(x_r, y_r, "o-", color=RAY_COLOR, linewidth=2, markersize=6, label="Ray RLlib")
    ax.set_xlabel("Workers / n_envs")
    ax.set_ylabel("Koraci/sec")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Speedup ---
    ax = axes[2]
    ax.set_title("Speedup (wall-clock vreme)")
    if len(ray_runs) >= 2:
        baseline_t = ray_runs[0]["duration_sec"]
        workers = [r["num_workers"] for r in ray_runs]
        speedups = [baseline_t / r["duration_sec"] for r in ray_runs]
        bars = ax.bar([str(w) for w in workers], speedups,
                      color=[blues[i % len(blues)] for i in range(len(workers))],
                      label="Ray RLlib")
        # Idealni linearni speedup
        ideal = [w / ray_runs[0]["num_workers"] for w in workers]
        ax.plot([str(w) for w in workers], ideal, "r--", alpha=0.5, label="Idealni linearni")
        for bar, sp in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{sp:.2f}x", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Broj env_runners")
    ax.set_ylabel("Speedup")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafik: {output_path}")


def plot_env_comparison(results_dir: Path, output_path: Path) -> None:
    """
    Poredni grafik sva 3 okruženja — throughput i speedup.
    Ovo ide direktno u poglavlje 4 master rada.
    """
    env_labels = {
        "CartPole-v1": "CartPole",
        "LunarLander-v3": "LunarLander",
        "BipedalWalker-v3": "BipedalWalker",
    }
    all_ray_runs = load_runs(results_dir, framework="ray_rllib")
    if not all_ray_runs:
        print("  Nema Ray rezultata za poredni grafik.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Poređenje okruženja — Ray RLlib skalabilnost", fontsize=13, fontweight="bold")

    colors = list(ENV_COLORS.values())

    # Grupiši po env-u
    envs_seen: dict[str, list[dict]] = {}
    for r in all_ray_runs:
        eid = r["env_id"]
        envs_seen.setdefault(eid, []).append(r)

    # Panel 1: Throughput po workeru po envu
    ax = axes[0]
    ax.set_title("Throughput po broju workera")
    for i, (env_id, runs) in enumerate(sorted(envs_seen.items())):
        runs = sorted(runs, key=lambda r: r["num_workers"])
        x = [r["num_workers"] for r in runs]
        y = [r.get("avg_throughput_steps_per_sec", 0) for r in runs]
        label = env_labels.get(env_id, env_id)
        ax.plot(x, y, "o-", color=colors[i % len(colors)], linewidth=2, markersize=6, label=label)
    ax.set_xlabel("Broj env_runners")
    ax.set_ylabel("Koraci/sec")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Speedup po envu
    ax = axes[1]
    ax.set_title("Speedup (w=max vs w=1)")
    env_names, speedup_vals, clrs = [], [], []
    for i, (env_id, runs) in enumerate(sorted(envs_seen.items())):
        runs = sorted(runs, key=lambda r: r["num_workers"])
        if len(runs) >= 2:
            sp = runs[0]["duration_sec"] / runs[-1]["duration_sec"]
            env_names.append(env_labels.get(env_id, env_id))
            speedup_vals.append(sp)
            clrs.append(colors[i % len(colors)])
    if env_names:
        bars = ax.bar(env_names, speedup_vals, color=clrs)
        for bar, sp in zip(bars, speedup_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{sp:.2f}x", ha="center", va="bottom", fontsize=10)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Speedup")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafik: {output_path}")


def plot_all(results_dir: Path, env_filter: str | None = None) -> None:
    plots_dir = results_dir / "plots"

    # Jedinstveni env-ovi u rezultatima
    all_runs = load_runs(results_dir)
    env_ids = sorted({r["env_id"] for r in all_runs})
    if env_filter:
        env_ids = [e for e in env_ids if env_filter.lower() in e.lower()]

    print(f"\nGenerišem grafike za: {env_ids}")

    for env_id in env_ids:
        env_slug = env_id.replace("/", "_").replace("-", "_").lower()
        plot_single_env(env_id, results_dir, plots_dir / f"scalability_{env_slug}.png")

    if len(env_ids) > 1:
        plot_env_comparison(results_dir, plots_dir / "comparison_all_envs.png")

    print(f"\nSvi grafici u: {plots_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generiši grafike iz rezultata")
    parser.add_argument("--results", default=str(RESULTS_DEFAULT))
    parser.add_argument("--env", default=None, help="Filter: samo ovaj env (npr. 'LunarLander')")
    args = parser.parse_args()

    plot_all(Path(args.results), env_filter=args.env)


if __name__ == "__main__":
    main()
