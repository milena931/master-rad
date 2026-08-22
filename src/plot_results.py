"""
Generiše grafike iz sačuvanih eksperimenata za master rad.

Pokretanje:
  python src/plot_results.py --cartpole-gcp
      # agregira results/gcp_cartpole_*/results → results/plots/cartpole/

  python src/plot_results.py --results results/gcp_cartpole_1/results
      # jedan folder sa JSON-ovima (stari 3-panel, samo PPO)

  python src/plot_results.py --env cartpole
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RESULTS_DEFAULT = ROOT / "results"

# Boje po envu za konzistentnost kroz grafike
ENV_COLORS = {
    "cartpole": "steelblue",
    "lunarlander": "darkorange",
    "bipedalwalker": "mediumseagreen",
}
RAY_COLOR = "steelblue"

ALGO_LABEL = {
    "ray_rllib": "PPO",
    "appo": "APPO",
    "dqn": "DQN",
    "sac": "SAC",
}
# Okabe–Ito, čitljivo u štampi i za daltonizam
ALGO_COLOR = {
    "ray_rllib": "#0072B2",
    "appo": "#E69F00",
    "dqn": "#009E73",
    "sac": "#CC79A7",
}
ALGO_MARKER = {
    "ray_rllib": "o",
    "appo": "s",
    "dqn": "D",
    "sac": "^",
}
ALGO_ORDER = ["ray_rllib", "appo", "dqn", "sac"]
N_TICKS = [1, 2, 4, 8, 16]


def _thesis_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "axes.titlepad": 8,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "axes.grid": False,
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _clean_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#ededed", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7)


def _n_index(workers: list[int]) -> list[float]:
    return [float(N_TICKS.index(w)) for w in workers]


def _set_n_axis(ax) -> None:
    ax.set_xticks(range(len(N_TICKS)))
    ax.set_xticklabels([str(t) for t in N_TICKS])
    ax.set_xlim(-0.4, len(N_TICKS) - 0.6)
    ax.set_xlabel("Broj radnika")


def _algo_kwargs(fw: str) -> dict:
    return dict(
        color=ALGO_COLOR[fw],
        marker=ALGO_MARKER[fw],
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.6,
        linewidth=2.1,
        label=ALGO_LABEL[fw],
        zorder=4,
    )


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


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.fmean(vals), statistics.stdev(vals)


def _dedupe_runs(runs: list[dict]) -> list[dict]:
    """Poslednji zapis za (framework, workers) — APPO w=0 se kod nas čuva kao drugi w=1."""
    by: dict[tuple[str, int], dict] = {}
    for r in runs:
        by[(r["framework"], int(r["num_workers"]))] = r
    return list(by.values())


def load_cartpole_gcp_seeds(results_root: Path) -> list[tuple[str, list[dict]]]:
    seeds = []
    for folder in sorted(results_root.glob("gcp_cartpole_*")):
        summary = folder / "results" / "summary_cartpole.json"
        if not summary.exists():
            continue
        runs = _dedupe_runs(json.loads(summary.read_text(encoding="utf-8")))
        seeds.append((folder.name, runs))
    return seeds


def aggregate_seeds(seeds: list[tuple[str, list[dict]]]) -> dict[str, dict[int, dict]]:
    """framework -> workers -> liste metrika po seedu."""
    buckets: dict[str, dict[int, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for _name, runs in seeds:
        for r in runs:
            fw, w = r["framework"], int(r["num_workers"])
            b = buckets[fw][w]
            b["duration"].append(r["duration_sec"])
            b["throughput"].append(r.get("avg_throughput_steps_per_sec", 0.0))
            b["best"].append(r.get("best_reward", 0.0))
            ev = r.get("evaluation") or {}
            if "mean_reward" in ev:
                b["eval"].append(ev["mean_reward"])
    return buckets


def _series(buckets: dict[int, dict], key: str) -> tuple[list[int], list[float], list[float]]:
    workers = sorted(buckets)
    means, stds = [], []
    for w in workers:
        m, s = _mean_std(buckets[w][key])
        means.append(m)
        stds.append(s)
    return workers, means, stds


def plot_cartpole_gcp(results_root: Path) -> None:
    """Grafici za poglavlje 6.1 iz results/gcp_cartpole_*/."""
    seeds = load_cartpole_gcp_seeds(results_root)
    if not seeds:
        print("  Nema results/gcp_cartpole_*/results/summary_cartpole.json")
        return

    buckets = aggregate_seeds(seeds)
    out = results_root / "plots" / "cartpole"
    out.mkdir(parents=True, exist_ok=True)
    _thesis_style()

    algos = [a for a in ALGO_ORDER if a in buckets]
    _plot_thr_speed_eff(buckets, algos, out)
    _plot_eval(buckets, algos, out)
    _plot_learning_ppo(seeds[0][1], out / "cartpole_ppo_kriva_ucenja.png")
    _write_table(buckets, algos, out / "cartpole_tabela.csv")
    print(f"\n  Grafici (poglavlje 6.1): {out}")
    print(f"  Seedovi: {', '.join(n for n, _ in seeds)}")


def _workers_in_ticks(buckets_fw: dict) -> list[int]:
    return [w for w in N_TICKS if w in buckets_fw]


def _save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  Grafik: {path}")


def _plot_one_metric(ax, buckets, algos, kind: str) -> None:
    if kind == "propusnost":
        for fw in algos:
            ws = _workers_in_ticks(buckets[fw])
            ys = [_mean_std(buckets[fw][w]["throughput"])[0] for w in ws]
            ax.plot(_n_index(ws), ys, **_algo_kwargs(fw))
        ax.set_ylabel("Koraci / s")
        ax.set_title("Propusnost")
    elif kind == "ubrzanje":
        ax.plot(
            range(len(N_TICKS)), N_TICKS, color="#b0b0b0",
            linestyle="--", linewidth=1.25, zorder=2, label="Idealno",
        )
        for fw in algos:
            if 1 not in buckets[fw]:
                continue
            t1, _ = _mean_std(buckets[fw][1]["duration"])
            ws = _workers_in_ticks(buckets[fw])
            ys = [t1 / _mean_std(buckets[fw][w]["duration"])[0] for w in ws]
            ax.plot(_n_index(ws), ys, **_algo_kwargs(fw))
        ax.set_ylabel("T(1) / T(N)")
        ax.set_title("Ubrzanje")
        ax.set_ylim(0, 17)
    else:
        ax.axhline(1.0, color="#b0b0b0", linestyle="--", linewidth=1.25, zorder=2)
        for fw in algos:
            if 1 not in buckets[fw]:
                continue
            t1, _ = _mean_std(buckets[fw][1]["duration"])
            ws = _workers_in_ticks(buckets[fw])
            ys = [(t1 / _mean_std(buckets[fw][w]["duration"])[0]) / w for w in ws]
            ax.plot(_n_index(ws), ys, **_algo_kwargs(fw))
        ax.set_ylabel("S(N) / N")
        ax.set_title("Efikasnost")
        ax.set_ylim(0, 1.12)
    _set_n_axis(ax)
    _clean_ax(ax)


def _plot_thr_speed_eff(buckets, algos, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), layout="constrained")
    for ax, kind in zip(axes, ("propusnost", "ubrzanje", "efikasnost")):
        _plot_one_metric(ax, buckets, algos, kind)

    handles = [
        plt.Line2D(
            [0], [0], color=ALGO_COLOR[fw], marker=ALGO_MARKER[fw],
            markerfacecolor="white", markeredgewidth=1.5, linewidth=2.0,
            markersize=7, label=ALGO_LABEL[fw],
        )
        for fw in algos
    ]
    handles.append(plt.Line2D(
        [0], [0], color="#b0b0b0", linestyle="--", linewidth=1.25, label="Idealno",
    ))
    fig.legend(handles, [h.get_label() for h in handles],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.16))
    _save(fig, out / "cartpole_skalabilnost.png")

    for kind, fname in (
        ("propusnost", "cartpole_propusnost.png"),
        ("ubrzanje", "cartpole_ubrzanje.png"),
        ("efikasnost", "cartpole_efikasnost.png"),
    ):
        fig, ax = plt.subplots(figsize=(5.7, 3.65), layout="constrained")
        _plot_one_metric(ax, buckets, algos, kind)
        ax.legend(loc="best")
        _save(fig, out / fname)


def _plot_eval(buckets, algos, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.9, 3.8), layout="constrained")
    ax.axhline(450, color="#b0b0b0", linestyle="--", linewidth=1.25, zorder=1)
    offsets = {"ray_rllib": -0.16, "appo": 0.0, "dqn": 0.16}
    for fw in algos:
        ws = _workers_in_ticks(buckets[fw])
        means = []
        for w in ws:
            vals = buckets[fw][w]["eval"]
            means.append(_mean_std(vals)[0])
            xi = N_TICKS.index(w) + offsets.get(fw, 0)
            ax.scatter(
                [xi] * len(vals), vals,
                color=ALGO_COLOR[fw], alpha=0.35, s=22,
                linewidths=0, zorder=3, clip_on=False,
            )
        ax.plot(_n_index(ws), means, **_algo_kwargs(fw))
    ax.set_title("Greedy evaluacija (10 epizoda)")
    ax.set_ylabel("Srednja nagrada")
    ax.set_ylim(0, 530)
    _set_n_axis(ax)
    _clean_ax(ax)
    ax.plot([], [], color="#b0b0b0", linestyle="--", linewidth=1.25, label="Cilj (450)")
    ax.legend(loc="lower right")
    _save(fig, out / "cartpole_eval.png")


def _plot_learning_ppo(runs: list[dict], path: Path) -> None:
    ppo = sorted(
        [r for r in runs if r["framework"] == "ray_rllib" and int(r["num_workers"]) in [0, 1, 2, 4, 8, 16]],
        key=lambda r: r["num_workers"],
    )
    if not ppo:
        return
    palette = ["#56B4E9", "#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
    styles = ["-", "-", "--", "--", "-.", "-."]
    fig, ax = plt.subplots(figsize=(5.9, 3.8), layout="constrained")
    ax.axhline(450, color="#b0b0b0", linestyle="--", linewidth=1.25, zorder=1)
    for i, run in enumerate(ppo):
        it = [x["iteration"] for x in run["iterations"]]
        rw = [x["episode_return_mean"] for x in run["iterations"]]
        ax.plot(
            it, rw, styles[i % len(styles)],
            color=palette[i % len(palette)],
            linewidth=1.9,
            label=f"N = {run['num_workers']}",
        )
    ax.set_title("PPO — kriva učenja")
    ax.set_xlabel("Iteracija")
    ax.set_ylabel("Srednja nagrada epizode")
    ax.set_xlim(left=0)
    _clean_ax(ax)
    ax.legend(ncol=2, loc="lower right")
    _save(fig, path)


def _write_table(buckets, algos, path: Path) -> None:
    lines = ["algoritam,N,n_seed,trajanje_s,propusnost,eval_mean,eval_std,speedup,efikasnost"]
    for fw in algos:
        t1 = _mean_std(buckets[fw][1]["duration"])[0] if 1 in buckets[fw] else None
        for w in sorted(buckets[fw]):
            d_m, _ = _mean_std(buckets[fw][w]["duration"])
            t_m, _ = _mean_std(buckets[fw][w]["throughput"])
            ev = buckets[fw][w]["eval"]
            e_m, e_s = _mean_std(ev) if ev else (float("nan"), float("nan"))
            n = max(len(buckets[fw][w]["duration"]), 1)
            if t1 and w >= 1:
                su = t1 / d_m
                eff = su / w
            else:
                su, eff = float("nan"), float("nan")
            lines.append(
                f"{ALGO_LABEL[fw]},{w},{n},{d_m:.1f},{t_m:.1f},{e_m:.1f},{e_s:.1f},{su:.3f},{eff:.3f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Tabela: {path}")


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
    parser.add_argument(
        "--cartpole-gcp",
        action="store_true",
        help="Agregiraj results/gcp_cartpole_* i nacrtaj grafike za poglavlje 6.1",
    )
    args = parser.parse_args()

    if args.cartpole_gcp:
        plot_cartpole_gcp(Path(args.results))
        return

    plot_all(Path(args.results), env_filter=args.env)


if __name__ == "__main__":
    main()
