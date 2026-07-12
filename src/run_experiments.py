"""
Orkestrator skalabilnost eksperimenata za master rad.

Šta radi:
  Za svako odabrano okruženje (CartPole, LunarLander, BipedalWalker):
    1. Snimi random agenta (GIF)
    2. Pokreni Ray sa 1, 2, 4 workera → JSON metrike
    3. Pokreni SB3 baseline (opciono)
    4. Generiši GIF-ove: random, naučen, evolution
  Na kraju ispiše tabelu speedup/efikasnost za sve envove.

Pokretanje:
  # Sve tri igre, samo Ray (brže):
  python run_experiments.py --envs cartpole lunarlander bipedalwalker --skip-sb3

  # Samo LunarLander, sa SB3 poređenjem:
  python run_experiments.py --envs lunarlander

  # Na GCP klasteru (Ray na GCP, SB3 lokalno):
  python run_experiments.py --envs lunarlander --ray-address ray://34.90.x.x:10001

  # Sa GIF-ovima:
  python run_experiments.py --envs cartpole --gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from train_ray import train as train_ray
from train_sb3 import train as train_sb3

ROOT = Path(__file__).parent.parent
CONFIG_DEFAULT = ROOT / "config" / "experiments.yaml"
RESULTS_DEFAULT = ROOT / "results"

ALL_ENVS = ["cartpole", "lunarlander", "bipedalwalker"]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_all_experiments(
    env_keys: list[str],
    config_path: Path = CONFIG_DEFAULT,
    output_dir: Path = RESULTS_DEFAULT,
    worker_counts: list[int] | None = None,
    skip_sb3: bool = False,
    ray_address: str | None = None,
    generate_gifs: bool = True,
    scaling_only: bool = False,
) -> dict[str, list[dict]]:
    """
    Pokreće eksperimente za sve tražene envove.

    scaling_only — ako True, za envove koji imaju scaling_max_iterations/
                   scaling_worker_counts u config-u, koristi te vrednosti.
                   Rezultat: kratki run koji meri throughput i speedup,
                   bez čekanja na konvergenciju (idealno za BipedalWalker lokalno).
    """
    cfg = load_config(config_path)
    workers = worker_counts or cfg["worker_counts"]
    ppo_defaults = cfg.get("ppo_defaults", cfg.get("ppo", {}))
    all_results: dict[str, list[dict]] = {}

    print(f"\n{'#'*65}")
    print(f"  MASTER RAD — {'SCALING STUDY' if scaling_only else 'SKALABILNOST EKSPERIMENTI'}")
    print(f"  Okruženja: {env_keys}")
    print(f"  Workers:   {workers}")
    print(f"  GIF-ovi:   {'da' if generate_gifs else 'ne'}")
    if scaling_only:
        print("  Mod:       SCALING ONLY (kratko, samo throughput/speedup)")
    if ray_address:
        print(f"  Ray:       {ray_address}  (GCP)")
    else:
        print("  Ray:       lokalni klaster")
    print(f"{'#'*65}")

    for env_key in env_keys:
        if env_key not in cfg["environments"]:
            print(f"\n[UPOZORENJE] Okruženje '{env_key}' nije u config-u, preskačem.")
            continue

        env_cfg = cfg["environments"][env_key]
        env_id = env_cfg["id"]
        gif_dir = output_dir / "gifs" / env_key if generate_gifs else None

        # Scaling-only mod: koristi kraće parametre ako postoje
        if scaling_only and "scaling_max_iterations" in env_cfg:
            effective_max_iter = env_cfg["scaling_max_iterations"]
            effective_workers = worker_counts or env_cfg.get("scaling_worker_counts", workers)
            print(f"\n  [Scaling only] {env_key}: {effective_max_iter} iteracija, workers={effective_workers}")
        elif ray_address is None and "max_iterations_local" in env_cfg:
            # Lokalno: koristi manji broj iteracija ako postoji
            effective_max_iter = env_cfg["max_iterations_local"]
            effective_workers = workers
            total_steps = effective_max_iter * env_cfg.get("ppo_override", {}).get("train_batch_size", 4000)
            print(f"\n  [Lokalni mod] {env_key}: {effective_max_iter} iteracija (~{total_steps/1e6:.1f}M koraka)")
        else:
            # GCP ili nema lokalne vrednosti: koristi puno
            effective_max_iter = env_cfg["max_iterations"]
            effective_workers = workers

        # Merge: defaults + env-specifični override
        ppo_params = {**ppo_defaults, **env_cfg.get("ppo_override", {})}

        print(f"\n{'='*65}")
        print(f"  Okruženje: {env_id}")
        print(f"  {env_cfg.get('description', '')}")
        print(f"{'='*65}")

        env_results: list[dict] = []

        # --- Ray RLlib run-ovi ---
        for n_workers in effective_workers:
            print(f"\n>>> Ray RLlib | env_runners={n_workers} | {env_id}")
            run = train_ray(
                env_id=env_id,
                num_env_runners=n_workers,
                max_iterations=effective_max_iter,
                target_reward=env_cfg["target_reward"],
                train_batch_size=ppo_params.get("train_batch_size", cfg["ppo_defaults"].get("train_batch_size", 4000)),
                lr=ppo_params.get("lr", 3e-4),
                ray_address=ray_address,
                output_dir=str(output_dir),
                gif_dir=str(gif_dir) if gif_dir else None,
                evolution_every=env_cfg.get("evolution_every", 10),
                ppo_params=ppo_params,
            )
            env_results.append(run.to_dict())

        # --- SB3 baseline run-ovi (opciono) ---
        if not skip_sb3:
            for n_envs in effective_workers:
                print(f"\n>>> SB3 baseline | n_envs={n_envs} | {env_id}")
                run = train_sb3(
                    env_id=env_id,
                    n_envs=n_envs,
                    total_timesteps=env_cfg.get("sb3_timesteps", cfg["sb3"]["total_timesteps"]),
                    output_dir=str(output_dir),
                    checkpoint_every=env_cfg.get("checkpoint_every", 25_000),
                    gif_dir=str(gif_dir) if gif_dir else None,
                    ppo_params=ppo_params,
                )
                env_results.append(run.to_dict())

        # Sačuvaj summary za ovaj env
        summary_path = output_dir / f"summary_{env_key}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(env_results, indent=2), encoding="utf-8")
        print(f"\nSummary: {summary_path}")

        all_results[env_key] = env_results
        _print_env_summary(env_key, env_id, env_results)

    # Ukupna tabela za sve envove
    if len(env_keys) > 1:
        _print_grand_summary(all_results)

    return all_results


def _print_env_summary(env_key: str, env_id: str, results: list[dict]) -> None:
    print(f"\n  --- Rezultati: {env_id} ---")
    print(f"  {'Framework':<22} {'Workers':>7} {'Duration':>10} {'Best Rwd':>10} {'Steps/s':>9}")
    print(f"  {'-'*63}")
    for r in results:
        print(
            f"  {r['framework']:<22} {r['num_workers']:>7} "
            f"{r['duration_sec']:>9.1f}s {r.get('best_reward', 0):>10.1f} "
            f"{r.get('avg_throughput_steps_per_sec', 0):>9.0f}"
        )

    ray_runs = sorted(
        [r for r in results if r["framework"] == "ray_rllib"],
        key=lambda r: r["num_workers"],
    )
    if len(ray_runs) >= 2:
        baseline = ray_runs[0]
        print(f"\n  Speedup (baseline = w={baseline['num_workers']}):")
        for r in ray_runs:
            speedup = baseline["duration_sec"] / r["duration_sec"] if r["duration_sec"] > 0 else 0
            n_ratio = r["num_workers"] / baseline["num_workers"]
            efficiency = speedup / n_ratio if n_ratio > 0 else 0
            thr_speedup = (
                r.get("avg_throughput_steps_per_sec", 1)
                / max(baseline.get("avg_throughput_steps_per_sec", 1), 1)
            )
            print(
                f"    w={r['num_workers']:>2}: "
                f"speedup={speedup:.2f}x  "
                f"efikasnost={efficiency:.2f}  "
                f"thr_speedup={thr_speedup:.2f}x"
            )


def _print_grand_summary(all_results: dict[str, list[dict]]) -> None:
    print(f"\n{'#'*65}")
    print("  UKUPNA TABELA — SVI ENVOVI")
    print(f"{'#'*65}")
    print(f"  {'Env':<16} {'Framework':<22} {'W':>3} {'Rwd':>8} {'thr(k/s)':>10} {'t(s)':>8}")
    print(f"  {'-'*70}")
    for env_key, results in all_results.items():
        for r in results:
            print(
                f"  {env_key:<16} {r['framework']:<22} {r['num_workers']:>3} "
                f"{r.get('best_reward', 0):>8.1f} "
                f"{r.get('avg_throughput_steps_per_sec', 0):>10.0f} "
                f"{r['duration_sec']:>8.1f}"
            )
    print(f"  {'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pokreni skalabilnost eksperimente za master rad",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["cartpole"],
        choices=ALL_ENVS,
        help="Lista okruženja za testiranje",
    )
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--output", default=str(RESULTS_DEFAULT))
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=None,
        help="Worker counts (default iz config-a: [1, 2, 4])",
    )
    parser.add_argument("--skip-sb3", action="store_true", help="Preskoči SB3 baseline")
    parser.add_argument("--ray-address", default=None, help='GCP: "ray://IP:10001"')
    parser.add_argument("--gif", action="store_true", help="Generiši GIF-ove za svaki env")
    parser.add_argument(
        "--scaling-only",
        action="store_true",
        help=(
            "Kratki mod za merenje throughputa i speedupa — koristi "
            "scaling_max_iterations i scaling_worker_counts iz config-a. "
            "Idealno za BipedalWalker na lokalnoj mašini (traje ~80 min umesto 15h)."
        ),
    )
    args = parser.parse_args()

    run_all_experiments(
        env_keys=args.envs,
        config_path=Path(args.config),
        output_dir=Path(args.output),
        worker_counts=args.workers,
        skip_sb3=args.skip_sb3,
        ray_address=args.ray_address,
        generate_gifs=args.gif,
        scaling_only=args.scaling_only,
    )


if __name__ == "__main__":
    main()
