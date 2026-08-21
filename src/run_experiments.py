"""
Orkestrator skalabilnost eksperimenata za master rad.

Šta radi:
  Za svako odabrano okruženje (CartPole, LunarLander, BipedalWalker):
    1. Pokreni Ray sa 1, 2, 4 workera → JSON metrike
    2. Generiši GIF-ove: random, naučen, evolution
  Na kraju ispiše tabelu speedup/efikasnost za sve envove.

Pokretanje:
  # Sve tri igre:
  python run_experiments.py --envs cartpole lunarlander bipedalwalker

  # Samo CartPole sa GIF-ovima:
  python run_experiments.py --envs cartpole --gif

  # Na GCP (ista config/experiments.yaml):
  python src/run_experiments.py --envs lunarlander --algo ppo appo dqn \
      --workers 1 2 4 8 --gif --evaluate 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Osigurava line-buffering kada se skripta pokreće kroz pipe (npr. | tee)
# Bez ovoga Python baferuje ispis u blokove → iteracije se ne vide u realnom vremenu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import yaml

from train_ray import train as train_ray
from train_appo import train as train_appo
from train_sac import train as train_sac
from train_dqn import train as train_dqn
from evaluate_agent import evaluate as evaluate_agent

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
    ray_address: str | None = None,
    generate_gifs: bool = True,
    scaling_only: bool = False,
    evaluate_episodes: int = 0,
    algorithms: list[str] | None = None,
    max_iterations_override: int | None = None,
) -> dict[str, list[dict]]:
    """
    Pokreće eksperimente za sve tražene envove.

    algorithms       — lista algoritama: ["ppo"], ["appo"] ili ["ppo", "appo", "sac", "dqn"].
                       Default: ["ppo"]
    scaling_only     — ako True, za envove koji imaju scaling_max_iterations/
                       scaling_worker_counts u config-u, koristi te vrednosti.
    evaluate_episodes — ako > 0, posle svakog treninga pokrene N epizoda za mean ± std.
    max_iterations_override — ako je dato, prepisuje max_iterations iz config-a za sve envove.
                       Korisno za brze testove: --max-iterations 30.
    """
    cfg = load_config(config_path)
    workers = worker_counts or cfg["worker_counts"]
    ppo_defaults = cfg.get("ppo_defaults", cfg.get("ppo", {}))
    appo_defaults = cfg.get("appo_defaults", {})
    sac_defaults = cfg.get("sac_defaults", {})
    dqn_defaults = cfg.get("dqn_defaults", {})
    run_algos = algorithms or ["ppo"]
    all_results: dict[str, list[dict]] = {}

    print(f"\n{'#'*65}")
    print(f"  MASTER RAD — {'SCALING STUDY' if scaling_only else 'SKALABILNOST EKSPERIMENTI'}")
    print(f"  Okruženja: {env_keys}")
    print(f"  Algoritmi: {run_algos}")
    print(f"  Workers:   {workers}")
    print(f"  GIF-ovi:   {'da' if generate_gifs else 'ne'}")
    if evaluate_episodes > 0:
        print(f"  Evaluacija: {evaluate_episodes} epizoda po runu (mean ± std)")
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

        # Ručni override (--max-iterations flag) prepisuje sve gore
        if max_iterations_override is not None:
            effective_max_iter = max_iterations_override

        # Merge: defaults + env-specifični override
        ppo_params = {**ppo_defaults, **env_cfg.get("ppo_override", {})}

        print(f"\n{'='*65}")
        print(f"  Okruženje: {env_id}")
        print(f"  {env_cfg.get('description', '')}")
        print(f"{'='*65}")

        env_results: list[dict] = []

        # ── PPO run-ovi ──────────────────────────────────────────────────────
        if "ppo" in run_algos:
            for n_workers in effective_workers:
                print(f"\n>>> PPO | env_runners={n_workers} | {env_id}")

                ckpt_dir = str(output_dir / "checkpoints" / f"ppo_{env_key}_w{n_workers}")

                run = train_ray(
                    env_id=env_id,
                    num_env_runners=n_workers,
                    max_iterations=effective_max_iter,
                    target_reward=env_cfg["target_reward"],
                    train_batch_size=ppo_params.get("train_batch_size", ppo_defaults.get("train_batch_size", 4000)),
                    lr=ppo_params.get("lr", 3e-4),
                    ray_address=ray_address,
                    output_dir=str(output_dir),
                    gif_dir=str(gif_dir) if gif_dir else None,
                    evolution_every=env_cfg.get("evolution_every", 10),
                    ppo_params=ppo_params,
                    checkpoint_dir=ckpt_dir,
                )

                run_dict = run.to_dict()
                _run_evaluation(
                    run_dict, run, env_id, evaluate_episodes,
                    gif_dir, algo_tag=f"ppo_w{n_workers}",
                    ckpt_dir=ckpt_dir,
                )
                env_results.append(run_dict)

        # ── A2C run-ovi ──────────────────────────────────────────────────────
        if "appo" in run_algos:
            appo_params = {**appo_defaults, **env_cfg.get("appo_override", {})}
            # A2C može imati poseban max_iterations (mnogo malih iteracija vs PPO)
            appo_max_iter = env_cfg.get("appo_max_iterations", effective_max_iter)

            for n_workers in effective_workers:
                print(f"\n>>> APPO | env_runners={n_workers} | {env_id}")

                ckpt_dir = str(output_dir / "checkpoints" / f"appo_{env_key}_w{n_workers}")

                run = train_appo(
                    env_id=env_id,
                    num_env_runners=n_workers,
                    max_iterations=appo_max_iter,
                    target_reward=env_cfg["target_reward"],
                    lr=appo_params.get("lr", 3e-4),
                    ray_address=ray_address,
                    output_dir=str(output_dir),
                    gif_dir=str(gif_dir) if gif_dir else None,
                    evolution_every=env_cfg.get(
                        "appo_evolution_every",
                        env_cfg.get("evolution_every", 10),
                    ),
                    appo_params=appo_params,
                    checkpoint_dir=ckpt_dir,
                )

                run_dict = run.to_dict()
                _run_evaluation(
                    run_dict, run, env_id, evaluate_episodes,
                    gif_dir, algo_tag=f"appo_w{n_workers}",
                    ckpt_dir=ckpt_dir,
                )
                env_results.append(run_dict)

        # ── SAC run-ovi (samo za envove sa sac_override — kontinualne akcije) ──
        if "sac" in run_algos:
            if "sac_override" not in env_cfg:
                print(f"\n  [PRESKAČEM] SAC nije konfigurisan za {env_key} (nema sac_override — koristiti za kontinualne akcije)")
            else:
                sac_params = {**sac_defaults, **env_cfg.get("sac_override", {})}
                sac_max_iter = env_cfg.get("sac_max_iterations", effective_max_iter)
                sac_workers = [max(w, 1) for w in effective_workers]
                seen_sac: set[int] = set()
                sac_workers = [w for w in sac_workers if not (w in seen_sac or seen_sac.add(w))]

                for n_workers in sac_workers:
                    print(f"\n>>> SAC | env_runners={n_workers} | {env_id}")

                    ckpt_dir = str(output_dir / "checkpoints" / f"sac_{env_key}_w{n_workers}")

                    run = train_sac(
                        env_id=env_id,
                        num_env_runners=n_workers,
                        max_iterations=sac_max_iter,
                        target_reward=env_cfg["target_reward"],
                        train_batch_size=sac_params.get("train_batch_size", 256),
                        lr=sac_params.get("lr", 0.00073),
                        ray_address=ray_address,
                        output_dir=str(output_dir),
                        gif_dir=str(gif_dir) if gif_dir else None,
                        evolution_every=env_cfg.get(
                            "sac_evolution_every",
                            env_cfg.get("evolution_every", 400),
                        ),
                        sac_params=sac_params,
                        checkpoint_dir=ckpt_dir,
                    )

                    run_dict = run.to_dict()
                    _run_evaluation(
                        run_dict, run, env_id, evaluate_episodes,
                        gif_dir, algo_tag=f"sac_w{n_workers}",
                        ckpt_dir=ckpt_dir,
                    )
                    env_results.append(run_dict)

        # ── DQN run-ovi (samo za envove sa dqn_override — diskretne akcije) ──
        if "dqn" in run_algos:
            if "dqn_override" not in env_cfg:
                print(f"\n  [PRESKAČEM] DQN nije konfigurisan za {env_key} (nema dqn_override — koristiti za diskretne akcije)")
            else:
                dqn_params = {**dqn_defaults, **env_cfg.get("dqn_override", {})}
                dqn_max_iter = env_cfg.get("dqn_max_iterations", effective_max_iter)
                dqn_workers = [max(w, 1) for w in effective_workers]
                seen_dqn: set[int] = set()
                dqn_workers = [w for w in dqn_workers if not (w in seen_dqn or seen_dqn.add(w))]

                for n_workers in dqn_workers:
                    print(f"\n>>> DQN | env_runners={n_workers} | {env_id}")

                    ckpt_dir = str(output_dir / "checkpoints" / f"dqn_{env_key}_w{n_workers}")

                    run = train_dqn(
                        env_id=env_id,
                        num_env_runners=n_workers,
                        max_iterations=dqn_max_iter,
                        target_reward=env_cfg["target_reward"],
                        batch_size=dqn_params.get("batch_size", 128),
                        lr=dqn_params.get("lr", 0.00063),
                        ray_address=ray_address,
                        output_dir=str(output_dir),
                        gif_dir=str(gif_dir) if gif_dir else None,
                        evolution_every=env_cfg.get(
                            "dqn_evolution_every",
                            env_cfg.get("evolution_every", 1300),
                        ),
                        dqn_params=dqn_params,
                        checkpoint_dir=ckpt_dir,
                    )

                    run_dict = run.to_dict()
                    _run_evaluation(
                        run_dict, run, env_id, evaluate_episodes,
                        gif_dir, algo_tag=f"dqn_w{n_workers}",
                        ckpt_dir=ckpt_dir,
                    )
                    env_results.append(run_dict)

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


def _run_evaluation(
    run_dict: dict,
    run,
    env_id: str,
    evaluate_episodes: int,
    gif_dir,
    algo_tag: str,
    ckpt_dir: str,
) -> None:
    """Pokreće post-trening evaluaciju i dodaje rezultate u run_dict."""
    if evaluate_episodes <= 0:
        return
    ckpt = run.checkpoint_path
    if not ckpt or ckpt == "None":
        print(f"  [UPOZORENJE] Checkpoint nije sačuvan — evaluacija preskočena.")
        print(f"  Proveri da li je {ckpt_dir} validan RLlib checkpoint.")
        return
    n_w = run_dict.get("num_workers", "?")
    print(f"\n>>> Evaluacija ({evaluate_episodes} epizoda) | {algo_tag}")
    try:
        eval_result = evaluate_agent(
            checkpoint_path=ckpt,
            env_id=env_id,
            n_episodes=evaluate_episodes,
            gif_dir=str(gif_dir) if gif_dir else None,
            gif_tag=algo_tag,
        )
        run_dict["evaluation"] = {
            "n_episodes": eval_result["n_episodes"],
            "mean_reward": eval_result["mean"],
            "std_reward": eval_result["std"],
            "min_reward": eval_result["min"],
            "max_reward": eval_result["max"],
            "gif_path": eval_result["gif_path"],
        }
        print(f"  {algo_tag}: mean={eval_result['mean']:.2f} ± {eval_result['std']:.2f}")
    except Exception as exc:
        print(f"  [UPOZORENJE] Evaluacija nije uspela: {exc}")
        print(f"  Možeš pokrenuti ručno: python src/evaluate_agent.py "
              f"--checkpoint {ckpt} --env {env_id} --episodes {evaluate_episodes}")


def _print_env_summary(env_key: str, env_id: str, results: list[dict]) -> None:
    has_eval = any("evaluation" in r for r in results)
    print(f"\n  --- Rezultati: {env_id} ---")
    if has_eval:
        print(f"  {'Framework':<22} {'Workers':>7} {'Duration':>10} {'Best Rwd':>10} {'Steps/s':>9} {'Eval mean±std':>18}")
        print(f"  {'-'*82}")
    else:
        print(f"  {'Framework':<22} {'Workers':>7} {'Duration':>10} {'Best Rwd':>10} {'Steps/s':>9}")
        print(f"  {'-'*63}")
    for r in results:
        line = (
            f"  {r['framework']:<22} {r['num_workers']:>7} "
            f"{r['duration_sec']:>9.1f}s {r.get('best_reward', 0):>10.1f} "
            f"{r.get('avg_throughput_steps_per_sec', 0):>9.0f}"
        )
        if has_eval and "evaluation" in r:
            ev = r["evaluation"]
            line += f"  {ev['mean_reward']:>7.2f} ± {ev['std_reward']:<7.2f}"
        print(line)

    _print_speedup_table(results, framework="ray_rllib", label="PPO")
    _print_speedup_table(results, framework="appo", label="APPO")
    _print_speedup_table(results, framework="sac", label="SAC")
    _print_speedup_table(results, framework="dqn", label="DQN")


def _print_speedup_table(results: list[dict], framework: str, label: str) -> None:
    """Štampa speedup/efikasnost tabelu za jedan algoritam."""
    runs = sorted(
        [r for r in results if r["framework"] == framework],
        key=lambda r: r["num_workers"],
    )
    if len(runs) < 2:
        return
    baseline = runs[0]
    print(f"\n  Speedup {label} (baseline = w={baseline['num_workers']}):")
    for r in runs:
        speedup = baseline["duration_sec"] / r["duration_sec"] if r["duration_sec"] > 0 else 0
        base_w = baseline["num_workers"]
        cur_w = r["num_workers"]
        if base_w > 0 and cur_w > 0:
            n_ratio = cur_w / base_w
            efficiency = speedup / n_ratio
        else:
            n_ratio = float("inf") if cur_w > 0 else 1.0
            efficiency = float("nan")
        thr_speedup = (
            r.get("avg_throughput_steps_per_sec", 1)
            / max(baseline.get("avg_throughput_steps_per_sec", 1), 1)
        )
        eff_str = f"{efficiency:.2f}" if efficiency == efficiency else "N/A"
        print(
            f"    w={r['num_workers']:>2}: "
            f"speedup={speedup:.2f}x  "
            f"efikasnost={eff_str}  "
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
    parser.add_argument("--ray-address", default=None, help='GCP: "ray://IP:10001"')
    parser.add_argument(
        "--algo",
        nargs="+",
        default=["ppo"],
        choices=["ppo", "appo", "sac", "dqn"],
        help="Algoritmi za pokretanje. Primeri: --algo ppo  --algo appo  --algo dqn  --algo ppo appo sac dqn",
    )
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
    parser.add_argument(
        "--evaluate",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Posle svakog treninga pokreni N evaluacionih epizoda. "
            "Ispisuje mean ± std nagrade i čuva best epizodu kao GIF. "
            "Primer: --evaluate 15"
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        metavar="N",
        help="Prepiši max_iterations iz config-a (korisno za brze testove).",
    )
    args = parser.parse_args()

    run_all_experiments(
        env_keys=args.envs,
        config_path=Path(args.config),
        output_dir=Path(args.output),
        worker_counts=args.workers,
        ray_address=args.ray_address,
        generate_gifs=args.gif,
        scaling_only=args.scaling_only,
        evaluate_episodes=args.evaluate,
        algorithms=args.algo,
        max_iterations_override=args.max_iterations,
    )


if __name__ == "__main__":
    main()
