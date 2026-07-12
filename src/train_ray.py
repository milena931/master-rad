"""
Distribuirano treniranje PPO agenta pomoću Ray RLlib.

Šta je PPO:
  PPO (Proximal Policy Optimization) uči agenta da bira dobre akcije.
  Intuicija: agent proba akciju → vidi koliko je bila dobra (advantage) →
  pomalo promeni strategiju, ali ne previše odjednom (clip = "proximal").

Zašto Ray RLlib:
  Ray pokreće N paralelnih EnvRunner procesa koji igraju igru istovremeno.
  Više runnera → više iskustva u sekundi (throughput) → brže učenje.
  Može da skalira na više mašina (GCP klaster) za razliku od SB3.

GIF-ovi koje ovaj fajl generiše (ako se zada --gif):
  - random_agent.gif       — agent pre treninga (haos)
  - ray_wN_trained.gif     — naučen agent sa N workera
  - ray_wN_evolution.gif   — kako je agent napredovao tokom treninga
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env

from metrics import TrainingRun, save_run


class _ClipObsWrapper(gym.ObservationWrapper):
    """Klipuje opservacije na granice observation_space.

    BipedalWalker-v3 povremeno vraća ugao tela koji malo prelazi π (3.14159)
    zbog numeričke nepreciznosti u Box2D fizičkom engine-u. Ray RLlib crashuje
    workera kad detektuje opservaciju izvan prostora. Ovo je najjednostavniji fix.
    """

    def observation(self, obs):
        return np.clip(obs, self.observation_space.low, self.observation_space.high)


def _register_clipped_envs() -> None:
    """Registruje klipovane verzije envova koji imaju OOB opservacije."""
    def _make_clipped_bipedal(cfg):
        env = gym.make("BipedalWalker-v3")
        return _ClipObsWrapper(env)

    register_env("BipedalWalker-v3-clipped", _make_clipped_bipedal)


_register_clipped_envs()
from play_game import (
    build_evolution_gif,
    record_episode,
    record_ray_algo,
    save_gif,
)

ROOT = Path(__file__).parent.parent

# Mapiranje env_id → klipovana verzija za envove sa OOB opservacijama
_CLIPPED_ENV_MAP = {
    "BipedalWalker-v3": "BipedalWalker-v3-clipped",
    "BipedalWalkerHardcore-v3": "BipedalWalker-v3-clipped",
}


def _resolve_env_id(env_id: str) -> str:
    """Vraća klipovanu verziju env_id ako postoji, inače originalni."""
    return _CLIPPED_ENV_MAP.get(env_id, env_id)


def build_config(
    env_id: str,
    num_env_runners: int,
    train_batch_size: int,
    lr: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.0,
    clip_range: float = 0.2,
    num_gpus: float = 0,
) -> PPOConfig:
    """
    Pravi PPO konfiguraciju za Ray RLlib.

    num_env_runners = KLJUČNI PARAMETAR za master rad:
      0 → lokalno u jednom procesu (sekvencijalno, kao baseline)
      N → N paralelnih Ray aktora koji igraju igru istovremeno

    train_batch_size = ukupno koraka pre svakog ažuriranja policy-ja.
    Sa N runner-a, svaki sakuplja ~train_batch_size/N koraka.

    Env-specifični parametri (iz ppo_override u experiments.yaml):
      gamma     — discount faktor: 0.999 za duge epizode (BipedalWalker)
      gae_lambda — GAE parametar za procenu advantage
      ent_coef  — entropy koeficijent: 0.001 za stabilnost, 0.01 za istraživanje
      clip_range — PPO clip parametar
    """
    return (
        PPOConfig()
        .environment(_resolve_env_id(env_id), clip_actions=True)
        .env_runners(num_env_runners=num_env_runners)
        .training(
            train_batch_size=train_batch_size,
            lr=lr,
            gamma=gamma,
            lambda_=gae_lambda,
            entropy_coeff=ent_coef,
            clip_param=clip_range,
        )
        .resources(num_gpus=num_gpus)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )


def train(
    env_id: str = "CartPole-v1",
    num_env_runners: int = 2,
    max_iterations: int = 30,
    target_reward: float = 450,
    train_batch_size: int = 4000,
    lr: float = 3e-4,
    seed: int = 42,
    num_gpus: float = 0,
    ray_address: str | None = None,
    output_dir: str | None = None,
    gif_dir: str | None = None,
    evolution_every: int = 10,
    ppo_params: dict | None = None,
) -> TrainingRun:
    """
    Trenira PPO agenta sa Ray RLlib.

    ppo_params — env-specifični PPO hiperparametri (iz experiments.yaml ppo_override).
                 Prepisuju default vrednosti. Npr. za BipedalWalker:
                   {"gamma": 0.999, "gae_lambda": 0.95, "ent_coef": 0.001, ...}
    """
    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
        print(f"Spojen na Ray klaster: {ray_address}")
    else:
        ray.init(ignore_reinit_error=True)
        print("Lokalni Ray klaster podignut.")

    p = ppo_params or {}
    out_path = Path(output_dir) if output_dir else ROOT / "results"
    gif_path = Path(gif_dir) if gif_dir else None

    run = TrainingRun(
        framework="ray_rllib",
        env_id=env_id,
        num_workers=num_env_runners,
        seed=seed,
    )

    config = build_config(
        env_id=env_id,
        num_env_runners=num_env_runners,
        train_batch_size=p.get("train_batch_size", train_batch_size),
        lr=p.get("lr", lr),
        gamma=p.get("gamma", 0.99),
        gae_lambda=p.get("gae_lambda", 0.95),
        ent_coef=p.get("ent_coef", 0.0),
        clip_range=p.get("clip_range", 0.2),
        num_gpus=num_gpus,
    )
    algo = config.build_algo()

    print(f"\n{'='*65}")
    print(f"  Ray RLlib PPO")
    print(f"  Env:         {env_id}")
    print(f"  EnvRunners:  {num_env_runners}  ← paralelni procesi")
    print(f"  BatchSize:   {p.get('train_batch_size', train_batch_size)} koraka/iter")
    print(f"  Cilj:        nagrada >= {target_reward}")
    print(f"  PPO params:  lr={p.get('lr', lr)}, gamma={p.get('gamma', 0.99)}, "
          f"ent_coef={p.get('ent_coef', 0.0)}")
    if gif_path:
        print(f"  GIF-ovi:     {gif_path}")
    print(f"{'='*65}\n")
    print(f"{'Iter':>4} | {'Reward':>10} | {'Steps/s':>9} | {'Iter(s)':>8} | {'Total(s)':>9}")
    print(f"{'-'*60}")

    # Snimamo frejmove tokom treninga za evolution GIF
    evolution_segments: list[tuple[str, list]] = []

    # Snimimo random agenta pre treninga (prvi segment)
    if gif_path:
        print("\n  Snimam random agenta (pre treninga)...")
        r_frames, r_reward = record_episode(env_id, model=None, max_steps=250)
        print(f"  → nagrada={r_reward:.0f}")
        evolution_segments.append(("Random (0 iteracija)", r_frames))

    try:
        for i in range(max_iterations):
            t0 = time.time()
            result = algo.train()
            iter_sec = time.time() - t0

            env_stats = result.get("env_runners", result.get("sampler_results", {}))
            reward = env_stats.get(
                "episode_return_mean",
                env_stats.get("episode_reward_mean", 0.0),
            )
            reward = float(reward) if reward is not None else 0.0

            steps_this_iter = result.get("num_env_steps_sampled_this_iter", train_batch_size)
            throughput = steps_this_iter / iter_sec if iter_sec > 0 else 0.0

            run.add_iteration(i, reward, throughput=throughput, extra={"iter_sec": round(iter_sec, 2)})

            # Stani ako su gradienti eksplodirali (NaN)
            if math.isnan(reward) or math.isinf(reward):
                print(f"\n  [NaN] Gradienti su eksplodirali na iteraciji {i}. Stajem.")
                break

            print(
                f"{i:4d} | {reward:10.2f} | {throughput:9.0f} | "
                f"{iter_sec:8.1f} | {run.duration_sec:9.0f}"
            )

            # Snimi kratku epizodu za evolution GIF
            if gif_path and (i + 1) % evolution_every == 0:
                frames, ep_reward = record_ray_algo(algo, env_id, max_steps=300)
                label = f"Iter {i+1} | nagrada≈{ep_reward:.0f}"
                evolution_segments.append((label, frames))
                print(f"  [Evolution] {label}")

            if reward >= target_reward:
                print(f"\n  Ciljna nagrada {target_reward} dostignuta za {run.duration_sec:.0f}s!")
                break

    finally:
        run.finish()

        # Generiši GIF-ove od finalnog modela
        if gif_path:
            _generate_gifs(
                algo=algo,
                env_id=env_id,
                gif_path=gif_path,
                num_env_runners=num_env_runners,
                evolution_segments=evolution_segments,
            )

        algo.stop()
        ray.shutdown()

    saved = save_run(run, out_path)
    print(f"\nProsečni throughput: {run.avg_throughput():.0f} koraka/sec")
    print(f"Metrike: {saved}")
    return run


def _generate_gifs(
    algo,
    env_id: str,
    gif_path: Path,
    num_env_runners: int,
    evolution_segments: list[tuple[str, list]],
) -> None:
    """
    Generiše GIF-ove dok je Ray algo još u memoriji.

    Mora da se pozove PRE algo.stop() i ray.shutdown()!
    """
    gif_path.mkdir(parents=True, exist_ok=True)
    print(f"\n  Generiše GIF-ove → {gif_path}")

    # Random agent GIF (samo jednom po env-u)
    random_gif = gif_path / "random_agent.gif"
    if not random_gif.exists():
        r_frames, r_reward = record_episode(env_id, model=None, max_steps=300)
        print(f"    Slučajan agent: nagrada={r_reward:.0f}")
        save_gif(r_frames, random_gif)

    # Naučen agent (best model)
    try:
        trained_frames, t_reward = record_ray_algo(algo, env_id, max_steps=500)
        print(f"    Naučen agent (Ray w={num_env_runners}): nagrada={t_reward:.0f}")
        save_gif(trained_frames, gif_path / f"ray_w{num_env_runners}_trained.gif")
    except Exception as e:
        print(f"    [UPOZORENJE] Nije mogao da snimi trained GIF: {e}")
        print(f"    Pokušaj ručno: python src/play_game.py --env {env_id}")

    # Evolution GIF
    if len(evolution_segments) >= 2:
        try:
            f_frames, f_reward = record_ray_algo(algo, env_id, max_steps=400)
            evolution_segments.append((f"Finalni | nagrada≈{f_reward:.0f}", f_frames))
        except Exception:
            pass

        evolution_frames = build_evolution_gif(evolution_segments)
        save_gif(evolution_frames, gif_path / f"ray_w{num_env_runners}_evolution.gif", fps=24)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ray RLlib PPO trening",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--workers", type=int, default=2, help="Broj env_runners (paralelnih procesa)")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--target-reward", type=float, default=450)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=float, default=0)
    parser.add_argument("--ray-address", default=None, help='GCP: "ray://IP:10001"')
    parser.add_argument("--output", default=None)
    parser.add_argument("--gif", action="store_true", help="Generiši GIF-ove tokom/posle treninga")
    parser.add_argument("--gif-dir", default=None, help="Folder za GIF-ove")
    parser.add_argument("--evolution-every", type=int, default=10,
                        help="Snimi epizodu za evolution GIF svakih N iteracija")
    args = parser.parse_args()

    env_key = args.env.replace("/", "_").replace("-", "_").lower()
    gif_dir = (
        args.gif_dir
        if args.gif_dir
        else (str(ROOT / "results" / "gifs" / env_key) if args.gif else None)
    )

    train(
        env_id=args.env,
        num_env_runners=args.workers,
        max_iterations=args.iterations,
        target_reward=args.target_reward,
        train_batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_gpus=args.gpu,
        ray_address=args.ray_address,
        output_dir=args.output,
        gif_dir=gif_dir,
        evolution_every=args.evolution_every,
    )


if __name__ == "__main__":
    main()
