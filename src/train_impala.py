"""
Distribuirano treniranje IMPALA agenta pomoću Ray RLlib.

Šta je IMPALA:
  IMPALA (Importance Weighted Actor-Learner Architecture) je asinhroni
  distribuirani RL algoritam. Za razliku od PPO koji čeka sve workere pre
  svakog update-a (sinhron), IMPALA learner ažurira policy KONTINUALNO
  dok workeri šalju iskustvo.

  Intuicija: workeri igraju igru i šalju trajektorije u centralni red čekanja.
  Learner uzima iz reda i radi update bez čekanja. V-trace korekcija
  kompenzuje zastarelost iskustva (off-policy correction).

Zašto IMPALA za poređenje sa PPO:
  PPO  — sinhron: learner čeka najsporijeg workera → idle vreme sa w=8+
  IMPALA — asinhron: learner nikad ne čeka → bolje iskorišćenje resursa
  Ovo poređenje je centralna eksperimentalna priča master rada.

GIF-ovi koje ovaj fajl generiše (ako se zada --gif):
  - random_agent.gif          — agent pre treninga
  - impala_wN_trained.gif     — naučen agent sa N workera
  - impala_wN_evolution.gif   — napredak tokom treninga
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import gymnasium as gym
import ray
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.tune import register_env

from metrics import TrainingRun, save_run
from play_game import (
    build_evolution_gif,
    record_episode,
    record_ray_algo,
    save_gif,
)

ROOT = Path(__file__).parent.parent

_CLIPPED_ENV_MAP = {
    "BipedalWalker-v3": "BipedalWalker-v3-clipped",
    "BipedalWalkerHardcore-v3": "BipedalWalker-v3-clipped",
}


class _ClipObsWrapper(gym.ObservationWrapper):
    """Klipuje opservacije na granice observation_space (fix za BipedalWalker Box2D)."""
    def observation(self, obs):
        return np.clip(obs, self.observation_space.low, self.observation_space.high)


def _register_clipped_envs() -> None:
    def _make_clipped_bipedal(cfg):
        env = gym.make("BipedalWalker-v3")
        return _ClipObsWrapper(env)
    register_env("BipedalWalker-v3-clipped", _make_clipped_bipedal)


def _resolve_env_id(env_id: str) -> str:
    return _CLIPPED_ENV_MAP.get(env_id, env_id)


def build_config(
    env_id: str,
    num_env_runners: int,
    train_batch_size: int,
    lr: float,
    entropy_coeff: float = 0.01,
    vf_loss_coeff: float = 0.5,
    rollout_fragment_length: int = 50,
    num_gpus: float = 0,
    min_time_s_per_iteration: float = 0.0,
    normalize_obs: bool = False,
    grad_clip: float | None = 40.0,
    num_epochs: int = 1,
    replay_proportion: float = 0.0,
    replay_buffer_num_slots: int = 0,
) -> ImpalaConfig:
    """
    Pravi IMPALA konfiguraciju za Ray RLlib.

    num_env_runners — isti parametar kao u PPO, isti efekat paralelizacije.
                      Razlika: IMPALA ih koristi asinhrono.

    rollout_fragment_length — koliko koraka svaki worker skupi pre slanja learner-u.
                              Manji = learner dobija iskustvo češće (asinhronost++).
                              Veći = manje komunikacije, ali zastalelije iskustvo.

    min_time_s_per_iteration — minimalno vreme čekanja po iteraciji (Ray default: 10s!).
                               0.0 = bez čekanja, iteracije teku koliko god brzo mogu.
                               Ovo je ključno za brze envove kao CartPole.

    normalize_obs — MeanStdFilter normalizacija opservacija (kao normalize_obs u PPO).
                    KRITIČNO za BipedalWalker — policy prima opservacije različitih
                    opsega i bez normalizacije ne može da nauči.

    vf_loss_coeff — težina value function greške u ukupnom loss-u.
    entropy_coeff — podstiče istraživanje (isto kao ent_coef u PPO).
    grad_clip     — maksimalna norma gradijenata (originalni IMPALA paper: 40).
                    Sprečava catastrophic forgetting zbog prevelikih update-a.
                    None = bez klipovanja (nije preporučeno za IMPALA).
    num_epochs    — ostaviti na 1 (default). Više epoha ne poboljšava efikasnost jer:
                    - Manji lr + više epoha = isti efektivni update kao originalni lr
                    - Veći lr + više epoha = divergencija (V-trace IS weights postaju netačni)
                    Ovo je fundamentalna arhitekturalna razlika od PPO.
    replay_proportion — odnos replay:novi podaci (0 = bez replay-a).
                    Izbjegavati: cold-start problem puni buffer lošim podacima.
    replay_buffer_num_slots — koliko batcha čuvamo u replay memoriji.
    """
    obs_filter = "MeanStdFilter" if normalize_obs else "NoFilter"
    training_kwargs: dict = dict(
        train_batch_size=train_batch_size,
        lr=lr,
        entropy_coeff=entropy_coeff,
        vf_loss_coeff=vf_loss_coeff,
        num_epochs=num_epochs,
    )
    if grad_clip is not None:
        training_kwargs["grad_clip"] = grad_clip
    if replay_proportion > 0.0:
        training_kwargs["replay_proportion"] = replay_proportion
        training_kwargs["replay_buffer_num_slots"] = max(replay_buffer_num_slots, 1)
    cfg = (
        ImpalaConfig()
        .environment(_resolve_env_id(env_id), clip_actions=True)
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length=rollout_fragment_length,
            observation_filter=obs_filter,
        )
        .training(**training_kwargs)
        .resources(num_gpus=num_gpus)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )
    # min_time_s_per_iteration je atribut baznog AlgorithmConfig, ne .training() arg
    cfg.min_time_s_per_iteration = min_time_s_per_iteration
    return cfg


def _is_valid_checkpoint(path: Path) -> bool:
    """Proverava da li direktorijum sadrži validan RLlib checkpoint."""
    return path.is_dir() and (
        (path / "rllib_checkpoint.json").exists()
        or (path / "algorithm_state.pkl").exists()
    )


def _save_checkpoint(algo, ckpt_path: Path) -> str | None:
    """
    Čuva RLlib checkpoint i vraća putanju kao string.

    Proverava da li je putanja dobijena od API-ja zaista validan checkpoint
    pre nego što je vrati — izbegava vraćanje pogrešnih putanja.
    """
    ckpt_path.mkdir(parents=True, exist_ok=True)

    saved_path: str | None = None
    try:
        result = algo.save(str(ckpt_path))
        if hasattr(result, "checkpoint") and result.checkpoint is not None:
            p = getattr(result.checkpoint, "path", None)
            if p:
                saved_path = str(p)
        if not saved_path:
            p = getattr(result, "path", None)
            if p:
                saved_path = str(p)
    except Exception:
        pass

    if saved_path and _is_valid_checkpoint(Path(saved_path)):
        return saved_path

    try:
        result = algo.save_checkpoint(str(ckpt_path))
        if isinstance(result, str) and _is_valid_checkpoint(Path(result)):
            return result
    except Exception:
        pass

    if _is_valid_checkpoint(ckpt_path):
        return str(ckpt_path)
    for subdir in sorted(ckpt_path.glob("checkpoint_*"), reverse=True):
        if _is_valid_checkpoint(subdir):
            return str(subdir)

    try:
        if any(ckpt_path.iterdir()):
            return str(ckpt_path)
    except Exception:
        pass

    return None


def train(
    env_id: str = "CartPole-v1",
    num_env_runners: int = 2,
    max_iterations: int = 30,
    target_reward: float = 450,
    train_batch_size: int = 4000,
    lr: float = 3e-4,
    num_gpus: float = 0,
    ray_address: str | None = None,
    output_dir: str | None = None,
    gif_dir: str | None = None,
    evolution_every: int = 10,
    impala_params: dict | None = None,
    checkpoint_dir: str | None = None,
) -> TrainingRun:
    """
    Trenira IMPALA agenta sa Ray RLlib.

    impala_params — env-specifični hiperparametri (iz experiments.yaml impala_override).
                    Podržani ključevi: lr, entropy_coeff, vf_loss_coeff,
                                       rollout_fragment_length, train_batch_size
    checkpoint_dir — folder u koji se čuva checkpoint posle treninga.
                     Ako None, checkpoint se ne čuva.
    """
    # IMPALA je asinhroni algoritam — zahteva bar jednog actor workera
    if num_env_runners < 1:
        print(f"  [INFO] IMPALA zahteva num_env_runners >= 1. Postavljam na 1 (bilo: {num_env_runners}).")
        num_env_runners = 1

    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
        print(f"Spojen na Ray klaster: {ray_address}")
    else:
        ray.init(ignore_reinit_error=True)
        print("Lokalni Ray klaster podignut.")

    _register_clipped_envs()

    p = impala_params or {}
    out_path = Path(output_dir) if output_dir else ROOT / "results"
    gif_path = Path(gif_dir) if gif_dir else None
    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else None

    run = TrainingRun(
        framework="impala",
        env_id=env_id,
        num_workers=num_env_runners,
    )

    config = build_config(
        env_id=env_id,
        num_env_runners=num_env_runners,
        train_batch_size=p.get("train_batch_size", train_batch_size),
        lr=p.get("lr", lr),
        entropy_coeff=p.get("entropy_coeff", 0.01),
        vf_loss_coeff=p.get("vf_loss_coeff", 0.5),
        rollout_fragment_length=p.get("rollout_fragment_length", 50),
        num_gpus=num_gpus,
        min_time_s_per_iteration=p.get("min_time_s_per_iteration", 0.0),
        normalize_obs=p.get("normalize_obs", False),
        grad_clip=p.get("grad_clip", 40.0),
        num_epochs=p.get("num_epochs", 1),
        replay_proportion=p.get("replay_proportion", 0.0),
        replay_buffer_num_slots=p.get("replay_buffer_num_slots", 0),
    )
    algo = config.build_algo()

    print(f"\n{'='*65}")
    print(f"  Ray RLlib IMPALA  (asinhrono)")
    print(f"  Env:              {env_id}")
    print(f"  EnvRunners:       {num_env_runners}  ← asinhroni procesi")
    print(f"  BatchSize:        {p.get('train_batch_size', train_batch_size)} koraka/iter")
    print(f"  RolloutFragment:  {p.get('rollout_fragment_length', 50)} koraka/worker/slanje")
    print(f"  Cilj:             nagrada >= {target_reward}")
    print(f"  NormObs:          {p.get('normalize_obs', False)}")
    print(f"  lr:               {p.get('lr', lr)}")
    print(f"  num_epochs:       {p.get('num_epochs', 1)}  ← SGD prolaza po batchu")
    print(f"  entropy_coeff:    {p.get('entropy_coeff', 0.01)}")
    print(f"  grad_clip:        {p.get('grad_clip', 40.0)}")
    print(f"  replay_proportion:{p.get('replay_proportion', 0.0)}")
    print(f"  min_time/iter:    {p.get('min_time_s_per_iteration', 0.0)}s")
    if gif_path:
        print(f"  GIF-ovi:          {gif_path}")
    print(f"{'='*65}\n")
    print(f"{'Iter':>4} | {'Reward':>10} | {'Steps/s':>9} | {'Iter(s)':>8} | {'Total(s)':>9}")
    print(f"{'-'*60}")

    evolution_segments: list[tuple[str, list]] = []

    if gif_path:
        print("\n  Snimam random agenta (pre treninga)...")
        r_frames, r_reward = record_episode(env_id, max_steps=250)
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

            if math.isnan(reward) or math.isinf(reward):
                print(f"\n  [NaN] Gradienti su eksplodirali na iteraciji {i}. Stajem.")
                break

            print(
                f"{i:4d} | {reward:10.2f} | {throughput:9.0f} | "
                f"{iter_sec:8.1f} | {run.duration_sec:9.0f}"
            )

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

        # Sačuvaj checkpoint pre gašenja algo-a
        if ckpt_path:
            run.checkpoint_path = _save_checkpoint(algo, ckpt_path)
            print(f"\n  Checkpoint sačuvan: {run.checkpoint_path}")

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
    gif_path.mkdir(parents=True, exist_ok=True)
    print(f"\n  Generiše GIF-ove → {gif_path}")

    random_gif = gif_path / "random_agent.gif"
    if not random_gif.exists():
        r_frames, r_reward = record_episode(env_id, max_steps=300)
        print(f"    Slučajan agent: nagrada={r_reward:.0f}")
        save_gif(r_frames, random_gif)

    try:
        trained_frames, t_reward = record_ray_algo(algo, env_id, max_steps=500)
        print(f"    Naučen agent (IMPALA w={num_env_runners}): nagrada={t_reward:.0f}")
        save_gif(trained_frames, gif_path / f"impala_w{num_env_runners}_trained.gif")
    except Exception as e:
        print(f"    [UPOZORENJE] Nije mogao da snimi trained GIF: {e}")

    if len(evolution_segments) >= 2:
        try:
            f_frames, f_reward = record_ray_algo(algo, env_id, max_steps=400)
            evolution_segments.append((f"Finalni | nagrada≈{f_reward:.0f}", f_frames))
        except Exception:
            pass

        evolution_frames = build_evolution_gif(evolution_segments)
        save_gif(evolution_frames, gif_path / f"impala_w{num_env_runners}_evolution.gif", fps=24)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ray RLlib IMPALA trening",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--workers", type=int, default=2, help="Broj env_runners (asinhroni procesi)")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--target-reward", type=float, default=450)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout-fragment", type=int, default=50,
                        help="Koraka po workeru pre slanja learner-u")
    parser.add_argument("--gpu", type=float, default=0)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif-dir", default=None)
    parser.add_argument("--evolution-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Folder za čuvanje checkpointa posle treninga")
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
        num_gpus=args.gpu,
        ray_address=args.ray_address,
        output_dir=args.output,
        gif_dir=gif_dir,
        evolution_every=args.evolution_every,
        impala_params={"rollout_fragment_length": args.rollout_fragment},
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
