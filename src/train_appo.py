"""
Distribuirano treniranje APPO agenta pomoću Ray RLlib.

Šta je APPO:
  APPO (Asynchronous Proximal Policy Optimization) je asinhroni RL algoritam
  koji kombinuje prednosti IMPALA-e i PPO-a:
  - Kao IMPALA: radnici skupljaju iskustvo ASINHRONO, learner ne čeka najsporijeg
  - Kao PPO: koristi klipovan surrogate objektiv (ne V-trace), stabilan update

  Intuicija: radnici igraju igru i šalju trajektorije u centralni red čekanja.
  Learner uzima iz reda i radi PPO update — ali umesto V-trace IS korekcije,
  koristi PPO-ovo klipovanje koje je robusnije na zastalelost iskustva.

Zašto APPO za poređenje sa PPO:
  PPO  — sinhroni: learner čeka SVE radnike → idle vreme raste sa brojem radnika
  APPO — asinhroni: learner nikad ne čeka → bolje iskorišćenje resursa na klasteru
  ISTI cilj (PPO clipping) → razlika dolazi isključivo od modela izvršavanja.
  Ovo je centralna eksperimentalna priča: sync vs async, isti algoritam, ista stabilnost.

GIF-ovi koje ovaj fajl generiše (ako se zada --gif):
  - random_agent.gif       — agent pre treninga
  - appo_wN_trained.gif    — naučen agent sa N workera
  - appo_wN_evolution.gif  — napredak tokom treninga
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
from ray.rllib.algorithms.appo import APPOConfig
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
    "LunarLander-v3": "LunarLander-v3-clipped",
}


def _register_clipped_envs() -> None:
    """Klipuje opservacije u Box granice.

    LunarLander-v3 i BipedalWalker-v3 mogu da vrate vrednosti van space-a
    (npr. visina landera y>2.5). RLlib preprocessor tad diže ValueError.
    """

    def _make_clipped(base_id: str):
        def _factory(cfg):
            import numpy as _np
            import gymnasium as _gym

            class _ClipWrapper(_gym.ObservationWrapper):
                def observation(self, obs):
                    return _np.clip(
                        obs, self.observation_space.low, self.observation_space.high
                    )

            return _ClipWrapper(_gym.make(base_id))

        return _factory

    register_env("BipedalWalker-v3-clipped", _make_clipped("BipedalWalker-v3"))
    register_env("LunarLander-v3-clipped", _make_clipped("LunarLander-v3"))


def _resolve_env_id(env_id: str) -> str:
    return _CLIPPED_ENV_MAP.get(env_id, env_id)


def build_config(
    env_id: str,
    num_env_runners: int,
    rollout_fragment_length: int,
    train_batch_size: int,
    lr: float,
    entropy_coeff: float = 0.01,
    vf_loss_coeff: float = 0.5,
    gae_lambda: float = 0.95,
    gamma: float = 0.99,
    clip_param: float = 0.2,
    num_sgd_iter: int = 1,
    sgd_minibatch_size: int | None = None,
    num_gpus: float = 0,
    normalize_obs: bool = False,
    grad_clip: float | None = 40.0,
    min_time_s_per_iteration: float = 0.0,
    min_sample_timesteps_per_iteration: int | None = None,
    vtrace: bool = True,
    use_circular_buffer: bool = False,
) -> APPOConfig:
    """
    Pravi APPO konfiguraciju za Ray RLlib.

    APPO = Asynchronous PPO: isti parametri kao PPO, asinhrono izvršavanje.

    num_env_runners       — broj asinhrohnih env radnika.
    rollout_fragment_length — koraka koje svaki worker skupi pre slanja learner-u.
    train_batch_size      — ukupno koraka pre svakog gradient update-a.
    clip_param            — PPO clipping (0.2 = standardni PPO).
    num_sgd_iter          — SGD prolaza po batchu (default 1 za async; PPO=4).
    gae_lambda            — GAE faktor (isti kao PPO).
    grad_clip             — maksimalna norma gradijenta (40 = IMPALA standard).
    normalize_obs         — MeanStdFilter normalizacija (kritično za BipedalWalker).
    min_time_s_per_iteration — 0; gating ide preko min_sample, ne preko zida od 10s.
    min_sample_timesteps_per_iteration — koliko env koraka mora da se sakupi po
        algo.train(). Default = train_batch_size. Bez ovoga APPO (async) vraća
        prazne iteracije od ~0.2s.
    """
    obs_filter = "MeanStdFilter" if normalize_obs else "NoFilter"

    training_kwargs: dict = dict(
        lr=lr,
        entropy_coeff=entropy_coeff,
        vf_loss_coeff=vf_loss_coeff,
        lambda_=gae_lambda,
        gamma=gamma,
        clip_param=clip_param,
        num_epochs=num_sgd_iter,
        train_batch_size=train_batch_size,
        vtrace=vtrace,
        use_circular_buffer=use_circular_buffer,
    )
    if sgd_minibatch_size is not None:
        # APPO/IMPALA: AlgorithmConfig.training(minibatch_size=...), ne sgd_minibatch_size.
        training_kwargs["minibatch_size"] = sgd_minibatch_size
    if grad_clip is not None:
        training_kwargs["grad_clip"] = grad_clip

    cfg = (
        APPOConfig()
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
    cfg.min_time_s_per_iteration = min_time_s_per_iteration
    # APPO default min_sample=0 + min_time=10s. Ako min_time spustimo na 0
    # bez min_sample, train() se vrati čim learner uradi prazan korak.
    cfg.min_sample_timesteps_per_iteration = (
        min_sample_timesteps_per_iteration
        if min_sample_timesteps_per_iteration is not None
        else train_batch_size
    )
    return cfg


def _is_valid_checkpoint(path: Path) -> bool:
    return path.is_dir() and (
        (path / "rllib_checkpoint.json").exists()
        or (path / "algorithm_state.pkl").exists()
    )


def _save_checkpoint(algo, ckpt_path: Path) -> str | None:
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
    num_env_runners: int = 4,
    max_iterations: int = 100,
    target_reward: float = 450,
    train_batch_size: int = 4000,
    lr: float = 3e-4,
    num_gpus: float = 0,
    ray_address: str | None = None,
    output_dir: str | None = None,
    gif_dir: str | None = None,
    evolution_every: int = 15,
    appo_params: dict | None = None,
    checkpoint_dir: str | None = None,
) -> TrainingRun:
    """
    Trenira APPO agenta sa Ray RLlib.

    appo_params — env-specifični hiperparametri (iz experiments.yaml appo_override).
                  Podržani ključevi: lr, entropy_coeff, vf_loss_coeff, gae_lambda,
                                     gamma, clip_param, num_sgd_iter, sgd_minibatch_size,
                                     rollout_fragment_length, train_batch_size,
                                     normalize_obs, grad_clip, min_time_s_per_iteration
    """
    if num_env_runners < 1:
        print(f"  [INFO] APPO zahteva num_env_runners >= 1. Postavljam na 1 (bilo: {num_env_runners}).")
        num_env_runners = 1

    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
        print(f"Spojen na Ray klaster: {ray_address}")
    else:
        ray.init(ignore_reinit_error=True)
        print("Lokalni Ray klaster podignut.")

    _register_clipped_envs()

    p = appo_params or {}
    out_path = Path(output_dir) if output_dir else ROOT / "results"
    gif_path = Path(gif_dir) if gif_dir else None
    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else None

    effective_batch = p.get("train_batch_size", train_batch_size)
    rollout_fragment = p.get("rollout_fragment_length", 100)
    min_sample = p.get("min_sample_timesteps_per_iteration", effective_batch)

    run = TrainingRun(
        framework="appo",
        env_id=env_id,
        num_workers=num_env_runners,
    )

    config = build_config(
        env_id=env_id,
        num_env_runners=num_env_runners,
        rollout_fragment_length=rollout_fragment,
        train_batch_size=effective_batch,
        lr=p.get("lr", lr),
        entropy_coeff=p.get("entropy_coeff", 0.01),
        vf_loss_coeff=p.get("vf_loss_coeff", 0.5),
        gae_lambda=p.get("gae_lambda", 0.95),
        gamma=p.get("gamma", 0.99),
        clip_param=p.get("clip_param", 0.2),
        num_sgd_iter=p.get("num_sgd_iter", 1),
        sgd_minibatch_size=p.get("sgd_minibatch_size", None),
        num_gpus=num_gpus,
        normalize_obs=p.get("normalize_obs", False),
        grad_clip=p.get("grad_clip", 40.0),
        min_time_s_per_iteration=p.get("min_time_s_per_iteration", 0.0),
        min_sample_timesteps_per_iteration=min_sample,
        vtrace=p.get("vtrace", True),
        use_circular_buffer=p.get("use_circular_buffer", False),
    )
    algo = config.build_algo()

    print(f"\n{'='*65}")
    print(f"  Ray RLlib APPO  (async PPO)")
    print(f"  Env:              {env_id}")
    print(f"  EnvRunners:       {num_env_runners}  ← asinhroni procesi")
    print(f"  BatchSize:        {effective_batch} koraka/iter")
    print(f"  RolloutFragment:  {rollout_fragment} koraka/worker/slanje")
    print(f"  Cilj:             nagrada >= {target_reward}")
    print(f"  NormObs:          {p.get('normalize_obs', False)}")
    print(f"  lr:               {p.get('lr', lr)}")
    print(f"  num_sgd_iter:     {p.get('num_sgd_iter', 1)}  ← SGD prolaza/batchu")
    print(f"  clip_param:       {p.get('clip_param', 0.2)}  ← PPO clipping")
    print(f"  vtrace:           {p.get('vtrace', True)}  ← RLlib APPO zahteva V-trace")
    print(f"  entropy_coeff:    {p.get('entropy_coeff', 0.01)}")
    print(f"  gae_lambda:       {p.get('gae_lambda', 0.95)}")
    print(f"  grad_clip:        {p.get('grad_clip', 40.0)}")
    print(f"  min_sample/iter:  {min_sample} koraka")
    print(f"  Ukupno koraka:    ~{max_iterations * min_sample / 1e3:.0f}k ({max_iterations} itera)")
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

    # APPO je asinhroni: radnici skupljaju podatke paralelno s learnerom.
    # num_env_steps_sampled_this_iter je uvijek 0 za APPO jer async workers ne
    # prijavljuju po-iteracijske korake. Koristimo razliku kumulativnog brojača.
    grad_iter = 0
    last_reward = 0.0
    prev_total_steps = 0

    try:
        while grad_iter < max_iterations:
            t0 = time.time()
            result = algo.train()
            iter_sec = time.time() - t0

            # Kumulativni koraci → razlika daje stvarni broj koraka u ovoj iteraciji
            total_steps = (
                result.get("num_env_steps_sampled", 0)
                or result.get("num_agent_steps_sampled", 0)
                or 0
            )
            steps_this_iter = max(0, total_steps - prev_total_steps)
            prev_total_steps = total_steps

            env_stats = result.get("env_runners", result.get("sampler_results", {}))
            reward = env_stats.get(
                "episode_return_mean",
                env_stats.get("episode_reward_mean", None),
            )
            if reward is None or (isinstance(reward, float) and math.isnan(reward)):
                reward = last_reward
            else:
                reward = float(reward)
                last_reward = reward

            throughput = steps_this_iter / iter_sec if (iter_sec > 0 and steps_this_iter > 0) else 0.0

            run.add_iteration(grad_iter, reward, throughput=throughput, extra={"iter_sec": round(iter_sec, 2)})

            print(
                f"{grad_iter:4d} | {reward:10.2f} | {throughput:9.0f} | "
                f"{iter_sec:8.1f} | {run.duration_sec:9.0f}"
            )

            if gif_path and (grad_iter + 1) % evolution_every == 0:
                frames, ep_reward = record_ray_algo(algo, env_id, max_steps=300)
                label = f"Iter {grad_iter+1} | nagrada≈{ep_reward:.0f}"
                evolution_segments.append((label, frames))
                print(f"  [Evolution] {label}")

            if reward >= target_reward:
                print(f"\n  Ciljna nagrada {target_reward} dostignuta za {run.duration_sec:.0f}s!")
                break

            grad_iter += 1

    finally:
        run.finish()

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
        print(f"    Naučen agent (APPO w={num_env_runners}): nagrada={t_reward:.0f}")
        save_gif(trained_frames, gif_path / f"appo_w{num_env_runners}_trained.gif")
    except Exception as e:
        print(f"    [UPOZORENJE] Nije mogao da snimi trained GIF: {e}")

    if len(evolution_segments) >= 2:
        try:
            f_frames, f_reward = record_ray_algo(algo, env_id, max_steps=400)
            evolution_segments.append((f"Finalni | nagrada≈{f_reward:.0f}", f_frames))
        except Exception:
            pass

        evolution_frames = build_evolution_gif(evolution_segments)
        save_gif(evolution_frames, gif_path / f"appo_w{num_env_runners}_evolution.gif", fps=24)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ray RLlib APPO trening (async PPO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--target-reward", type=float, default=450)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout-fragment", type=int, default=100)
    parser.add_argument("--gpu", type=float, default=0)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif-dir", default=None)
    parser.add_argument("--evolution-every", type=int, default=15)
    parser.add_argument("--checkpoint-dir", default=None)
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
        appo_params={"rollout_fragment_length": args.rollout_fragment},
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
