"""
Distribuirano treniranje SAC agenta pomoću Ray RLlib.

Šta je SAC:
  SAC (Soft Actor-Critic) je off-policy RL algoritam za kontinualne akcije.
  Koristi replay buffer (kao DQN) i actor-critic arhitekturu sa automatskim
  podešavanjem entropijskog koeficijenta.

  Intuicija: umesto da teži SAMO maksimalnoj nagradi, SAC teži MAKSIMALNOJ
  nagradi + MAKSIMALNOJ ENTROPIJI (istraživanju). Ovo sprečava prerano
  kolapsiranje politike u lokalni optimum.

  Ključne razlike od PPO/APPO:
  - Off-policy: uči iz starih iskustava u replay bufferu (data efficiency++)
  - Continuous-only: za diskretne akcije ne radi dobro (koristiti PPO/DQN)
  - Nema paralelizaciju učenja kao PPO (1 learner, više data collectora)

Zašto SAC za BipedalWalker:
  BipedalWalker ima kontinualne akcije → PPO-ova discreta ne važi.
  SAC je state-of-the-art za kontinualne kontrole i u SB3 referenci
  dostiže 300.53 ± 0.76 za samo 500k koraka (vs PPO-ovih 288 za 5M koraka)
  → demonstrira prednost off-policy algoritama u sample efikasnosti.

Referentni hiperparametri:
  https://huggingface.co/sb3/sac-BipedalWalker-v3 (nagrada 300.53 ± 0.76)

GIF-ovi koje ovaj fajl generiše (ako se zada --gif):
  - random_agent.gif       — agent pre treninga
  - sac_wN_trained.gif     — naučen agent sa N workera
  - sac_wN_evolution.gif   — napredak tokom treninga
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
from ray.rllib.algorithms.sac import SACConfig
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


def _register_clipped_envs() -> None:
    def _make_clipped_bipedal(cfg):
        # Klasa se definiše UNUTAR factory funkcije da cloudpickle može da je serializuje
        # bez importovanja modula u remote workerima (koji nemaju src/ u Python putu).
        import numpy as _np
        import gymnasium as _gym

        class _ClipWrapper(_gym.ObservationWrapper):
            def observation(self, obs):
                return _np.clip(obs, self.observation_space.low, self.observation_space.high)

        return _ClipWrapper(_gym.make("BipedalWalker-v3"))

    register_env("BipedalWalker-v3-clipped", _make_clipped_bipedal)


def _resolve_env_id(env_id: str) -> str:
    return _CLIPPED_ENV_MAP.get(env_id, env_id)


def _intensity_for_gradient_steps(
    gradient_steps: int,
    train_batch_size: int,
    rollout_fragment_length: int,
    num_env_runners: int,
    min_sample: int | None = None,
) -> float:
    """RLlib training_intensity za ukupno `gradient_steps` SGD po algo.train().

    algo.train() ponavlja training_step dok ne skupi min_sample koraka.
    Svaki training_step uzme fragment×workers koraka i uradi `weight` SGD.
    Ako to ne podelimo, 1024/256 = 4 kola × 64 SGD = 256 SGD/iter (4 min tišine).
    """
    native = train_batch_size / (
        rollout_fragment_length * max(num_env_runners + 1, 1)
    )
    steps_per_round = rollout_fragment_length * max(num_env_runners, 1)
    rounds = 1.0
    if min_sample and steps_per_round > 0:
        rounds = max(1.0, min_sample / steps_per_round)
    sgd_per_round = max(1.0, gradient_steps / rounds)
    return float(sgd_per_round) * native


def _set_log_std_init(algo, value: float) -> int:
    """Postavlja free log_std parametre na SB3 log_std_init (npr. -3)."""
    import torch as _torch

    n = 0
    policy = algo.get_policy()
    for name, param in policy.model.named_parameters():
        if "log_std" in name:
            with _torch.no_grad():
                param.fill_(value)
            n += 1
    if n and hasattr(algo, "env_runner_group") and algo.env_runner_group is not None:
        try:
            algo.env_runner_group.sync_weights()
        except Exception:
            pass
    return n


def build_config(
    env_id: str,
    num_env_runners: int,
    rollout_fragment_length: int,
    train_batch_size: int,
    lr: float,
    gamma: float = 0.98,
    tau: float = 0.02,
    buffer_size: int = 300_000,
    learning_starts: int = 10_000,
    training_intensity: float | None = None,
    gradient_steps: int = 64,
    log_std_init: float | None = None,
    target_update_freq: int = 1,
    net_arch: list[int] | None = None,
    normalize_obs: bool = False,
    min_sample_timesteps_per_iteration: int | None = None,
    num_gpus: float = 0,
) -> SACConfig:
    """
    Pravi SAC konfiguraciju za Ray RLlib.

    rollout_fragment_length — env koraci koje svaki worker skupi po pozivu (SB3: train_freq).
    train_batch_size        — mini-batch za svaki gradient update (SB3: batch_size).
    training_intensity      — gradient_steps / rollout_fragment (SB3: gradient_steps/train_freq).
                              1.0 = 1 gradient update po env koraku (SB3 default za ovaj env).
    learning_starts         — env koraka pre prvog gradient update-a (SB3: learning_starts).
    buffer_size             — kapacitet replay buffer-a (SB3: buffer_size).
    tau                     — koeficijent soft-update target mreže (SB3: tau).
    target_update_freq      — koliko gradient steps između target network update-a (SB3: 1).
    net_arch                — arhitektura mreže: [400, 300] za BipedalWalker (SB3 referenca).
    lr                      — isti learning rate za actor, critic i alpha (SB3: learning_rate).
    normalize_obs           — MeanStdFilter normalizacija opservacija.
    """
    if net_arch is None:
        net_arch = [400, 300]

    obs_filter = "MeanStdFilter" if normalize_obs else "NoFilter"

    if training_intensity is None:
        training_intensity = _intensity_for_gradient_steps(
            gradient_steps,
            train_batch_size,
            rollout_fragment_length,
            num_env_runners,
            min_sample_timesteps_per_iteration,
        )

    cfg = (
        SACConfig()
        .environment(_resolve_env_id(env_id), clip_actions=True)
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length=rollout_fragment_length,
            observation_filter=obs_filter,
        )
        .training(
            gamma=gamma,
            train_batch_size=train_batch_size,
            twin_q=True,
            tau=tau,
            initial_alpha=1.0,
            target_entropy="auto",
            replay_buffer_config={
                # Uniformni buffer kao SB3 (ne prioritized).
                # PrioritizedReplayBuffer sa alpha≈0 puca:
                #   weight = (p * n)**(-beta) → inf → IndexError.
                "type": "MultiAgentReplayBuffer",
                "capacity": buffer_size,
                "storage_unit": "timesteps",
            },
            training_intensity=training_intensity,
            num_steps_sampled_before_learning_starts=learning_starts,
            # Stari API stack čita lr iz optimization{}, ne iz actor_lr.
            optimization_config={
                "actor_learning_rate": lr,
                "critic_learning_rate": lr,
                "entropy_learning_rate": lr,
            },
            actor_lr=lr,
            critic_lr=lr,
            alpha_lr=lr,
            policy_model_config={
                "fcnet_hiddens": net_arch,
                "fcnet_activation": "relu",
                **({"free_log_std": True} if log_std_init is not None else {}),
            },
            q_model_config={
                "fcnet_hiddens": net_arch,
                "fcnet_activation": "relu",
            },
            target_network_update_freq=target_update_freq,
        )
        .resources(num_gpus=num_gpus)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )
    # RLlib SAC default: min_time=1s, min_sample=100 → ~1.4k koraka/iter i skoro
    # nula SGD (1 update na ~260 env koraka umesto 1:1 kao SB3).
    steps_gate = (
        min_sample_timesteps_per_iteration
        if min_sample_timesteps_per_iteration is not None
        else num_env_runners * rollout_fragment_length
    )
    cfg.min_time_s_per_iteration = 0
    cfg.min_sample_timesteps_per_iteration = steps_gate
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
    env_id: str = "BipedalWalker-v3",
    num_env_runners: int = 4,
    max_iterations: int = 2000,
    target_reward: float = 300,
    train_batch_size: int = 256,
    lr: float = 0.00073,
    num_gpus: float = 0,
    ray_address: str | None = None,
    output_dir: str | None = None,
    gif_dir: str | None = None,
    evolution_every: int = 400,
    sac_params: dict | None = None,
    checkpoint_dir: str | None = None,
) -> TrainingRun:
    """
    Trenira SAC agenta sa Ray RLlib.

    sac_params — env-specifični hiperparametri (iz experiments.yaml sac_override).
                 Podržani ključevi: lr, gamma, tau, buffer_size, train_batch_size,
                                    learning_starts, training_intensity, rollout_fragment_length,
                                    target_update_freq, net_arch, normalize_obs, print_every
    max_iterations — broj algo.train() poziva. Svaki poziv skuplja
                     rollout_fragment_length × num_workers env koraka.
                     Ukupno koraka ≈ max_iterations × rollout_fragment × num_workers.
    """
    if num_env_runners < 1:
        print(f"  [INFO] SAC zahteva num_env_runners >= 1. Postavljam na 1.")
        num_env_runners = 1

    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
        print(f"Spojen na Ray klaster: {ray_address}")
    else:
        ray.init(ignore_reinit_error=True)
        print("Lokalni Ray klaster podignut.")

    _register_clipped_envs()

    p = sac_params or {}
    out_path = Path(output_dir) if output_dir else ROOT / "results"
    gif_path = Path(gif_dir) if gif_dir else None
    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else None

    rollout_fragment = p.get("rollout_fragment_length", 64)
    effective_batch = p.get("train_batch_size", train_batch_size)
    effective_lr = p.get("lr", lr)
    net_arch = p.get("net_arch", [400, 300])
    print_every = p.get("print_every", 50)
    gradient_steps = int(p.get("gradient_steps", 64))
    log_std_init = p.get("log_std_init", None)
    if log_std_init is not None:
        log_std_init = float(log_std_init)
    min_sample = p.get(
        "min_sample_timesteps_per_iteration",
        num_env_runners * rollout_fragment,
    )

    intensity = p.get("training_intensity")
    if intensity is None:
        intensity = _intensity_for_gradient_steps(
            gradient_steps,
            effective_batch,
            rollout_fragment,
            num_env_runners,
            min_sample,
        )

    steps_per_iter = min_sample
    total_steps_estimate = max_iterations * steps_per_iter

    run = TrainingRun(
        framework="sac",
        env_id=env_id,
        num_workers=num_env_runners,
    )

    config = build_config(
        env_id=env_id,
        num_env_runners=num_env_runners,
        rollout_fragment_length=rollout_fragment,
        train_batch_size=effective_batch,
        lr=effective_lr,
        gamma=p.get("gamma", 0.98),
        tau=p.get("tau", 0.02),
        buffer_size=p.get("buffer_size", 300_000),
        learning_starts=p.get("learning_starts", 10_000),
        training_intensity=intensity,
        gradient_steps=gradient_steps,
        log_std_init=log_std_init,
        target_update_freq=p.get("target_update_freq", 1),
        net_arch=net_arch,
        normalize_obs=p.get("normalize_obs", False),
        min_sample_timesteps_per_iteration=min_sample,
        num_gpus=num_gpus,
    )
    algo = config.build_algo()
    n_logstd = _set_log_std_init(algo, log_std_init) if log_std_init is not None else 0

    print(f"\n{'='*65}")
    print(f"  Ray RLlib SAC  (off-policy, kontinualne akcije)")
    print(f"  Env:              {env_id}")
    print(f"  EnvRunners:       {num_env_runners}  ← data kolektori")
    print(f"  train_freq:       {rollout_fragment}  (SB3: train_freq=64)")
    print(f"  gradient_steps:   {gradient_steps} ukupno po iteraciji  (SB3: 64)")
    print(f"  StepsPerIter:     ~{steps_per_iter} ({num_env_runners}w × {rollout_fragment})")
    print(f"  batch_size:       {effective_batch}  (SB3: 256)")
    print(f"  buffer_size:      {p.get('buffer_size', 300_000)}")
    print(f"  learning_starts:  {p.get('learning_starts', 10_000)}")
    print(f"  n_timesteps:      ~{total_steps_estimate/1e3:.0f}k ({max_iterations} itera)")
    print(f"  lr:               {effective_lr}  (SB3: 0.00073)")
    print(f"  gamma / tau:      {p.get('gamma', 0.98)} / {p.get('tau', 0.02)}")
    print(f"  net_arch:         {net_arch}")
    print(f"  log_std_init:     {log_std_init if log_std_init is not None else 'default (mreža; -3 samo uz gSDE)'}")
    print(f"  ent_coef:         auto")
    print(f"  normalize:        {p.get('normalize_obs', False)}")
    print(f"  use_sde:          NIJE u RLlib SAC — nema ekvivalenta")
    print(f"  Cilj:             nagrada >= {target_reward}")
    print(f"  Štampanje:        svaka {print_every}. iteracija")
    if gif_path:
        print(f"  GIF-ovi:          {gif_path}")
    print(f"{'='*65}\n")
    print(f"{'Iter':>6} | {'Steps':>8} | {'Reward':>10} | {'Steps/s':>9} | {'Total(s)':>9}")
    print(f"{'-'*62}")

    evolution_segments: list[tuple[str, list]] = []

    if gif_path:
        print("\n  Snimam random agenta (pre treninga)...")
        r_frames, r_reward = record_episode(env_id, max_steps=500)
        print(f"  → nagrada={r_reward:.0f}")
        evolution_segments.append(("Random (0 iteracija)", r_frames))

    total_steps = 0
    last_reward = 0.0

    try:
        for iteration in range(max_iterations):
            t0 = time.time()
            result = algo.train()
            iter_sec = time.time() - t0

            steps_this_iter = result.get("num_env_steps_sampled_this_iter", 0) or 0
            total_steps += steps_this_iter

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

            run.add_iteration(iteration, reward, throughput=throughput, extra={
                "iter_sec": round(iter_sec, 3),
                "total_steps": total_steps,
            })

            if (iteration + 1) % print_every == 0 or iteration == 0:
                print(
                    f"{iteration+1:6d} | {total_steps:8d} | {reward:10.2f} | "
                    f"{throughput:9.0f} | {run.duration_sec:9.0f}"
                )

            if gif_path and (iteration + 1) % evolution_every == 0:
                frames, ep_reward = record_ray_algo(algo, env_id, max_steps=1000)
                label = f"Iter {iteration+1} | nagrada≈{ep_reward:.0f}"
                evolution_segments.append((label, frames))
                print(f"  [Evolution] {label}")

            if reward >= target_reward:
                print(f"\n  Ciljna nagrada {target_reward} dostignuta za {run.duration_sec:.0f}s!")
                break

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
    print(f"\nUkupno prikupljenih koraka: {total_steps:,}")
    print(f"Prosečni throughput: {run.avg_throughput():.0f} koraka/sec")
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
        r_frames, r_reward = record_episode(env_id, max_steps=600)
        print(f"    Slučajan agent: nagrada={r_reward:.0f}")
        save_gif(r_frames, random_gif)

    try:
        trained_frames, t_reward = record_ray_algo(algo, env_id, max_steps=1600)
        print(f"    Naučen agent (SAC w={num_env_runners}): nagrada={t_reward:.0f}")
        save_gif(trained_frames, gif_path / f"sac_w{num_env_runners}_trained.gif")
    except Exception as e:
        print(f"    [UPOZORENJE] Nije mogao da snimi trained GIF: {e}")

    if len(evolution_segments) >= 2:
        try:
            f_frames, f_reward = record_ray_algo(algo, env_id, max_steps=1600)
            evolution_segments.append((f"Finalni | nagrada≈{f_reward:.0f}", f_frames))
        except Exception:
            pass

        evolution_frames = build_evolution_gif(evolution_segments)
        save_gif(evolution_frames, gif_path / f"sac_w{num_env_runners}_evolution.gif", fps=24)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ray RLlib SAC trening (off-policy, kontinualne akcije)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="BipedalWalker-v3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=2000,
                        help="Broj algo.train() poziva (koraci = iters × workers × rollout)")
    parser.add_argument("--target-reward", type=float, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.00073)
    parser.add_argument("--rollout-fragment", type=int, default=64,
                        help="Env koraka po workeru po pozivu (SB3: train_freq)")
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--gpu", type=float, default=0)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif-dir", default=None)
    parser.add_argument("--evolution-every", type=int, default=400)
    parser.add_argument("--print-every", type=int, default=50)
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
        sac_params={
            "rollout_fragment_length": args.rollout_fragment,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "print_every": args.print_every,
        },
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
