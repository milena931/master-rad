"""
Distribuirano treniranje DQN agenta pomoću Ray RLlib.

Šta je DQN:
  DQN (Deep Q-Network) je off-policy RL algoritam za DISKRETNE akcije.
  Koristi replay buffer za break korelacije između uzoraka i target mrežu
  za stabilno učenje.

  Arhitektura: jedna Q-mreža procenjuje Q(s,a) za sve akcije odjednom.
  Akcija se bira ε-greedy: nasumično sa verovatnoćom ε, inače argmax Q(s,a).

  Ključne razlike od PPO:
  - Off-policy: uči iz replay buffer-a (prošlih iskustava) → data efficiency++
  - Diskretne akcije ONLY: LunarLander, CartPole (ne BipedalWalker)
  - Double DQN: dve mreže za stabilniju procenu Q vrednosti (default ON)

Referentni hiperparametri za LunarLander:
  https://huggingface.co/sb3/dqn-LunarLander-v2
  Nagrada: 136.79 ± 42.72 za 100k koraka (DQN je sample-efficient ali slabiji od PPO)

GIF-ovi koje ovaj fajl generiše (ako se zada --gif):
  - random_agent.gif       — agent pre treninga
  - dqn_wN_trained.gif     — naučen agent sa N workera
  - dqn_wN_evolution.gif   — napredak tokom treninga
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.utils.from_config import from_config as _from_config

from metrics import TrainingRun, save_run
from play_game import (
    build_evolution_gif,
    record_episode,
    record_ray_algo,
    save_gif,
)

ROOT = Path(__file__).parent.parent


def _patch_replay_buffer_check() -> None:
    """
    Workaround za Ray bug koji se pojavljuje samo za DQN (ne SAC/APPO).

    Razlog: DQNConfig.validate() poziva validate_buffer_config() koji konvertuje
    'type' string (npr. "MultiAgentPrioritizedReplayBuffer") u Python klasu.
    Zatim Algorithm._create_local_replay_buffer_if_necessary() pokušava:
        if "EpisodeReplayBuffer" in config["replay_buffer_config"]["type"]:
    što pada sa TypeError jer je type sada klasa (ABCMeta), ne string.

    SAC nema ovaj problem jer SACConfig.MRO = [SACConfig, AlgorithmConfig]
    (ne prolazi kroz DQNConfig.validate i validate_buffer_config).

    Fix: zameniti proveru sa verzijom koja handluje i string i klasu.
    """
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer

    if getattr(Algorithm._create_local_replay_buffer_if_necessary, "_dqn_patched", False):
        return

    def _patched(self, config):
        rbc = None
        try:
            rbc = config.get("replay_buffer_config")
        except Exception:
            pass
        if not rbc:
            return None
        if rbc.get("no_local_replay_buffer"):
            return None

        type_ = rbc.get("type")
        if isinstance(type_, type):
            is_episode_buf = "EpisodeReplayBuffer" in type_.__name__
        elif isinstance(type_, str):
            is_episode_buf = "EpisodeReplayBuffer" in type_
        else:
            is_episode_buf = False

        if is_episode_buf:
            rbc["metrics_num_episodes_for_smoothing"] = (
                self.config.metrics_num_episodes_for_smoothing
            )
        return _from_config(ReplayBuffer, rbc)

    _patched._dqn_patched = True
    Algorithm._create_local_replay_buffer_if_necessary = _patched


def build_config(
    env_id: str,
    num_env_runners: int,
    train_freq: int,
    batch_size: int,
    lr: float,
    gamma: float = 0.99,
    buffer_size: int = 50_000,
    learning_starts: int = 0,
    exploration_timesteps: int = 12_000,
    initial_eps: float = 1.0,
    final_eps: float = 0.1,
    target_update_interval: int = 250,
    net_arch: list[int] | None = None,
    double_q: bool = True,
    dueling: bool = False,
    training_intensity: float | None = None,
    min_sample_timesteps_per_iteration: int = 1000,
    num_gpus: float = 0,
) -> DQNConfig:
    """
    Pravi DQN konfiguraciju za Ray RLlib.

    train_freq              — env koraka po workeru pre gradient update-a (SB3: train_freq).
    batch_size              — mini-batch za gradient update (SB3: batch_size).
    exploration_timesteps   — ukupnih env koraka za epsilon decay (SB3: exploration_fraction × n_timesteps).
    initial_eps             — početni epsilon za ε-greedy istraživanje.
    final_eps               — finalni epsilon (SB3: exploration_final_eps).
    target_update_interval  — gradient koraka između target mreža update-a (SB3: target_update_interval).
    net_arch                — arhitektura Q-mreže (SB3: policy_kwargs.net_arch).
    double_q                — Double DQN (SB3 default: True).
    dueling                 — Dueling DQN (SB3 DQN default: False).
    training_intensity      — gradient_steps/env_step ratio. None = Ray default (1/call).
                              Ako je zadat gradient_steps, računa se kao
                              gradient_steps * batch_size / train_freq (SB3).
    """
    _patch_replay_buffer_check()

    if net_arch is None:
        net_arch = [256, 256]

    training_kwargs: dict = dict(
        gamma=gamma,
        lr=lr,
        train_batch_size=batch_size,
        target_network_update_freq=target_update_interval,
        replay_buffer_config={
                # MultiAgentPrioritizedReplayBuffer je jedini tip podržan za stari API stack.
                # alpha≈0 = uniformno uzorkovanje (SB3 DQN ne koristi prioritized replay).
                # validate_buffer_config() konvertuje string u klasu → patchujemo
                # _create_local_replay_buffer_if_necessary da to handluje.
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": buffer_size,
                "prioritized_replay_alpha": 1e-5,
                "prioritized_replay_beta": 1.0,
                "prioritized_replay_eps": 1e-6,
            },
        epsilon=[(0, initial_eps), (exploration_timesteps, final_eps)],
        num_steps_sampled_before_learning_starts=learning_starts,
        double_q=double_q,
        dueling=dueling,
        hiddens=net_arch,
        n_step=1,
    )
    if training_intensity is not None:
        training_kwargs["training_intensity"] = training_intensity

    cfg = (
        DQNConfig()
        .environment(env_id)
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length=train_freq,
        )
        .training(**training_kwargs)
        .reporting(min_sample_timesteps_per_iteration=min_sample_timesteps_per_iteration)
        .resources(num_gpus=num_gpus)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )
    # CPU: bez ovoga RLlib koristi MultiGPULearnerThread → _queue.Empty
    # i učenje stane (CartPole eval 9 / kolaps posle iter 20).
    if num_gpus == 0:
        cfg.simple_optimizer = True
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
    env_id: str = "LunarLander-v3",
    num_env_runners: int = 4,
    max_iterations: int = 6500,
    target_reward: float = 200,
    batch_size: int = 128,
    lr: float = 0.00063,
    num_gpus: float = 0,
    ray_address: str | None = None,
    output_dir: str | None = None,
    gif_dir: str | None = None,
    evolution_every: int = 1300,
    dqn_params: dict | None = None,
    checkpoint_dir: str | None = None,
) -> TrainingRun:
    """
    Trenira DQN agenta sa Ray RLlib.

    dqn_params — env-specifični hiperparametri (iz experiments.yaml dqn_override).
                 Podržani ključevi: lr, gamma, batch_size, buffer_size, learning_starts,
                                    exploration_timesteps, initial_eps, final_eps,
                                    target_update_interval, train_freq, net_arch,
                                    double_q, dueling, training_intensity, gradient_steps, print_every
    max_iterations — broj algo.train() poziva.
                     Ukupno koraka ≈ max_iterations × train_freq × num_workers.
    """
    if num_env_runners < 1:
        print(f"  [INFO] DQN zahteva num_env_runners >= 1. Postavljam na 1.")
        num_env_runners = 1

    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
        print(f"Spojen na Ray klaster: {ray_address}")
    else:
        ray.init(ignore_reinit_error=True)
        print("Lokalni Ray klaster podignut.")

    p = dqn_params or {}
    out_path = Path(output_dir) if output_dir else ROOT / "results"
    gif_path = Path(gif_dir) if gif_dir else None
    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else None

    train_freq = int(p.get("train_freq", 4))
    effective_batch = p.get("batch_size", batch_size)
    effective_lr = p.get("lr", lr)
    net_arch = p.get("net_arch", [256, 256])
    print_every = p.get("print_every", 5)
    min_sample = p.get("min_sample_timesteps_per_iteration", 1000)
    gradient_steps = p.get("gradient_steps")
    intensity = p.get("training_intensity", None)
    if intensity is None and gradient_steps is not None:
        # SB3: 128 SGD svakih 256 env koraka → intensity = 128*64/256 = 32
        tfreq = max(train_freq, 1)
        intensity = float(gradient_steps) * float(effective_batch) / float(tfreq)

    # RLlib DQN default je min_sample_timesteps_per_iteration=1000:
    # svaki algo.train() sakuplja bar toliko koraka, ne samo train_freq × workers.
    steps_per_iter = max(min_sample, num_env_runners * train_freq)
    total_steps_estimate = max_iterations * steps_per_iter

    run = TrainingRun(
        framework="dqn",
        env_id=env_id,
        num_workers=num_env_runners,
    )

    config = build_config(
        env_id=env_id,
        num_env_runners=num_env_runners,
        train_freq=train_freq,
        batch_size=effective_batch,
        lr=effective_lr,
        gamma=p.get("gamma", 0.99),
        buffer_size=p.get("buffer_size", 50_000),
        learning_starts=p.get("learning_starts", 0),
        exploration_timesteps=p.get("exploration_timesteps", 12_000),
        initial_eps=p.get("initial_eps", 1.0),
        final_eps=p.get("final_eps", 0.1),
        target_update_interval=p.get("target_update_interval", 250),
        net_arch=net_arch,
        double_q=p.get("double_q", True),
        dueling=p.get("dueling", False),
        training_intensity=intensity,
        min_sample_timesteps_per_iteration=min_sample,
        num_gpus=num_gpus,
    )
    algo = config.build_algo()

    explore_steps = p.get("exploration_timesteps", 12_000)
    print(f"\n{'='*65}")
    print(f"  Ray RLlib DQN  (off-policy, diskretne akcije)")
    print(f"  Env:              {env_id}")
    print(f"  EnvRunners:       {num_env_runners}  ← data kolektori")
    print(f"  TrainFreq:        {train_freq} koraka/worker/fragment  (SB3: train_freq)")
    print(f"  StepsPerIter:     ~{steps_per_iter} (RLlib min_sample={min_sample})")
    print(f"  BatchSize:        {effective_batch}  (SB3: batch_size)")
    print(f"  BufferSize:       {p.get('buffer_size', 50_000)}")
    print(f"  LearningStarts:   {p.get('learning_starts', 0)} koraka")
    print(f"  Epsilon:          1.0 → {p.get('final_eps', 0.1)} za {explore_steps} koraka")
    print(f"  TargetUpdate:     svaka {p.get('target_update_interval', 250)} grad. koraka")
    if gradient_steps is not None:
        print(f"  gradient_steps:   {int(gradient_steps)}  (SB3) → intensity={intensity:.1f}")
    print(f"  Ukupno koraka:    ~{total_steps_estimate/1e3:.0f}k ({max_iterations} itera)")
    print(f"  Cilj:             nagrada >= {target_reward}")
    print(f"  lr:               {effective_lr}")
    print(f"  gamma:            {p.get('gamma', 0.99)}")
    print(f"  net_arch:         {net_arch}")
    print(f"  double_q:         {p.get('double_q', True)}")
    print(f"  dueling:          {p.get('dueling', False)}")
    print(f"  Štampanje:        svaka {print_every}. iteracija")
    if gif_path:
        print(f"  GIF-ovi:          {gif_path}")
    print(f"{'='*65}\n")
    print(f"{'Iter':>6} | {'Steps':>8} | {'Reward':>10} | {'Steps/s':>9} | {'Total(s)':>9}")
    print(f"{'-'*62}")

    evolution_segments: list[tuple[str, list]] = []

    if gif_path:
        print("\n  Snimam random agenta (pre treninga)...")
        r_frames, r_reward = record_episode(env_id, max_steps=1000)
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
        r_frames, r_reward = record_episode(env_id, max_steps=1000)
        print(f"    Slučajan agent: nagrada={r_reward:.0f}")
        save_gif(r_frames, random_gif)

    try:
        trained_frames, t_reward = record_ray_algo(algo, env_id, max_steps=1000)
        print(f"    Naučen agent (DQN w={num_env_runners}): nagrada={t_reward:.0f}")
        save_gif(trained_frames, gif_path / f"dqn_w{num_env_runners}_trained.gif")
    except Exception as e:
        print(f"    [UPOZORENJE] Nije mogao da snimi trained GIF: {e}")

    if len(evolution_segments) >= 2:
        try:
            f_frames, f_reward = record_ray_algo(algo, env_id, max_steps=1000)
            evolution_segments.append((f"Finalni | nagrada≈{f_reward:.0f}", f_frames))
        except Exception:
            pass

        evolution_frames = build_evolution_gif(evolution_segments)
        save_gif(evolution_frames, gif_path / f"dqn_w{num_env_runners}_evolution.gif", fps=24)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ray RLlib DQN trening (off-policy, diskretne akcije)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=6500,
                        help="Broj algo.train() poziva (koraci = iters × workers × train_freq)")
    parser.add_argument("--target-reward", type=float, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.00063)
    parser.add_argument("--train-freq", type=int, default=4,
                        help="Env koraka po workeru pre gradient update-a (SB3: train_freq)")
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=0)
    parser.add_argument("--exploration-timesteps", type=int, default=12_000)
    parser.add_argument("--final-eps", type=float, default=0.1)
    parser.add_argument("--target-update", type=int, default=250)
    parser.add_argument("--gpu", type=float, default=0)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif-dir", default=None)
    parser.add_argument("--evolution-every", type=int, default=1300)
    parser.add_argument("--print-every", type=int, default=100)
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
        batch_size=args.batch_size,
        lr=args.lr,
        num_gpus=args.gpu,
        ray_address=args.ray_address,
        output_dir=args.output,
        gif_dir=gif_dir,
        evolution_every=args.evolution_every,
        dqn_params={
            "train_freq": args.train_freq,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "exploration_timesteps": args.exploration_timesteps,
            "final_eps": args.final_eps,
            "target_update_interval": args.target_update,
            "print_every": args.print_every,
        },
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
