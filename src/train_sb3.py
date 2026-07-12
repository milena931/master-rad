"""
Baseline treniranje PPO agenta pomoću Stable-Baselines3.

Šta ovaj fajl radi:
  - Trenira PPO agenta u bilo kom Gymnasium okruženju
  - Čuva model svakih N koraka (međukoraci)
  - Na kraju generiše 3 GIF-a:
      1. random_agent.gif    — kako izgleda bez treninga
      2. trained_agent.gif   — naučen agent
      3. evolution.gif       — razvoj od random do naučenog (spojeni svi međukoraci)

Zašto SB3 u master radu:
  Koristimo ga kao baseline — isti PPO algoritam, jedna mašina, bez Ray distribucije.
  Poređenje: SB3 n_envs=4 vs Ray num_env_runners=4 → koji daje veći throughput?
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from metrics import TrainingRun, save_run
from play_game import (
    build_evolution_gif,
    record_episode,
    save_gif,
)

ROOT = Path(__file__).parent.parent


class MetricsCallback(BaseCallback):
    """
    Hvata metrike tokom SB3 treninga (nagrada + throughput).
    Poziva se na svakom vektorisanom koraku (jednom po n_envs paralelnih koraka).
    Loguje svakih ~rollout_steps koraka po env-u.
    """

    def __init__(self, run: TrainingRun, rollout_steps: int = 2048):
        super().__init__()
        self.run = run
        self.rollout_steps = rollout_steps
        self.iteration = 0
        self._last_log_call = 0

    def _on_step(self) -> bool:
        steps_per_log = self.training_env.num_envs * self.rollout_steps
        if self.n_calls - self._last_log_call < steps_per_log:
            return True
        if len(self.model.ep_info_buffer) == 0:
            return True

        self._last_log_call = self.n_calls
        rewards = [ep["r"] for ep in self.model.ep_info_buffer]
        mean_reward = sum(rewards) / len(rewards)
        elapsed = self.run.duration_sec
        throughput = self.num_timesteps / elapsed if elapsed > 0 else 0.0

        self.run.add_iteration(self.iteration, mean_reward, throughput=throughput)
        print(
            f"Step {self.num_timesteps:7d} | reward={mean_reward:8.2f} | "
            f"throughput={throughput:7.0f} k/s | total={elapsed:.0f}s"
        )
        self.iteration += 1
        return True


def train(
    env_id: str = "CartPole-v1",
    n_envs: int = 1,
    total_timesteps: int = 100_000,
    seed: int = 42,
    output_dir: str | None = None,
    use_subproc: bool = True,
    checkpoint_every: int = 25_000,
    gif_dir: str | None = None,
    ppo_params: dict | None = None,
) -> TrainingRun:
    """
    Trenira PPO sa SB3.

    ppo_params — rečnik PPO hiperparametara (iz experiments.yaml ppo_override).
                 Ključni parametri za BipedalWalker:
                   lr, gamma, gae_lambda, ent_coef, clip_range,
                   n_steps, batch_size, n_epochs
                 Ako je None, koriste se SB3 defaulti.

    checkpoint_every — čuva snapshot modela svakih N UKUPNIH env koraka.
                       Ovi snimci se koriste za evolution GIF.
    """
    out_path = Path(output_dir) if output_dir else ROOT / "results"
    ckpt_path = out_path / "checkpoints" / f"sb3_{env_id.replace('/', '_')}_n{n_envs}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    p = ppo_params or {}

    run = TrainingRun(
        framework="stable_baselines3",
        env_id=env_id,
        num_workers=n_envs,
        seed=seed,
    )

    vec_env_cls = SubprocVecEnv if use_subproc and n_envs > 1 else DummyVecEnv
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)

    # PPO parametri — koristimo env-specifične ako postoje
    ppo_kwargs: dict = {
        "learning_rate": p.get("lr", 3e-4),
        "gamma":         p.get("gamma", 0.99),
        "gae_lambda":    p.get("gae_lambda", 0.95),
        "ent_coef":      p.get("ent_coef", 0.0),
        "clip_range":    p.get("clip_range", 0.2),
        "n_steps":       p.get("n_steps", 2048),
        "batch_size":    p.get("batch_size", 64),
        "n_epochs":      p.get("n_epochs", 10),
    }

    print(f"\n{'='*65}")
    print(f"  Stable-Baselines3 PPO (baseline)")
    print(f"  Env:         {env_id}")
    print(f"  n_envs:      {n_envs}  ({'SubprocVecEnv' if use_subproc and n_envs > 1 else 'DummyVecEnv'})")
    print(f"  Koraci:      {total_timesteps:,}")
    print(f"  Checkpoint:  svakih {checkpoint_every:,} koraka")
    print(f"  PPO params:  lr={ppo_kwargs['learning_rate']}, gamma={ppo_kwargs['gamma']}, "
          f"ent_coef={ppo_kwargs['ent_coef']}, n_steps={ppo_kwargs['n_steps']}")
    print(f"{'='*65}\n")

    model = PPO("MlpPolicy", env, verbose=0, seed=seed, **ppo_kwargs)

    # CheckpointCallback čuva model svakih N koraka
    # save_freq je u brojevima poziva _on_step (= jednom po vect. koraku)
    # pa ga skaliramo sa n_envs da dobijemo pravi broj ukupnih env koraka
    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_every // n_envs, 1),
        save_path=str(ckpt_path),
        name_prefix="ppo",
        verbose=0,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[MetricsCallback(run), checkpoint_callback],
    )
    run.finish()

    # Čuvaj finalni model
    final_model_path = ckpt_path / "ppo_final.zip"
    model.save(str(final_model_path.with_suffix("")))
    print(f"\nFinalni model: {final_model_path}")
    print(f"Prosečni throughput: {run.avg_throughput():.0f} koraka/sec")

    saved = save_run(run, out_path)
    print(f"Metrike: {saved}")

    # --- Generisanje GIF-ova ---
    if gif_dir:
        _generate_gifs(
            env_id=env_id,
            model=model,
            ckpt_path=ckpt_path,
            gif_dir=Path(gif_dir),
            n_envs=n_envs,
        )

    env.close()
    return run


def _generate_gifs(
    env_id: str,
    model: PPO,
    ckpt_path: Path,
    gif_dir: Path,
    n_envs: int,
) -> None:
    """
    Generiše tri GIF-a za jedan SB3 run.

    random_agent.gif  — prikazuje koliko je loš agent bez treninga
    sb3_nN_trained.gif — agent na kraju treninga
    sb3_nN_evolution.gif — razvoj kroz sve međukorake (za master rad)
    """
    gif_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Generišem GIF-ove → {gif_dir}")

    # 1. Slučajan agent (uvek isti, bez obzira na n_envs)
    random_gif = gif_dir / "random_agent.gif"
    if not random_gif.exists():
        random_frames, rr = record_episode(env_id, model=None, max_steps=300)
        print(f"    Slučajan agent: nagrada={rr:.0f}")
        save_gif(random_frames, random_gif)

    # 2. Naučen agent
    trained_frames, tr = record_episode(env_id, model=model, max_steps=500)
    print(f"    Naučen agent (SB3 n={n_envs}): nagrada={tr:.0f}")
    save_gif(trained_frames, gif_dir / f"sb3_n{n_envs}_trained.gif")

    # 3. Evolution GIF — spaja random + sve međukorake + finalni
    ckpt_files = sorted(ckpt_path.glob("ppo_*_steps.zip"), key=_ckpt_sort_key)

    segments: list[tuple[str, list]] = []

    # Počinjemo sa random agentom
    r_frames, _ = record_episode(env_id, model=None, max_steps=200)
    segments.append(("Random (početak)", r_frames))

    # Svaki međukorak (max 4 da GIF ne bude prevelik)
    step = max(1, len(ckpt_files) // 4)
    for ckpt_file in ckpt_files[::step]:
        steps_label = _extract_steps(ckpt_file)
        ckpt_model = PPO.load(str(ckpt_file))
        frames, reward = record_episode(env_id, model=ckpt_model, max_steps=300)
        del ckpt_model
        segments.append((f"Korak {steps_label} | nagrada≈{reward:.0f}", frames))

    # Finalni model
    f_frames, f_reward = record_episode(env_id, model=model, max_steps=400)
    segments.append((f"Finalni model | nagrada≈{f_reward:.0f}", f_frames))

    evolution_frames = build_evolution_gif(segments)
    save_gif(evolution_frames, gif_dir / f"sb3_n{n_envs}_evolution.gif", fps=24)


def _ckpt_sort_key(path: Path) -> int:
    """Sortira checkpoint fajlove po broju koraka."""
    return _extract_steps_int(path)


def _extract_steps_int(path: Path) -> int:
    """Izvlači broj koraka iz naziva fajla (ppo_10000_steps.zip → 10000)."""
    try:
        return int(path.stem.split("_")[-2])
    except (IndexError, ValueError):
        return 0


def _extract_steps(path: Path) -> str:
    n = _extract_steps_int(path)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n//1_000}K"
    return str(n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SB3 PPO baseline trening",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--n-envs", type=int, default=1, help="Broj paralelnih env kopija")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    parser.add_argument("--gif", action="store_true", help="Generiši GIF-ove posle treninga")
    parser.add_argument("--gif-dir", default=None, help="Folder za GIF-ove")
    parser.add_argument("--dummy", action="store_true")
    # PPO override parametri (za BipedalWalker i LunarLander)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 3e-4)")
    parser.add_argument("--gamma", type=float, default=None, help="Discount faktor")
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy koef (exploration)")
    parser.add_argument("--n-steps", type=int, default=None, help="Rollout dužina po env-u")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    args = parser.parse_args()

    # Skupi samo parametre koji su eksplicitno zadati
    ppo_params = {k: v for k, v in {
        "lr": args.lr, "gamma": args.gamma, "gae_lambda": args.gae_lambda,
        "ent_coef": args.ent_coef, "n_steps": args.n_steps,
        "batch_size": args.batch_size, "n_epochs": args.n_epochs,
    }.items() if v is not None}

    env_key = args.env.replace("/", "_").replace("-", "_").lower()
    gif_dir = (
        Path(args.gif_dir)
        if args.gif_dir
        else (ROOT / "results" / "gifs" / env_key if args.gif else None)
    )

    train(
        env_id=args.env,
        n_envs=args.n_envs,
        total_timesteps=args.timesteps,
        seed=args.seed,
        output_dir=args.output,
        use_subproc=not args.dummy,
        checkpoint_every=args.checkpoint_every,
        gif_dir=str(gif_dir) if gif_dir else None,
        ppo_params=ppo_params or None,
    )


if __name__ == "__main__":
    main()
