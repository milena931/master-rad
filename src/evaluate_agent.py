"""
evaluate_agent.py — Evaluacija sačuvanog agenta iz checkpointa.

Učitava Ray RLlib checkpoint (PPO ili IMPALA), pokreće N epizoda,
računa prosečnu nagradu ± standardnu devijaciju i čuva best epizodu kao GIF.

Primer korišćenja:
  # Posle treninga sa --checkpoint-dir:
  python src/evaluate_agent.py \\
      --checkpoint results/checkpoints/ppo_cartpole \\
      --env CartPole-v1 \\
      --episodes 15 \\
      --gif-dir results/gifs/cartpole_v1

  # Kratak test (5 epizoda, bez GIF-a):
  python src/evaluate_agent.py \\
      --checkpoint results/checkpoints/ppo_cartpole \\
      --env CartPole-v1 \\
      --episodes 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import gymnasium as gym
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune import register_env

from play_game import record_ray_algo, save_gif


def _register_custom_envs() -> None:
    """
    Registruje custom env varijante koje su bile aktivne tokom treninga.

    BipedalWalker checkpoint je snimljen sa env="BipedalWalker-v3-clipped"
    (custom wrapper koji klipuje OOB opservacije). Mora biti registrovan
    pre Algorithm.from_checkpoint(), inače RolloutWorker ne može da ga napravi.
    """
    class _ClipObsWrapper(gym.ObservationWrapper):
        def observation(self, obs):
            return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _make_clipped_bipedal(cfg):
        return _ClipObsWrapper(gym.make("BipedalWalker-v3"))

    register_env("BipedalWalker-v3-clipped", _make_clipped_bipedal)

ROOT = Path(__file__).parent.parent


def evaluate(
    checkpoint_path: str,
    env_id: str,
    n_episodes: int = 15,
    max_steps: int = 500,
    gif_dir: str | None = None,
    gif_tag: str = "best",
) -> dict:
    """
    Učitava checkpoint i pokreće n_episodes epizoda.

    Vraća dict sa:
      rewards        — lista nagrada po epizodi
      mean           — prosek
      std            — standardna devijacija
      best_reward    — nagrada najboje epizode
      gif_path       — putanja do GIF-a (ako gif_dir nije None)

    gif_tag — prefiks za ime GIF fajla (npr. "ppo_w4" → "ppo_w4_best.gif")
    """
    ray.init(ignore_reinit_error=True)

    # Mora pre from_checkpoint() — checkpoint može da zahteva custom env
    _register_custom_envs()

    print(f"\n{'='*60}")
    print(f"  Evaluacija agenta")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Env:         {env_id}")
    print(f"  Epizode:     {n_episodes}")
    print(f"{'='*60}\n")

    ckpt = Path(checkpoint_path)
    if not ckpt.is_absolute():
        ckpt = (ROOT / ckpt).resolve()
    algo = Algorithm.from_checkpoint(str(ckpt))

    rewards: list[float] = []
    best_frames: list = []
    best_reward = float("-inf")

    print(f"{'Epizoda':>8} | {'Nagrada':>10}")
    print(f"{'-'*22}")

    for ep in range(n_episodes):
        frames, reward = record_ray_algo(algo, env_id, max_steps=max_steps)
        rewards.append(reward)

        if reward > best_reward:
            best_reward = reward
            best_frames = frames

        print(f"{ep+1:8d} | {reward:10.2f}")

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))

    print(f"\n{'─'*40}")
    print(f"  Prosek:  {mean_r:.2f}  ±  {std_r:.2f}  (std)")
    print(f"  Min:     {min(rewards):.2f}")
    print(f"  Max:     {max(rewards):.2f}")
    print(f"{'─'*40}\n")

    gif_out: str | None = None
    if gif_dir and best_frames:
        gif_path = Path(gif_dir)
        gif_path.mkdir(parents=True, exist_ok=True)
        out_file = gif_path / f"{gif_tag}_best.gif"
        save_gif(best_frames, out_file)
        gif_out = str(out_file)
        print(f"  Best epizoda ({best_reward:.0f}) sačuvana: {out_file}")

    algo.stop()
    ray.shutdown()

    result = {
        "checkpoint": checkpoint_path,
        "env_id": env_id,
        "n_episodes": n_episodes,
        "rewards": rewards,
        "mean": round(mean_r, 3),
        "std": round(std_r, 3),
        "min": round(min(rewards), 3),
        "max": round(max(rewards), 3),
        "best_reward": round(best_reward, 3),
        "gif_path": gif_out,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluacija sačuvanog Ray RLlib agenta",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Putanja do Ray RLlib checkpointa")
    parser.add_argument("--env", required=True,
                        help="Gymnasium env ID (mora biti isti kao pri treningu)")
    parser.add_argument("--episodes", type=int, default=15,
                        help="Broj epizoda za evaluaciju")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maksimalan broj koraka po epizodi")
    parser.add_argument("--gif-dir", default=None,
                        help="Folder za čuvanje best GIF-a")
    parser.add_argument("--gif-tag", default="eval",
                        help="Prefiks za ime GIF fajla (npr. 'ppo_w4' → 'ppo_w4_best.gif')")
    parser.add_argument("--output", default=None,
                        help="Sačuvaj rezultate u JSON fajl")
    args = parser.parse_args()

    env_key = args.env.replace("/", "_").replace("-", "_").lower()
    gif_dir = (
        args.gif_dir
        if args.gif_dir
        else str(ROOT / "results" / "gifs" / env_key)
        if args.gif_tag
        else None
    )

    result = evaluate(
        checkpoint_path=args.checkpoint,
        env_id=args.env,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        gif_dir=gif_dir,
        gif_tag=args.gif_tag,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Rezultati sačuvani: {out}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
