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
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
from ray.tune import register_env

from play_game import save_gif

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


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


def _find_policy_dir(ckpt: Path) -> Path | None:
    """Traži default_policy direktorijum unutar checkpointa."""
    candidates = [
        ckpt / "policies" / "default_policy",
    ]
    for subdir in sorted(ckpt.glob("checkpoint_*"), reverse=True):
        candidates.append(subdir / "policies" / "default_policy")
    for c in candidates:
        if c.exists():
            return c
    return None


def _run_episodes_with_policy(
    policy: Policy,
    env_id: str,
    n_episodes: int,
    max_steps: int,
) -> tuple[list[float], list]:
    """
    Pokreće epizode koristeći samo Policy objekat — bez Ray workera.

    Mnogo lakše od učitavanja punog Algorithm-a:
    nema remote workera, nema Ray aktorskih procesa, mala potrošnja memorije.
    """
    rewards: list[float] = []
    best_frames: list = []
    best_reward = float("-inf")

    env = gym.make(env_id, render_mode="rgb_array")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        frames: list = []

        for _ in range(max_steps):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            action = policy.compute_single_action(obs, explore=False)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_frames = frames

        print(f"{ep+1:8d} | {ep_reward:10.2f}")

    env.close()
    return rewards, best_frames


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

    Koristi Policy.from_checkpoint (bez Ray workera) da izbegne OOM
    koji se javlja kada se puni Algorithm.from_checkpoint pokreće odmah
    posle treninga dok mašina još nije oslobodila memoriju.

    Vraća dict sa:
      rewards        — lista nagrada po epizodi
      mean           — prosek
      std            — standardna devijacija
      best_reward    — nagrada najboje epizode
      gif_path       — putanja do GIF-a (ako gif_dir nije None)
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.is_absolute():
        ckpt = (ROOT / ckpt).resolve()

    print(f"\n{'='*60}")
    print(f"  Evaluacija agenta (Policy inference, bez Ray workera)")
    print(f"  Checkpoint:  {ckpt}")
    print(f"  Env:         {env_id}")
    print(f"  Epizode:     {n_episodes}")
    print(f"{'='*60}\n")

    print(f"{'Epizoda':>8} | {'Nagrada':>10}")
    print(f"{'-'*22}")

    policy_dir = _find_policy_dir(ckpt)
    if policy_dir is None:
        raise FileNotFoundError(
            f"Nije pronađen policies/default_policy unutar {ckpt}. "
            "Provjeri da je checkpoint ispravan."
        )

    _register_custom_envs()
    policy = Policy.from_checkpoint(str(policy_dir))
    rewards, best_frames = _run_episodes_with_policy(policy, env_id, n_episodes, max_steps)

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
        print(f"  Best epizoda ({max(rewards):.0f}) sačuvana: {out_file}")

    result = {
        "checkpoint": str(ckpt),
        "env_id": env_id,
        "n_episodes": n_episodes,
        "rewards": rewards,
        "mean": round(mean_r, 3),
        "std": round(std_r, 3),
        "min": round(min(rewards), 3),
        "max": round(max(rewards), 3),
        "best_reward": round(max(rewards), 3),
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
