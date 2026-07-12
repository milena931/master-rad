"""
play_game.py — Generički snimač gameplay GIF-ova za master rad.

Podržava sva tri okruženja: CartPole, LunarLander, BipedalWalker.

GIF tipovi koje generiše:
  1. random_agent.gif    — agent radi random akcije (bez treninga)
  2. trained_agent.gif   — naučen agent na kraju treninga
  3. evolution.gif       — agent na različitim tačkama treninga spojeno u jedan GIF
                          (vidi se kako napreduje od haosa do majstorstva)

Korišćenje kao skripta:
  python play_game.py --env LunarLander-v3 --model ../checkpoints/lunarlander_sb3.zip
  python play_game.py --env CartPole-v1 --random   # samo random agent
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import imageio
import numpy as np
import gymnasium as gym

if TYPE_CHECKING:
    from stable_baselines3 import PPO as SB3PPO

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Snimanje epizoda
# ---------------------------------------------------------------------------

def record_episode(
    env_id: str,
    model: "SB3PPO | None" = None,
    max_steps: int = 500,
) -> tuple[list[np.ndarray], float]:
    """
    Odigra jednu epizodu i vrati listu frejmova + ukupnu nagradu.
    model=None → slučajan agent (nasumične akcije).
    model=SB3 model → naučen agent.
    """
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    total_reward = 0.0

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    env.close()
    return frames, total_reward


def record_ray_algo(
    algo,
    env_id: str,
    max_steps: int = 500,
) -> tuple[list[np.ndarray], float]:
    """
    Snima epizodu koristeći Ray algo koji je JOŠ U MEMORIJI (tokom treninga).
    Ne treba učitavati checkpoint — algo je živ.
    """
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    total_reward = 0.0

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        # compute_single_action radi za i diskretne i kontinualne akcije
        action = algo.compute_single_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    env.close()
    return frames, total_reward


# ---------------------------------------------------------------------------
# Pomoćne funkcije za GIF kreiranje
# ---------------------------------------------------------------------------

def make_separator(reference_frame: np.ndarray, n_frames: int = 12) -> list[np.ndarray]:
    """
    Tamni separator između epizoda u evolution GIF-u.
    Vizuelno razdvaja "fazu 1" od "faze 2" treninga.
    """
    sep = np.full_like(reference_frame, fill_value=30)  # tamno sivo
    return [sep] * n_frames


def add_label_overlay(frame: np.ndarray, text: str) -> np.ndarray:
    """
    Dodaje prost tekstualni overlay na frame (bez cv2 zavisnosti).
    Samo postavlja tamnu traku pri dnu — čist i jednostavan.
    Vraća originalni frame ako PIL nije dostupan.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        w, h = img.size
        draw.rectangle([(0, h - 28), (w, h)], fill=(0, 0, 0, 180))
        draw.text((8, h - 22), text, fill=(255, 255, 255))
        return np.array(img)
    except Exception:
        return frame  # PIL nije instaliran, vrati original


def build_evolution_gif(
    segments: list[tuple[str, list[np.ndarray]]],
) -> list[np.ndarray]:
    """
    Spaja više segmenata (label + frejmovi) u jedan GIF.
    Svaki segment je npr. ("Iteracija 0 (random)", frejmovi).
    Između segmenata dodaje tamni separator.
    """
    all_frames: list[np.ndarray] = []
    for label, frames in segments:
        if not frames:
            continue
        labeled = [add_label_overlay(f, label) for f in frames]
        if all_frames:
            all_frames.extend(make_separator(frames[0]))
        all_frames.extend(labeled)
    return all_frames


# ---------------------------------------------------------------------------
# Čuvanje GIF-a
# ---------------------------------------------------------------------------

def save_gif(frames: list[np.ndarray], path: Path | str, fps: int = 30) -> Path:
    """Čuva listu frejmova kao GIF fajl."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        print(f"  [UPOZORENJE] Nema frejmova za {out.name} — preskačem.")
        return out
    imageio.mimsave(str(out), frames, fps=fps)
    size_kb = out.stat().st_size // 1024
    print(f"  GIF: {out}  ({len(frames)} frejmova, {size_kb} KB)")
    return out


# ---------------------------------------------------------------------------
# Glavni snimač za jedan env
# ---------------------------------------------------------------------------

def record_env_demos(
    env_id: str,
    gif_dir: Path,
    sb3_model_path: Path | None = None,
    fps: int = 30,
    max_steps_random: int = 300,
    max_steps_trained: int = 500,
) -> dict[str, Path]:
    """
    Snima sve GIF-ove za jedno okruženje pomoću SB3 modela.
    Vraća rečnik {naziv: putanja_GIF-a}.

    Generiše:
      - random_agent.gif  (uvek)
      - trained_agent.gif (ako postoji sb3_model_path)
    """
    from stable_baselines3 import PPO

    results: dict[str, Path] = {}
    env_key = env_id.replace("/", "_").replace("-", "_").lower()

    print(f"\n  Snimam GIF-ove za {env_id}...")

    # 1. Slučajan agent
    random_frames, random_reward = record_episode(env_id, model=None, max_steps=max_steps_random)
    print(f"    Slučajan agent: nagrada={random_reward:.0f}, koraka={len(random_frames)}")
    p = save_gif(random_frames, gif_dir / "random_agent.gif", fps=fps)
    results["random"] = p

    # 2. Naučen agent (SB3)
    if sb3_model_path and Path(sb3_model_path).exists():
        model = PPO.load(str(sb3_model_path))
        trained_frames, trained_reward = record_episode(env_id, model=model, max_steps=max_steps_trained)
        print(f"    Naučen agent (SB3): nagrada={trained_reward:.0f}, koraka={len(trained_frames)}")
        p = save_gif(trained_frames, gif_dir / "sb3_trained.gif", fps=fps)
        results["sb3_trained"] = p

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snimi GIF-ove za Gymnasium okruženje",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium env ID")
    parser.add_argument("--model", default=None, help="Putanja do SB3 .zip modela")
    parser.add_argument("--output", default=None, help="Folder za GIF-ove")
    parser.add_argument("--random", action="store_true", help="Samo random agent")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    env_key = args.env.replace("/", "_").replace("-", "_").lower()
    gif_dir = Path(args.output) if args.output else ROOT / "results" / "gifs" / env_key

    if args.random or args.model is None:
        frames, reward = record_episode(args.env, model=None, max_steps=400)
        print(f"Random agent — nagrada: {reward:.0f}")
        save_gif(frames, gif_dir / "random_agent.gif", fps=args.fps)
    else:
        record_env_demos(args.env, gif_dir, sb3_model_path=Path(args.model), fps=args.fps)


if __name__ == "__main__":
    main()
