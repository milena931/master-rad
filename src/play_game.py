"""
play_game.py — Generički snimač gameplay GIF-ova za master rad.

Podržava sva tri okruženja: CartPole, LunarLander, BipedalWalker.

GIF tipovi koje generiše:
  1. random_agent.gif  — agent radi random akcije (bez treninga)
  2. ray_wN_trained.gif — naučen Ray agent sa N workera
  3. ray_wN_evolution.gif — napredak agenta tokom treninga

Korišćenje kao skripta:
  python play_game.py --env CartPole-v1 --random   # samo random agent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
import gymnasium as gym

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Snimanje epizoda
# ---------------------------------------------------------------------------

def record_episode(
    env_id: str,
    max_steps: int = 500,
) -> tuple[list[np.ndarray], float]:
    """
    Snima jednu epizodu slučajnog agenta i vraća listu frejmova + ukupnu nagradu.
    Za snimanje naučenog Ray agenta koristi record_ray_algo.
    """
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    total_reward = 0.0

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = env.action_space.sample()
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

    Važno: prolazi kroz local_worker().compute_single_action() umesto
    direktno algo.compute_single_action(), jer local_worker primenjuje
    isti MeanStdFilter koji se koristi pri treningu. Bez toga, policy
    prima nenormalizovane opservacije i loše se ponaša (čak i ako je
    dobro istreniran).
    """
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    total_reward = 0.0

    # Pokušaj da koristimo local_worker koji ima obs filter u svom pipeline-u.
    # Ovo je potrebno kada je normalize_obs=True (MeanStdFilter):
    # algo.compute_single_action() zaobilazi filter i šalje sirove opservacije
    # policy-ju → policy vidi ulaze koji su drugačiji od treninga → loš rezultat.
    try:
        _worker = algo.workers.local_worker()
        def _infer(o):
            return _worker.compute_single_action(o, explore=False)
    except Exception:
        def _infer(o):
            return algo.compute_single_action(o, explore=False)

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = _infer(obs)
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
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snimi GIF slučajnog agenta za Gymnasium okruženje",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium env ID")
    parser.add_argument("--output", default=None, help="Folder za GIF-ove")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    env_key = args.env.replace("/", "_").replace("-", "_").lower()
    gif_dir = Path(args.output) if args.output else ROOT / "results" / "gifs" / env_key

    frames, reward = record_episode(args.env, max_steps=400)
    print(f"Random agent — nagrada: {reward:.0f}")
    save_gif(frames, gif_dir / "random_agent.gif", fps=args.fps)


if __name__ == "__main__":
    main()
