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
import warnings
from pathlib import Path

import imageio
import numpy as np
import gymnasium as gym

ROOT = Path(__file__).parent.parent

# Skup algo ID-ova za koje je filter već dijagnostikovan (da ne spamuje)
_filter_debugged: set[int] = set()


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


def _try_get_filter_from_worker(w) -> object | None:
    """Izvlači MeanStdFilter iz jednog RolloutWorker objekta."""
    try:
        filters = getattr(w, "filters", None)
        if filters is None:
            return None
        if isinstance(filters, dict):
            # Probaj "default_policy" ili prvi dostupni ključ
            f = filters.get("default_policy")
            if f is not None:
                return f
            return next((v for v in filters.values() if v is not None), None)
        # Ako filters nije dict (direktan filter objekat)
        return filters if callable(filters) else None
    except Exception:
        return None


def _resolve_worker_set(algo):
    """
    Vraća EnvRunnerGroup / WorkerSet objekat iz algoritma.

    U Ray 2.x, pravi atribut je `algo.env_runner_group` (EnvRunnerGroup).
    `algo.workers` je deprecated i vraća metod/property koji može da pukne.
    Probamo oba, pa još privatne atribute kao fallback.
    """
    # ── Primarno: Ray 2.x ─────────────────────────────────────────────────────
    for attr in ("env_runner_group", "_env_runner_group"):
        ws = getattr(algo, attr, None)
        if ws is not None and not callable(ws):
            return ws

    # ── Sekundarno: stari API (workers kao WorkerSet direktno) ─────────────────
    for attr in ("workers", "_workers", "_rollout_workers"):
        ws = getattr(algo, attr, None)
        if ws is None:
            continue
        # Ako je callable (deprecated property/metod), pokušaj pozvati
        if callable(ws) and not isinstance(ws, type):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ws = ws()
            except Exception:
                continue
        if ws is not None:
            return ws

    return None


def _get_obs_filter(algo, _debug: bool = False) -> object | None:
    """
    Izvlači MeanStdFilter iz Ray RLlib algoritma.

    Svih 5 pristupa su pokrivena; ako nijedno ne nađe filter, štampa dijagnostiku.
    """
    ws = _resolve_worker_set(algo)
    if ws is None:
        if _debug:
            print("  [DEBUG-FILTER] WorkerSet nije dostupan")
        return None

    # ── Pristup 1: lokalni env runner / worker (Ray 2.x i stari API) ───────────
    for lw_attr in ("_local_env_runner", "_local_worker", "local_worker"):
        try:
            w = getattr(ws, lw_attr, None)
            if w is None:
                continue
            if callable(w) and not isinstance(w, type):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w = w()
            f = _try_get_filter_from_worker(w)
            if f is not None:
                return f
        except Exception:
            pass

    # ── Pristup 2: remote workeri ─────────────────────────────────────────────
    try:
        import ray as _ray
        for rattr in ("_remote_workers", "remote_workers"):
            remote_workers = getattr(ws, rattr, None) or []
            if callable(remote_workers) and not isinstance(remote_workers, type):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        remote_workers = remote_workers() or []
                except Exception:
                    remote_workers = []
            for rw in list(remote_workers)[:1]:
                try:
                    f = _ray.get(rw.apply.remote(lambda w: _try_get_filter_from_worker(w)))
                    if f is not None:
                        return f
                except Exception:
                    pass
    except Exception:
        pass

    # ── Pristup 3: foreach_worker / foreach_env_runner ────────────────────────
    for fw_method in ("foreach_worker", "foreach_env_runner"):
        try:
            found: list = [None]

            def _grab_filter(w, found=found):
                if found[0] is None:
                    found[0] = _try_get_filter_from_worker(w)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                getattr(ws, fw_method)(_grab_filter)
            if found[0] is not None:
                return found[0]
        except Exception:
            pass

    # ── Pristup 4: rekonstrukcija iz stanja workera (get_filter_state) ────────
    try:
        from ray.rllib.utils.filter import MeanStdFilter as _MSF  # type: ignore
        for state_method in ("get_filter_state",):
            state_fn = getattr(ws, state_method, None)
            if state_fn is None:
                continue
            state = state_fn()
            pid = next(iter(state), None)
            if pid is not None:
                obs_shape = algo.get_policy(pid).observation_space.shape
                f = _MSF(obs_shape)
                f.from_state(state[pid])
                return f
    except Exception:
        pass

    # ── Dijagnostika (aktivira se samo prvi put) ──────────────────────────────
    if _debug:
        _diagnose_filter(algo, ws)

    return None


def _diagnose_filter(algo, ws=None) -> None:
    """Štampa informacije o stanju workera da pomogne u dijagnostici."""
    print("  [DEBUG-FILTER] ─── Dijagnostika filtera ─────────────────────────")
    for a in ("workers", "env_runner_group", "_env_runner_group"):
        v = getattr(algo, a, "NEMA")
        print(f"  [DEBUG-FILTER] algo.{a}: {type(v).__name__}")
    if ws is None:
        print("  [DEBUG-FILTER] WorkerSet/EnvRunnerGroup nije dostupan")
        print("  [DEBUG-FILTER] ────────────────────────────────────────────────────")
        return
    print(f"  [DEBUG-FILTER] WorkerSet type: {type(ws).__name__}")
    for attr in ("_local_env_runner", "_local_worker", "_remote_workers",
                 "local_worker", "foreach_env_runner", "foreach_worker"):
        val = getattr(ws, attr, "NEMA")
        vr = repr(val)[:80] if val != "NEMA" else "NEMA"
        print(f"  [DEBUG-FILTER] ws.{attr}: {type(val).__name__} = {vr}")
    # Pokušaj dohvatiti lokalni worker
    for lw_attr in ("_local_env_runner", "_local_worker"):
        w = getattr(ws, lw_attr, None)
        if w is not None:
            filters = getattr(w, "filters", "NEMA_ATTR")
            print(f"  [DEBUG-FILTER] ws.{lw_attr}.filters: {filters}")
            break
    print("  [DEBUG-FILTER] ────────────────────────────────────────────────────")


def record_ray_algo(
    algo,
    env_id: str,
    max_steps: int = 500,
) -> tuple[list[np.ndarray], float]:
    """
    Snima epizodu koristeći Ray algo.

    Kritično: mora da primeni MeanStdFilter pre slanja opservacija policy-ju.
    Bez filtera, policy koji je treniran sa normalize_obs=True dobija sirove
    opservacije → ponaša se haotično čak i ako je dobro istreniran.

    Rešava problem deprecated local_worker() API-ja u novijim Ray verzijama
    tako što pristupa filteru direktno i primenjuje ga ručno.
    """
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    total_reward = 0.0

    algo_id = id(algo)
    first_time = algo_id not in _filter_debugged
    obs_filter = _get_obs_filter(algo, _debug=first_time)
    if obs_filter is None:
        if first_time:
            _filter_debugged.add(algo_id)
            print("  [UPOZORENJE] MeanStdFilter nije pronađen — obs se šalju nenormalizovane.")

    def _infer(o: np.ndarray) -> np.ndarray:
        if obs_filter is not None:
            o = obs_filter(o, update=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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
