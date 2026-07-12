"""Trenira kratko SB3 agenta i snima CartPole epizode u GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO


def record_episode(env: gym.Env, model: PPO | None, max_steps: int = 500) -> tuple[list[np.ndarray], float]:
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
        total_reward += reward
        if terminated or truncated:
            break

    return frames, total_reward


def train_and_play(
    env_id: str = "CartPole-v1",
    timesteps: int = 30_000,
    output_gif: str = "../results/demo_cartpole_trained.gif",
    model_path: str = "../checkpoints/cartpole_ppo.zip",
) -> None:
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    if model_file.with_suffix(".zip").exists() or model_file.exists():
        print(f"Učitavam postojeći model: {model_file}")
        model = PPO.load(str(model_file.with_suffix("")))
    else:
        print(f"Treniram PPO agenta na {env_id} ({timesteps} koraka)...")
        train_env = gym.make(env_id)
        model = PPO("MlpPolicy", train_env, verbose=1, seed=42)
        model.learn(total_timesteps=timesteps)
        model.save(str(model_file.with_suffix("")))
        train_env.close()
        print(f"Model sačuvan: {model_file}")

    render_env = gym.make(env_id, render_mode="rgb_array")

    print("\nSnimam slučajnog agenta...")
    random_frames, random_reward = record_episode(render_env, model=None)
    print(f"  Slučajan agent — nagrada: {random_reward:.0f}, koraka: {len(random_frames)}")

    print("Snimam naučenog agenta...")
    trained_frames, trained_reward = record_episode(render_env, model=model)
    print(f"  Naučen agent — nagrada: {trained_reward:.0f}, koraka: {len(trained_frames)}")

    render_env.close()

    out = Path(output_gif)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Spoji: prvo random (kratko), pa trained
    combined = random_frames + trained_frames
    imageio.mimsave(out, combined, fps=30)
    print(f"\nGIF sačuvan: {out.resolve()}")
    print("Otvaranje: xdg-open", out.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Pusti CartPole i snimi GIF")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--output", default="../results/demo_cartpole_trained.gif")
    parser.add_argument("--model", default="../checkpoints/cartpole_ppo.zip")
    args = parser.parse_args()

    train_and_play(args.env, args.timesteps, args.output, args.model)


if __name__ == "__main__":
    main()
