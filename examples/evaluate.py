"""Example: Evaluate and visualize agent behavior in the Voice Synthesis environment.

Can run with a random agent (no dependencies) or a trained stable-baselines3 model.
"""

from __future__ import annotations

import argparse

import numpy as np

from voice_rl_env import VoiceSynthesisEnv


def run_episode(
    env: VoiceSynthesisEnv,
    model: object | None = None,
    seed: int = 42,
) -> dict:
    """Run a single episode and collect metrics.

    Args:
        env: The voice synthesis environment.
        model: Optional trained model with a predict() method.
        seed: Random seed.

    Returns:
        Dictionary of episode metrics.
    """
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    step_rewards = []
    target_matches = []

    steps_taken = 0
    for _step in range(env.max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)  # type: ignore[union-attr]
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_rewards.append(reward)
        target_matches.append(info["reward_components"]["target_matching"])
        steps_taken += 1

        if env.render_mode == "ansi":
            env.render()
            print(f"  Reward: {reward:.4f}")
            print(f"  Target Match: {info['reward_components']['target_matching']:.4f}")
            print()

        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": steps_taken,
        "best_target_match": max(target_matches),
        "final_target_match": target_matches[-1],
        "mean_reward": np.mean(step_rewards),
        "final_params": info["current_params"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Voice Synthesis Agent")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--target", type=str, default="neutral", help="Target speaker profile")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render environment (ansi mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    render_mode = "ansi" if args.render else None
    env = VoiceSynthesisEnv(
        target_profile=args.target,
        max_steps=200,
        render_mode=render_mode,
    )

    model = None
    if args.model_path:
        try:
            from stable_baselines3 import PPO, SAC

            # Try loading as PPO first, then SAC
            try:
                model = PPO.load(args.model_path)
                print(f"Loaded PPO model from {args.model_path}")
            except Exception:
                model = SAC.load(args.model_path)
                print(f"Loaded SAC model from {args.model_path}")
        except ImportError:
            print("stable-baselines3 required to load models.")
            return
    else:
        print("No model provided, using random agent.")

    # Run evaluation
    results = []
    for ep in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'='*60}")

        metrics = run_episode(env, model=model, seed=args.seed + ep)
        results.append(metrics)

        print(f"  Total Reward:       {metrics['total_reward']:.3f}")
        print(f"  Steps:              {metrics['steps']}")
        print(f"  Best Target Match:  {metrics['best_target_match']:.4f}")
        print(f"  Final Target Match: {metrics['final_target_match']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    rewards = [r["total_reward"] for r in results]
    matches = [r["best_target_match"] for r in results]
    print(f"  Mean Reward:        {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")
    print(f"  Mean Best Match:    {np.mean(matches):.4f} +/- {np.std(matches):.4f}")

    env.close()


if __name__ == "__main__":
    main()
