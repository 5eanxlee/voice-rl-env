"""Example: Train a PPO agent on the Voice Synthesis environment.

Requires: pip install voice-rl-env[train]
"""

from __future__ import annotations

import argparse

import gymnasium as gym

from voice_rl_env import VoiceSynthesisEnv
from voice_rl_env.rewards import RewardConfig
from voice_rl_env.wrappers import CurriculumWrapper, FeatureDifferenceWrapper


def make_env(
    target_profile: str | None = None,
    use_curriculum: bool = True,
    use_feature_diff: bool = True,
) -> gym.Env:
    """Create and wrap the voice synthesis environment."""
    reward_config = RewardConfig(
        target_weight=1.0,
        naturalness_weight=0.3,
        stability_weight=0.1,
        boundary_weight=0.05,
    )

    env: gym.Env = VoiceSynthesisEnv(
        target_profile=target_profile,
        max_steps=200,
        action_scale=0.1,
        noise_std=0.02,
        reward_config=reward_config,
    )

    if use_feature_diff:
        env = FeatureDifferenceWrapper(env)
    if use_curriculum:
        env = CurriculumWrapper(env, initial_difficulty=0.1, difficulty_increment=0.002)

    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Voice Synthesis Env")
    parser.add_argument("--target", type=str, default=None, help="Target speaker profile name")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-path", type=str, default="ppo_voice", help="Model save path")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("stable-baselines3 is required. Install with: pip install voice-rl-env[train]")
        return

    # Create training and eval environments
    train_env = DummyVecEnv([lambda: make_env(args.target)])
    eval_env = DummyVecEnv([lambda: make_env(args.target, use_curriculum=False)])

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
    )

    # Eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./{args.save_path}_best/",
        log_path=f"./{args.save_path}_logs/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train
    print(f"Training PPO for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")

    # Quick evaluation
    print("\nEvaluating trained agent...")
    obs = eval_env.reset()
    total_reward = 0.0
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += float(reward[0])
        if done[0]:
            break
    print(f"Evaluation episode reward: {total_reward:.3f}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
