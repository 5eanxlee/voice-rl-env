"""Gymnasium environment registration."""

import gymnasium as gym


def register_envs() -> None:
    """Register voice RL environments with Gymnasium."""
    gym.register(
        id="VoiceSynthesis-v0",
        entry_point="voice_rl_env.env:VoiceSynthesisEnv",
        max_episode_steps=200,
    )
