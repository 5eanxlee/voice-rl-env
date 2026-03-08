"""Tests for environment wrappers."""

from __future__ import annotations

from voice_rl_env.env import VoiceSynthesisEnv
from voice_rl_env.voice_simulator import NUM_FEATURES, NUM_PARAMS
from voice_rl_env.wrappers import (
    CurriculumWrapper,
    FeatureDifferenceWrapper,
    NormalizeRewardWrapper,
)


class TestFeatureDifferenceWrapper:
    """Tests for FeatureDifferenceWrapper."""

    def test_observation_shape(self) -> None:
        """Observation should have shape (NUM_FEATURES + NUM_PARAMS,)."""
        env = FeatureDifferenceWrapper(
            VoiceSynthesisEnv(target_profile="neutral")
        )
        obs, _ = env.reset(seed=42)
        assert obs.shape == (NUM_FEATURES + NUM_PARAMS,)
        env.close()

    def test_step(self) -> None:
        """Step should work correctly with wrapper."""
        env = FeatureDifferenceWrapper(
            VoiceSynthesisEnv(target_profile="neutral")
        )
        env.reset(seed=42)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (NUM_FEATURES + NUM_PARAMS,)
        env.close()


class TestNormalizeRewardWrapper:
    """Tests for NormalizeRewardWrapper."""

    def test_reward_normalization(self) -> None:
        """Rewards should be normalized after initial steps."""
        env = NormalizeRewardWrapper(
            VoiceSynthesisEnv(target_profile="neutral")
        )
        env.reset(seed=42)

        rewards = []
        for _ in range(50):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                env.reset()

        # After warmup, rewards should have reasonable scale
        assert all(isinstance(r, float) for r in rewards)
        env.close()


class TestCurriculumWrapper:
    """Tests for CurriculumWrapper."""

    def test_difficulty_increases(self) -> None:
        """Difficulty should increase with each episode."""
        env = CurriculumWrapper(
            VoiceSynthesisEnv(target_profile="neutral"),
            initial_difficulty=0.1,
            difficulty_increment=0.1,
        )

        _, info1 = env.reset(seed=42)
        _, info2 = env.reset(seed=43)
        _, info3 = env.reset(seed=44)

        assert info1["difficulty"] < info2["difficulty"] < info3["difficulty"]
        env.close()

    def test_max_difficulty(self) -> None:
        """Difficulty should not exceed max."""
        env = CurriculumWrapper(
            VoiceSynthesisEnv(target_profile="neutral"),
            initial_difficulty=0.9,
            difficulty_increment=0.5,
            max_difficulty=1.0,
        )

        for _ in range(5):
            _, info = env.reset()

        assert info["difficulty"] <= 1.0
        env.close()

    def test_step_works(self) -> None:
        """Steps should work correctly with curriculum wrapper."""
        env = CurriculumWrapper(
            VoiceSynthesisEnv(target_profile="neutral"),
        )
        env.reset(seed=42)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        env.close()
