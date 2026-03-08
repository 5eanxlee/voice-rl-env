"""Tests for the VoiceSynthesisEnv."""

from __future__ import annotations

import numpy as np
import pytest

from voice_rl_env.env import VoiceSynthesisEnv
from voice_rl_env.rewards import RewardConfig
from voice_rl_env.voice_simulator import NUM_FEATURES, NUM_PARAMS, SpeakerProfile


class TestVoiceSynthesisEnv:
    """Tests for the core environment."""

    def test_creation(self) -> None:
        """Test basic environment creation."""
        env = VoiceSynthesisEnv()
        assert env.observation_space.shape == (2 * NUM_FEATURES + NUM_PARAMS,)
        assert env.action_space.shape == (NUM_PARAMS,)
        env.close()

    def test_creation_with_profile_name(self) -> None:
        """Test creation with a named profile."""
        env = VoiceSynthesisEnv(target_profile="neutral")
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_creation_with_profile_object(self) -> None:
        """Test creation with a SpeakerProfile object."""
        profile = SpeakerProfile(name="test", pitch_shift=3.0)
        env = VoiceSynthesisEnv(target_profile=profile)
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_creation_invalid_profile(self) -> None:
        """Test that invalid profile name raises error."""
        with pytest.raises(ValueError, match="Unknown profile"):
            VoiceSynthesisEnv(target_profile="nonexistent")

    def test_reset(self) -> None:
        """Test environment reset."""
        env = VoiceSynthesisEnv(target_profile="neutral")
        obs, info = env.reset(seed=42)

        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        assert info["step_count"] == 0
        assert info["episode_reward"] == 0.0
        assert "current_params" in info
        env.close()

    def test_reset_reproducibility(self) -> None:
        """Test that reset with same seed produces same results."""
        env = VoiceSynthesisEnv(target_profile="neutral")

        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)

        np.testing.assert_array_equal(obs1, obs2)
        env.close()

    def test_reset_different_seeds(self) -> None:
        """Test that different seeds produce different results."""
        env = VoiceSynthesisEnv()

        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)

        assert not np.array_equal(obs1, obs2)
        env.close()

    def test_step(self) -> None:
        """Test a single step."""
        env = VoiceSynthesisEnv(target_profile="neutral")
        env.reset(seed=42)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["step_count"] == 1
        assert "reward_components" in info
        env.close()

    def test_step_action_clipping(self) -> None:
        """Test that out-of-range actions are clipped."""
        env = VoiceSynthesisEnv(target_profile="neutral")
        env.reset(seed=42)

        # Action outside [-1, 1]
        action = np.ones(NUM_PARAMS) * 5.0
        obs, reward, terminated, truncated, info = env.step(action)

        # Should still work without error
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_episode_truncation(self) -> None:
        """Test that episode truncates after max_steps."""
        max_steps = 10
        env = VoiceSynthesisEnv(target_profile="neutral", max_steps=max_steps)
        env.reset(seed=42)

        for i in range(max_steps):
            action = np.zeros(NUM_PARAMS)
            obs, reward, terminated, truncated, info = env.step(action)

            if i < max_steps - 1:
                assert not truncated
            else:
                assert truncated
                assert info["step_count"] == max_steps

        env.close()

    def test_reward_components(self) -> None:
        """Test that reward components are present and reasonable."""
        env = VoiceSynthesisEnv(target_profile="neutral")
        env.reset(seed=42)

        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)

        components = info["reward_components"]
        assert "target_matching" in components
        assert "naturalness" in components
        assert "stability_penalty" in components
        assert "boundary_penalty" in components
        assert "total" in components

        assert 0.0 <= components["target_matching"] <= 1.0
        assert 0.0 <= components["naturalness"] <= 1.0
        assert 0.0 <= components["stability_penalty"] <= 1.0
        assert 0.0 <= components["boundary_penalty"] <= 1.0
        env.close()

    def test_custom_reward_config(self) -> None:
        """Test environment with custom reward configuration."""
        config = RewardConfig(
            target_weight=2.0,
            naturalness_weight=0.0,
            stability_weight=0.0,
            boundary_weight=0.0,
        )
        env = VoiceSynthesisEnv(target_profile="neutral", reward_config=config)
        env.reset(seed=42)

        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)

        # With only target weight, reward should be 2x target matching
        expected = 2.0 * info["reward_components"]["target_matching"]
        assert abs(reward - expected) < 1e-6
        env.close()

    def test_render_ansi(self) -> None:
        """Test ANSI rendering."""
        env = VoiceSynthesisEnv(target_profile="neutral", render_mode="ansi")
        env.reset(seed=42)

        output = env.render()
        assert output is not None
        assert "Step:" in output
        assert "Current Parameters:" in output
        env.close()

    def test_render_no_mode(self) -> None:
        """Test rendering with no mode set."""
        env = VoiceSynthesisEnv(target_profile="neutral")
        env.reset(seed=42)

        output = env.render()
        assert output is None
        env.close()

    def test_multiple_episodes(self) -> None:
        """Test running multiple episodes."""
        env = VoiceSynthesisEnv(target_profile="neutral")

        for _ in range(3):
            obs, info = env.reset()
            assert info["step_count"] == 0
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

        env.close()

    def test_reset_with_options(self) -> None:
        """Test reset with override options."""
        env = VoiceSynthesisEnv()
        obs, info = env.reset(
            seed=42,
            options={"target_profile": "breathy"},
        )
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_gymnasium_check_env(self) -> None:
        """Test compatibility with gymnasium's env checker."""
        from gymnasium.utils.env_checker import check_env

        env = VoiceSynthesisEnv(target_profile="neutral")
        # check_env will raise if there are issues
        check_env(env.unwrapped, skip_render_check=True)
        env.close()
