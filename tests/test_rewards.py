"""Tests for reward functions."""

from __future__ import annotations

import numpy as np

from voice_rl_env.rewards import (
    RewardConfig,
    boundary_penalty,
    compute_reward,
    naturalness_reward,
    stability_penalty,
    target_matching_reward,
)
from voice_rl_env.voice_simulator import (
    NUM_FEATURES,
    NUM_PARAMS,
    PARAM_RANGES,
    SPEAKER_PROFILES,
    VoiceSimulator,
)


class TestTargetMatchingReward:
    """Tests for target_matching_reward."""

    def test_perfect_match(self) -> None:
        """Perfect match should give reward of 1.0."""
        features = np.ones(NUM_FEATURES)
        reward = target_matching_reward(features, features)
        assert abs(reward - 1.0) < 1e-6

    def test_large_difference(self) -> None:
        """Large difference should give reward near 0."""
        f1 = np.zeros(NUM_FEATURES)
        f2 = np.ones(NUM_FEATURES) * 10.0
        reward = target_matching_reward(f1, f2)
        assert reward < 0.01

    def test_monotonicity(self) -> None:
        """Reward should decrease with increasing distance."""
        target = np.zeros(NUM_FEATURES)
        r1 = target_matching_reward(np.ones(NUM_FEATURES) * 0.1, target)
        r2 = target_matching_reward(np.ones(NUM_FEATURES) * 0.5, target)
        r3 = target_matching_reward(np.ones(NUM_FEATURES) * 1.0, target)
        assert r1 > r2 > r3

    def test_range(self) -> None:
        """Reward should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            f1 = rng.normal(size=NUM_FEATURES)
            f2 = rng.normal(size=NUM_FEATURES)
            reward = target_matching_reward(f1, f2)
            assert 0.0 <= reward <= 1.0


class TestNaturalnessReward:
    """Tests for naturalness_reward."""

    def test_neutral_profile(self) -> None:
        """Neutral profile should have high naturalness."""
        params = SPEAKER_PROFILES["neutral"].to_params()
        score = naturalness_reward(params)
        assert score > 0.7

    def test_extreme_params(self) -> None:
        """Extreme parameter combinations should have lower naturalness."""
        # High breathiness + high tension
        params = SPEAKER_PROFILES["neutral"].to_params().copy()
        params[3] = 0.9  # breathiness
        params[9] = 0.9  # tension
        score = naturalness_reward(params)
        assert score < 0.8

    def test_range(self) -> None:
        """Score should be in [0, 1]."""
        rng = np.random.default_rng(42)
        sim = VoiceSimulator(rng=rng)
        for _ in range(100):
            params = sim.random_params()
            score = naturalness_reward(params)
            assert 0.0 <= score <= 1.0


class TestStabilityPenalty:
    """Tests for stability_penalty."""

    def test_zero_action(self) -> None:
        """Zero action should have zero penalty."""
        action = np.zeros(NUM_PARAMS)
        penalty = stability_penalty(action)
        assert abs(penalty) < 1e-6

    def test_large_action(self) -> None:
        """Large action should have high penalty."""
        action = np.ones(NUM_PARAMS)
        penalty = stability_penalty(action)
        assert penalty > 0.5

    def test_monotonicity(self) -> None:
        """Penalty should increase with action magnitude."""
        p1 = stability_penalty(np.ones(NUM_PARAMS) * 0.1)
        p2 = stability_penalty(np.ones(NUM_PARAMS) * 0.5)
        p3 = stability_penalty(np.ones(NUM_PARAMS) * 1.0)
        assert p1 < p2 < p3


class TestBoundaryPenalty:
    """Tests for boundary_penalty."""

    def test_center_params(self) -> None:
        """Parameters at center of range should have no penalty."""
        center = (PARAM_RANGES[:, 0] + PARAM_RANGES[:, 1]) / 2.0
        penalty = boundary_penalty(center)
        assert abs(penalty) < 1e-6

    def test_extreme_params(self) -> None:
        """Parameters at extremes should have penalty."""
        # At minimum boundary
        penalty_min = boundary_penalty(PARAM_RANGES[:, 0])
        assert penalty_min > 0.0

        # At maximum boundary
        penalty_max = boundary_penalty(PARAM_RANGES[:, 1])
        assert penalty_max > 0.0


class TestComputeReward:
    """Tests for the composite compute_reward function."""

    def test_returns_tuple(self) -> None:
        """Should return (float, dict) tuple."""
        sim = VoiceSimulator(rng=np.random.default_rng(42))
        params = SPEAKER_PROFILES["neutral"].to_params()
        features = sim.params_to_features(params, add_noise=False)
        action = np.zeros(NUM_PARAMS)

        total, components = compute_reward(features, features, params, action)
        assert isinstance(total, float)
        assert isinstance(components, dict)

    def test_perfect_match_high_reward(self) -> None:
        """Perfect match with zero action should give high reward."""
        sim = VoiceSimulator(rng=np.random.default_rng(42))
        params = SPEAKER_PROFILES["neutral"].to_params()
        features = sim.params_to_features(params, add_noise=False)
        action = np.zeros(NUM_PARAMS)

        total, components = compute_reward(features, features, params, action)
        assert total > 1.0  # target_weight(1.0)*1.0 + naturalness > 0

    def test_custom_config(self) -> None:
        """Test with custom reward configuration."""
        config = RewardConfig(
            target_weight=0.0,
            naturalness_weight=1.0,
            stability_weight=0.0,
            boundary_weight=0.0,
        )
        sim = VoiceSimulator(rng=np.random.default_rng(42))
        params = SPEAKER_PROFILES["neutral"].to_params()
        features = sim.params_to_features(params, add_noise=False)
        action = np.zeros(NUM_PARAMS)

        total, components = compute_reward(features, features, params, action, config)
        # Should be just naturalness reward
        assert abs(total - components["naturalness"]) < 1e-6
