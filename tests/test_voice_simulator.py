"""Tests for the voice simulator."""

from __future__ import annotations

import numpy as np

from voice_rl_env.voice_simulator import (
    NUM_FEATURES,
    NUM_PARAMS,
    PARAM_RANGES,
    SPEAKER_PROFILES,
    SpeakerProfile,
    VoiceSimulator,
)


class TestVoiceSimulator:
    """Tests for VoiceSimulator."""

    def test_params_to_features_shape(self) -> None:
        """Test that feature output has correct shape."""
        sim = VoiceSimulator(rng=np.random.default_rng(42))
        params = SPEAKER_PROFILES["neutral"].to_params()
        features = sim.params_to_features(params)
        assert features.shape == (NUM_FEATURES,)

    def test_params_to_features_deterministic_without_noise(self) -> None:
        """Test deterministic output when noise is disabled."""
        sim = VoiceSimulator(rng=np.random.default_rng(42))
        params = SPEAKER_PROFILES["neutral"].to_params()

        f1 = sim.params_to_features(params, add_noise=False)
        f2 = sim.params_to_features(params, add_noise=False)

        np.testing.assert_array_equal(f1, f2)

    def test_params_to_features_noise(self) -> None:
        """Test that noise is added when enabled."""
        sim = VoiceSimulator(noise_std=0.1, rng=np.random.default_rng(42))
        params = SPEAKER_PROFILES["neutral"].to_params()

        f_clean = sim.params_to_features(params, add_noise=False)
        f_noisy = sim.params_to_features(params, add_noise=True)

        # Should be different due to noise
        assert not np.array_equal(f_clean, f_noisy)

    def test_different_params_different_features(self) -> None:
        """Test that different parameters produce different features."""
        sim = VoiceSimulator(rng=np.random.default_rng(42))

        f1 = sim.params_to_features(
            SPEAKER_PROFILES["neutral"].to_params(), add_noise=False
        )
        f2 = sim.params_to_features(
            SPEAKER_PROFILES["high_pitch"].to_params(), add_noise=False
        )

        assert not np.array_equal(f1, f2)

    def test_normalize_denormalize_roundtrip(self) -> None:
        """Test that normalize/denormalize are inverses."""
        sim = VoiceSimulator(rng=np.random.default_rng(42))
        params = SPEAKER_PROFILES["neutral"].to_params()

        normalized = sim.normalize_params(params)
        recovered = sim.denormalize_params(normalized)

        np.testing.assert_allclose(params, recovered, atol=1e-10)

    def test_normalize_range(self) -> None:
        """Test that normalized params are in [-1, 1]."""
        sim = VoiceSimulator(rng=np.random.default_rng(42))

        # Test with min values
        normalized_min = sim.normalize_params(PARAM_RANGES[:, 0])
        np.testing.assert_allclose(normalized_min, -np.ones(NUM_PARAMS), atol=1e-10)

        # Test with max values
        normalized_max = sim.normalize_params(PARAM_RANGES[:, 1])
        np.testing.assert_allclose(normalized_max, np.ones(NUM_PARAMS), atol=1e-10)

    def test_random_params_in_range(self) -> None:
        """Test that random params are within valid ranges."""
        sim = VoiceSimulator(rng=np.random.default_rng(42))

        for _ in range(100):
            params = sim.random_params()
            assert np.all(params >= PARAM_RANGES[:, 0])
            assert np.all(params <= PARAM_RANGES[:, 1])

    def test_random_profile(self) -> None:
        """Test random profile generation."""
        sim = VoiceSimulator(rng=np.random.default_rng(42))
        profile = sim.random_profile(name="test")

        assert profile.name == "test"
        params = profile.to_params()
        assert params.shape == (NUM_PARAMS,)
        assert np.all(params >= PARAM_RANGES[:, 0])
        assert np.all(params <= PARAM_RANGES[:, 1])


class TestSpeakerProfile:
    """Tests for SpeakerProfile."""

    def test_to_params(self) -> None:
        """Test conversion to parameter array."""
        profile = SpeakerProfile()
        params = profile.to_params()
        assert params.shape == (NUM_PARAMS,)
        assert params.dtype == np.float64

    def test_from_params(self) -> None:
        """Test creation from parameter array."""
        original = SPEAKER_PROFILES["neutral"]
        params = original.to_params()
        recovered = SpeakerProfile.from_params(params, name="recovered")

        assert recovered.name == "recovered"
        np.testing.assert_allclose(recovered.to_params(), params)

    def test_predefined_profiles(self) -> None:
        """Test that all predefined profiles have valid parameters."""
        for name, profile in SPEAKER_PROFILES.items():
            params = profile.to_params()
            assert params.shape == (NUM_PARAMS,), f"Profile {name} has wrong shape"
            assert np.all(params >= PARAM_RANGES[:, 0]), f"Profile {name} below min"
            assert np.all(params <= PARAM_RANGES[:, 1]), f"Profile {name} above max"
