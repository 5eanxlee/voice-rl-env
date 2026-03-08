"""Custom Gymnasium wrappers for the voice RL environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


class NormalizeRewardWrapper(gym.RewardWrapper):
    """Normalize rewards using a running mean and standard deviation.

    Keeps an exponential moving average of reward statistics and
    normalizes rewards to approximately zero mean and unit variance.

    Args:
        env: The environment to wrap.
        gamma: Discount factor for running statistics.
        epsilon: Small constant for numerical stability.
    """

    def __init__(
        self,
        env: gym.Env[NDArray[np.float64], NDArray[np.float64]],
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self._reward_mean: float = 0.0
        self._reward_var: float = 1.0
        self._return_val: float = 0.0

    def reward(self, reward: float) -> float:
        """Normalize the reward."""
        self._return_val = self._return_val * self.gamma + reward
        self._reward_mean = self._reward_mean * self.gamma + reward * (1.0 - self.gamma)
        self._reward_var = self._reward_var * self.gamma + (
            reward - self._reward_mean
        ) ** 2 * (1.0 - self.gamma)
        std = max(np.sqrt(self._reward_var), self.epsilon)
        return float((reward - self._reward_mean) / std)


class FeatureDifferenceWrapper(gym.ObservationWrapper):
    """Replace separate current/target features with their difference.

    Instead of observing current features and target features separately,
    the agent observes the difference (target - current) plus current params.
    This can simplify learning when the goal is to minimize the difference.

    The new observation space has shape (NUM_FEATURES + NUM_PARAMS,).
    """

    def __init__(
        self,
        env: gym.Env[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        super().__init__(env)
        from voice_rl_env.voice_simulator import NUM_FEATURES, NUM_PARAMS

        self._num_features = NUM_FEATURES
        self._num_params = NUM_PARAMS

        new_dim = NUM_FEATURES + NUM_PARAMS
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_dim,),
            dtype=np.float64,
        )

    def observation(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform observation to feature difference representation."""
        nf = self._num_features
        current_features = obs[:nf]
        target_features = obs[nf : 2 * nf]
        params = obs[2 * nf :]

        diff = target_features - current_features
        return np.concatenate([diff, params], dtype=np.float64)


class CurriculumWrapper(gym.Wrapper):
    """Curriculum learning wrapper that gradually increases task difficulty.

    Starts with the initial parameters close to the target and gradually
    increases the distance as the agent improves.

    Args:
        env: The environment to wrap.
        initial_difficulty: Starting difficulty (0 = trivial, 1 = full).
        difficulty_increment: How much to increase difficulty per episode.
        max_difficulty: Maximum difficulty level.
    """

    def __init__(
        self,
        env: gym.Env[NDArray[np.float64], NDArray[np.float64]],
        initial_difficulty: float = 0.1,
        difficulty_increment: float = 0.005,
        max_difficulty: float = 1.0,
    ) -> None:
        super().__init__(env)
        self.difficulty = initial_difficulty
        self.difficulty_increment = difficulty_increment
        self.max_difficulty = max_difficulty
        self._episode_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Reset with curriculum-adjusted initial parameters."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Access the unwrapped env to adjust params toward target
        base_env = self.unwrapped
        if hasattr(base_env, "_current_params") and hasattr(base_env, "_target_features"):
            assert base_env._simulator is not None
            # Interpolate initial params toward target
            from voice_rl_env.voice_simulator import SPEAKER_PROFILES

            # Get target params from features (approximate inverse)
            current = base_env._current_params
            # Move current params closer to a reasonable starting point
            # based on difficulty (lower difficulty = closer to a neutral start)
            neutral = SPEAKER_PROFILES["neutral"].to_params()
            alpha = self.difficulty
            base_env._current_params = (1.0 - alpha) * neutral + alpha * current

            # Recompute features and observation
            base_env._current_features = base_env._simulator.params_to_features(
                base_env._current_params
            )
            obs = base_env._get_observation()

        self._episode_count += 1
        self.difficulty = min(
            self.difficulty + self.difficulty_increment, self.max_difficulty
        )

        info["difficulty"] = self.difficulty
        info["episode_count"] = self._episode_count
        return obs, info
