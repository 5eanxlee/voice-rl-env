"""Core Gymnasium environment for voice synthesis RL training.

The VoiceSynthesisEnv simulates a voice synthesis pipeline where an RL agent
learns to adjust synthesis parameters to match a target speaker's voice profile.
Each episode presents a target voice, and the agent iteratively refines
parameters to minimize the difference between generated and target features.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from voice_rl_env.rewards import RewardConfig, compute_reward
from voice_rl_env.voice_simulator import (
    NUM_FEATURES,
    NUM_PARAMS,
    PARAM_RANGES,
    SPEAKER_PROFILES,
    SpeakerProfile,
    VoiceSimulator,
)


class VoiceSynthesisEnv(gym.Env[NDArray[np.float64], NDArray[np.float64]]):
    """Gymnasium environment for voice synthesis parameter optimization.

    The agent observes the current voice features and target voice features,
    then outputs continuous adjustments to synthesis parameters. The goal is
    to produce voice features that match the target speaker profile.

    Observation Space:
        Box of shape (2 * NUM_FEATURES + NUM_PARAMS,) containing:
        - Current voice features (NUM_FEATURES)
        - Target voice features (NUM_FEATURES)
        - Current normalized parameters (NUM_PARAMS)

    Action Space:
        Box of shape (NUM_PARAMS,) in [-1, 1], representing parameter
        adjustments scaled by action_scale.

    Reward:
        Composite reward combining target matching, naturalness,
        stability, and boundary penalties. See rewards.py.

    Args:
        target_profile: Target speaker profile or name from SPEAKER_PROFILES.
            If None, a random profile is sampled each episode.
        max_steps: Maximum steps per episode.
        action_scale: Scale factor for actions (controls step size).
        noise_std: Observation noise standard deviation.
        reward_config: Reward function configuration.
        render_mode: Gymnasium render mode ("human" or "ansi").
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(
        self,
        target_profile: str | SpeakerProfile | None = None,
        max_steps: int = 200,
        action_scale: float = 0.1,
        noise_std: float = 0.02,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.max_steps = max_steps
        self.action_scale = action_scale
        self.noise_std = noise_std
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode

        # Resolve target profile
        self._target_profile_spec = target_profile
        self._target_profile: SpeakerProfile | None = None
        if isinstance(target_profile, str):
            if target_profile in SPEAKER_PROFILES:
                self._target_profile = SPEAKER_PROFILES[target_profile]
            else:
                raise ValueError(
                    f"Unknown profile '{target_profile}'. "
                    f"Available: {list(SPEAKER_PROFILES.keys())}"
                )
        elif isinstance(target_profile, SpeakerProfile):
            self._target_profile = target_profile

        # Observation: current features + target features + current params
        obs_dim = 2 * NUM_FEATURES + NUM_PARAMS
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float64,
        )

        # Action: parameter adjustments in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(NUM_PARAMS,),
            dtype=np.float64,
        )

        # Internal state (initialized in reset)
        self._simulator: VoiceSimulator | None = None
        self._current_params: NDArray[np.float64] | None = None
        self._target_features: NDArray[np.float64] | None = None
        self._current_features: NDArray[np.float64] | None = None
        self._step_count: int = 0
        self._episode_reward: float = 0.0
        self._best_target_match: float = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Optional dict that may contain:
                - "target_profile": Override target profile for this episode.
                - "initial_params": Starting parameter values.

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        # Create simulator with environment's RNG
        rng = np.random.default_rng(seed if seed is not None else self.np_random.integers(2**31))
        self._simulator = VoiceSimulator(
            noise_std=self.noise_std,
            rng=rng,
        )

        # Determine target profile
        target = self._target_profile
        if options and "target_profile" in options:
            tp = options["target_profile"]
            if isinstance(tp, str):
                target = SPEAKER_PROFILES[tp]
            elif isinstance(tp, SpeakerProfile):
                target = tp
        if target is None:
            target = self._simulator.random_profile(name="episode_target")

        # Compute target features (without noise for a stable target)
        target_params = target.to_params()
        self._target_features = self._simulator.params_to_features(
            target_params, add_noise=False
        )

        # Initialize current parameters
        if options and "initial_params" in options:
            self._current_params = np.array(options["initial_params"], dtype=np.float64)
        else:
            self._current_params = self._simulator.random_params()

        # Compute initial features
        self._current_features = self._simulator.params_to_features(self._current_params)

        # Reset counters
        self._step_count = 0
        self._episode_reward = 0.0
        self._best_target_match = 0.0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Parameter adjustments in [-1, 1], shape (NUM_PARAMS,).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self._simulator is not None, "Must call reset() before step()"
        assert self._current_params is not None
        assert self._target_features is not None

        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Apply action as parameter adjustment
        param_ranges = PARAM_RANGES[:, 1] - PARAM_RANGES[:, 0]
        param_delta = action * self.action_scale * param_ranges
        new_params = self._current_params + param_delta

        # Clip to valid parameter ranges
        new_params = np.clip(new_params, PARAM_RANGES[:, 0], PARAM_RANGES[:, 1])
        self._current_params = new_params

        # Compute new features
        self._current_features = self._simulator.params_to_features(self._current_params)

        # Compute reward
        reward, reward_components = compute_reward(
            current_features=self._current_features,
            target_features=self._target_features,
            params=self._current_params,
            action=action,
            config=self.reward_config,
        )

        # Update counters
        self._step_count += 1
        self._episode_reward += reward
        self._best_target_match = max(
            self._best_target_match, reward_components["target_matching"]
        )

        # Check termination conditions
        terminated = reward_components["target_matching"] > 0.95
        truncated = self._step_count >= self.max_steps

        obs = self._get_observation()
        info = self._get_info()
        info["reward_components"] = reward_components

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> NDArray[np.float64]:
        """Construct the observation vector."""
        assert self._current_features is not None
        assert self._target_features is not None
        assert self._current_params is not None
        assert self._simulator is not None

        normalized_params = self._simulator.normalize_params(self._current_params)
        return np.concatenate(
            [self._current_features, self._target_features, normalized_params],
            dtype=np.float64,
        )

    def _get_info(self) -> dict[str, Any]:
        """Construct the info dictionary."""
        assert self._current_params is not None

        return {
            "step_count": self._step_count,
            "episode_reward": self._episode_reward,
            "best_target_match": self._best_target_match,
            "current_params": self._current_params.copy(),
        }

    def render(self) -> str | None:
        """Render the environment state.

        Returns:
            String representation when render_mode is "ansi", None otherwise.
        """
        if self._current_params is None:
            return None

        if self.render_mode == "ansi":
            from voice_rl_env.voice_simulator import PARAM_NAMES

            lines = [
                f"Step: {self._step_count}/{self.max_steps}",
                f"Episode Reward: {self._episode_reward:.3f}",
                f"Best Target Match: {self._best_target_match:.3f}",
                "",
                "Current Parameters:",
            ]
            for name, val in zip(PARAM_NAMES, self._current_params):
                lines.append(f"  {name:>20s}: {val:+.4f}")
            output = "\n".join(lines)
            print(output)
            return output

        return None
