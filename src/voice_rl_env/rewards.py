"""Reward functions for the voice RL environment.

Provides a configurable composite reward based on multiple voice quality metrics:
- Target matching: how close the current voice is to the target profile
- Naturalness: penalizes unnatural parameter combinations
- Stability: penalizes large parameter changes between steps
- Boundary: penalizes parameters near their extreme values
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from voice_rl_env.voice_simulator import PARAM_RANGES


@dataclass
class RewardConfig:
    """Configuration for reward computation weights and thresholds.

    Attributes:
        target_weight: Weight for target-matching reward component.
        naturalness_weight: Weight for naturalness reward component.
        stability_weight: Weight for stability penalty component.
        boundary_weight: Weight for boundary penalty component.
        target_threshold: Feature distance below which target reward is maximal.
        stability_scale: Scale factor for action magnitude penalty.
        boundary_margin: Fraction of range considered "near boundary".
    """

    target_weight: float = 1.0
    naturalness_weight: float = 0.3
    stability_weight: float = 0.1
    boundary_weight: float = 0.05
    target_threshold: float = 0.1
    stability_scale: float = 2.0
    boundary_margin: float = 0.1


def target_matching_reward(
    current_features: NDArray[np.float64],
    target_features: NDArray[np.float64],
    threshold: float = 0.1,
) -> float:
    """Compute reward based on closeness to target voice features.

    Uses a Gaussian-shaped reward that peaks when current features match
    the target. The threshold controls the sharpness of the reward.

    Args:
        current_features: Current voice feature vector (NUM_FEATURES,).
        target_features: Target voice feature vector (NUM_FEATURES,).
        threshold: Controls reward sharpness (smaller = sharper).

    Returns:
        Reward in [0, 1], where 1 means perfect match.
    """
    diff = current_features - target_features
    mse = float(np.mean(diff**2))
    return float(np.exp(-mse / (2.0 * threshold**2)))


def naturalness_reward(params: NDArray[np.float64]) -> float:
    """Compute naturalness score based on parameter combinations.

    Penalizes unlikely parameter combinations that would produce
    unnatural-sounding voice output. Based on heuristics from
    speech synthesis research.

    Args:
        params: Current raw synthesis parameters (NUM_PARAMS,).

    Returns:
        Score in [0, 1], where 1 is maximally natural.
    """
    score = 1.0

    breathiness = params[3]
    tension = params[9]
    energy_scale = params[2]
    speed_factor = params[1]
    nasality = params[8]

    # High breathiness + high tension is unnatural
    bt_penalty = breathiness * tension
    score -= 0.3 * bt_penalty

    # Very high energy + very high speed is unnatural
    if energy_scale > 2.0 and speed_factor > 1.5:
        score -= 0.2 * (energy_scale / 3.0) * (speed_factor / 2.0)

    # High nasality + high breathiness is unnatural
    nb_penalty = nasality * breathiness
    score -= 0.15 * nb_penalty

    # Extreme speed values are less natural
    speed_dev = abs(speed_factor - 1.0)
    if speed_dev > 0.5:
        score -= 0.1 * (speed_dev - 0.5) / 0.5

    return float(np.clip(score, 0.0, 1.0))


def stability_penalty(
    action: NDArray[np.float64],
    scale: float = 2.0,
) -> float:
    """Compute penalty for large parameter changes.

    Encourages smooth, gradual adjustments to synthesis parameters
    rather than erratic jumps.

    Args:
        action: The action taken (parameter adjustments), shape (NUM_PARAMS,).
        scale: Scale factor for the penalty.

    Returns:
        Penalty in [0, 1], where 0 means no penalty (small action).
    """
    action_magnitude = float(np.mean(action**2))
    return float(1.0 - np.exp(-scale * action_magnitude))


def boundary_penalty(
    params: NDArray[np.float64],
    margin: float = 0.1,
) -> float:
    """Compute penalty for parameters near their boundary values.

    Discourages the agent from pushing parameters to extreme values
    which tend to produce artifacts.

    Args:
        params: Current raw synthesis parameters (NUM_PARAMS,).
        margin: Fraction of parameter range considered "near boundary".

    Returns:
        Penalty in [0, 1], accumulated over all parameters.
    """
    mins = PARAM_RANGES[:, 0]
    maxs = PARAM_RANGES[:, 1]
    ranges = maxs - mins

    # Distance from boundaries as fraction of range
    lower_dist = (params - mins) / ranges
    upper_dist = (maxs - params) / ranges

    # Penalty when closer than margin to either boundary
    lower_penalty = np.maximum(0.0, margin - lower_dist) / margin
    upper_penalty = np.maximum(0.0, margin - upper_dist) / margin

    total_penalty = float(np.mean(lower_penalty + upper_penalty))
    return float(np.clip(total_penalty, 0.0, 1.0))


def compute_reward(
    current_features: NDArray[np.float64],
    target_features: NDArray[np.float64],
    params: NDArray[np.float64],
    action: NDArray[np.float64],
    config: RewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute the composite reward for a step.

    Combines target matching, naturalness, stability, and boundary
    components into a single scalar reward.

    Args:
        current_features: Current voice feature vector.
        target_features: Target voice feature vector.
        params: Current raw synthesis parameters.
        action: The action taken this step.
        config: Reward configuration (uses defaults if None).

    Returns:
        Tuple of (total_reward, component_dict) where component_dict
        contains individual reward component values for logging.
    """
    if config is None:
        config = RewardConfig()

    # Compute individual components
    target_r = target_matching_reward(
        current_features, target_features, config.target_threshold
    )
    natural_r = naturalness_reward(params)
    stability_p = stability_penalty(action, config.stability_scale)
    boundary_p = boundary_penalty(params, config.boundary_margin)

    # Weighted combination
    total = (
        config.target_weight * target_r
        + config.naturalness_weight * natural_r
        - config.stability_weight * stability_p
        - config.boundary_weight * boundary_p
    )

    components = {
        "target_matching": target_r,
        "naturalness": natural_r,
        "stability_penalty": stability_p,
        "boundary_penalty": boundary_p,
        "total": total,
    }

    return total, components
