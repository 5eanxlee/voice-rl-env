"""Voice RL Environment - A Gymnasium-compatible RL environment for training voice models."""

from voice_rl_env.env import VoiceSynthesisEnv
from voice_rl_env.rewards import RewardConfig, compute_reward
from voice_rl_env.voice_simulator import SpeakerProfile, VoiceSimulator

__all__ = [
    "VoiceSynthesisEnv",
    "RewardConfig",
    "compute_reward",
    "VoiceSimulator",
    "SpeakerProfile",
]

__version__ = "0.1.0"
