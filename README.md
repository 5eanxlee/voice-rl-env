# Voice RL Environment

A [Gymnasium](https://gymnasium.farama.org/)-compatible reinforcement learning environment for training voice synthesis models. The agent learns to adjust synthesis parameters (pitch, speed, energy, breathiness, vibrato, formant shift, etc.) to match a target speaker's voice profile.

## Features

- **Gymnasium API**: Fully compatible with Gymnasium's `Env` interface, works with any RL library (stable-baselines3, RLlib, CleanRL, etc.)
- **Rich observation space**: 51-dimensional voice features including mel-spectrogram statistics, pitch contour, energy, duration, spectral, and formant features
- **10 continuous actions**: Control pitch shift, speed, energy, breathiness, vibrato (depth + rate), formant shift, spectral tilt, nasality, and tension
- **Composite reward**: Configurable blend of target matching, naturalness, stability, and boundary penalties
- **Speaker profiles**: 6 built-in profiles (neutral, high_pitch, low_pitch, breathy, energetic, nasal) plus random generation
- **Wrappers**: Feature-difference observations, reward normalization, curriculum learning
- **No heavy dependencies**: Core environment uses only NumPy and SciPy

## Installation

```bash
# Core environment only
pip install -e .

# With training dependencies (stable-baselines3 + PyTorch)
pip install -e ".[train]"

# With development tools
pip install -e ".[dev]"
```

## Quick Start

```python
import gymnasium as gym
from voice_rl_env import VoiceSynthesisEnv

# Create environment with a target speaker profile
env = VoiceSynthesisEnv(target_profile="neutral")

obs, info = env.reset(seed=42)
for _ in range(200):
    action = env.action_space.sample()  # Replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Best target match: {info['best_target_match']:.4f}")
env.close()
```

## Environment Details

### Observation Space

`Box(shape=(112,))` — concatenation of:

| Component | Dimensions | Description |
|---|---|---|
| Current voice features | 51 | Mel stats (32) + pitch (4) + energy (3) + duration (2) + spectral (4) + formants (6) |
| Target voice features | 51 | Same features for the target speaker |
| Current parameters | 10 | Normalized synthesis parameters in [-1, 1] |

### Action Space

`Box(shape=(10,), low=-1, high=1)` — parameter adjustments:

| Index | Parameter | Range | Description |
|---|---|---|---|
| 0 | pitch_shift | -12 to +12 | Semitones relative to base |
| 1 | speed_factor | 0.5 to 2.0 | Speaking rate multiplier |
| 2 | energy_scale | 0.2 to 3.0 | Overall energy |
| 3 | breathiness | 0.0 to 1.0 | Breath amount |
| 4 | vibrato_depth | 0.0 to 1.0 | Vibrato depth |
| 5 | vibrato_rate | 3.0 to 8.0 | Vibrato frequency (Hz) |
| 6 | formant_shift | 0.8 to 1.2 | Formant frequency multiplier |
| 7 | spectral_tilt | -6.0 to +6.0 | Spectral slope (dB/octave) |
| 8 | nasality | 0.0 to 1.0 | Nasal resonance |
| 9 | tension | 0.0 to 1.0 | Vocal tension |

### Reward

Composite reward with configurable weights:

- **Target matching** (default weight: 1.0): Gaussian similarity between current and target features
- **Naturalness** (default weight: 0.3): Penalizes unnatural parameter combinations
- **Stability penalty** (default weight: 0.1): Penalizes large parameter jumps
- **Boundary penalty** (default weight: 0.05): Penalizes parameters near extremes

### Episode Termination

- **Success**: Target matching score exceeds 0.95
- **Truncation**: Maximum steps reached (default: 200)

## Wrappers

```python
from voice_rl_env.wrappers import (
    FeatureDifferenceWrapper,  # Observe (target - current) instead of both
    NormalizeRewardWrapper,    # Running reward normalization
    CurriculumWrapper,         # Gradually increase difficulty
)
```

## Training Examples

```bash
# Train with PPO
python examples/train_ppo.py --target neutral --timesteps 100000

# Train with SAC
python examples/train_sac.py --target breathy --timesteps 100000

# Evaluate a trained model
python examples/evaluate.py --model-path ppo_voice --target neutral --render

# Evaluate with random agent
python examples/evaluate.py --target neutral --episodes 10 --render
```

## Custom Speaker Profiles

```python
from voice_rl_env import VoiceSynthesisEnv, SpeakerProfile

custom = SpeakerProfile(
    name="my_speaker",
    pitch_shift=3.0,
    speed_factor=1.1,
    energy_scale=1.0,
    breathiness=0.4,
    vibrato_depth=0.3,
    vibrato_rate=5.5,
    formant_shift=1.05,
    spectral_tilt=1.0,
    nasality=0.1,
    tension=0.2,
)

env = VoiceSynthesisEnv(target_profile=custom)
```

## Custom Reward Configuration

```python
from voice_rl_env import VoiceSynthesisEnv, RewardConfig

config = RewardConfig(
    target_weight=2.0,       # Emphasize target matching
    naturalness_weight=0.5,  # Increase naturalness importance
    stability_weight=0.2,    # Penalize jerky actions more
    boundary_weight=0.1,     # Stronger boundary avoidance
)

env = VoiceSynthesisEnv(reward_config=config)
```

## License

MIT
