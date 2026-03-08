"""Voice synthesis simulator using numpy-based feature generation.

Simulates the relationship between synthesis parameters and resulting voice features
without requiring actual audio processing. Uses realistic parameter-to-feature mappings
derived from speech science.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# --- Constants ---

# Synthesis parameter names and their valid ranges
PARAM_NAMES: list[str] = [
    "pitch_shift",      # semitones relative to base (-12 to +12)
    "speed_factor",     # speaking rate multiplier (0.5 to 2.0)
    "energy_scale",     # overall energy multiplier (0.2 to 3.0)
    "breathiness",      # breathiness amount (0.0 to 1.0)
    "vibrato_depth",    # vibrato depth in semitones (0.0 to 1.0)
    "vibrato_rate",     # vibrato rate in Hz (3.0 to 8.0)
    "formant_shift",    # formant frequency shift factor (0.8 to 1.2)
    "spectral_tilt",    # spectral tilt in dB/octave (-6.0 to 6.0)
    "nasality",         # nasality amount (0.0 to 1.0)
    "tension",          # vocal tension (0.0 to 1.0)
]

NUM_PARAMS: int = len(PARAM_NAMES)

# Raw parameter ranges (min, max) before normalization
PARAM_RANGES: NDArray[np.float64] = np.array(
    [
        [-12.0, 12.0],    # pitch_shift
        [0.5, 2.0],       # speed_factor
        [0.2, 3.0],       # energy_scale
        [0.0, 1.0],       # breathiness
        [0.0, 1.0],       # vibrato_depth
        [3.0, 8.0],       # vibrato_rate
        [0.8, 1.2],       # formant_shift
        [-6.0, 6.0],      # spectral_tilt
        [0.0, 1.0],       # nasality
        [0.0, 1.0],       # tension
    ],
    dtype=np.float64,
)

# Feature dimension breakdown
NUM_MEL_BANDS: int = 8
NUM_MEL_STATS: int = 4  # mean, std, skew, kurtosis per band
NUM_PITCH_FEATURES: int = 4  # mean, std, range, jitter
NUM_ENERGY_FEATURES: int = 3  # mean, std, dynamic_range
NUM_DURATION_FEATURES: int = 2  # speaking_rate, pause_ratio
NUM_SPECTRAL_FEATURES: int = 4  # centroid, bandwidth, rolloff, flux
NUM_FORMANT_FEATURES: int = 6  # F1-F3 freq + bandwidth

NUM_FEATURES: int = (
    NUM_MEL_BANDS * NUM_MEL_STATS
    + NUM_PITCH_FEATURES
    + NUM_ENERGY_FEATURES
    + NUM_DURATION_FEATURES
    + NUM_SPECTRAL_FEATURES
    + NUM_FORMANT_FEATURES
)  # = 32 + 4 + 3 + 2 + 4 + 6 = 51


@dataclass
class SpeakerProfile:
    """Defines a target speaker's voice characteristics.

    Each field represents a synthesis parameter value that characterizes
    the speaker's natural voice. The RL agent's goal is to find parameters
    that produce features matching this profile.
    """

    name: str = "default"
    pitch_shift: float = 0.0
    speed_factor: float = 1.0
    energy_scale: float = 1.0
    breathiness: float = 0.15
    vibrato_depth: float = 0.3
    vibrato_rate: float = 5.5
    formant_shift: float = 1.0
    spectral_tilt: float = 0.0
    nasality: float = 0.1
    tension: float = 0.3

    def to_params(self) -> NDArray[np.float64]:
        """Convert profile to parameter array."""
        return np.array(
            [
                self.pitch_shift,
                self.speed_factor,
                self.energy_scale,
                self.breathiness,
                self.vibrato_depth,
                self.vibrato_rate,
                self.formant_shift,
                self.spectral_tilt,
                self.nasality,
                self.tension,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_params(cls, params: NDArray[np.float64], name: str = "custom") -> SpeakerProfile:
        """Create a profile from a parameter array."""
        return cls(
            name=name,
            pitch_shift=float(params[0]),
            speed_factor=float(params[1]),
            energy_scale=float(params[2]),
            breathiness=float(params[3]),
            vibrato_depth=float(params[4]),
            vibrato_rate=float(params[5]),
            formant_shift=float(params[6]),
            spectral_tilt=float(params[7]),
            nasality=float(params[8]),
            tension=float(params[9]),
        )


# Predefined speaker profiles for diverse training scenarios
SPEAKER_PROFILES: dict[str, SpeakerProfile] = {
    "neutral": SpeakerProfile(
        name="neutral",
        pitch_shift=0.0,
        speed_factor=1.0,
        energy_scale=1.0,
        breathiness=0.15,
        vibrato_depth=0.2,
        vibrato_rate=5.5,
        formant_shift=1.0,
        spectral_tilt=0.0,
        nasality=0.1,
        tension=0.3,
    ),
    "high_pitch": SpeakerProfile(
        name="high_pitch",
        pitch_shift=5.0,
        speed_factor=1.1,
        energy_scale=0.9,
        breathiness=0.2,
        vibrato_depth=0.35,
        vibrato_rate=6.0,
        formant_shift=1.1,
        spectral_tilt=1.5,
        nasality=0.08,
        tension=0.4,
    ),
    "low_pitch": SpeakerProfile(
        name="low_pitch",
        pitch_shift=-4.0,
        speed_factor=0.9,
        energy_scale=1.2,
        breathiness=0.1,
        vibrato_depth=0.15,
        vibrato_rate=4.5,
        formant_shift=0.9,
        spectral_tilt=-2.0,
        nasality=0.15,
        tension=0.25,
    ),
    "breathy": SpeakerProfile(
        name="breathy",
        pitch_shift=1.0,
        speed_factor=0.95,
        energy_scale=0.7,
        breathiness=0.7,
        vibrato_depth=0.4,
        vibrato_rate=5.0,
        formant_shift=1.02,
        spectral_tilt=3.0,
        nasality=0.05,
        tension=0.15,
    ),
    "energetic": SpeakerProfile(
        name="energetic",
        pitch_shift=2.0,
        speed_factor=1.3,
        energy_scale=1.8,
        breathiness=0.05,
        vibrato_depth=0.25,
        vibrato_rate=6.5,
        formant_shift=1.05,
        spectral_tilt=-1.0,
        nasality=0.12,
        tension=0.6,
    ),
    "nasal": SpeakerProfile(
        name="nasal",
        pitch_shift=-1.0,
        speed_factor=1.05,
        energy_scale=0.85,
        breathiness=0.1,
        vibrato_depth=0.15,
        vibrato_rate=5.0,
        formant_shift=0.95,
        spectral_tilt=0.5,
        nasality=0.7,
        tension=0.35,
    ),
}


@dataclass
class VoiceSimulator:
    """Simulates voice synthesis by mapping parameters to voice features.

    Uses a physics-inspired model to generate realistic voice features from
    synthesis parameters. The mapping is nonlinear and includes interactions
    between parameters to simulate real voice characteristics.

    Attributes:
        noise_std: Standard deviation of observation noise.
        interaction_strength: Strength of parameter interaction effects.
        rng: NumPy random number generator for reproducibility.
    """

    noise_std: float = 0.02
    interaction_strength: float = 0.3
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def _compute_mel_features(
        self, params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute mel-spectrogram statistics from synthesis parameters.

        Models how pitch, energy, breathiness, and spectral tilt affect
        the mel-spectrogram distribution across frequency bands.
        """
        pitch_shift = params[0]
        energy_scale = params[2]
        breathiness = params[3]
        spectral_tilt = params[7]
        tension = params[9]

        features = np.zeros(NUM_MEL_BANDS * NUM_MEL_STATS, dtype=np.float64)

        for band in range(NUM_MEL_BANDS):
            band_center = (band + 0.5) / NUM_MEL_BANDS
            idx = band * NUM_MEL_STATS

            # Mean energy in band: affected by energy scale, spectral tilt, pitch
            base_energy = energy_scale * np.exp(-0.5 * spectral_tilt * band_center)
            pitch_effect = 0.1 * pitch_shift / 12.0 * (1.0 - band_center)
            features[idx] = base_energy + pitch_effect

            # Std: affected by breathiness and tension
            features[idx + 1] = 0.1 + 0.3 * breathiness + 0.15 * tension

            # Skew: affected by tension and spectral tilt
            features[idx + 2] = 0.2 * tension - 0.1 * spectral_tilt / 6.0

            # Kurtosis: affected by breathiness
            features[idx + 3] = 3.0 + 1.5 * breathiness - 0.5 * tension

        return features

    def _compute_pitch_features(
        self, params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute pitch contour statistics.

        Models pitch mean, variation, range, and jitter based on
        synthesis parameters.
        """
        pitch_shift = params[0]
        vibrato_depth = params[4]
        vibrato_rate = params[5]
        tension = params[9]

        # Base pitch in Hz (relative to 150 Hz reference)
        base_pitch = 150.0 * (2.0 ** (pitch_shift / 12.0))

        # Pitch mean (normalized)
        pitch_mean = base_pitch / 300.0

        # Pitch std: vibrato adds periodic variation, tension adds irregularity
        pitch_std = 0.05 + 0.15 * vibrato_depth + 0.08 * tension

        # Pitch range: affected by vibrato depth and rate interaction
        pitch_range = vibrato_depth * (1.0 + 0.1 * (vibrato_rate - 5.5))

        # Jitter: micro-perturbations, higher with tension, lower with breathiness
        jitter = 0.01 + 0.03 * tension + 0.01 * params[3]  # breathiness

        return np.array([pitch_mean, pitch_std, pitch_range, jitter], dtype=np.float64)

    def _compute_energy_features(
        self, params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute energy statistics."""
        energy_scale = params[2]
        breathiness = params[3]
        tension = params[9]

        # Mean energy (log scale)
        energy_mean = np.log1p(energy_scale) / np.log1p(3.0)

        # Energy std: breathiness adds fluctuations
        energy_std = 0.1 + 0.2 * breathiness + 0.1 * tension

        # Dynamic range: tension increases it, breathiness decreases it
        dynamic_range = 0.5 + 0.3 * tension - 0.2 * breathiness

        return np.array([energy_mean, energy_std, dynamic_range], dtype=np.float64)

    def _compute_duration_features(
        self, params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute duration-related features."""
        speed_factor = params[1]
        breathiness = params[3]

        # Speaking rate (normalized, 1.0 = normal)
        speaking_rate = speed_factor

        # Pause ratio: slower speech and breathy voice have more pauses
        pause_ratio = 0.15 + 0.1 / speed_factor + 0.05 * breathiness

        return np.array([speaking_rate, pause_ratio], dtype=np.float64)

    def _compute_spectral_features(
        self, params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute spectral features (centroid, bandwidth, rolloff, flux)."""
        pitch_shift = params[0]
        energy_scale = params[2]
        breathiness = params[3]
        spectral_tilt = params[7]
        tension = params[9]

        # Spectral centroid: affected by pitch, tilt, tension
        centroid = 0.4 + 0.1 * pitch_shift / 12.0 - 0.1 * spectral_tilt / 6.0 + 0.15 * tension

        # Spectral bandwidth: breathiness broadens spectrum
        bandwidth = 0.3 + 0.2 * breathiness + 0.1 * tension

        # Spectral rolloff: higher with tension, lower with breathiness
        rolloff = 0.6 + 0.15 * tension - 0.1 * breathiness + 0.05 * pitch_shift / 12.0

        # Spectral flux: rate of spectral change
        flux = 0.1 + 0.1 * energy_scale / 3.0 + 0.05 * tension

        return np.array([centroid, bandwidth, rolloff, flux], dtype=np.float64)

    def _compute_formant_features(
        self, params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute formant features (F1-F3 frequencies and bandwidths).

        Formants are resonant frequencies of the vocal tract, affected by
        formant shift, nasality, and tension.
        """
        formant_shift = params[6]
        nasality = params[8]
        tension = params[9]

        # Base formant frequencies (Hz), normalized to [0, 1] range
        f1_base = 500.0 / 4000.0  # ~0.125
        f2_base = 1500.0 / 4000.0  # ~0.375
        f3_base = 2500.0 / 4000.0  # ~0.625

        # Apply formant shift
        f1 = f1_base * formant_shift
        f2 = f2_base * formant_shift
        f3 = f3_base * formant_shift

        # Nasality lowers F1 and introduces anti-formant near F1
        f1 -= 0.02 * nasality
        f2 += 0.01 * nasality

        # Tension raises formants slightly
        f1 += 0.01 * tension
        f2 += 0.015 * tension
        f3 += 0.01 * tension

        # Bandwidths: nasality broadens, tension narrows
        bw1 = 0.05 + 0.03 * nasality - 0.015 * tension
        bw2 = 0.04 + 0.02 * nasality - 0.01 * tension
        bw3 = 0.06 + 0.025 * nasality - 0.01 * tension

        return np.array([f1, f2, f3, bw1, bw2, bw3], dtype=np.float64)

    def _apply_interactions(
        self,
        features: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply nonlinear parameter interactions to features.

        Real voice characteristics exhibit complex interactions between
        parameters (e.g., high tension with high breathiness creates a
        qualitatively different sound than either alone).
        """
        breathiness = params[3]
        tension = params[9]
        energy_scale = params[2]
        nasality = params[8]

        # Breathiness-tension interaction (they counteract each other)
        bt_interaction = breathiness * tension * self.interaction_strength
        features = features * (1.0 + 0.1 * bt_interaction)

        # Energy-tension interaction (high energy + tension = harsh)
        et_interaction = (energy_scale / 3.0) * tension * self.interaction_strength
        features = features + 0.05 * et_interaction

        # Nasality-formant interaction (nonlinear effect on spectral shape)
        nf_interaction = nasality * params[6] * self.interaction_strength
        features[-6:] *= 1.0 + 0.1 * nf_interaction

        return features

    def params_to_features(
        self,
        params: NDArray[np.float64],
        add_noise: bool = True,
    ) -> NDArray[np.float64]:
        """Convert synthesis parameters to voice features.

        Args:
            params: Array of synthesis parameters (shape: NUM_PARAMS).
            add_noise: Whether to add observation noise.

        Returns:
            Feature vector of shape (NUM_FEATURES,).
        """
        mel_features = self._compute_mel_features(params)
        pitch_features = self._compute_pitch_features(params)
        energy_features = self._compute_energy_features(params)
        duration_features = self._compute_duration_features(params)
        spectral_features = self._compute_spectral_features(params)
        formant_features = self._compute_formant_features(params)

        features = np.concatenate(
            [
                mel_features,
                pitch_features,
                energy_features,
                duration_features,
                spectral_features,
                formant_features,
            ]
        )

        # Apply parameter interactions
        features = self._apply_interactions(features, params)

        # Add observation noise
        if add_noise:
            noise = self.rng.normal(0.0, self.noise_std, size=features.shape)
            features = features + noise

        return features

    def normalize_params(
        self, raw_params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Normalize parameters from raw ranges to [-1, 1]."""
        mins = PARAM_RANGES[:, 0]
        maxs = PARAM_RANGES[:, 1]
        return 2.0 * (raw_params - mins) / (maxs - mins) - 1.0

    def denormalize_params(
        self, norm_params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Denormalize parameters from [-1, 1] to raw ranges."""
        mins = PARAM_RANGES[:, 0]
        maxs = PARAM_RANGES[:, 1]
        return mins + (norm_params + 1.0) / 2.0 * (maxs - mins)

    def random_params(self) -> NDArray[np.float64]:
        """Generate random synthesis parameters within valid ranges."""
        mins = PARAM_RANGES[:, 0]
        maxs = PARAM_RANGES[:, 1]
        return self.rng.uniform(mins, maxs).astype(np.float64)

    def random_profile(self, name: str | None = None) -> SpeakerProfile:
        """Generate a random speaker profile."""
        params = self.random_params()
        return SpeakerProfile.from_params(params, name=name or "random")
