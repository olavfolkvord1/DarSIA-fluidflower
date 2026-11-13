"""Phase-model inference: optical features and two logistic heads.

A phase model classifies each pixel of a corrected image by its optical change
against a baseline image of the same run and against the previous frame. It has
two heads on one standardized feature matrix: the probability that a pixel holds
CO2 (in any phase), and the probability that the CO2 in it is free gas.

A delivered model is a folder holding the two heads and the decision policy::

    models/generic/co2_presence_signal.json
    models/generic/gas_within_co2_signal.json
    phase_decision_policy.json

The model carries the decision threshold of its gas head, tuned with the model;
the phase rules apply it. The CO2 threshold is a setting of the phase rules.

The model carries its sizes in millimetres. A delivered head gives its texture
scale in pixels of the grid it was trained on; the reader converts it with that
grid's pixel size, and inference converts it to pixels of the grid of the given
images. All arithmetic runs in the floating dtype of the given images.

"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PHASE_FEATURE_NAMES = (
    "delta_r",
    "delta_g",
    "delta_b",
    "delta_intensity",
    "delta_red_fraction",
    "delta_green_fraction",
    "delta_blue_fraction",
    "delta_log_r_over_g",
    "delta_log_g_over_b",
    "delta_log_r_over_b",
    "delta_texture_highpass",
    "abs_delta_texture_highpass",
    "rgb_change_norm",
    "local_rgb_change_mean",
    "local_rgb_change_std",
    "local_rgb_change_coherence",
    "baseline_intensity",
    "baseline_signed_local_contrast",
    "current_signed_local_contrast",
    "delta_signed_local_contrast",
    "persistent_bright_contrast",
    "new_bright_contrast",
    "baseline_relative_local_contrast",
    "current_relative_local_contrast",
    "delta_relative_local_contrast",
    "persistent_relative_bright_contrast",
    "new_relative_bright_contrast",
    "relative_bright_contrast_agreement",
    "dynamic_available",
    "dynamic_delta_r",
    "dynamic_delta_g",
    "dynamic_delta_b",
    "dynamic_delta_intensity",
    "dynamic_texture_delta",
    "dynamic_rgb_change_norm",
    "local_dynamic_rgb_change_mean",
)
"""Feature columns, in the order of the model coefficients."""

_LINEAR_RGB = "camera-white-relative-linear-rgb"
"""Input colour space of the features: linear RGB without clipping or scaling."""


@dataclass(frozen=True)
class LogisticHead:
    """One logistic head on standardized features."""

    feature_mean: np.ndarray
    """Mean of each feature column in the training data."""
    feature_scale: np.ndarray
    """Standard deviation of each feature column in the training data."""
    coefficients: np.ndarray
    """Coefficient of each standardized feature column."""
    intercept: float
    """Intercept of the logit."""

    def probability(self, features: np.ndarray) -> np.ndarray:
        """Probability of the head's class for each row of a feature matrix.

        The parameters are cast to the dtype of the features, and the logit is
        clipped to [-30, 30] before the logistic function.

        Parameters
        ----------
        features : array
            Feature matrix of shape (pixels, features).

        Returns
        -------
        array
            Probabilities of shape (pixels,), in the dtype of the features.

        """
        dtype = features.dtype
        standardized = (features - self.feature_mean.astype(dtype)) / (
            self.feature_scale.astype(dtype)
        )
        logits = standardized @ self.coefficients.astype(dtype)
        logits = np.clip(logits + dtype.type(self.intercept), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


@dataclass(frozen=True)
class PhaseModel:
    """A phase model: two heads sharing one texture scale, and the gas threshold."""

    co2_presence: LogisticHead
    """Head giving the probability that a pixel holds CO2."""
    gas_within_co2: LogisticHead
    """Head giving the probability that the CO2 in a pixel is free gas."""
    texture_sigma_mm: float
    """Standard deviation (mm) of the Gaussian defining the image texture."""
    gas_threshold: float
    """Decision threshold of the gas head: CO2 is free gas where the gas head's
    probability reaches it."""


def read_phase_model(folder: str | Path, pixel_size_mm: float) -> PhaseModel:
    """Read the two heads of a delivered phase model folder.

    The gas threshold is the gas head's presence threshold in the folder's
    ``phase_decision_policy.json``.

    Parameters
    ----------
    folder : str or Path
        Model folder holding the heads in ``models/generic``.
    pixel_size_mm : float
        Pixel size of the grid the model was trained on; the heads give their
        texture scale in its pixels.

    Returns
    -------
    PhaseModel
        The two heads, their texture scale and the gas threshold.

    Raises
    ------
    ValueError
        If a head uses other features than :data:`PHASE_FEATURE_NAMES`, was
        trained on other than linear RGB, or the heads use different texture
        scales.

    """
    heads = {}
    for role in ("co2_presence", "gas_within_co2"):
        path = Path(folder) / "models" / "generic" / f"{role}_signal.json"
        heads[role] = json.loads(path.read_text(encoding="utf-8"))
    for payload in heads.values():
        if (
            tuple(payload["feature_names"]) != PHASE_FEATURE_NAMES
            or payload["feature_contract"]["input_color_space"] != _LINEAR_RGB
        ):
            raise ValueError(
                "Only phase heads on the listed features of linear RGB are supported."
            )
    policy = json.loads(
        (Path(folder) / "phase_decision_policy.json").read_text(encoding="utf-8")
    )
    gas = policy["models"]["generic"]["gas_within_co2"]
    sigmas = {payload["texture_sigma"] for payload in heads.values()}
    if len(sigmas) != 1:
        raise ValueError("The two phase heads use different texture scales.")
    return PhaseModel(
        **{
            role: LogisticHead(
                feature_mean=np.asarray(payload["feature_mean"], dtype=float),
                feature_scale=np.asarray(payload["feature_scale"], dtype=float),
                coefficients=np.asarray(payload["coefficients"], dtype=float),
                intercept=float(payload["intercept"]),
            )
            for role, payload in heads.items()
        },
        texture_sigma_mm=float(sigmas.pop()) * pixel_size_mm,
        gas_threshold=float(gas["presence_threshold"]),
    )
