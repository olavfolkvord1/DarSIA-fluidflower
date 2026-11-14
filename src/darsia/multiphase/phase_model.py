"""Phase-model inference: optical features and two logistic heads.

A phase model classifies each pixel of a corrected image by its optical change
against a baseline image of the same run and against the previous frame. It has
two heads on one standardized feature matrix: the probability that a pixel holds
CO2 (in any phase), and the probability that the CO2 in it is free gas.

A model is a folder holding the two heads and the decision policy::

    models/generic/co2_presence_signal.json
    models/generic/gas_within_co2_signal.json
    phase_decision_policy.json

The model carries the decision threshold of its gas head, tuned with the model;
the phase rules apply it. The CO2 threshold is a setting of the phase rules.

The model carries its sizes in millimetres. A trained head records its texture
scale in mm; a delivered head gives it in pixels of the grid it was trained on,
and the reader converts it with that grid's pixel size. Inference converts it to
pixels of the grid of the given images. All arithmetic runs in the floating
dtype of the given images.

"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import darsia

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

_SPATIAL_CONTEXT_FRACTION = 0.0015
"""Width of the local change statistics, as a fraction of the shorter physical
side of the image (on square pixels, of its shorter pixel count)."""

_ROWS_PER_CHUNK = 64
"""Image rows per feature chunk; the full feature matrix is never held."""


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


def _head_path(folder: str | Path, role: str) -> Path:
    """Path of one head in a model folder."""
    return Path(folder) / "models" / "generic" / f"{role}_signal.json"


def read_phase_model(
    folder: str | Path, pixel_size_mm: float | None = None
) -> PhaseModel:
    """Read the two heads of a phase model folder.

    A trained folder (:func:`write_phase_model`) gives the texture scale in mm.
    A delivered folder gives it in pixels of the grid the model was trained on,
    whose pixel size must then be given. The gas threshold is the gas head's
    presence threshold in the folder's ``phase_decision_policy.json``.

    Parameters
    ----------
    folder : str or Path
        Model folder holding the heads in ``models/generic``.
    pixel_size_mm : float, optional
        Pixel size of the training grid of a delivered folder.

    Returns
    -------
    PhaseModel
        The two heads, their texture scale and the gas threshold.

    Raises
    ------
    ValueError
        If a head uses other features than :data:`PHASE_FEATURE_NAMES`, was
        trained on other than linear RGB, the heads use different texture
        scales, or a delivered folder is read without the pixel size.

    """
    heads = {}
    for role in ("co2_presence", "gas_within_co2"):
        heads[role] = json.loads(_head_path(folder, role).read_text(encoding="utf-8"))
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
    if "texture_sigma_mm" in heads["co2_presence"]:
        key, scale = "texture_sigma_mm", 1.0
    elif pixel_size_mm is None:
        raise ValueError(
            "A delivered phase model needs its training grid's pixel size."
        )
    else:
        key, scale = "texture_sigma", pixel_size_mm
    sigmas = {payload[key] for payload in heads.values()}
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
        texture_sigma_mm=float(sigmas.pop()) * scale,
        gas_threshold=float(gas["presence_threshold"]),
    )


def write_phase_model(
    folder: str | Path,
    model: PhaseModel,
    *,
    pixel_size_mm: float,
    co2_threshold: float,
    cross_validation: dict[str, dict],
) -> None:
    """Write a trained phase model as a folder :func:`read_phase_model` reads.

    Each head records the texture scale in mm and the pixel size of the grid it
    was trained on. Beside the heads, ``phase_decision_policy.json`` holds each
    head's decision threshold and ``cross_validation.json`` the cross-validation.

    Parameters
    ----------
    folder : str or Path
        New model folder.
    model : PhaseModel
        The two heads, their texture scale and the gas threshold.
    pixel_size_mm : float
        Pixel size of the training grid.
    co2_threshold : float
        Decision threshold of the CO2 head, tuned in training; the gas head's is
        the model's.
    cross_validation : dict
        Cross-validation summary of each head, JSON serializable.

    """
    _head_path(folder, "co2_presence").parent.mkdir(parents=True)
    for role in ("co2_presence", "gas_within_co2"):
        head = getattr(model, role)
        payload = {
            "feature_names": list(PHASE_FEATURE_NAMES),
            "feature_mean": [float(value) for value in head.feature_mean],
            "feature_scale": [float(value) for value in head.feature_scale],
            "coefficients": [float(value) for value in head.coefficients],
            "intercept": float(head.intercept),
            "texture_sigma_mm": float(model.texture_sigma_mm),
            "training_pixel_size_mm": float(pixel_size_mm),
            "feature_contract": {"input_color_space": _LINEAR_RGB},
        }
        _head_path(folder, role).write_text(json.dumps(payload, indent=1), "utf-8")
    policy = {
        "schema": 1,
        "models": {
            "generic": {
                "co2_presence": {"presence_threshold": float(co2_threshold)},
                "gas_within_co2": {"presence_threshold": float(model.gas_threshold)},
            }
        },
    }
    for name, document in (
        ("phase_decision_policy.json", policy),
        ("cross_validation.json", cross_validation),
    ):
        (Path(folder) / name).write_text(json.dumps(document, indent=1), "utf-8")


def _log_ratios(rgb: np.ndarray, epsilon: np.floating) -> np.ndarray:
    """Logarithms of red/green, green/blue and red/blue, one column each."""
    safe = np.maximum(rgb, epsilon)
    return np.column_stack(
        [
            np.log(safe[..., i] / safe[..., j]).reshape(-1)
            for i, j in ((0, 1), (1, 2), (0, 2))
        ]
    )


class PhaseFeatures:
    """The optical features of one frame, from filters of the whole frame.

    The features compare the current image with the baseline (colour,
    chromaticity, log ratios, texture and its local statistics, bright grains)
    and with the previous frame; without a previous frame the dynamic features
    are zero. The filters run once on the whole frame; :meth:`matrix` gives the
    feature rows of any pixels, so a frame's chunks and a training sample of its
    pixels read the same values.

    The texture scale is converted to pixels of the images' grid. A pixel with a
    non-finite channel is invalid and enters the filters as zero. A pixel is
    supported if all given images are valid in the square window of half-width
    four texture sigmas (rounded) around it. All arithmetic runs in the floating
    dtype of the given images.

    Parameters
    ----------
    current, baseline : array
        Linear RGB images of shape (rows, cols, 3) on one grid.
    previous : array, optional
        Linear RGB image of the run's previous frame on the same grid.
    texture_sigma_mm : float
        Standard deviation (mm) of the Gaussian defining the image texture.
    pixel_size_mm : float
        Pixel size of the images' grid.

    Attributes
    ----------
    dtype : type
        Floating dtype of the features.
    support : array
        Boolean mask of the supported pixels.

    """

    def __init__(
        self,
        current: np.ndarray,
        baseline: np.ndarray,
        previous: np.ndarray | None = None,
        *,
        texture_sigma_mm: float,
        pixel_size_mm: float,
    ) -> None:
        given = [i for i in (current, baseline, previous) if i is not None]
        dtype = np.result_type(*given).type
        if len({image.shape for image in given}) != 1:
            raise ValueError("The images of a frame differ in shape.")
        valid = np.ones(current.shape[:2], dtype=bool)
        images = []
        for image in given:
            finite = np.isfinite(image).all(axis=-1)
            valid &= finite
            images.append(
                np.where(finite[..., None], image, 0).astype(dtype, copy=False)
            )
        current, baseline = images[:2]

        sigma = texture_sigma_mm / pixel_size_mm
        spatial = dtype(max(sigma, _SPATIAL_CONTEXT_FRACTION * min(current.shape[:2])))
        gray = [np.mean(image, axis=2, dtype=dtype) for image in images]
        self._smooth = [darsia.gaussian_blur(g, sigma) for g in gray]
        self._contrast = [g - s for g, s in zip(gray, self._smooth)]
        self._delta = current - baseline
        self._norm = np.linalg.norm(self._delta, axis=-1)
        self._local_mean = darsia.gaussian_blur(self._norm, spatial)
        local_squared = darsia.gaussian_blur(np.square(self._norm), spatial)
        self._local_std = np.sqrt(
            np.maximum(local_squared - np.square(self._local_mean), 0)
        )
        self._dynamic = previous is not None
        if self._dynamic:
            self._dynamic_delta = current - images[2]
            self._dynamic_norm = np.linalg.norm(self._dynamic_delta, axis=-1)
            self._local_dynamic_mean = darsia.gaussian_blur(self._dynamic_norm, spatial)
        self._images = images
        self.dtype = dtype
        width = 2 * round(4 * sigma) + 1
        self.support = cv2.erode(
            valid.astype(np.uint8), np.ones((width, width), np.uint8)
        ).astype(bool)

    def matrix(self, selection) -> np.ndarray:
        """Feature matrix of the selected pixels.

        Parameters
        ----------
        selection : slice or tuple of arrays
            Pixels to read: a slice of image rows, or arrays of row and column
            indices.

        Returns
        -------
        array
            Feature matrix of shape (pixels, 36), the columns in the order of
            :data:`PHASE_FEATURE_NAMES`, pixels in row-major order of the
            selection.

        """
        dtype = self.dtype
        epsilon, floor = dtype(1e-4), dtype(0.05)
        rgb = [image[selection] for image in self._images]
        contrast = [c[selection] for c in self._contrast]
        smooth = [s[selection] for s in self._smooth[:2]]
        local_mean = self._local_mean[selection]
        local_std = self._local_std[selection]
        intensity = [np.mean(image, axis=-1, dtype=dtype) for image in rgb]
        fractions = [i / (i.sum(axis=-1, keepdims=True) + epsilon) for i in rgb[:2]]
        texture = [np.abs(c) for c in contrast]
        texture_delta = texture[0] - texture[1]
        bright = [np.maximum(c, 0) for c in contrast[:2]]
        rel_current, rel_baseline = (
            np.clip(c / np.maximum(s, floor), -1.0, 4.0)
            for c, s in zip(contrast[:2], smooth)
        )
        rel_bright = [np.maximum(rel_current, 0), np.maximum(rel_baseline, 0)]
        persistent = np.minimum(rel_bright[1], rel_bright[0])
        columns = [
            self._delta[selection],
            intensity[0] - intensity[1],
            fractions[0] - fractions[1],
            _log_ratios(rgb[0], epsilon) - _log_ratios(rgb[1], epsilon),
            texture_delta,
            np.abs(texture_delta),
            self._norm[selection],
            local_mean,
            local_std,
            local_mean / (local_std + epsilon),
            intensity[1],
            contrast[1],
            contrast[0],
            contrast[0] - contrast[1],
            np.minimum(bright[1], bright[0]),
            np.maximum(bright[0] - bright[1], 0),
            rel_baseline,
            rel_current,
            rel_current - rel_baseline,
            persistent,
            np.maximum(rel_bright[0] - rel_bright[1], 0),
            persistent
            * persistent
            / (np.maximum(rel_bright[1], rel_bright[0]) + floor),
        ]
        count = intensity[0].size
        if self._dynamic:
            columns += [
                np.ones(count, dtype),
                self._dynamic_delta[selection],
                intensity[0] - intensity[2],
                texture[0] - texture[2],
                self._dynamic_norm[selection],
                self._local_dynamic_mean[selection],
            ]
        else:
            columns.append(np.zeros((count, 8), dtype))
        return np.column_stack([c.reshape(count, -1) for c in columns])


def phase_probabilities(
    model: PhaseModel,
    current: np.ndarray,
    baseline: np.ndarray,
    previous: np.ndarray | None = None,
    *,
    pixel_size_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """CO2-presence and gas-within-CO2 probabilities of one frame.

    Both heads read the frame's :class:`PhaseFeatures` with the model's texture
    scale, built in chunks of rows. Outside the features' support both
    probabilities are NaN, so the support travels with the result.

    Parameters
    ----------
    model : PhaseModel
        The delivered model.
    current, baseline : array
        Linear RGB images of shape (rows, cols, 3) on one grid.
    previous : array, optional
        Linear RGB image of the run's previous frame on the same grid.
    pixel_size_mm : float
        Pixel size of the images' grid.

    Returns
    -------
    co2, gas : array
        Probabilities of shape (rows, cols) in the images' floating dtype, NaN
        outside the support.

    """
    features = PhaseFeatures(
        current,
        baseline,
        previous,
        texture_sigma_mm=model.texture_sigma_mm,
        pixel_size_mm=pixel_size_mm,
    )
    rows, cols = features.support.shape
    co2 = np.empty((rows, cols), features.dtype)
    gas = np.empty((rows, cols), features.dtype)
    for start in range(0, rows, _ROWS_PER_CHUNK):
        chunk = np.s_[start : start + _ROWS_PER_CHUNK]
        matrix = features.matrix(chunk)
        co2[chunk] = model.co2_presence.probability(matrix).reshape(-1, cols)
        gas[chunk] = model.gas_within_co2.probability(matrix).reshape(-1, cols)
    co2[~features.support] = np.nan
    gas[~features.support] = np.nan
    return co2, gas


def limit_probabilities(
    co2: np.ndarray,
    gas: np.ndarray,
    keep_co2: np.ndarray | None = None,
    keep_gas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A frame's phase probabilities limited by expert knowledge.

    Where no CO2 may be (outside ``keep_co2``) both probabilities are zero, so the
    phase rules see water there; where no free gas may be (outside ``keep_gas``)
    the gas-within-CO2 probability is zero, so CO2 there is dissolved. Outside the
    features' support the probabilities stay unknown (NaN). Without a limit the
    probabilities are returned as given.

    Parameters
    ----------
    co2, gas : array
        CO2-presence and gas-within-CO2 probabilities of shape (rows, cols)
        (:func:`phase_probabilities`).
    keep_co2, keep_gas : array, optional
        Where CO2, dissolved or as gas, and where free gas may be, of shape
        (rows, cols), e.g. an expert-knowledge adapter's masks of
        ``concentration_aq`` and ``saturation_g``; None limits nothing.

    Returns
    -------
    co2, gas : array
        The limited probabilities in their dtype, new arrays where limited.

    Raises
    ------
    ValueError
        If a limit is not of the probabilities' shape.

    """
    if keep_co2 is None and keep_gas is None:
        return co2, gas
    for keep in (keep_co2, keep_gas):
        if keep is not None and keep.shape != co2.shape:
            raise ValueError(
                f"A limit of shape {keep.shape} does not fit probabilities of "
                f"shape {co2.shape}."
            )
    no_co2 = np.zeros(co2.shape, bool) if keep_co2 is None else ~keep_co2
    no_gas = no_co2 if keep_gas is None else no_co2 | ~keep_gas
    co2 = np.where(no_co2 & ~np.isnan(co2), co2.dtype.type(0), co2)
    gas = np.where(no_gas & ~np.isnan(gas), gas.dtype.type(0), gas)
    return co2, gas
