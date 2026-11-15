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

A trained model also records the canvas it was trained on (``canvas.json``, the
rig's description of its corrected images). The canvas is checked once, when a
run's model is read for the canvas of its images. Another resolution of the
canvas passes (the grid's shape and pixel size, with the same extent), since the
model's sizes are in mm; a model of another canvas is read only by the caller's
explicit choice, as is a delivered model, which records none.

"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import darsia
from darsia.multiphase.label_set import check_canvas

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

_PIXELS_PER_BLOCK = 4096
"""Pixels whose features are computed together, few enough for the processor's
cache."""


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
    canvas: dict | None
    """Canvas of the images the model was trained on (``Rig.canvas``); None for a
    folder that records none, e.g. a delivered one."""


def _head_path(folder: str | Path, role: str) -> Path:
    """Path of one head in a model folder."""
    return Path(folder) / "models" / "generic" / f"{role}_signal.json"


def read_phase_model(
    folder: str | Path,
    pixel_size_mm: float | None = None,
    *,
    canvas: dict | None,
    accept_canvas: bool = False,
) -> PhaseModel:
    """Read the two heads of a phase model folder for images of a canvas.

    A trained folder (:func:`write_phase_model`) gives the texture scale in mm.
    A delivered folder gives it in pixels of the grid the model was trained on,
    whose pixel size must then be given. The gas threshold is the gas head's
    presence threshold in the folder's ``phase_decision_policy.json``, and the
    model's canvas is that of ``canvas.json``, unknown (None) without one. The
    model's canvas is checked once, here, against the canvas of the images it is
    read for (:func:`~darsia.check_canvas`); another resolution of it passes.

    Parameters
    ----------
    folder : str or Path
        Model folder holding the heads in ``models/generic``.
    pixel_size_mm : float, optional
        Pixel size of the training grid of a delivered folder.
    canvas : dict or None
        Canvas of the images the model is read for (``Rig.canvas``).
    accept_canvas : bool
        Whether to read a model of another canvas, or one recording none, by the
        caller's explicit choice.

    Returns
    -------
    PhaseModel
        The two heads, their texture scale, the gas threshold and the canvas.

    Raises
    ------
    ValueError
        If a head uses other features than :data:`PHASE_FEATURE_NAMES`, was
        trained on other than linear RGB, the heads use different texture
        scales, or a delivered folder is read without the pixel size. If the
        model's canvas differs from the images' or is not recorded, without
        ``accept_canvas``; the message names the differing settings.

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
    record = Path(folder) / "canvas.json"
    recorded = None
    if record.exists():
        recorded = json.loads(record.read_text(encoding="utf-8"))["canvas"]
    check_canvas(
        recorded, canvas, accept=accept_canvas, subject="the model", resolution=True
    )
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
        canvas=recorded,
    )


def write_phase_model(
    folder: str | Path,
    model: PhaseModel,
    *,
    pixel_size_mm: float,
    co2_threshold: float,
    cross_validation: dict[str, dict],
    accepted_canvases: tuple[dict | None, ...],
    moved_from: tuple[dict, ...],
) -> None:
    """Write a trained phase model as a folder :func:`read_phase_model` reads.

    Each head records the texture scale in mm and the pixel size of the grid it
    was trained on. Beside the heads, ``phase_decision_policy.json`` holds each
    head's decision threshold, ``cross_validation.json`` the cross-validation
    and ``canvas.json`` the model's canvas, with the canvases of label sets the
    training accepted explicitly and those its label sets were moved from.

    Parameters
    ----------
    folder : str or Path
        New model folder.
    model : PhaseModel
        The two heads, their texture scale, the gas threshold and the canvas.
    pixel_size_mm : float
        Pixel size of the training grid.
    co2_threshold : float
        Decision threshold of the CO2 head, tuned in training; the gas head's is
        the model's.
    cross_validation : dict
        Cross-validation summary of each head, JSON serializable.
    accepted_canvases : tuple
        Canvases of the label sets accepted although they differ
        (``TrainingSamples.accepted_canvases``).
    moved_from : tuple
        Canvases the label sets were moved from (``TrainingSamples.moved_from``).

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
    canvases = {
        "canvas": model.canvas,
        "accepted_label_canvases": list(accepted_canvases),
        "labels_moved_from": list(moved_from),
    }
    for name, document in (
        ("phase_decision_policy.json", policy),
        ("cross_validation.json", cross_validation),
        ("canvas.json", canvases),
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


def phase_feature_sizes(
    shape: tuple[int, ...], texture_sigma_mm: float, pixel_size_mm: float, dtype: type
) -> tuple[float, np.floating, int]:
    """The sizes of a frame's features in pixels of its grid.

    Parameters
    ----------
    shape : tuple of int
        Shape of the images, rows and columns first.
    texture_sigma_mm : float
        Standard deviation (mm) of the Gaussian defining the image texture.
    pixel_size_mm : float
        Pixel size of the images' grid.
    dtype : type
        Floating dtype of the features.

    Returns
    -------
    sigma : float
        The texture's standard deviation in pixels.
    spatial : floating
        The standard deviation of the local change statistics in pixels: a fraction
        of the shorter side, at least the texture's, in the features' dtype.
    width : int
        Side of the support window: twice four texture sigmas, rounded, plus one.

    """
    sigma = texture_sigma_mm / pixel_size_mm
    spatial = dtype(max(sigma, _SPATIAL_CONTEXT_FRACTION * min(shape[:2])))
    return sigma, spatial, 2 * round(4 * sigma) + 1


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

        sigma, spatial, width = phase_feature_sizes(
            current.shape, texture_sigma_mm, pixel_size_mm, dtype
        )
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
        self.support = cv2.erode(
            valid.astype(np.uint8), np.ones((width, width), np.uint8)
        ).astype(bool)

    def matrix(self, selection, out: np.ndarray | None = None) -> np.ndarray:
        """Feature matrix of the selected pixels.

        Each feature depends on its pixel alone, so the features are computed in
        blocks of pixels small enough for the processor's cache, and each block
        is written into its rows of the matrix.

        Parameters
        ----------
        selection : slice or tuple of arrays
            Pixels to read: a slice of image rows, or arrays of row and column
            indices.
        out : array, optional
            Matrix of shape (pixels, 36) in the features' dtype and C order to write
            into, e.g. one matrix for all chunks of a frame.

        Returns
        -------
        array
            Feature matrix of shape (pixels, 36), the columns in the order of
            :data:`PHASE_FEATURE_NAMES`, pixels in row-major order of the
            selection.

        Raises
        ------
        ValueError
            If ``out`` has another shape, dtype or memory order.

        """
        fields = [
            self._images,
            self._contrast,
            self._smooth[:2],
            [self._delta, self._norm, self._local_mean, self._local_std],
        ]
        if self._dynamic:
            fields.append(
                [self._dynamic_delta, self._dynamic_norm, self._local_dynamic_mean]
            )
        count = self._norm[selection].size
        pixels = [
            [field[selection].reshape(count, *field.shape[2:]) for field in group]
            for group in fields
        ]
        shape = (count, len(PHASE_FEATURE_NAMES))
        if out is None:
            out = np.empty(shape, self.dtype)
        elif (
            out.shape != shape or out.dtype != self.dtype or not out.flags.c_contiguous
        ):
            raise ValueError(
                f"The feature matrix must be a C-ordered {shape} array of "
                f"{np.dtype(self.dtype).name}."
            )
        for start in range(0, count, _PIXELS_PER_BLOCK):
            block = slice(start, start + _PIXELS_PER_BLOCK)
            self._block(out[block], *([f[block] for f in group] for group in pixels))
        return out

    def _block(self, out, rgb, contrast, smooth, change, dynamic=None) -> None:
        """Write the features of one block of pixels into its rows of the matrix."""
        dtype = self.dtype
        epsilon, floor = dtype(1e-4), dtype(0.05)
        delta, norm, local_mean, local_std = change
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
            delta,
            intensity[0] - intensity[1],
            fractions[0] - fractions[1],
            _log_ratios(rgb[0], epsilon) - _log_ratios(rgb[1], epsilon),
            texture_delta,
            np.abs(texture_delta),
            norm,
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
        count = len(out)
        if dynamic:
            dynamic_delta, dynamic_norm, local_dynamic_mean = dynamic
            columns += [
                np.ones(count, dtype),
                dynamic_delta,
                intensity[0] - intensity[2],
                texture[0] - texture[2],
                dynamic_norm,
                local_dynamic_mean,
            ]
        else:
            columns.append(np.zeros((count, 8), dtype))
        start = 0
        for column in columns:
            values = column.reshape(count, -1)
            out[:, start : start + values.shape[1]] = values
            start += values.shape[1]


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
    scale, on the process's compute backend (:func:`darsia.compute_backend`).
    Outside the features' support both probabilities are NaN, so the support
    travels with the result.

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
    return darsia.compute_backend().phase_probabilities(
        current,
        baseline,
        previous,
        texture_sigma_mm=model.texture_sigma_mm,
        pixel_size_mm=pixel_size_mm,
        heads=(model.co2_presence, model.gas_within_co2),
    )
