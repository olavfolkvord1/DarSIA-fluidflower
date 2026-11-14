"""Material illumination correction: smooth positive RGB gains per material.

The correction multiplies linear RGB by a spatially smooth gain. The gain is the
exponential of a quadratic polynomial in the image coordinates, one per material
(e.g., per sand type), and one common polynomial used for all other pixels, e.g.,
for water. The polynomials are fitted such that samples of each material match a
constant target colour in the logarithmic domain. The model thus separates the
spatial variation of the illumination from the colour of the materials.

"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg

import darsia


@dataclass(frozen=True)
class MaterialFitSettings:
    """Settings of the material illumination fit: the block samples of the
    baseline (:func:`darsia.fit_material_illumination`) and the robust fit of
    the gains; part of the correction's configuration table."""

    spatial_ridge: float
    """Ridge penalty of the spatial terms of the common field."""
    offset_ridge: float
    """Ridge penalty of the material offsets of the common field."""
    residual_constant_ridge: float
    """Ridge penalty of the constant term of each material's residual field."""
    residual_spatial_ridge: float
    """Ridge penalty of the spatial terms of each material's residual field."""
    iterations: int
    """Number of robust solves per channel and stage."""
    huber_threshold: float
    """Residuals beyond this many robust standard deviations are down-weighted."""
    clip_level: float
    """Values at or above it, in units of the white level, are not fitted."""
    dark_level: float
    """Cells at or below this value in a channel, in target units, are not fitted."""
    minimum_block_cells: int
    """Fewest usable cells of one material for which a block gives a median."""
    chroma_cutoff: float
    """Smallest chromaticity deviation of a colour outlier block."""
    chroma_sigmas: float
    """Chromaticity deviation of a colour outlier block, in robust standard
    deviations."""


def _polynomial(positions: np.ndarray) -> np.ndarray:
    """Quadratic monomials [1, x, y, x², xy, y²] of normalized positions (N, 2)."""
    x, y = positions[:, 0], positions[:, 1]
    return np.column_stack((np.ones_like(x), x, y, x * x, x * y, y * y))


def _pixel_resize(source_shape: tuple, target_shape: tuple) -> np.ndarray:
    """Homography of pixel centres under an area resize between two shapes."""
    sy, sx = np.asarray(target_shape, dtype=np.float64) / np.asarray(
        source_shape, dtype=np.float64
    )
    return np.asarray(
        [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=np.float64
    )


def fit_material_gain(
    positions: np.ndarray,
    materials_of_samples: np.ndarray,
    measured_rgb: np.ndarray,
    target_log_rgb: np.ndarray,
    *,
    fit_mask: np.ndarray,
    materials: list[int],
    settings: MaterialFitSettings,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Fit a common quadratic log-gain field and one field per material.

    The first stage fits, for all samples together, one quadratic log-gain field
    plus one constant offset per material; the offsets sum to zero. Each sample
    has the weight one over the sample count of its material. The second stage
    fits, per material, a quadratic residual around the common field plus its
    offset. Both stages use the robust ridge regression of
    :func:`darsia.linalg.robust_ridge`. The fit computes in the floating dtype of
    the positions, at least single precision; the rank and condition of the
    common design come from its singular values in that dtype.

    Parameters
    ----------
    positions : array
        Normalized sample positions (N, 2), x = (column + 0.5) * 2 / width - 1 and
        y likewise with rows.
    materials_of_samples : array
        Integer material of each sample (N,).
    measured_rgb : array
        Positive measured linear RGB of each sample (N, 3).
    target_log_rgb : array
        Natural logarithm of the target RGB of each sample (N, 3).
    fit_mask : array
        Boolean selection of the samples used in the fit (N,).
    materials : list of int
        Fitted materials, in order.
    settings : MaterialFitSettings
        Ridge penalties, iterations and Huber threshold of the robust fit.

    Returns
    -------
    common : array
        Coefficients (6, 3) of the common log-gain field.
    fields : dict
        Coefficients (6, 3) of the log-gain field of each material.

    Raises
    ------
    ValueError
        If a material has fewer than eight samples, or the common field is not
        identifiable (fewer than 32 samples, rank deficient, or a condition
        number above 1000).

    """
    ids = [int(material) for material in materials]
    dtype = np.result_type(positions, np.float32)
    positions = np.asarray(positions[fit_mask], dtype=dtype)
    labels = np.asarray(materials_of_samples)[fit_mask]
    measured = np.asarray(measured_rgb)[fit_mask]
    targets = np.asarray(target_log_rgb)[fit_mask]
    if any(np.count_nonzero(labels == material) < 8 for material in ids):
        raise ValueError("At least eight samples per material are required.")

    # Design of the common stage: quadratic field and sum-zero material offsets.
    polynomial = _polynomial(positions)
    effects = np.zeros((len(labels), len(ids) - 1), dtype=dtype)
    for index, material in enumerate(ids[:-1]):
        effects[labels == material, index] = 1
    effects[labels == ids[-1]] = -1
    design = np.column_stack((polynomial, effects))
    singular = scipy.linalg.svdvals(design, check_finite=False)
    rank = np.count_nonzero(
        singular > singular[0] * max(design.shape) * np.finfo(dtype).eps
    )
    if len(design) < 32 or rank != design.shape[1] or singular[0] / singular[-1] > 1000:
        raise ValueError("The common illumination field is not identifiable.")

    # Log ratios in the samples' dtype, fitted in the fit's dtype.
    response = np.asarray(targets - np.log(measured), dtype=dtype)
    base = np.reciprocal(
        np.array(
            [np.count_nonzero(labels == material) for material in labels],
            dtype=dtype,
        )
    )
    base /= base.sum()
    penalty = np.zeros((design.shape[1], design.shape[1]), dtype=dtype)
    penalty[1:6, 1:6] = np.eye(5, dtype=dtype) * settings.spatial_ridge
    penalty[6:, 6:] = settings.offset_ridge * (
        np.eye(len(ids) - 1, dtype=dtype)
        + np.ones((len(ids) - 1, len(ids) - 1), dtype=dtype)
    )
    shared = darsia.linalg.robust_ridge(
        design, response, base, penalty, settings.iterations, settings.huber_threshold
    )
    offsets = np.vstack((shared[6:], -np.sum(shared[6:], axis=0)))

    # Residual field per material around the common field and its offset.
    residual_penalty = np.diag(
        np.array(
            [
                settings.residual_constant_ridge,
                *([settings.residual_spatial_ridge] * 5),
            ],
            dtype=dtype,
        )
    )
    fields = {}
    for index, material in enumerate(ids):
        keep = labels == material
        z = polynomial[keep]
        prior = shared[:6].copy()
        prior[0] += offsets[index]
        delta = darsia.linalg.robust_ridge(
            z,
            response[keep] - z @ prior,
            np.full(len(z), 1 / dtype.type(len(z)), dtype=dtype),
            residual_penalty,
            settings.iterations,
            settings.huber_threshold,
        )
        fields[material] = prior + delta
    return shared[:6].copy(), fields


class MaterialIlluminationCorrection(darsia.BaseCorrection):
    """Multiply linear RGB by a smooth positive gain per material.

    At each pixel, the gain is ``exp(P @ B)`` with ``P = [1, x, y, x², xy, y²]``
    the normalized position of the pixel on the fitting grid and ``B`` the
    coefficients of the pixel's material, or the common coefficients if the
    pixel's material has no field of its own (e.g., water). Pixel centres are
    mapped to the fitting grid as by an area resize. Pixels with a non-finite or
    negative channel become NaN; there is no clipping and no additive term. The
    gain is evaluated in the image's floating dtype.

    """

    def __init__(
        self,
        common_coefficients: np.ndarray | None = None,
        material_coefficients: dict[int, np.ndarray] | None = None,
        material_map: np.ndarray | None = None,
        field_shape: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the correction, or an empty object to be loaded from file.

        Parameters
        ----------
        common_coefficients : array, optional
            Coefficients (6, 3) of the common log-gain field.
        material_coefficients : dict, optional
            Coefficients (6, 3) of the log-gain field of each material.
        material_map : array, optional
            Integer material of each pixel of the images to be corrected.
        field_shape : tuple of int, optional
            Rows and columns of the fitting grid.

        """
        if common_coefficients is None:
            return
        self.common_coefficients = np.asarray(common_coefficients)
        """Coefficients of the common log-gain field."""
        self.material_coefficients = {
            int(material): np.asarray(coefficients)
            for material, coefficients in material_coefficients.items()
        }
        """Coefficients of the log-gain field of each material."""
        self.material_map = np.asarray(material_map)
        """Material of each pixel of the images to be corrected."""
        self.field_shape = tuple(int(n) for n in field_shape)
        """Rows and columns of the fitting grid."""
        self._grid_to_field = _pixel_resize(self.material_map.shape, self.field_shape)
        self._gain = None

    def _gain_rows(self, first: int, last: int, dtype: np.dtype) -> np.ndarray:
        """Gain (rows, cols, 3) of one block of rows."""
        width = self.material_map.shape[1]
        row, column = np.meshgrid(
            np.arange(first, last, dtype=dtype),
            np.arange(width, dtype=dtype),
            indexing="ij",
        )
        points = (
            np.stack((column, row, np.ones_like(column)), axis=-1)
            @ self._grid_to_field.astype(dtype).T
        )
        field_rows, field_cols = self.field_shape
        x = (points[..., 0] / points[..., 2] + 0.5) * 2.0 / field_cols - 1.0
        y = (points[..., 1] / points[..., 2] + 0.5) * 2.0 / field_rows - 1.0
        p = np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=-1)
        materials = self.material_map[first:last, :, None]
        gain = np.exp(p @ self.common_coefficients.astype(dtype))
        for material, coefficients in self.material_coefficients.items():
            gain = np.where(
                materials == material, np.exp(p @ coefficients.astype(dtype)), gain
            )
        return gain

    def gain(self, dtype: np.dtype) -> np.ndarray:
        """Gain field (rows, cols, 3), evaluated once per dtype and kept.

        Parameters
        ----------
        dtype : dtype
            Floating dtype of the evaluation.

        Returns
        -------
        array
            Gain of each pixel and channel.

        """
        if self._gain is None or self._gain.dtype != dtype:
            rows = self.material_map.shape[0]
            self._gain = np.empty((*self.material_map.shape, 3), dtype)
            for first in range(0, rows, 64):
                self._gain[first : first + 64] = self._gain_rows(
                    first, min(first + 64, rows), dtype
                )
        return self._gain

    def correct_array(self, image: np.ndarray) -> np.ndarray:
        """Multiply a linear RGB image by the gain.

        Parameters
        ----------
        image : array
            Linear RGB image (rows, cols, 3) on the grid of the material map.

        Returns
        -------
        array
            Corrected image in the image's floating dtype; NaN where a channel is
            not finite or negative.

        """
        values = np.asarray(image)
        dtype = np.result_type(values, np.float32)
        valid = np.isfinite(values).all(axis=-1) & (values >= 0).all(axis=-1)
        result = np.full(values.shape, np.nan, dtype)
        result[valid] = values[valid] * self.gain(dtype)[valid]
        return result

    # ! ---- I/O ----

    def save(self, path: Path) -> None:
        """Save the correction to a npz file.

        Parameters
        ----------
        path : Path
            Path to the file.

        """
        materials = list(self.material_coefficients)
        np.savez_compressed(
            path,
            class_name=type(self).__name__,
            common_coefficients=self.common_coefficients,
            materials=np.array(materials),
            material_coefficients=np.array(
                [self.material_coefficients[material] for material in materials]
            ),
            material_map=self.material_map,
            field_shape=np.array(self.field_shape),
        )

    def load(self, path: Path) -> None:
        """Load the correction from a npz file.

        Parameters
        ----------
        path : Path
            Path to the file.

        """
        with np.load(path) as data:
            self.__init__(
                common_coefficients=data["common_coefficients"],
                material_coefficients=dict(
                    zip(data["materials"].tolist(), data["material_coefficients"])
                ),
                material_map=data["material_map"],
                field_shape=tuple(data["field_shape"]),
            )
