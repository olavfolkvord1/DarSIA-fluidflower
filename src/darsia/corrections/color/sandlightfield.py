"""A light field shared by the sands of a photograph, with one level per sand.

Sand of one material is the same sand everywhere in a rig, so after its own level
the smooth variation of its brightness across the photograph is the light. The
field is fitted on the logarithm of the green channel of a baseline photograph,
reduced to a coarse working grid: blocks of working cells that lie wholly in one
material inside a mask are reduced to their median, which removes the grain
texture, and a polynomial without constant term plus one level per material is
fitted to them by least squares with Huber weights. Dividing the field out keeps
the light of one chosen point of the photograph, e.g. the colour checker's, so
that the photograph's colours are those the sands would have there.

"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

_MAD_TO_SIGMA = 1.4826
"""Robust standard deviation of a normal distribution per median absolute
deviation."""
_TILE_ROWS = 256
"""Rows of a photograph flattened at once."""


@dataclass(frozen=True)
class SandLightFieldSettings:
    """Settings of the light field shared by the sands of a photograph."""

    cell_pixels: int
    """Side length of a cell of the working grid, in pixels of the photograph:
    the grid has its rows and columns divided by it, rounded down."""
    degree: int
    """Degree of the field's polynomial in the image coordinates."""
    block_cells: int
    """Side length of a block of working cells reduced to one sample."""
    minimum_blocks: int
    """Fewest blocks a material must have to take part in the fit."""
    iterations: int
    """Number of weighted least-squares solves."""
    huber: float
    """Residuals beyond this many robust standard deviations are down-weighted."""
    samples_per_unknown: int
    """Fewest blocks per unknown of the fit (the levels and the polynomial's
    terms)."""


def polynomial_terms(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Monomials ``x**i * y**(d - i)`` for ``d = 1, ..., degree`` and
    ``i = 0, ..., d``, in this order, as the last axis.

    Parameters
    ----------
    x, y : array
        Normalized image coordinates.
    degree : int
        Highest degree.

    Returns
    -------
    array
        The terms, in double precision.

    """
    return np.stack(
        [
            np.asarray(x, float) ** i * np.asarray(y, float) ** (d - i)
            for d in range(1, int(degree) + 1)
            for i in range(d + 1)
        ],
        axis=-1,
    )


@dataclass(frozen=True)
class SandLightField:
    """The fitted light field of a photograph, the log of its light without
    constant term, and the log level of each sand."""

    coefficients: np.ndarray
    """Coefficients of :func:`polynomial_terms` of the normalized coordinates of
    the working grid."""
    levels: dict[int, float]
    """Log level of the green channel of each fitted material."""
    degree: int
    """Degree of the polynomial."""
    grid_shape: tuple[int, int]
    """Rows and columns of the working grid."""
    image_shape: tuple[int, int]
    """Rows and columns of the photograph."""
    samples: int
    """Number of blocks fitted."""
    spread: dict[int, tuple[float, float]]
    """Interquartile range of each material's log green blocks about its level,
    before and after the field is divided out."""

    def coordinates(
        self, columns: np.ndarray, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalized coordinates of positions of the photograph's pixels: their
        positions on the working grid (pixel centres under an area resize), with
        the grid's centre at 0 and its edges at -1 and 1.

        Parameters
        ----------
        columns, rows : array
            Positions in pixels of the photograph.

        Returns
        -------
        tuple of array
            x and y.

        """
        (gh, gw), (nh, nw) = self.grid_shape, self.image_shape
        x = (((np.asarray(columns, float) + 0.5) * gw / nw - 0.5) - gw / 2) / (gw / 2)
        y = (((np.asarray(rows, float) + 0.5) * gh / nh - 0.5) - gh / 2) / (gh / 2)
        return x, y

    def flatten(self, image: np.ndarray, anchor: Sequence[float]) -> np.ndarray:
        """The photograph with the light field divided out, keeping the light at
        a point: each pixel times the light at the point over the light at the
        pixel. The field is evaluated in double precision and applied in the
        image's floating precision.

        Parameters
        ----------
        image : array
            The photograph (rows, cols, channels) on the grid of the fit.
        anchor : sequence of float
            Column and row, in pixels of the photograph, of the point whose light
            is kept, e.g. the centre of the colour checker.

        Returns
        -------
        array
            The flattened photograph.

        Raises
        ------
        ValueError
            If the image is not on the grid of the fit.

        """
        image = np.asarray(image)
        if image.shape[:2] != tuple(self.image_shape):
            raise ValueError("The light field was fitted on another grid.")
        coefficients = [float(c) for c in self.coefficients]
        x, y = self.coordinates(*np.asarray(anchor, float))
        kept = float(polynomial_terms(x, y, self.degree) @ self.coefficients)
        flattened = np.empty_like(image)
        # Rows in tiles, to bound the memory of the field's terms.
        for first in range(0, image.shape[0], _TILE_ROWS):
            stop = min(first + _TILE_ROWS, image.shape[0])
            rows, columns = np.mgrid[first:stop, 0 : image.shape[1]]
            x, y = self.coordinates(columns, rows)
            field = np.full(x.shape, kept)
            index = 0
            for d in range(1, self.degree + 1):
                for i in range(d + 1):
                    field += -coefficients[index] * x**i * y ** (d - i)
                    index += 1
            flattened[first:stop] = (
                image[first:stop] * np.exp(field).astype(image.dtype)[..., None]
            )
        return flattened


def _blocks(values: np.ndarray, block: int) -> np.ndarray:
    """The whole blocks of ``block`` x ``block`` cells of a grid, as
    (block rows, block columns, cells)."""
    h, w = (values.shape[0] // block) * block, (values.shape[1] // block) * block
    return (
        values[:h, :w]
        .reshape(h // block, block, w // block, block)
        .swapaxes(1, 2)
        .reshape(h // block, w // block, block * block)
    )


def fit_sand_light_field(
    baseline: np.ndarray,
    material_map: np.ndarray,
    mask: np.ndarray,
    settings: SandLightFieldSettings,
) -> SandLightField:
    """Fit the light field shared by the sands of a baseline photograph.

    The baseline is area-reduced to the working grid. A working cell belongs to a
    material if it is fully covered by the material's pixels inside the mask, and
    is usable if its green value lies strictly between 0 and 1. Blocks of usable
    cells of one material give the median of their log green. Materials with
    fewer blocks than the minimum are left out. The log green of a block is its
    material's level plus the field; the fit is iterated with Huber weights of the
    residuals over their robust standard deviation.

    Parameters
    ----------
    baseline : array
        Linear RGB baseline (rows, cols, 3) in units of the sensor's white level.
    material_map : array
        Integer material of each pixel (rows, cols).
    mask : array
        Boolean pixels of sand that may be fitted (rows, cols), e.g. the sand
        outside a band along the rig's frame.
    settings : SandLightFieldSettings
        Settings of the fit.

    Returns
    -------
    SandLightField
        The fitted field.

    Raises
    ------
    ValueError
        If the inputs are not of one grid, or too few blocks remain for the fit.

    """
    baseline = np.asarray(baseline)
    material_map, mask = np.asarray(material_map), np.asarray(mask, bool)
    image_shape = baseline.shape[:2]
    if material_map.shape != image_shape or mask.shape != image_shape:
        raise ValueError("The baseline, its materials and mask need one grid.")
    cell, degree = int(settings.cell_pixels), int(settings.degree)
    grid_shape = tuple(max(1, int(n) // cell) for n in image_shape)
    size = (grid_shape[1], grid_shape[0])
    cells = np.full(grid_shape, -1, np.int64)
    for material in np.unique(material_map[mask]):
        coverage = cv2.resize(
            ((material_map == material) & mask).astype(np.float32),
            size,
            interpolation=cv2.INTER_AREA,
        )
        cells[coverage >= 0.999] = int(material)
    green = cv2.resize(
        np.asarray(baseline, np.float32), size, interpolation=cv2.INTER_AREA
    )[..., 1].astype(np.float64)
    usable = (cells >= 0) & np.isfinite(green) & (green > 0) & (green < 1)
    log_green = np.full(grid_shape, np.nan)
    log_green[usable] = np.log(green[usable])
    cells = np.where(usable, cells, -1)

    block = int(settings.block_cells)
    ids, values = _blocks(cells, block), _blocks(log_green, block)
    whole = (
        (ids >= 0).all(axis=2)
        & (ids == ids[..., :1]).all(axis=2)
        & np.isfinite(values).all(axis=2)
    )
    rows, cols = np.nonzero(whole)
    sample_materials = ids[rows, cols, 0]
    target = np.median(values[rows, cols], axis=1)
    present = [
        int(material)
        for material in np.unique(sample_materials)
        if np.count_nonzero(sample_materials == material) >= settings.minimum_blocks
    ]
    keep = np.isin(sample_materials, present)
    gh, gw = grid_shape
    terms = polynomial_terms(
        ((cols + 0.5) * block - 0.5 - gw / 2) / (gw / 2),
        ((rows + 0.5) * block - 0.5 - gh / 2) / (gh / 2),
        degree,
    )[keep]
    unknowns = len(present) + terms.shape[1]
    if not present or np.count_nonzero(keep) < settings.samples_per_unknown * unknowns:
        raise ValueError("Too few blocks of one sand to fit a light field.")
    levels = np.stack(
        [sample_materials[keep] == material for material in present], axis=-1
    ).astype(float)
    design, target = np.concatenate([levels, terms], axis=1), target[keep]
    weights = np.ones(len(target))
    for _ in range(int(settings.iterations)):
        root = np.sqrt(weights)
        solution = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)[0]
        residual = target - design @ solution
        scale = _MAD_TO_SIGMA * float(np.median(np.abs(residual - np.median(residual))))
        limit = settings.huber * max(scale, 1e-9)
        weights = np.where(
            np.abs(residual) <= limit,
            1.0,
            limit / np.maximum(np.abs(residual), 1e-300),
        )
    coefficients = solution[len(present) :]
    remaining = target - levels @ solution[: len(present)]
    flattened = remaining - terms @ coefficients
    spread = {}
    for material in present:
        mine = sample_materials[keep] == material
        spread[material] = tuple(
            float(np.subtract(*np.percentile(part[mine], [75, 25])))
            for part in (remaining, flattened)
        )
    return SandLightField(
        coefficients=coefficients,
        levels={m: float(level) for m, level in zip(present, solution[: len(present)])},
        degree=degree,
        grid_shape=grid_shape,
        image_shape=tuple(image_shape),
        samples=int(len(target)),
        spread=spread,
    )
