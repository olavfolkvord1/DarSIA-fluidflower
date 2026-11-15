"""Fit of the material illumination correction on block medians of a baseline.

The baseline image is reduced onto a coarse fitting grid. On that grid, the fit
takes the channel-wise median of each block of pure material, compares it with a
constant target colour of the material, excludes blocks of outlying chromaticity,
and fits the gain fields of :class:`MaterialIlluminationCorrection` on the
training folds of the blocks. Optionally, the fitting grid is a reference run's:
the baseline is placed on it through a homography, and only blocks pure in both
runs are fitted.

"""

from dataclasses import dataclass

import cv2
import numpy as np

from darsia.corrections.color.materialilluminationcorrection import (
    MaterialFitSettings,
    MaterialIlluminationCorrection,
    _pixel_resize,
    fit_material_gain,
)
from darsia.image.subregions import warp_perspective
from darsia.utils.linalg import solve

_TRAINING_FOLDS = (0, 3, 4)
"""Folds of the blocks used in the fit; the remaining folds are held out."""


@dataclass(frozen=True)
class MaterialFitReference:
    """The reference run of a material illumination fit: its fitting grid is the
    fitting grid of the fit, and a block is fitted only where the reference holds
    the block's material too."""

    baseline: np.ndarray
    """Linear RGB baseline (rows, cols, 3) of the reference, in units of the
    sensor's white level."""
    material_map: np.ndarray
    """Integer material of each pixel of the reference (rows, cols)."""
    fit_mask: np.ndarray
    """Boolean pixels of the reference eligible for the fit (rows, cols)."""
    dimensions: tuple[float, float]
    """Physical height and width of the reference's image."""
    homography: np.ndarray
    """Homography (3, 3) from the pixel centres (column, row, 1) of the fitted
    baseline to those of the reference's baseline, used in the fitted baseline's
    floating dtype."""


def _reduce_rgb(rgb: np.ndarray, size: tuple) -> np.ndarray:
    """Area reduction normalized by the finite support; partly finite cells are NaN."""
    valid = np.isfinite(rgb).all(-1)
    weights = cv2.resize(valid.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    means = cv2.resize(
        np.where(valid[..., None], rgb, 0), size, interpolation=cv2.INTER_AREA
    )
    means /= np.maximum(weights[..., None], np.float32(1e-6))
    means[weights < 0.999] = np.nan
    return means


def _warp(values: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Bilinear warp of a single precision array onto a grid of its own shape, in
    the precision of the homography (:func:`darsia.warp_perspective`): positions
    in single precision sampled by OpenCV's remap, or OpenCV's perspective warp in
    double precision. Positions outside give zero.
    """
    height, width = values.shape[:2]
    return warp_perspective(values, matrix, (width, height), matrix.dtype)


def _warp_rgb(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Warp of reduced RGB normalized by the warped finite support; cells partly
    supported are NaN."""
    valid = np.isfinite(rgb).all(-1)
    values = _warp(np.where(valid[..., None], rgb, 0), matrix)
    support = _warp(valid.astype(np.float32), matrix)
    values /= np.maximum(support[..., None], np.float32(1e-9))
    values[support < 0.999] = np.nan
    return values


def _strict(
    mask: np.ndarray, erode: bool, size: tuple, matrix: np.ndarray | None = None
) -> np.ndarray:
    """Cells of the fitting grid fully covered by a mask, optionally eroded by one.

    With a homography ``matrix``, the fully covered cells are warped and a cell
    must again be fully covered."""
    coverage = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    full = coverage >= 0.999 if erode else coverage > 0.999
    if matrix is not None:
        covered = _warp(full.astype(np.float32), matrix)
        full = covered >= 0.999 if erode else covered > 0.999
    if not erode:
        return full
    return cv2.erode(
        full.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def project_to_fit_grid(
    rgb: np.ndarray,
    material_map: np.ndarray,
    fit_mask: np.ndarray,
    fit_shape: tuple[int, int],
    materials: list[int],
    settings: MaterialFitSettings,
    grid_to_field: np.ndarray | None = None,
) -> dict:
    """Reduce a baseline and its masks onto the fitting grid.

    A pixel is usable if all its channels are finite and nonnegative. The image is
    area-reduced to the fitting grid, normalized by the usable support; cells
    partly covered by usable pixels are NaN. A cell is safe for fitting if it is
    fully covered by usable pixels below the clip level in all channels; it is of
    pure material if it is fully covered by eligible pixels of that material,
    after an erosion by one cell. With a homography to another fitting grid, e.g.,
    a reference run's, the reduced image and the fully covered cells are warped
    onto it (bilinear; the image normalized by its warped support) and a cell must
    be fully covered again, before the erosion.

    Parameters
    ----------
    rgb : array
        Linear RGB baseline (rows, cols, 3) in units of the sensor's white level.
    material_map : array
        Integer material of each pixel (rows, cols).
    fit_mask : array
        Boolean pixels eligible for the fit (rows, cols).
    fit_shape : tuple of int
        Rows and columns of the fitting grid.
    materials : list of int
        Fitted materials.
    settings : MaterialFitSettings
        Settings of the fit; the clip level.
    grid_to_field : array, optional
        Homography (3, 3) from the pixel centres (column, row, 1) of the baseline
        to those of another fitting grid of the same shape, used in the baseline's
        floating dtype (at least single precision). Defaults to the area resize
        onto the fitting grid.

    Returns
    -------
    dict
        Single precision ``rgb`` on the fitting grid, the boolean ``fit_safe`` and
        ``frame_safe`` cells, and the pure ``materials`` of the cells (-1 for
        none).

    """
    rgb = np.asarray(rgb)
    usable = np.isfinite(rgb).all(-1) & (rgb >= 0).all(-1)
    size = (int(fit_shape[1]), int(fit_shape[0]))
    matrix = None
    if grid_to_field is not None:
        # From the baseline's own fitting grid to the given one.
        dtype = np.result_type(rgb, np.float32)
        resize = _pixel_resize(rgb.shape[:2], fit_shape, dtype)
        matrix = np.asarray(grid_to_field, dtype) @ solve(
            resize, np.eye(3, dtype=dtype)
        )
        matrix /= matrix[2, 2]
    rgb = np.where(usable[..., None], rgb, np.nan).astype(np.float32, copy=False)
    frame_safe = _strict(fit_mask, True, size, matrix)
    pure = np.full(tuple(fit_shape), -1, np.int32)
    for material in materials:
        mask = fit_mask & (material_map == material)
        pure[_strict(mask, True, size, matrix) & frame_safe] = material
    reduced = _reduce_rgb(rgb, size)
    safe = usable & (rgb < settings.clip_level).all(-1)
    return {
        "rgb": reduced if matrix is None else _warp_rgb(reduced, matrix),
        "fit_safe": _strict(safe, False, size, matrix),
        "frame_safe": frame_safe,
        "materials": pure,
    }


def collect_block_samples(
    projection: dict,
    targets: dict[int, np.ndarray],
    *,
    dimensions: tuple[float, float],
    edge_margin: float,
    block_size: int,
    dn_scale: float,
    settings: MaterialFitSettings,
    dtype: np.dtype,
    reference: dict | None = None,
) -> dict:
    """Median colour of each block of pure material on the fitting grid.

    A cell enters a block median only if it is safe, its values exceed the dark
    level in all channels, it lies farther than ``edge_margin`` from the grid's
    edge, and it holds pure material; with a reference projection on the same
    grid, the cell must also be safe and of the same pure material in the
    reference. Blocks with too few such cells of a
    material are skipped. Each block is assigned to one of five folds by its
    position. Positions and margins are computed in ``dtype``, so the fit of
    :func:`fit_block_field` runs in it; the colours stay single precision.

    Parameters
    ----------
    projection : dict
        Projection of :func:`project_to_fit_grid`.
    targets : dict
        Natural logarithm of the target RGB of each material, in the units of
        ``dn_scale`` times the image values.
    dimensions : tuple of float
        Physical height and width of the image.
    edge_margin : float
        Physical margin along the edges of the image.
    block_size : int
        Block side length in cells of the fitting grid.
    dn_scale : float
        Factor from image units to the units of the targets, e.g., the sensor's
        white level in camera digital numbers.
    settings : MaterialFitSettings
        Settings of the fit; the dark level and the fewest cells of a block.
    dtype : dtype
        Floating dtype of the positions, e.g., the baseline's.
    reference : dict, optional
        Projection of a reference run's baseline onto the same fitting grid, see
        :func:`project_to_fit_grid`; its colours are not used.

    Returns
    -------
    dict
        Arrays over the blocks: normalized position ``x`` and ``y``, ``material``,
        ``fold``, measured median ``before`` and target RGB ``target``.

    """
    materials_map = projection["materials"]
    shape = materials_map.shape
    before = projection["rgb"] * np.float32(dn_scale)
    keep = projection["frame_safe"] & projection["fit_safe"]
    if reference is not None:
        keep &= reference["frame_safe"] & reference["fit_safe"]
        keep &= reference["materials"] == materials_map
    keep &= np.isfinite(before).all(-1)
    keep &= (before > settings.dark_level).all(-1)

    # Physical margin along the edges.
    height, width = shape
    dtype = np.dtype(dtype)
    yy, xx = np.indices(shape)
    y, x = yy.astype(dtype), xx.astype(dtype)
    physical_h, physical_w = (dtype.type(d) for d in dimensions)
    keep &= ((x + 0.5) * physical_w / width > edge_margin) & (
        (width - x - 0.5) * physical_w / width > edge_margin
    )
    keep &= ((y + 0.5) * physical_h / height > edge_margin) & (
        (height - y - 0.5) * physical_h / height > edge_margin
    )

    records = []
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            block = (
                slice(row, min(row + block_size, height)),
                slice(col, min(col + block_size, width)),
            )
            for material in sorted(targets):
                selected = keep[block] & (materials_map[block] == material)
                if int(selected.sum()) < settings.minimum_block_cells:
                    continue
                columns, rows = xx[block][selected], yy[block][selected]
                records.append(
                    (
                        (columns.astype(dtype).mean() + 0.5) * 2 / width - 1,
                        (rows.astype(dtype).mean() + 0.5) * 2 / height - 1,
                        material,
                        (3 * (row // block_size) + 7 * (col // block_size)) % 5,
                        np.median(
                            before.reshape(-1, 3)[rows * width + columns], axis=0
                        ),
                        np.exp(np.asarray(targets[material], dtype=before.dtype)),
                    )
                )
    if not records:
        raise ValueError("No block of pure material survives the masks.")
    names = ("x", "y", "material", "fold", "before", "target")
    samples = {
        name: np.asarray([record[i] for record in records])
        for i, name in enumerate(names)
    }
    samples["material"] = samples["material"].astype(np.int64)
    samples["fold"] = samples["fold"].astype(np.int64)
    return samples


def block_color_outliers(
    samples: dict, materials: list[int], settings: MaterialFitSettings
) -> np.ndarray:
    """Blocks whose chromaticity deviates from their material's training blocks.

    The chromaticity of a block is the red and blue log ratio to green of its
    measured over its target colour. Per material, the centre and the robust
    standard deviation are taken over the training folds; a block is an outlier
    if a component deviates by more than the larger of a fixed cutoff and a
    multiple of the robust standard deviation.

    Parameters
    ----------
    samples : dict
        Block samples of :func:`collect_block_samples`.
    materials : list of int
        Fitted materials.
    settings : MaterialFitSettings
        Settings of the fit; the outlier cutoffs.

    Returns
    -------
    array
        Boolean outlier flag of each block.

    Raises
    ------
    ValueError
        If a material has fewer than eight training blocks.

    """
    residual = np.log(samples["before"]) - np.log(samples["target"])
    chroma = residual[:, [0, 2]] - residual[:, 1:2]
    training = np.isin(samples["fold"], _TRAINING_FOLDS)
    outliers = np.zeros(len(chroma), dtype=bool)
    for material in materials:
        use = training & (samples["material"] == material)
        if int(use.sum()) < 8:
            raise ValueError(f"Too few training blocks for material {material}.")
        centre = np.median(chroma[use], axis=0)
        sigma = 1.4826 * np.median(np.abs(chroma[use] - centre), axis=0)
        cutoff = np.maximum(settings.chroma_cutoff, settings.chroma_sigmas * sigma)
        members = samples["material"] == material
        outliers[members] = np.any(np.abs(chroma[members] - centre) > cutoff, axis=1)
    return outliers


def fit_block_field(
    samples: dict, materials: list[int], settings: MaterialFitSettings
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Fit the gain fields on the training blocks without colour outliers.

    Parameters
    ----------
    samples : dict
        Block samples of :func:`collect_block_samples`.
    materials : list of int
        Fitted materials.
    settings : MaterialFitSettings
        Settings of the robust fit.

    Returns
    -------
    tuple
        Common coefficients and coefficients per material, see
        :func:`fit_material_gain`.

    """
    selected = np.isin(samples["fold"], _TRAINING_FOLDS) & ~block_color_outliers(
        samples, materials, settings
    )
    return fit_material_gain(
        np.column_stack((samples["x"], samples["y"])),
        samples["material"],
        samples["before"],
        np.log(samples["target"]),
        fit_mask=selected,
        materials=materials,
        settings=settings,
    )


def fit_material_illumination(
    baseline: np.ndarray,
    material_map: np.ndarray,
    fit_mask: np.ndarray,
    targets: dict[int, np.ndarray],
    *,
    dimensions: tuple[float, float],
    fit_shape: tuple[int, int],
    block_size: int,
    edge_margin: float,
    dn_scale: float,
    settings: MaterialFitSettings,
    reference: MaterialFitReference | None = None,
) -> MaterialIlluminationCorrection:
    """Fit the material illumination correction on a baseline image.

    The fit computes in the baseline's floating dtype (at least single
    precision); the reduction to the fitting grid is single precision for every
    baseline. With a reference run, the fitting grid is the reference's: the
    baseline is placed on it through the reference's homography, only blocks pure
    in both runs and safe in both are fitted, the block positions and the edge
    margin are the reference's, and the correction evaluates the gain at the
    pixels' positions on the reference's fitting grid.

    Parameters
    ----------
    baseline : array
        Linear RGB baseline (rows, cols, 3) in units of the sensor's white level;
        pixels with a non-finite or negative channel are not used.
    material_map : array
        Integer material of each pixel (rows, cols). Pixels of materials without
        a target are corrected with the common field.
    fit_mask : array
        Boolean pixels eligible for the fit (rows, cols).
    targets : dict
        Natural logarithm of the target RGB of each fitted material, in units of
        ``dn_scale`` times the image values.
    dimensions : tuple of float
        Physical height and width of the image; not used with a reference.
    fit_shape : tuple of int
        Rows and columns of the fitting grid, each at most the baseline's (and
        the reference's).
    block_size : int
        Block side length in cells of the fitting grid.
    edge_margin : float
        Physical margin along the edges left out of the fit.
    dn_scale : float
        Factor from image units to the units of the targets.
    settings : MaterialFitSettings
        Settings of the robust fit.
    reference : MaterialFitReference, optional
        Reference run whose fitting grid the fit uses. Defaults to the baseline's
        own fitting grid.

    Returns
    -------
    MaterialIlluminationCorrection
        Correction for images on the grid of the baseline.

    """
    # The fitting grid is no finer than the images, e.g., a coarse run grid.
    fit_shape = tuple(min(int(n), m) for n, m in zip(fit_shape, np.shape(baseline)))
    materials = sorted(int(material) for material in targets)
    dtype = np.result_type(baseline, np.float32)
    grid_to_field = reference_projection = None
    if reference is not None:
        shape = np.shape(reference.baseline)
        fit_shape = tuple(min(n, m) for n, m in zip(fit_shape, shape))
        dimensions = reference.dimensions
        grid_to_field = _pixel_resize(shape[:2], fit_shape, dtype) @ np.asarray(
            reference.homography, dtype
        )
        reference_projection = project_to_fit_grid(
            reference.baseline,
            reference.material_map,
            reference.fit_mask,
            fit_shape,
            materials,
            settings,
        )
    projection = project_to_fit_grid(
        baseline, material_map, fit_mask, fit_shape, materials, settings, grid_to_field
    )
    samples = collect_block_samples(
        projection,
        targets,
        dimensions=dimensions,
        edge_margin=edge_margin,
        block_size=block_size,
        dn_scale=dn_scale,
        settings=settings,
        dtype=dtype,
        reference=reference_projection,
    )
    common, fields = fit_block_field(samples, materials, settings)
    return MaterialIlluminationCorrection(
        common, fields, material_map, fit_shape, grid_to_field
    )
