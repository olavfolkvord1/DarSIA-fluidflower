"""A proposal of the reference run of the material light correction: the medoid,
the run whose optical correction to all other runs is least on average.

Each run is described by its prepared baseline on the run grid: the median
colour of each cell of a coarse grid over the stable sand along the rig frame
(:func:`reference_samples`), and the colour checker's patches. The correction
between two runs (:func:`propose_reference`) is the mean of its parts: the light,
the median over their shared stable cells of the absolute mean log difference,
and the colour, the median over the shared usable patches of the log difference's
norm over sqrt(3), so that a grey shift and a colour cast both count. Runs are
ranked by the mean correction to the others, then by the largest, then by name.
It is a proposal only; nothing is chosen by it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ReferenceProposalSettings:
    """How the reference run of the light correction is proposed; no defaults."""

    stable_band_mm: float
    """Width of the stable sand band, measured inwards from the sand facing the
    rig frame, in millimetres."""
    use_lighting: bool
    """Whether the light of the stable sand counts."""
    use_color: bool
    """Whether the colour checker's patches count."""
    grid_shape: tuple[int, int]
    """Rows and columns of the coarse grid the stable sand is reduced on."""
    minimum_stable_cells: int
    """Fewest stable cells a run, and two runs together, must have."""
    minimum_chart_patches: int
    """Fewest usable colour-checker patches a run, and two runs together, must
    have."""


@dataclass(frozen=True)
class ReferenceSamples:
    """What one run's prepared baseline gives the proposal."""

    run: str
    stable_rgb: np.ndarray
    """The median colour of each cell of the coarse grid (cells, 3); NaN where a
    cell has no stable sand."""
    stable_valid: np.ndarray
    """Whether each cell has stable sand."""
    chart_rgb: np.ndarray | None = None
    """The colour checker's patches (patches, 3), or None."""


def frame_facing_sand_boundary(
    sand_domain: np.ndarray, other_material: np.ndarray | None = None
) -> np.ndarray:
    """The sand that faces the outside of the map, not water or enclosed holes.

    Parameters
    ----------
    sand_domain : np.ndarray
        Boolean image of the sand.
    other_material : np.ndarray, optional
        Boolean image of mapped material that is not sand (e.g. free water or a
        colour checker), which is not a gap towards the frame.

    Returns
    -------
    np.ndarray
        Boolean image of the sand next to the outside.

    Raises
    ------
    ValueError
        If the images are not boolean images of one shape, or overlap.

    """
    sand = np.asarray(sand_domain)
    if sand.ndim != 2 or sand.dtype != bool:
        raise ValueError("The mapped sand domain must be a boolean image")
    other = np.zeros(sand.shape, bool) if other_material is None else other_material
    other = np.asarray(other)
    if other.dtype != bool or other.shape != sand.shape:
        raise ValueError("Other mapped material must use the same boolean grid")
    if np.any(sand & other):
        raise ValueError("Sand and other material cannot overlap")
    # The padding stands for the part of the frame outside a cropped grid.
    space = np.pad(~(sand | other), 1, constant_values=True)
    seeds = np.zeros(space.shape, bool)
    seeds[[0, -1], :] = True
    seeds[:, [0, -1]] = True
    outside = ndimage.binary_propagation(seeds, mask=space)
    return sand & ndimage.binary_dilation(outside)[1:-1, 1:-1]


def frame_distance_mm(
    sand_domain: np.ndarray,
    pixel_size_mm: Sequence[float],
    frame_facing_boundary: np.ndarray | None = None,
) -> np.ndarray | None:
    """Each pixel's distance in millimetres from the sand's outer boundary, the
    part facing the frame where it is given; None without a boundary."""
    domain = np.asarray(sand_domain)
    outer = ndimage.binary_fill_holes(domain)
    boundary = outer & ~ndimage.binary_erosion(outer, border_value=0)
    if frame_facing_boundary is not None:
        selected = np.asarray(frame_facing_boundary)
        if selected.dtype != bool or selected.shape != domain.shape:
            raise ValueError("Frame-facing boundary must use the same boolean grid")
        boundary &= selected
    if not boundary.any():
        return None
    spacing = np.asarray(pixel_size_mm, dtype=float)
    return ndimage.distance_transform_edt(~boundary, sampling=spacing)


def stable_band_mask(
    sand_domain: np.ndarray,
    width_mm: float,
    pixel_size_mm: Sequence[float],
    frame_facing_boundary: np.ndarray | None = None,
) -> np.ndarray:
    """The stable sand: within ``width_mm`` inwards from the sand's outer
    boundary, the part facing the frame where it is given. Neither phases nor
    image values define it.

    Raises
    ------
    ValueError
        If the domain is not a boolean image, or the width or the pixel sizes are
        not positive millimetres.

    """
    domain = np.asarray(sand_domain)
    spacing = np.asarray(pixel_size_mm, dtype=float)
    if domain.ndim != 2 or domain.dtype != bool:
        raise ValueError("The stable area needs a boolean sand-domain mask")
    if (
        spacing.shape != (2,)
        or not np.isfinite(spacing).all()
        or np.any(spacing <= 0)
        or isinstance(width_mm, bool)
        or not np.isfinite(width_mm)
        or width_mm <= 0
    ):
        raise ValueError(
            "Stable-area width and pixel sizes must be positive millimetres"
        )
    distance = frame_distance_mm(domain, spacing, frame_facing_boundary)
    if distance is None:
        return np.zeros(domain.shape, bool)
    return domain & (distance <= float(width_mm))


def reference_samples(
    run: str,
    rgb: np.ndarray,
    stable_mask: np.ndarray,
    chart_rgb: np.ndarray | None,
    grid_shape: tuple[int, int],
) -> ReferenceSamples:
    """A run's samples: its stable sand reduced to the median colour of each cell
    of a coarse grid, each cell equally weighted and kept at its index, so that
    missing pixels never shift which parts of two runs are compared.

    Parameters
    ----------
    run : str
        The run.
    rgb : np.ndarray
        The prepared baseline, linear RGB in the same exposure units for every
        run; black, saturated and non-finite values are not used.
    stable_mask : np.ndarray
        Boolean image of the stable sand (:func:`stable_band_mask`).
    chart_rgb : np.ndarray or None
        The colour checker's patches, or None where colour does not count.
    grid_shape : tuple of int
        Rows and columns of the coarse grid.

    Returns
    -------
    ReferenceSamples
        The run's samples.

    Raises
    ------
    ValueError
        If the inputs do not describe one named RGB image and its mask.

    """
    image = np.asarray(rgb)
    mask = np.asarray(stable_mask)
    if (
        not isinstance(run, str)
        or not run
        or image.ndim != 3
        or image.shape[-1] != 3
        or mask.dtype != bool
        or mask.shape != image.shape[:2]
        or len(grid_shape) != 2
        or any(int(n) != n or n < 1 for n in grid_shape)
    ):
        raise ValueError("Reference samples require one named RGB image and its mask")
    usable = (
        mask
        & np.isfinite(image).all(axis=2)
        & (image > 0).all(axis=2)
        & (image < 1).all(axis=2)
    )
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    cells = np.full((rows * cols, 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(cells), bool)
    yy = np.linspace(0, image.shape[0], rows + 1, dtype=int)
    xx = np.linspace(0, image.shape[1], cols + 1, dtype=int)
    for iy in range(rows):
        for ix in range(cols):
            part = np.s_[yy[iy] : yy[iy + 1], xx[ix] : xx[ix + 1]]
            selected = image[part][usable[part]]
            if len(selected):
                cells[iy * cols + ix] = np.median(selected, axis=0)
                valid[iy * cols + ix] = True
    chart = (
        None if chart_rgb is None else np.asarray(chart_rgb, np.float64).reshape(-1, 3)
    )
    return ReferenceSamples(run, cells, valid, chart)


def _usable(chart: np.ndarray) -> np.ndarray:
    """The colour-checker patches with finite colours strictly inside (0, 1)."""
    return (
        np.isfinite(chart).all(axis=1)
        & (chart > 0).all(axis=1)
        & (chart < 1).all(axis=1)
    )


def propose_reference(
    samples: Sequence[ReferenceSamples], settings: ReferenceProposalSettings
) -> dict:
    """The medoid of the runs' optical corrections, with every run's ranking.

    Parameters
    ----------
    samples : Sequence[ReferenceSamples]
        Each run's samples, one per run.
    settings : ReferenceProposalSettings
        Which parts count and how many cells and patches are needed.

    Returns
    -------
    dict
        ``reference_run`` (the proposal), ``ranking`` (each run's
        ``mean_log_correction``, ``maximum_log_correction`` and, for each part
        used, ``mean_light_correction`` and ``mean_colour_correction``, ranked),
        ``uses_lighting``, ``uses_color`` and ``selected`` (always False: a
        proposal only).

    Raises
    ------
    ValueError
        If the runs repeat, no part is used, or a run or a pair has too few
        stable cells or usable patches.

    """
    samples = tuple(samples)
    use_lighting, use_color = settings.use_lighting, settings.use_color
    cells, patches = settings.minimum_stable_cells, settings.minimum_chart_patches
    if (
        not samples
        or len({row.run for row in samples}) != len(samples)
        or not (use_lighting or use_color)
    ):
        raise ValueError("Choose distinct optical inputs and at least one correction")
    for row in samples:
        if use_lighting and np.count_nonzero(row.stable_valid) < cells:
            raise ValueError("Too few stable areas for " + row.run)
        if use_color:
            if row.chart_rgb is None:
                raise ValueError("Matching ColorChecker observations are required")
            if np.count_nonzero(_usable(row.chart_rgb)) < patches:
                raise ValueError("Too few usable ColorChecker patches for " + row.run)
    count = len(samples)
    costs = np.zeros((count, count))
    light = np.zeros_like(costs) if use_lighting else None
    colour = np.zeros_like(costs) if use_color else None
    for i, left in enumerate(samples):
        for j in range(i + 1, count):
            right, parts = samples[j], []
            pair = f"{left.run} / {right.run}"
            if use_lighting:
                if left.stable_rgb.shape != right.stable_rgb.shape:
                    raise ValueError("Stable areas must use the same coarse grid")
                common = left.stable_valid & right.stable_valid
                if np.count_nonzero(common) < cells:
                    raise ValueError("Too few shared stable areas for " + pair)
                delta = np.log(left.stable_rgb[common]) - np.log(
                    right.stable_rgb[common]
                )
                parts.append(float(np.median(np.abs(delta.mean(axis=1)))))
                light[i, j] = light[j, i] = parts[-1]
            if use_color:
                a, b = left.chart_rgb, right.chart_rgb
                if a.shape != b.shape:
                    raise ValueError("Matching ColorChecker observations are required")
                usable = _usable(a) & _usable(b)
                if np.count_nonzero(usable) < patches:
                    raise ValueError("Too few shared ColorChecker patches for " + pair)
                delta = np.log(a[usable]) - np.log(b[usable])
                # Both a grey shift and a colour cast count.
                parts.append(
                    float(np.median(np.linalg.norm(delta, axis=1) / np.sqrt(3)))
                )
                colour[i, j] = colour[j, i] = parts[-1]
            costs[i, j] = costs[j, i] = float(np.mean(parts))
    denominator = max(1, count - 1)
    rows = []
    for i, row in enumerate(samples):
        rows.append(
            {
                "run": row.run,
                "mean_log_correction": float(costs[i].sum() / denominator),
                "maximum_log_correction": float(costs[i].max()),
            }
        )
        if light is not None:
            rows[-1]["mean_light_correction"] = float(light[i].sum() / denominator)
        if colour is not None:
            rows[-1]["mean_colour_correction"] = float(colour[i].sum() / denominator)
    ranking = sorted(
        rows,
        key=lambda r: (r["mean_log_correction"], r["maximum_log_correction"], r["run"]),
    )
    return {
        "reference_run": ranking[0]["run"],
        "method": "minimum-mean-pairwise-optical-correction",
        "ranking": ranking,
        "uses_lighting": bool(use_lighting),
        "uses_color": bool(use_color),
        "selected": False,
    }
