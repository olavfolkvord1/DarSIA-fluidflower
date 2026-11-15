"""Beer–Lambert response per sand: each sand's colour change read as the reference sand's.

The concentration curve is one and the same for every sand; sands differ only
through optics that can be named. By Beer–Lambert's law, the log colour change
y = ln(current / baseline) of a sand is the absorbance of its pore water times
the optical path through that water, and the sand's background brightness, its
light and any shadow divide out. At equal concentration, sand k shows
y_k = p_k y_ref, where p_k, the sand's path ratio, is its effective optical path
relative to that of the reference sand.

Every sand's log colour change, the reference sand's included (p = 1), is read as
the reference sand's colour change at the same place on one smooth background B
of the reference sand, B (exp(y_k / p_k) - 1) (:class:`BeerLambertReading`). The
linear change a sand shows scales with its local background brightness: brighter
mixed grains next to a boundary and the grain texture would read as
concentration, and a shadow would need its own correction. Reading every sand
from its log change on the same background removes all three alike, and applies
p in the log space it is measured in. The change is read on the reference sand's
colour path, without a cap at the path's end (:class:`darsia.ColorPathSignal`),
and takes the reference sand's factor. No sand is divided by a shadow's light.

The background (:func:`reference_background`) is one smooth field over the rig.
The camera sees background = illumination x reflectance, and the light of the
corrected images varies slowly, so the reference sand's background where another
sand lies is a smooth surface through the reference sand's own background: a
Legendre surface in the log colour of each channel, fitted robustly to the log of
block means of reference-sand pixels away from other materials and outside a
real shadow (:func:`shadow_light`). The block means are arithmetic, the level a
colour change scales with: with the change proportional to the background, the
mean change B mean(dI / I0) equals mean(dI) only for B = mean(I0), while a
geometric mean sits below it by the grain texture. Grains of two sands mix at a
boundary, a painted boundary is uncertain by about a millimetre, and the
reference sand reads brighter there. A shadow is no background: a large one,
such as a colour chart's, would pull the surface down under it, and every sand
read there would lose that share of its change.

The path ratios are measured where two sands meet (:func:`path_ratios`): there,
connected pore water has the same concentration on both sides once diffusion has
evened it out, so the ratio of the two sides' log colour changes is the ratio of
their optical paths.

Every length and area given in millimetres is an optical one, converted to pixels
with the grid's pitch on the sand plane (:func:`darsia.optical_pitch_m`). Every
setting is a field of :class:`BeerLambertSettings`, :class:`ShadowSettings` or
:class:`PathRatioSettings`.

"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre
from scipy import ndimage

import darsia

_BLOCKS_PER_TERM = 4
"""Fewest block means per term of the background's surface; with fewer the fit is
refused."""

_DARKEST_MEAN = 1e-9
"""Floor of the colour means before their logarithm."""


@dataclass(frozen=True)
class ShadowSettings:
    """Settings of the shadow light; one configuration table."""

    fine_mm: float
    """Side [mm] of the square of the fine mean of a sand's brightness, which is
    the light inside a shadow, and over which the shadow fades in."""
    broad_mm: float
    """Side [mm] of the square of the broad mean of a sand's brightness."""
    below: float
    """A shadow's broad mean lies below this share of its sand's median."""
    reach_mm: float
    """Distance [mm] by which a shadow reaches beyond the area so found."""
    smallest_mm2: float
    """Smallest area [mm²] of a shadow."""
    darkest: float
    """Lowest light of a shadow, as a share of its sand's median."""


@dataclass(frozen=True)
class BeerLambertSettings:
    """Settings of the Beer–Lambert response; one configuration table."""

    reference_sand: int
    """Facies id of the reference sand, whose path ratio is one."""
    mean_mm: float
    """Side [mm] of the square over which a sand's colours are averaged inside its
    dissolved CO2 before their log change, and of the background's blocks."""
    background_degree: int
    """Degree of the background's Legendre surface."""
    background_margin_mm: float
    """Smallest distance [mm] of a background pixel from any other material."""
    unshaded_light: float
    """Lowest shadow light of a background pixel."""
    background_fill: float
    """Smallest share of a block's pixels that gives a block mean of the
    background; a block without pixels never gives one."""
    background_iterations: int
    """Robust solves of the background's surface per channel."""
    background_huber_threshold: float
    """Huber threshold of the background's robust fit, in robust standard
    deviations of its residuals."""
    path_ratio: Mapping[int, float] | None = None
    """Path ratio of every other sand read, by facies id, as measured where two
    sands meet (:func:`path_ratios`); None until they are measured, which the
    reading refuses."""


def shadow_light(
    baseline: np.ndarray,
    sand_map: darsia.Image,
    optical_pitch: tuple[float, float],
    sands: Iterable[int],
    settings: ShadowSettings,
) -> np.ndarray:
    """The light of each pixel in a real shadow, as a share of its sand's median.

    A sand's ordinary tone is no shadow. A shadow is a large area where the broad
    mean of a sand's brightness (the mean of the channels) lies below a share of
    the sand's median; that edge lies inside the true shadow, so the area reaches
    out by ``reach_mm``. Inside it the light is the fine mean of the sand's
    brightness over its median, no brighter than one and no darker than
    ``darkest``, faded in over ``fine_mm``, so that the rule draws no edge of its
    own. The light is one elsewhere. The means, the shadow's area and its reach
    are on the sand plane. Computed in the dtype of the baseline.

    Parameters
    ----------
    baseline : array
        RGB image of the baseline (rows, cols, 3).
    sand_map : darsia.Image
        Facies ids of the grid.
    optical_pitch : tuple of float
        Pitch [m] of the grid on the sand plane, down a column and along a row
        (:func:`darsia.optical_pitch_m`).
    sands : iterable of int
        Facies ids of the sands whose brightness is followed.
    settings : ShadowSettings
        Settings of the shadow light.

    Returns
    -------
    array
        The light (rows, cols).

    """
    facies = sand_map.img
    gray = np.mean(baseline, axis=-1)
    valid = np.isfinite(gray) & (gray > 0)
    gray = np.where(valid, gray, 0)
    masks = {sand: facies == sand for sand in sands}
    optical = darsia.optical_grid(sand_map, optical_pitch, baseline.dtype.type)
    fine_rev = darsia.REV(size=settings.fine_mm / 1000, img=optical)
    broad_rev = darsia.REV(size=settings.broad_mm / 1000, img=optical)
    fine, broad = np.ones_like(gray), np.ones_like(gray)
    for target, rev in ((fine, fine_rev), (broad, broad_rev)):
        mean = darsia.sand_mean(gray, valid, masks, rev)
        for sand_mask in masks.values():
            mask = valid & sand_mask
            if mask.any():
                target[mask] = mean[mask] / np.median(mean[mask])
    dy, dx = (float(size) for size in optical.voxel_size)
    regions, count = ndimage.label(broad < settings.below)
    sizes = np.bincount(regions.ravel(), minlength=count + 1) * (dx * dy * 1e6)
    sizes[0] = 0
    core = (sizes >= settings.smallest_mm2)[regions]
    if not core.any():
        return np.ones_like(gray)
    reach = ndimage.distance_transform_edt(~core, sampling=(dy * 1000, dx * 1000))
    shade = (reach <= settings.reach_mm).astype(gray.dtype)
    weight = darsia.sand_mean(shade, valid, masks, fine_rev)
    darkest = gray.dtype.type(settings.darkest)
    return 1 + weight * (np.clip(fine, darkest, 1) - 1)


def reference_background(
    baseline: np.ndarray,
    sand_map: darsia.Image,
    optical_pitch: tuple[float, float],
    support: np.ndarray,
    light: np.ndarray,
    settings: BeerLambertSettings,
) -> np.ndarray:
    """The reference sand's smooth background over the whole grid.

    Reference-sand pixels of ``support`` count when they lie at least
    ``background_margin_mm`` from any other facies and their shadow light is at
    least ``unshaded_light``; a pixel counts only if all its channels are finite
    and positive. Each channel is averaged over square blocks of ``mean_mm``
    (a block needs ``background_fill`` of its pixels, and one at least), and a
    Legendre surface of ``background_degree`` in the normalised pixel position is
    fitted to the log of these arithmetic block means, weighted by their pixel
    count, with the robust reweighting of :func:`darsia.linalg.robust_ridge`. The
    surface is held inside the range of the block means, so that far from the
    reference sand it never reads a background brighter or darker than the
    reference sand shows. The margin and the blocks are on the sand plane. The
    block sums and the fit run in double precision; the field is in the dtype of
    the baseline.

    Parameters
    ----------
    baseline : array
        RGB image of the baseline (rows, cols, 3).
    sand_map : darsia.Image
        Facies ids of the grid.
    optical_pitch : tuple of float
        Pitch [m] of the grid on the sand plane, down a column and along a row
        (:func:`darsia.optical_pitch_m`).
    support : array
        Pixels of the baseline that may enter the background.
    light : array
        Shadow light of each pixel (:func:`shadow_light`; one without shadows).
    settings : BeerLambertSettings
        Settings of the response.

    Returns
    -------
    array
        The background (rows, cols, 3).

    """
    facies = sand_map.img
    own = facies == settings.reference_sand
    optical = darsia.optical_grid(sand_map, optical_pitch, baseline.dtype.type)
    dy, dx = optical.voxel_size
    pixel_mm = float(np.sqrt(float(dx) * float(dy))) * 1000
    margin = ndimage.distance_transform_edt(own) * pixel_mm
    mask = own & support & (margin >= settings.background_margin_mm)
    mask &= light >= settings.unshaded_light
    mask &= np.isfinite(baseline).all(-1) & (baseline > 0).all(-1)
    block = max(1, int(round(settings.mean_mm / pixel_mm)))
    degree = settings.background_degree

    rows, cols = mask.shape
    hb, wb = rows // block, cols // block
    cut = mask[: hb * block, : wb * block]
    count = cut.reshape(hb, block, wb, block).sum((1, 3)).astype(np.float64)
    values = np.where(cut[..., None], baseline[: hb * block, : wb * block], 0)
    sums = values.reshape(hb, block, wb, block, 3).sum((1, 3), dtype=np.float64)
    keep = (count > 0) & (count >= settings.background_fill * block * block)
    terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    if keep.sum() < _BLOCKS_PER_TERM * len(terms):
        raise ValueError(
            f"The reference sand fills {int(keep.sum())} blocks, too few for a "
            f"surface of degree {degree}"
        )
    block_rows, block_cols = np.nonzero(keep)
    bx = legendre.legvander((block_cols + 0.5) * block / cols * 2 - 1, degree)
    by = legendre.legvander((block_rows + 0.5) * block / rows * 2 - 1, degree)
    design = np.stack([bx[:, i] * by[:, j] for i, j in terms], -1)
    weight = count[keep] / count[keep].max()
    means = np.log(sums[keep] / count[keep][:, None])
    penalty = np.zeros((len(terms), len(terms)))
    coefficients = darsia.linalg.robust_ridge(
        design,
        means,
        weight,
        penalty,
        settings.background_iterations,
        settings.background_huber_threshold,
    )
    vx = legendre.legvander((np.arange(cols) + 0.5) / cols * 2 - 1, degree)
    vy = legendre.legvander((np.arange(rows) + 0.5) / rows * 2 - 1, degree)
    field = np.empty((rows, cols, 3), baseline.dtype)
    for channel in range(3):
        table = np.zeros((degree + 1, degree + 1))
        for k, (i, j) in enumerate(terms):
            table[i, j] = coefficients[k, channel]
        surface = np.clip(
            vy @ table.T @ vx.T, means[:, channel].min(), means[:, channel].max()
        )
        field[..., channel] = np.exp(surface)
    return field


class BeerLambertReading:
    """Each sand's dissolved CO2 read from its log colour change as the reference
    sand's change on the reference sand's background.

    """

    def __init__(
        self,
        path: darsia.ColorPathSignal,
        factor: float,
        background: np.ndarray,
        sand_map: darsia.Image,
        optical_pitch: tuple[float, float],
        settings: BeerLambertSettings,
    ) -> None:
        """Reading of the sands named in the settings.

        Parameters
        ----------
        path : darsia.ColorPathSignal
            Colour path of the reference sand.
        factor : float
            Factor of the reference sand's signal: its path's scale over its
            measured colour range. Every sand takes it.
        background : array
            The reference sand's background (:func:`reference_background`) for the
            baseline of the frames read.
        sand_map : darsia.Image
            Facies ids of the grid.
        optical_pitch : tuple of float
            Pitch [m] of the grid on the sand plane, down a column and along a
            row (:func:`darsia.optical_pitch_m`), which sizes the colour means.
        settings : BeerLambertSettings
            Settings of the response.

        Raises
        ------
        ValueError
            If the settings hold no path ratios.

        """
        if settings.path_ratio is None:
            raise ValueError("The reading needs the sands' measured path ratios.")
        self.path = path
        """Colour path of the reference sand."""
        self.factor = float(factor)
        """Factor of every sand's signal."""
        self.background = background
        """The reference sand's background."""
        self.ratios = {
            int(settings.reference_sand): 1.0,
            **{int(k): float(v) for k, v in settings.path_ratio.items()},
        }
        """Path ratio of each sand read, the reference sand's included."""
        self.sands = sorted(self.ratios)
        """Facies ids of the sands read."""
        optical = darsia.optical_grid(sand_map, optical_pitch, background.dtype.type)
        self.rev = darsia.REV(size=settings.mean_mm / 1000, img=optical)
        """Square of the colour means."""

    def __call__(
        self,
        current: np.ndarray,
        baseline: np.ndarray,
        support: np.ndarray,
        dissolved: np.ndarray,
        facies: np.ndarray,
    ) -> np.ndarray:
        """Signal of the sands' dissolved CO2 in a frame; zero elsewhere.

        The current and baseline colours are averaged over the square inside each
        sand's dissolved pixels; their log change, the background's change at the
        same log change over the sand's path ratio, is computed in double
        precision and read on the reference sand's colour path in the dtype of
        the images.

        Parameters
        ----------
        current, baseline : array
            RGB images of the frame and of the baseline, of shape (rows, cols, 3).
        support : array
            Pixels whose colour change is measured; the dissolved pixels lie in it.
        dissolved : array
            Pixels of dissolved CO2 read.
        facies : array
            Facies id of each pixel.

        Returns
        -------
        array
            The signal, in the dtype of the images.

        """
        dtype = current.dtype
        masks = {sand: facies == sand for sand in self.sands}
        means = [
            darsia.sand_mean(
                np.where(dissolved[..., None], rgb, 0), dissolved, masks, self.rev
            )
            for rgb in (current, baseline)
        ]
        signal = np.zeros(facies.shape, dtype)
        for sand, ratio in self.ratios.items():
            mask = dissolved & masks[sand]
            if not mask.any():
                continue
            now, then = (
                np.log(np.maximum(mean[mask].astype(np.float64), _DARKEST_MEAN))
                for mean in means
            )
            change = self.background[mask].astype(np.float64) * np.expm1(
                (now - then) / ratio
            )
            values = self.path(np.ascontiguousarray(change, dtype))
            signal[mask] = values * values.dtype.type(self.factor)
        return signal


@dataclass(frozen=True)
class PathRatioSettings:
    """Settings of the path-ratio measurement; one configuration table."""

    mean_mm: float
    """Side [mm] of the square of the signal's mean inside each sand's dissolved
    CO2, as in the run statistics."""
    tile_mm: float
    """Side [mm] of a boundary tile."""
    band_mm: tuple[float, float]
    """Distances [mm] from the boundary between which each side's log colour
    change is measured; nearer pixels are left out for the uncertainty of the
    boundary."""
    far_mm: float
    """Distance [mm] from the boundary up to which each side must be dissolved
    CO2; from the band's outer edge to it, each side is checked for flatness."""
    min_dissolved_share: float
    """Smallest share of dissolved pixels on each side up to ``far_mm``."""
    gas_free_mm: float
    """Distance [mm] from a tile's pixels within which no free gas may lie."""
    min_band_mm2: float
    """Smallest area [mm²] of dissolved pixels of each side in each half of the
    band; the far band needs it too for its flatness check."""
    flat_tol_per_mm: float
    """Largest change of a side's log change per mm, relative to it."""
    arrived_share: float
    """Share of dissolved pixels in both sides' bands from which a tile carries
    CO2."""
    equalise_h: float
    """Time [h] both sides must have carried CO2 for the boundary to even out."""
    min_level: float
    """Smallest signal, and smallest log change, of each side."""
    ceiling_margin: float
    """Largest mean signal of each side, as a share of the ceiling."""
    min_tiles: int
    """Fewest qualified tiles of a pair that give its log ratio."""
    block_mm: float
    """Side [mm] of the blocks of tiles drawn by the bootstrap."""
    bootstrap: int
    """Number of bootstrap samples of a pair's log ratio."""
    seed: int
    """Seed of the bootstrap."""


@dataclass(frozen=True)
class PairRatio:
    """Log ratio of the optical paths of two sands that meet."""

    log_ratio: float
    """Median over the qualified tiles of the log of the first sand's change over
    the second sand's."""
    log_sd: float
    """Standard deviation of that median over the block bootstrap."""
    tiles: int
    """Number of qualified tiles."""


@dataclass(frozen=True)
class PathRatios:
    """Path ratios of the sands to the reference sand."""

    ratio: dict[int, float]
    """Path ratio of each other sand."""
    sd_rel: dict[int, float]
    """Relative standard deviation of each ratio."""
    pairs: dict[tuple[int, int], PairRatio]
    """Log ratio of each pair with enough qualified tiles."""


_SMALLEST_SCALE = 1e-6
"""Floor of a side's log change in the flatness rule."""

_SMALLEST_SD = 1e-3
"""Floor of a pair's log standard deviation in the weights of the fit."""

_BAND, _FAR = [1, 2], [3, 4]
"""Distance bins of the band's halves and of the far band's halves."""


class _Tiles:
    """The boundary bands of a sand map, grouped by pair, side, distance and tile.

    For each pair (a, b) of the sands (a < b) that touch, every pixel within
    ``far_mm`` of the other sand, and nearer to it than to any third facies, gets
    its side (0 for a, 1 for b), its distance bin and the tile of its nearest
    boundary pixel (a's pixels next to b), so that both sides of a tile face the
    same piece of boundary. The distance bins are the part inside the band, the
    band's two halves and the far band's two halves.

    """

    def __init__(self, facies, pixel_mm, sands, settings):
        inner, outer = settings.band_mm
        far = settings.far_mm
        self.edges = np.asarray(
            (0.0, inner, (inner + outer) / 2, outer, (outer + far) / 2, far),
            np.float32,
        )
        bins = len(self.edges) - 1
        labels = [int(v) for v in np.unique(facies)]
        dist = {
            label: (ndimage.distance_transform_edt(facies != label) * pixel_mm).astype(
                np.float32
            )
            for label in labels
        }
        tile_px = max(1, int(round(settings.tile_mm / pixel_mm)))
        ncol = facies.shape[1] // tile_px + 1
        index, group = [], []
        self.pairs, self.blocks, self.corners = [], [], []
        offset = 0
        for a, b in combinations(sorted(set(sands) & set(labels)), 2):
            boundary = (facies == a) & (dist[b] <= 1.5 * pixel_mm)
            if not boundary.any():
                continue
            others = [dist[label] for label in labels if label not in (a, b)]
            nearest_other = (
                np.minimum.reduce(others)
                if others
                else np.full(facies.shape, np.inf, np.float32)
            )
            sides = [
                (facies == own)
                & (dist[other] < self.edges[-1])
                & (dist[other] < nearest_other)
                for own, other in ((a, b), (b, a))
            ]
            band = sides[0] | sides[1]
            _, nearest = ndimage.distance_transform_edt(~boundary, return_indices=True)
            cells, tile_of_band = np.unique(
                (nearest[0][band] // tile_px).astype(np.int64) * ncol
                + nearest[1][band] // tile_px,
                return_inverse=True,
            )
            del nearest
            tiles = np.full(facies.shape, -1, np.int64)
            tiles[band] = tile_of_band.ravel()
            count = len(cells)
            for side, (selected, other) in enumerate(zip(sides, (b, a))):
                flat = np.flatnonzero(selected)
                distance = dist[other].ravel()[flat]
                where = np.searchsorted(self.edges, distance, side="right") - 1
                where = np.clip(where, 0, bins - 1)
                index.append(flat)
                group.append(
                    offset + (side * bins + where) * count + tiles.ravel()[flat]
                )
            self.pairs.append((a, b))
            self.blocks.append((offset, count))
            corner = np.column_stack(
                [(cells // ncol) * tile_px, (cells % ncol) * tile_px]
            )
            self.corners.append(corner * pixel_mm)
            offset += 2 * bins * count
        self.bins = bins
        self.index = np.concatenate(index) if index else np.zeros(0, np.int64)
        self.group = np.concatenate(group) if group else np.zeros(0, np.int64)
        self.group = self.group.astype(np.int64)
        self.groups = offset
        self.pixels = np.bincount(self.group, minlength=offset)

    def sums(self, dissolved, gas_near, current, baseline, signal, ceiling):
        """Per group of one frame: its dissolved pixels, its pixels near free gas,
        the sums of the log colour change and of the signal over its dissolved
        pixels (in double precision), and its dissolved pixels at the ceiling."""
        index, group, size = self.index, self.group, self.groups
        picked = dissolved.ravel()[index]
        g, pixel = group[picked], index[picked]
        now, then = (
            np.log(
                np.maximum(rgb.reshape(-1, 3)[pixel].astype(np.float64), _DARKEST_MEAN)
            )
            for rgb in (current, baseline)
        )
        log_change = now - then
        x = signal.ravel()[pixel].astype(np.float64)
        return dict(
            dissolved=np.bincount(g, minlength=size),
            gas=np.bincount(group[gas_near.ravel()[index]], minlength=size),
            log=np.stack(
                [np.bincount(g, log_change[:, c], minlength=size) for c in range(3)], -1
            ),
            signal=np.bincount(g, x, minlength=size),
            ceiling=np.bincount(g[x >= ceiling], minlength=size),
        )


class _Observations:
    """Qualified tiles of a run's frames, added in time order."""

    def __init__(self, tiles, reference, min_pixels, ceiling, settings):
        self.tiles, self.reference, self.settings = tiles, reference, settings
        self.min_pixels, self.ceiling = min_pixels, ceiling
        self.arrival = [np.full(count, np.inf) for _, count in tiles.blocks]
        self.direction = np.zeros(3)
        self.kept = {k: [] for k in ("pair", "tile", "near", "far", "first", "second")}
        self.kept["has_far"] = []

    def add(self, hours, sums):
        """Qualify one frame's tiles, except for flatness and the log level, which
        need the log-change direction of all frames."""
        settings, bins = self.settings, self.tiles.bins
        for p, ((a, b), (offset, count)) in enumerate(
            zip(self.tiles.pairs, self.tiles.blocks)
        ):
            block = offset + np.arange(2 * bins * count).reshape(2, bins, count)
            n_d, n_all = sums["dissolved"][block], self.tiles.pixels[block]

            def mean(key, where):
                total = sums[key][block][:, where].sum(1)
                number = np.maximum(n_d[:, where].sum(1), 1)
                return total / (number[..., None] if total.ndim == 3 else number)

            share = n_d[:, _BAND].sum(1) / np.maximum(n_all[:, _BAND].sum(1), 1)
            first = self.arrival[p]
            arrived = (share >= settings.arrived_share).all(0)
            first[arrived & ~np.isfinite(first)] = hours
            if self.reference in (a, b):
                side = block[(a, b).index(self.reference)][_BAND]
                self.direction += sums["log"][side].sum((0, 1))
            signal = mean("signal", _BAND)
            dissolved = n_d.sum(1) / np.maximum(n_all.sum(1), 1)
            ok = (
                (n_d[:, _BAND].sum(1) > 0).all(0)
                & (dissolved >= settings.min_dissolved_share).all(0)
                & (n_all.sum(1) > 0).all(0)
                & (sums["gas"][block].sum((0, 1)) == 0)
                & (n_d[:, _BAND] >= self.min_pixels).all((0, 1))
                & (signal >= settings.min_level).all(0)
                & (signal <= settings.ceiling_margin * self.ceiling).all(0)
                & (sums["ceiling"][block][:, _BAND].sum(1) == 0).all(0)
                & (hours - first >= settings.equalise_h)
            )
            keep = np.flatnonzero(ok)
            if not keep.size:
                continue
            self.kept["pair"].append(np.full(keep.size, p))
            self.kept["tile"].append(keep)
            for key, where in (
                ("near", _BAND),
                ("far", _FAR),
                ("first", _BAND[:1]),
                ("second", _BAND[1:]),
            ):
                self.kept[key].append(np.moveaxis(mean("log", where)[:, keep], 0, 1))
            far = (n_d[:, _FAR] >= self.min_pixels).all(1)
            self.kept["has_far"].append(far[:, keep].T)

    def log_changes(self):
        """Pair, tile and both sides' log change, on the reference sand's mean
        log-change direction, of every qualified tile that is flat and above the
        level."""
        if not self.kept["pair"]:
            return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2))
        kept = {key: np.concatenate(value) for key, value in self.kept.items()}
        norm = np.linalg.norm(self.direction)
        if not norm > 0:
            raise ValueError(
                "The reference sand has no dissolved CO2 at its boundaries"
            )
        direction = self.direction / norm
        inner, outer = self.settings.band_mm
        near, far = kept["near"] @ direction, kept["far"] @ direction
        halves = kept["second"] @ direction - kept["first"] @ direction
        scale = np.maximum(np.abs(near), _SMALLEST_SCALE)
        gradient = np.where(
            kept["has_far"],
            np.abs(far - near) / scale / ((self.settings.far_mm - inner) / 2),
            np.abs(halves) / scale / ((outer - inner) / 2),
        )
        ok = (gradient <= self.settings.flat_tol_per_mm).all(1)
        ok &= (near >= self.settings.min_level).all(1)
        return kept["pair"][ok], kept["tile"][ok], near[ok]


def _pair_ratios(tiles, pair, tile, near, settings):
    """Log ratio of each pair with enough qualified tiles, and its block-bootstrap
    standard deviation."""
    rng = np.random.default_rng(settings.seed)
    result = {}
    for p, (a, b) in enumerate(tiles.pairs):
        selected = pair == p
        if selected.sum() < settings.min_tiles:
            continue
        log_ratio = np.log(near[selected, 0]) - np.log(near[selected, 1])
        corner = np.floor(tiles.corners[p][tile[selected]] / settings.block_mm)
        corner = corner.astype(np.int64)
        _, block = np.unique(corner[:, 0] * 100000 + corner[:, 1], return_inverse=True)
        members = [np.flatnonzero(block == k) for k in range(block.max() + 1)]
        medians = [
            np.median(
                log_ratio[
                    np.concatenate(
                        [
                            members[k]
                            for k in rng.integers(0, len(members), len(members))
                        ]
                    )
                ]
            )
            for _ in range(settings.bootstrap)
        ]
        result[(a, b)] = PairRatio(
            float(np.median(log_ratio)), float(np.std(medians)), int(selected.sum())
        )
    return result


def path_ratios(
    frames: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    hours: Sequence[float],
    baseline: np.ndarray,
    sand_map: darsia.Image,
    optical_pitch: tuple[float, float],
    counted: np.ndarray,
    reading: darsia.SandReading,
    light: np.ndarray,
    reference: int,
    ceiling: float,
    settings: PathRatioSettings,
) -> PathRatios:
    """Each sand's path ratio to the reference sand, measured where two sands meet.

    Dissolved CO2 in connected pore water has the same concentration on both
    sides of a sand boundary once diffusion has had time to even it out, so the
    ratio of the two sides' log colour changes there is the ratio of their
    optical paths. One observation is a tile of one boundary in one frame: the
    mean log colour change in the band on each side. A tile counts only where
    continuity holds physically:

    - each side is dissolved CO2 up to ``far_mm`` (``min_dissolved_share``), so
      that no front, gas or unmeasured pixel lies in the tile;
    - no free gas lies within ``gas_free_mm``, since gas changes the optics;
    - each side is flat: its log change varies by at most ``flat_tol_per_mm``
      relative to it, between the band and the far band, or between the band's
      halves where the far band is too thin;
    - both sides have carried CO2 for ``equalise_h``, from the first frame where
      ``arrived_share`` of both bands is dissolved, so that the boundary has had
      time to even out;
    - no pixel is at the ceiling of the concentration curve and both sides' mean
      signals lie below ``ceiling_margin`` of it, since a capped value says
      nothing of the optics;
    - both sides' signals and log changes are at least ``min_level``, below
      which the ratio is noise.

    The free gas the phase model found (:func:`darsia.model_gas`, which gravity
    or a peel made dissolved CO2) is free gas here, not dissolved CO2, as in the
    run statistics. The signal the level and ceiling rules read is the reading's,
    in each sand's ordinary light (divided by the shadow light), averaged over
    ``mean_mm`` inside each sand's dissolved CO2. The log change is projected on the
    reference sand's mean log-change direction at its boundaries. Per pair, the
    log ratio is the median of its tiles', with a bootstrap over blocks of tiles.
    The path ratios follow from all pairs at once by weighted least squares on the
    log ratios, with the reference sand at one. Every length and area is on the
    sand plane: the mean on the pitch of each axis, the tiles, bands and
    distances on the side of the pitch's cell.

    Parameters
    ----------
    frames : sequence of (array, array, array, array)
        For each frame of the run, in time order: its RGB image (rows, cols, 3) on
        the grid of the sand map, its phase map (:class:`darsia.PhaseLabel`) and
        rule map (:class:`darsia.PhaseRule`), and the pixels whose colour change
        is measured. Read by index.
    hours : sequence of float
        Time [h] of each frame.
    baseline : array
        RGB image of the run's baseline.
    sand_map : darsia.Image
        Facies ids of the run's grid.
    optical_pitch : tuple of float
        Pitch [m] of the grid on the sand plane, down a column and along a row
        (:func:`darsia.optical_pitch_m`).
    counted : array
        Pixels that count in the tiles.
    reading : darsia.SandReading
        The sands measured (``reading.sands``) and the signal of their dissolved
        CO2, ``reading(current, baseline, support, dissolved, facies)``.
    light : array
        Shadow light of each pixel (:func:`shadow_light`).
    reference : int
        Facies id of the reference sand.
    ceiling : float
        Signal at the top of the concentration curve (its top over the gain).
    settings : PathRatioSettings
        Settings of the measurement.

    Returns
    -------
    PathRatios
        The ratios, their uncertainty and the pairs they come from.

    Raises
    ------
    ValueError
        If a sand is not connected to the reference sand by pairs with a ratio.

    """
    facies = sand_map.img
    sands = list(reading.sands)
    pixel_mm = float(np.sqrt(optical_pitch[0] * optical_pitch[1])) * 1000
    tiles = _Tiles(facies, pixel_mm, sands, settings)
    min_pixels = settings.min_band_mm2 / pixel_mm**2
    observations = _Observations(tiles, reference, min_pixels, ceiling, settings)
    optical = darsia.optical_grid(sand_map, optical_pitch, baseline.dtype.type)
    rev = darsia.REV(size=settings.mean_mm / 1000, img=optical)
    masks = {sand: facies == sand for sand in sands}
    for index in range(len(frames)):
        current, phase, rules, support = frames[index]
        gas = darsia.model_gas(phase, rules)
        dissolved = support & (phase == darsia.PhaseLabel.DISSOLVED) & ~gas
        signal = reading(current, baseline, support, dissolved, facies) / light
        signal = darsia.sand_mean(signal, dissolved, masks, rev)
        near_gas = np.zeros(facies.shape, bool)
        if gas.any():
            near_gas = (
                ndimage.distance_transform_edt(~gas) * pixel_mm < settings.gas_free_mm
            )
        sums = tiles.sums(
            dissolved & counted, near_gas, current, baseline, signal, ceiling
        )
        observations.add(hours[index], sums)
    pairs = _pair_ratios(tiles, *observations.log_changes(), settings)

    others = sorted(set(sands) - {reference})
    reached, frontier = {reference}, [reference]
    while frontier:
        sand = frontier.pop()
        for a, b in pairs:
            for this, that in ((a, b), (b, a)):
                if this == sand and that not in reached:
                    reached.add(that)
                    frontier.append(that)
    missing = [sand for sand in others if sand not in reached]
    if missing:
        raise ValueError(
            f"No qualified boundary pairs connect sand(s) {missing} to the "
            f"reference sand {reference}"
        )
    design = np.zeros((len(pairs), len(others)))
    for row, (a, b) in enumerate(pairs):
        for sand, sign in ((a, 1), (b, -1)):
            if sand in others:
                design[row, others.index(sand)] += sign
    log_ratio = np.array([pair.log_ratio for pair in pairs.values()])
    weight = np.array(
        [1 / max(pair.log_sd, _SMALLEST_SD) ** 2 for pair in pairs.values()]
    )
    covariance = np.linalg.inv(design.T @ (weight[:, None] * design))
    beta = covariance @ design.T @ (weight * log_ratio)
    return PathRatios(
        ratio={sand: float(np.exp(beta[k])) for k, sand in enumerate(others)},
        sd_rel={
            sand: float(np.sqrt(covariance[k, k])) for k, sand in enumerate(others)
        },
        pairs=pairs,
    )


PATH_RATIOS = "path_ratios.json"
"""The file of stored path ratios (:func:`write_path_ratios`)."""


def write_path_ratios(folder: Path, ratios: PathRatios) -> None:
    """Store path ratios (:func:`path_ratios`) in ``folder``, as JSON with every
    digit kept; a pair is named ``<sand>/<sand>``."""
    found = dict(
        ratio={str(sand): value for sand, value in ratios.ratio.items()},
        sd_rel={str(sand): value for sand, value in ratios.sd_rel.items()},
        pairs={f"{a}/{b}": asdict(pair) for (a, b), pair in ratios.pairs.items()},
    )
    text = json.dumps(found, indent=1, allow_nan=False)
    (Path(folder) / PATH_RATIOS).write_text(text + "\n", encoding="utf-8")


def read_path_ratios(folder: Path) -> PathRatios:
    """Stored path ratios (:func:`write_path_ratios`)."""
    found = json.loads((Path(folder) / PATH_RATIOS).read_text(encoding="utf-8"))
    pairs = {
        tuple(int(sand) for sand in name.split("/")): PairRatio(**pair)
        for name, pair in found["pairs"].items()
    }
    return PathRatios(
        ratio={int(sand): float(v) for sand, v in found["ratio"].items()},
        sd_rel={int(sand): float(v) for sand, v in found["sd_rel"].items()},
        pairs=pairs,
    )
