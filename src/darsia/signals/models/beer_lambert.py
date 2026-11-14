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

Every length and area given in millimetres is an optical one, converted to pixels
with the grid's pitch on the sand plane (:func:`darsia.optical_pitch_m`). Every
setting is a field of :class:`BeerLambertSettings` or :class:`ShadowSettings`.

"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

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
    path_ratio: Mapping[int, float]
    """Path ratio of every other sand read, by facies id, as measured where two
    sands meet."""
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

        """
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
