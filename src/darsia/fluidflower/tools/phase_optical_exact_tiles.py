"""Exact optical heads on 2-D tiles, optionally restricted to a grown search region.

Why this exists
---------------
``predict_image_tiled`` evaluates the serialized two-head model in full-width
strips.  This module evaluates the *same* code (``HierarchicalPhaseModel.
predict_image``) on square tiles with a halo.  Inside every computed tile the
result is bit-identical to the strip reference, because

* the halo covers the largest Gaussian support used by the features, and
* the left edge of every extended tile is aligned to 64 px, so OpenCV's SIMD
  row filter treats each pixel exactly as it does in a full-width strip (the
  only place an unaligned tile differed in tests was the last two columns at
  the right image border, by one float32 ulp).

That property is checked at run time against a reference strip
(:func:`verify_against_reference_strip`) instead of being assumed.

Two uses follow from exact tiles:

1. ``threads > 1`` evaluates tiles concurrently inside ONE process.  Temporaries
   are tile sized, so peak memory stays small and workers do not compete for
   memory bandwidth the way many full-strip processes do.
2. ``search`` restricts evaluation to a search region and *grows* it until every
   connected CO2 region found by the fine model is enclosed by evaluated water.
   The model itself decides where the plume edge is; the prior only says where
   to start looking.  Pixels outside the evaluated tiles are reported as "not
   evaluated" and receive the fill value; nothing is ever forced to CO2.

Measured on RUN6 (37 MP) the search region saves optical work early (0.23x of
the full cost at 0.75 h) and nothing late (0.96x at 96 h, plume = 44 % of the
domain).  It is therefore OFF by default and falls back to all tiles whenever
it would not pay.  It must not be used as a mask for the final phase map: in the
same run 25-32 % of the final CO2 area is added by the history/physics stages
where the optical head is below 0.5.

Nothing here changes a stored model, its 36-feature contract, or Studio.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any

import cv2
import numpy as np

from darsia.fluidflower.phase_features import (
    SPATIAL_CONTEXT_FRACTION, optical_feature_matrix, prepare_optical_feature_workspace,
)

METHOD = "optical-exact-tiles-v1"
ALIGN = 64
HEAD_FIELDS = ("co2_presence_probability", "gas_probability_conditional_on_co2")
# "reference": HierarchicalPhaseModel.predict_image, bit-identical to the strips.
# "packed32":  same features, standardisation folded into one float32 product for both
#              heads.  Not bit-identical (probabilities differ by about 1e-6); it removes
#              the float64 copies that are ~44 % of the reference CPU time.
HEAD_MODES = ("reference", "packed32")


@dataclass(frozen=True)
class SearchSettings:
    """Where to start looking; growth decides where to stop."""

    coarse_scale: int = 8               # the same model on block means; a detector only
    coarse_probability: float = 0.1     # low on purpose: a seed is cheap, a missed plume is not
    prior_margin_px: int = 64           # dilation of the previous FINAL map
    injection_radius_px: int = 300
    grow_band_px: int = 32              # CO2 this close to an unevaluated tile makes it evaluated
    grow_min_component_px: int = 10     # isolated specks do not drive growth
    full_frame_above_fraction: float = 0.6   # beyond this the search region cannot pay for itself

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TileReport:
    method: str = METHOD
    heads: str = "reference"
    tile: int = 256
    halo: int = 0
    threads: int = 1
    tiles_total: int = 0
    tiles_evaluated: int = 0
    evaluated_fraction: float = 1.0
    evaluated_window_px: int = 0
    evaluated_cost_fraction: float = 1.0   # evaluated window pixels (halos included) / image pixels
    search: dict[str, Any] | None = None
    seed_tiles: int = 0
    grow_rounds: int = 0
    full_frame_fallback: bool = False
    fallback_reason: str | None = None
    seconds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def packed_parameters(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """Fold (x - mean) / scale into the coefficients: logits = x @ weights + bias."""
    heads = (model.co2_presence, model.gas_within_co2)
    weights = np.column_stack([head.coefficients / head.scale for head in heads])
    bias = np.array([head.intercept - head.mean @ weights[:, index] for index, head in enumerate(heads)])
    return weights.astype(np.float32), bias.astype(np.float32)


def stable_sigmoid32(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    decay = np.exp(-np.abs(logits))
    return np.where(logits >= 0, 1 / (1 + decay), decay / (1 + decay)).astype(np.float32, copy=False)


def _checked_rgb(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected an RGB image, got {array.shape}")
    return array


def _tile_rgb(array: np.ndarray | None, window: tuple[slice, slice], scale: bool) -> np.ndarray | None:
    """One tile as float32, scaled to [0, 1] when the WHOLE image needs it.

    The reference ``_as_rgb_float`` divides a block by 255 when that block's
    maximum exceeds 1.5.  A full-width strip of a photograph always does; a small
    dark tile might not.  Deciding per image and converting here keeps every tile
    identical to the strip reference.
    """
    if array is None:
        return None
    block = np.asarray(array[window][..., :3], dtype=np.float32)
    return block / np.float32(255.0) if scale else block


def _needs_scaling(array: np.ndarray | None) -> bool:
    if array is None:
        return False
    if np.issubdtype(array.dtype, np.integer):
        return True
    # Float input follows the reference rule on the whole image.
    return bool(array.size) and float(np.nanmax(array)) > 1.5


def spatial_sigma(shape_hw: tuple[int, int], texture_sigma: float) -> float:
    return max(float(texture_sigma), SPATIAL_CONTEXT_FRACTION * float(min(shape_hw)))


def tile_halo(shape_hw: tuple[int, int], texture_sigma: float) -> int:
    """Same overlap rule as ``predict_image_tiled``."""
    return max(16, int(math.ceil(6.0 * spatial_sigma(shape_hw, texture_sigma))))


def tile_grid(shape_hw: tuple[int, int], tile: int) -> tuple[int, int]:
    return -(-shape_hw[0] // tile), -(-shape_hw[1] // tile)


def _extended_window(ty: int, tx: int, shape_hw: tuple[int, int], tile: int, halo: int, run: int = 1):
    """Core and halo window of ``run`` horizontally adjacent tiles starting at (ty, tx).

    Adjacent tiles are evaluated as one window so that their shared halos are
    not computed twice.  The left edge is aligned to ``ALIGN`` and the width is a
    multiple of ``ALIGN`` (or ends at the image border), see the module docstring.
    """
    height, width = shape_hw
    y0, x0 = ty * tile, tx * tile
    y1, x1 = min(height, y0 + tile), min(width, x0 + run * tile)
    a, b = max(0, y0 - halo), min(height, y1 + halo)
    c = (max(0, x0 - halo) // ALIGN) * ALIGN
    span = -(-(min(width, x1 + halo) - c) // ALIGN) * ALIGN
    d = min(width, c + span)
    return (y0, y1, x0, x1), (a, b, c, d)


def tile_runs(tiles: np.ndarray, max_run: int) -> list[tuple[int, int, int]]:
    """(row, first column, length) of horizontally adjacent True tiles, at most ``max_run`` long."""
    runs = []
    for ty in range(tiles.shape[0]):
        tx = 0
        while tx < tiles.shape[1]:
            if not tiles[ty, tx]:
                tx += 1
                continue
            length = 1
            while tx + length < tiles.shape[1] and tiles[ty, tx + length] and length < max_run:
                length += 1
            runs.append((ty, tx, length))
            tx += length
    return runs


def tiles_touching(mask: np.ndarray, tile: int) -> np.ndarray:
    """Boolean tile grid: True where ``mask`` has any pixel."""
    height, width = mask.shape
    rows, columns = tile_grid((height, width), tile)
    padded = np.zeros((rows * tile, columns * tile), dtype=bool)
    padded[:height, :width] = mask
    return padded.reshape(rows, tile, columns, tile).any(axis=(1, 3))


def tiles_to_mask(tiles: np.ndarray, tile: int, shape_hw: tuple[int, int]) -> np.ndarray:
    return np.kron(tiles, np.ones((tile, tile), dtype=bool))[: shape_hw[0], : shape_hw[1]]


class ExactTilePredictor:
    """Evaluate the reference optical model tile by tile."""

    def __init__(self, model: Any, baseline_rgb: Any, *, tile: int = 256, texture_sigma: float = 2.0,
                 threads: int = 1, radiometry_profile: Any | None = None, clearwater_rgb: Any | None = None,
                 max_window_px: int = 2048, heads: str = "reference"):
        if tile < ALIGN or tile % ALIGN:
            raise ValueError(f"Tile size must be a positive multiple of {ALIGN}")
        if heads not in HEAD_MODES:
            raise ValueError(f"Head evaluation must be one of {HEAD_MODES}")
        self.heads = heads
        if heads == "packed32":
            # predict_image() enforces the model's radiometric binding; keep that guard.
            from darsia.fluidflower.phase_training.radiometry import model_profile
            if radiometry_profile != model_profile(model):
                raise ValueError("Model requires its bound radiometric profile before image inference")
            self._weights, self._bias = packed_parameters(model)
        self.model, self.tile, self.texture_sigma = model, int(tile), float(texture_sigma)
        self.threads = max(1, int(threads))
        self.max_run = max(1, int(max_window_px) // int(tile))
        self.radiometry_profile = radiometry_profile
        self.baseline = _checked_rgb(baseline_rgb)
        self.clearwater = _checked_rgb(clearwater_rgb)
        self.shape = tuple(self.baseline.shape[:2])
        self.sigma = spatial_sigma(self.shape, self.texture_sigma)
        self.halo = tile_halo(self.shape, self.texture_sigma)
        self._scale_baseline = _needs_scaling(self.baseline)
        self._scale_clearwater = _needs_scaling(self.clearwater)

    # ------------------------------------------------------------------ evaluation
    def _predict_window(self, current, previous, scale_current, scale_previous, window):
        a, b, c, d = window
        region = (slice(a, b), slice(c, d))
        if self.heads == "packed32":
            workspace = prepare_optical_feature_workspace(
                _tile_rgb(current, region, scale_current), _tile_rgb(self.baseline, region, self._scale_baseline),
                previous_rgb=_tile_rgb(previous, region, scale_previous),
                clearwater_rgb=_tile_rgb(self.clearwater, region, self._scale_clearwater),
                texture_sigma=self.texture_sigma, spatial_context_sigma=self.sigma,
                contract=self.model.feature_contract)
            shape = workspace.current.shape[:2]
            features = optical_feature_matrix(workspace, (...,)).reshape(-1, self._weights.shape[0])
            probability = stable_sigmoid32(features @ self._weights + self._bias)
            return {name: probability[:, index].reshape(shape) for index, name in enumerate(HEAD_FIELDS)}
        kwargs = {} if self.radiometry_profile is None else {"radiometry_profile": self.radiometry_profile}
        return self.model.predict_image(
            _tile_rgb(current, region, scale_current), _tile_rgb(self.baseline, region, self._scale_baseline),
            previous_rgb=_tile_rgb(previous, region, scale_previous),
            clearwater_rgb=_tile_rgb(self.clearwater, region, self._scale_clearwater),
            texture_sigma=self.texture_sigma, spatial_context_sigma=self.sigma, **kwargs)

    def evaluate_tiles(self, current, previous, tiles: np.ndarray, outputs: dict[str, np.ndarray]) -> int:
        """Fill ``outputs`` for every True tile; return the number of pixels actually evaluated (with halos)."""
        scale_current, scale_previous = _needs_scaling(current), _needs_scaling(previous)

        def work(run):
            ty, tx, length = run
            (y0, y1, x0, x1), window = _extended_window(ty, tx, self.shape, self.tile, self.halo, length)
            predicted = self._predict_window(current, previous, scale_current, scale_previous, window)
            local = (slice(y0 - window[0], y1 - window[0]), slice(x0 - window[2], x1 - window[2]))
            for name in HEAD_FIELDS:
                outputs[name][y0:y1, x0:x1] = predicted[name][local]
            return (window[1] - window[0]) * (window[3] - window[2])

        # Longest first keeps the threads evenly loaded.
        runs = sorted(tile_runs(tiles, self.max_run), key=lambda run: -run[2])
        if self.threads == 1 or len(runs) < 2:
            return sum(work(run) for run in runs)
        with ThreadPoolExecutor(self.threads) as pool:
            # list() propagates the first worker exception instead of hiding it.
            return sum(list(pool.map(work, runs)))

    def predict(self, current_rgb: Any, previous_rgb: Any | None = None, *, search: SearchSettings | None = None,
                prior_mask: np.ndarray | None = None, injection_points_px=(), domain: np.ndarray | None = None,
                fill: float = 0.0) -> tuple[dict[str, np.ndarray], np.ndarray, TileReport]:
        """Return the two heads, the evaluated-pixel mask and a provenance report.

        ``search=None`` evaluates every tile (exact replacement of the strips).
        """
        started = time.perf_counter()
        current, previous = _checked_rgb(current_rgb), _checked_rgb(previous_rgb)
        if tuple(current.shape[:2]) != self.shape or (previous is not None and tuple(previous.shape[:2]) != self.shape):
            raise ValueError("Current, previous and baseline images must share one grid")
        grid = tile_grid(self.shape, self.tile)
        report = TileReport(heads=self.heads, tile=self.tile, halo=self.halo, threads=self.threads,
                            tiles_total=grid[0] * grid[1])
        outputs = {name: np.full(self.shape, np.float32(fill), dtype=np.float32) for name in HEAD_FIELDS}
        everything = np.ones(grid, dtype=bool)
        if search is None:
            tiles = everything
        else:
            report.search = search.to_dict()
            tiles, done = self._search_tiles(current, previous, search, prior_mask, injection_points_px, domain,
                                             outputs, report)
        if search is None or report.full_frame_fallback:
            remaining = everything if search is None else everything & ~done
            report.evaluated_window_px += self.evaluate_tiles(current, previous, remaining, outputs)
            tiles = everything
        evaluated = tiles_to_mask(tiles, self.tile, self.shape)
        report.tiles_evaluated = int(tiles.sum())
        report.evaluated_fraction = float(evaluated.mean())
        report.evaluated_cost_fraction = float(report.evaluated_window_px) / float(self.shape[0] * self.shape[1])
        report.seconds["total"] = time.perf_counter() - started
        return outputs, evaluated, report

    # ------------------------------------------------------------------ search region
    def coarse_probability(self, current, previous, scale: int) -> np.ndarray:
        """The serialized CO2 head on ``scale`` x block means.  A detector, never a result."""
        def block_mean(array, scaled):
            if array is None:
                return None
            height, width = (self.shape[0] // scale) * scale, (self.shape[1] // scale) * scale
            block = np.asarray(array[:height, :width, :3], dtype=np.float32)
            block = block.reshape(height // scale, scale, width // scale, scale, 3).mean(axis=(1, 3), dtype=np.float32)
            return block / np.float32(255.0) if scaled else block
        kwargs = {} if self.radiometry_profile is None else {"radiometry_profile": self.radiometry_profile}
        predicted = self.model.predict_image(
            block_mean(current, _needs_scaling(current)), block_mean(self.baseline, self._scale_baseline),
            previous_rgb=block_mean(previous, _needs_scaling(previous)),
            clearwater_rgb=block_mean(self.clearwater, self._scale_clearwater),
            texture_sigma=self.texture_sigma, **kwargs)
        return predicted[HEAD_FIELDS[0]]

    def _search_tiles(self, current, previous, settings, prior_mask, injection_points_px, domain, outputs, report):
        clock = time.perf_counter()
        inside = np.ones(self.shape, dtype=bool) if domain is None else np.asarray(domain, dtype=bool)
        if inside.shape != self.shape:
            raise ValueError("Domain mask must match the image grid")
        allowed = tiles_touching(inside, self.tile)
        seed = np.zeros(self.shape, dtype=np.uint8)
        if prior_mask is not None:
            prior = np.asarray(prior_mask, dtype=np.uint8)
            if prior.shape != self.shape:
                raise ValueError("Prior mask must match the image grid")
            size = 2 * int(settings.prior_margin_px) + 1
            seed |= cv2.dilate(prior, np.ones((size, size), dtype=np.uint8))
        for x, y in injection_points_px:
            cv2.circle(seed, (int(round(x)), int(round(y))), int(settings.injection_radius_px), 1, -1)
        coarse = self.coarse_probability(current, previous, int(settings.coarse_scale))
        detected = cv2.resize((coarse >= settings.coarse_probability).astype(np.uint8),
                              (self.shape[1], self.shape[0]), interpolation=cv2.INTER_NEAREST)
        seed |= detected
        report.seconds["seed"] = time.perf_counter() - clock
        tiles = tiles_touching(seed.astype(bool) & inside, self.tile) & allowed
        report.seed_tiles = int(tiles.sum())
        done = np.zeros_like(tiles)
        limit = settings.full_frame_above_fraction * float(allowed.sum())
        band = 2 * int(settings.grow_band_px) + 1
        clock = time.perf_counter()
        while True:
            if tiles.sum() > limit:
                report.full_frame_fallback = True
                report.fallback_reason = (f"search region reached {tiles.sum()} of {int(allowed.sum())} domain tiles "
                                          f"(> {settings.full_frame_above_fraction:.0%}); evaluating every tile")
                break
            report.evaluated_window_px += self.evaluate_tiles(current, previous, tiles & ~done, outputs)
            done = done | tiles
            evaluated = tiles_to_mask(done, self.tile, self.shape)
            positive = ((outputs[HEAD_FIELDS[0]] >= 0.5) & evaluated & inside).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(positive, 8)
            keep = np.zeros(count, dtype=bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= settings.grow_min_component_px
            reach = cv2.dilate(keep[labels].astype(np.uint8), np.ones((band, band), dtype=np.uint8)).astype(bool)
            wanted = tiles_touching(reach & inside, self.tile) & allowed & ~done
            if not wanted.any():
                break
            tiles = tiles | wanted
            report.grow_rounds += 1
        report.seconds["grow"] = time.perf_counter() - clock
        return tiles, done


def verify_against_reference_strip(predictor: ExactTilePredictor, current_rgb, previous_rgb, outputs,
                                   evaluated: np.ndarray, *, row: int | None = None, rows: int = 64) -> dict[str, Any]:
    """Compare evaluated tiles with one full-width reference strip, bit for bit.

    The strip is computed exactly as ``predict_image_tiled`` would (same halo,
    same sigma).  Returns counts, never raises: the caller decides what a
    mismatch means (the expected count is zero).
    """
    height = predictor.shape[0]
    if row is None:
        touched = np.nonzero(evaluated.any(axis=1))[0]
        row = int(touched[len(touched) // 2]) if touched.size else 0
    start, stop = max(0, min(row, height - rows)), max(0, min(row, height - rows)) + min(rows, height)
    a, b = max(0, start - predictor.halo), min(height, stop + predictor.halo)
    current, previous = _checked_rgb(current_rgb), _checked_rgb(previous_rgb)
    reference = predictor._predict_window(current, previous, _needs_scaling(current), _needs_scaling(previous),
                                          (a, b, 0, predictor.shape[1]))
    local = slice(start - a, stop - a)
    mask = evaluated[start:stop]
    result = dict(rows=[int(start), int(stop)], compared_pixels=int(mask.sum()), exact=True)
    for name in HEAD_FIELDS:
        theirs, mine = reference[name][local][mask], outputs[name][start:stop][mask]
        different = int(np.count_nonzero(theirs != mine))
        result[name] = dict(different_pixels=different,
                            max_abs_difference=float(np.max(np.abs(theirs - mine))) if mine.size else 0.0)
        result["exact"] = result["exact"] and different == 0
    return result


__all__ = ["ALIGN", "HEAD_MODES", "METHOD", "ExactTilePredictor", "SearchSettings", "TileReport", "tile_grid", "tile_halo",
           "tiles_to_mask", "tiles_touching", "verify_against_reference_strip"]
