"""Run-specific clear-water references and optical gas evidence for FF-AC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


CLEAR_REFERENCE_COUNT = 5


@dataclass(frozen=True)
class ClearReferenceSpec:
    folder: str
    edge: str


CLEAR_REFERENCE_SPECS = {
    "ac18": ClearReferenceSpec(
        "AC18/AC18_221011_134200_flush_30s", "start"
    ),
    "ac20": ClearReferenceSpec("AC20/221101_145800_AC20_flush", "start"),
    "ac24": ClearReferenceSpec("AC24/230110_142000_AC24_flush", "start"),
    # AC23 (2023-01-05; its folder name 230501 is a day/month typo) has no flush folder: the flush
    # photographed for AC24 five days later is borrowed (Olav, 2026-09-18: same packing assumed;
    # chart correlation 0.92, 8 px shift).
    "ac23": ClearReferenceSpec("AC24/230110_142000_AC24_flush", "start"),
    "ac26": ClearReferenceSpec("AC26/230123_180000_AC26_flush", "start"),
    "ac27": ClearReferenceSpec("AC27/230123_144300_AC27_flush", "start"),
    "ac32": ClearReferenceSpec("AC32/230213_155800_AC32_flush", "start"),
    "ac52": ClearReferenceSpec("AC52/230515_144800_AC52_flush_30s", "start"),
    "ac56": ClearReferenceSpec("AC56/230609_101430_AC56_wash", "end"),
    "ac60": ClearReferenceSpec("AC60/230706_101900_AC60_wash", "end"),
    "ac61": ClearReferenceSpec("AC61/230710_102130_AC61_wash", "end"),
    # Added 2026-09-18 for the series-wide phase-map batch (32 runs). Same rule as the entries above:
    # runs up to AC52 use the first photographs of the (first) flush folder, AC53 onwards the last
    # photographs of the wash folder. Every choice was checked against the photographs: the selected
    # edge shows the clear-water rig (interior mean RGB about (115-128, 95-105, 46-58), blue minus red
    # -64 to -76), the other edge is blue (BTB) or dirty (wash start). AC23 has no flush or wash folder
    # and stays unconfigured; AC53 has two washes, the one directly before its flush is used.
    "ac17": ClearReferenceSpec("AC17/AC17_221004_153000_flush_30s", "start"),
    "ac19": ClearReferenceSpec("AC19/221710_132300_AC19_flush", "start"),
    "ac21": ClearReferenceSpec("AC21/221107_143800_AC21_flush", "start"),
    "ac22": ClearReferenceSpec("AC22/221114_134000_AC22_flush", "start"),
    "ac25": ClearReferenceSpec("AC25/230113_150000_AC25_flush", "start"),
    "ac28": ClearReferenceSpec("AC28/230130_155400_AC28_flush", "start"),
    "ac29": ClearReferenceSpec("AC29/230202_133700_AC29_flush", "start"),
    "ac30": ClearReferenceSpec("AC30/230206_155800_AC30_flush", "start"),
    "ac31": ClearReferenceSpec("AC31/230208_160700_AC31_flush", "start"),
    "ac33": ClearReferenceSpec("AC33/230216_164000_AC33_flush", "start"),
    "ac34": ClearReferenceSpec("AC34/230220_141100_AC34_flush", "start"),
    "ac35": ClearReferenceSpec("AC35/230223_185300_AC35_flush", "start"),
    "ac38": ClearReferenceSpec("AC38/230305_143900_AC38_flush", "start"),
    "ac39": ClearReferenceSpec("AC39/230309_130800_AC39_flush", "start"),
    "ac40": ClearReferenceSpec("AC40/230314_173500_AC40_flush", "start"),
    # AC41: its own flush (2023-03-21) is not aligned with the injection photographs (2023-03-28;
    # chart correlation 0.17 for the first flush photograph, 0.96 with a 24x40 px shift for the
    # last), while the flush photographed for AC42 on 2023-03-23 matches AC41's camera position
    # exactly (correlation 0.975, zero shift); 2026-09-18.
    "ac41": ClearReferenceSpec("AC42/230323_144700_AC42_flush", "start"),
    "ac42": ClearReferenceSpec("AC42/230323_144700_AC42_flush", "start"),
    "ac43": ClearReferenceSpec("AC43/230411_154000_AC43_flush", "start"),
    "ac44": ClearReferenceSpec("AC44/230417_160500_AC44_flush", "start"),
    "ac45": ClearReferenceSpec("AC45/230420_141900_AC45_flush", "start"),
    "ac46": ClearReferenceSpec("AC46/230424_155000_AC46_flush", "start"),
    "ac47": ClearReferenceSpec("AC47/230427_172500_AC47_flush", "start"),
    "ac48": ClearReferenceSpec("AC48/230501_155200_AC48_flush", "start"),
    "ac49": ClearReferenceSpec("AC49/230504_173100_AC49_flush_30s", "start"),
    "ac50": ClearReferenceSpec("AC50/230508_142300_AC50_flush_30s", "start"),
    "ac51": ClearReferenceSpec("AC51/230511_143300_AC51_flush_30s", "start"),
    "ac53": ClearReferenceSpec("AC53/230529_121630_AC53_wash", "end"),
    "ac54": ClearReferenceSpec("AC54/230601_090800_AC54_wash", "end"),
    "ac55": ClearReferenceSpec("AC55/230606_091100_AC55_wash", "end"),
    "ac57": ClearReferenceSpec("AC57/230613_093030_AC57_wash", "end"),
    "ac58": ClearReferenceSpec("AC58/230616_104030_AC58_wash", "end"),
    "ac59": ClearReferenceSpec("AC59/230703_103430_AC59_wash", "end"),
}


def select_clear_reference_paths(
    raw_root: Path,
    run: str,
    count: int = CLEAR_REFERENCE_COUNT,
) -> tuple[Path, ...]:
    key = str(run).strip().lower()
    if key not in CLEAR_REFERENCE_SPECS:
        raise KeyError(f"No clear-water reference is configured for {run}")
    spec = CLEAR_REFERENCE_SPECS[key]
    folder = Path(raw_root) / Path(spec.folder)
    files = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    if len(files) < int(count):
        raise FileNotFoundError(
            f"{run}: expected at least {count} clear-water images in {folder}"
        )
    selected = files[:count] if spec.edge == "start" else files[-count:]
    return tuple(selected)


def clearwater_affinity(
    baseline_rgb: np.ndarray,
    clear_rgb: np.ndarray,
    observed_rgb: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the legacy endpoint-distance affinity used by the postflight."""

    baseline = np.asarray(baseline_rgb, dtype=np.float32)[..., :3]
    clear = np.asarray(clear_rgb, dtype=np.float32)[..., :3]
    observed = np.asarray(observed_rgb, dtype=np.float32)[..., :3]
    valid = np.asarray(valid_mask, dtype=bool)
    if baseline.shape != clear.shape or baseline.shape != observed.shape:
        raise ValueError("Baseline, clear-water, and observed RGB shapes differ")
    if valid.shape != baseline.shape[:2]:
        raise ValueError("Valid mask shape differs from RGB fields")

    span = np.linalg.norm(clear - baseline, axis=-1)
    distance_blue = np.linalg.norm(observed - baseline, axis=-1)
    distance_clear = np.linalg.norm(observed - clear, axis=-1)
    valid_span = span[valid & np.isfinite(span)]
    if valid_span.size == 0:
        raise ValueError("No valid clear-water reference pixels")
    span_floor = max(0.01, float(np.quantile(valid_span, 0.25)))
    informative = valid & np.isfinite(distance_blue) & np.isfinite(distance_clear)
    informative &= span >= span_floor
    changed = distance_blue >= np.maximum(0.01, 0.10 * span)
    evidence = informative & changed
    affinity = np.zeros(span.shape, dtype=np.float32)
    denominator = distance_blue + distance_clear
    affinity[evidence] = (
        distance_blue[evidence] / np.maximum(denominator[evidence], 1e-8)
    )
    clear_like = evidence & (affinity >= 0.60)
    return affinity, evidence, clear_like


def clearwater_projection_score(
    baseline_rgb: np.ndarray,
    clear_rgb: np.ndarray,
    observed_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    orthogonal_scale: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project RGB change toward the wash endpoint with off-axis rejection.

    The score is one only for a complete, well-aligned move from the blue
    baseline to the run-specific clear-water endpoint. A yellow aqueous colour
    change receives a low score when it points away from that RGB direction.
    """

    baseline = np.asarray(baseline_rgb, dtype=np.float32)[..., :3]
    clear = np.asarray(clear_rgb, dtype=np.float32)[..., :3]
    observed = np.asarray(observed_rgb, dtype=np.float32)[..., :3]
    valid = np.asarray(valid_mask, dtype=bool)
    if baseline.shape != clear.shape or baseline.shape != observed.shape:
        raise ValueError("Baseline, clear-water, and observed RGB shapes differ")
    if valid.shape != baseline.shape[:2]:
        raise ValueError("Valid mask shape differs from RGB fields")

    direction = clear - baseline
    span_squared = np.sum(direction * direction, axis=-1)
    span = np.sqrt(np.maximum(span_squared, 0.0))
    valid_span = span[valid & np.isfinite(span)]
    if valid_span.size == 0:
        raise ValueError("No valid clear-water reference pixels")
    span_floor = max(0.01, float(np.quantile(valid_span, 0.25)))

    delta = observed - baseline
    changed_distance = np.linalg.norm(delta, axis=-1)
    projection = np.sum(delta * direction, axis=-1) / np.maximum(
        span_squared, 1e-8
    )
    projected_delta = projection[..., None] * direction
    orthogonal = np.linalg.norm(delta - projected_delta, axis=-1) / np.maximum(
        span, 1e-6
    )
    finite = (
        np.isfinite(projection)
        & np.isfinite(orthogonal)
        & np.isfinite(changed_distance)
    )
    informative = valid & finite & (span >= span_floor)
    changed = changed_distance >= np.maximum(0.01, 0.10 * span)
    evidence = informative & changed

    scale = max(1e-6, float(orthogonal_scale))
    alignment = np.exp(-0.5 * np.square(orthogonal / scale))
    score = np.zeros(valid.shape, dtype=np.float32)
    score[evidence] = (
        np.clip(projection[evidence], 0.0, 1.0) * alignment[evidence]
    ).astype(np.float32)
    return score, evidence, orthogonal.astype(np.float32)


def average_rgb(images: Sequence[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("At least one clear-water image is required")
    shapes = {tuple(np.asarray(image).shape) for image in images}
    if len(shapes) != 1:
        raise ValueError(f"Clear-water image shapes differ: {sorted(shapes)}")
    return np.mean(
        np.stack(
            [np.asarray(image, dtype=np.float32)[..., :3] for image in images],
            axis=0,
        ),
        axis=0,
        dtype=np.float32,
    )
