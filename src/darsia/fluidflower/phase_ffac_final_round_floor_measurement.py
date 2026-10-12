"""Locked Step A pooling of source-bound photographic floor measurements.

No source pixels, domains or labels are written. The photographic measurement
method is unchanged; this module only pools its accepted 1 cm samples.
"""
from __future__ import annotations

import math
import numpy as np

from . import phase_ffac_p1_final_bottom_report as photo

CONTRACT = "ffac-final-round-floor-measurement-gate-v1"
SPACING_MM = 10.0
MINIMUM_RUNS_PER_NODE = 3
MAXIMUM_MEDIAN_RESIDUAL_MM = 1.0
MINIMUM_PASSING_RUNS = 39


def physical_samples(measurement):
    provenance = measurement["provenance"]
    H = np.asarray(provenance["H_frame_native_to_metric_pixel"], dtype=float)
    factor = float(provenance["metric_mm_factor"])
    ox, oy = provenance["native_origin_xy"]
    result = []
    for original in measurement["samples"]:
        if original.get("sand_bottom_canvas_y") is None:
            continue
        sample = dict(original)
        x = float(sample["column"]) + ox
        bottom = float(sample["bottom_canvas_y"]) + oy
        sand = float(sample["sand_bottom_canvas_y"]) + oy
        u, v = photo.project(H, x, bottom)
        a, b = photo.project(H, x, sand)
        sample["metric_x_mm"] = float(u * factor)
        sample["band_distance_mm"] = float(np.hypot(a-u, b-v) * factor)
        result.append(sample)
    return result


def pool_line(references, *, anchor_mm):
    """Nearest shared 1 cm node; at most one accepted sample per run/node.

    Camera-dependent sub-centimetre sample offsets are retained in the receipt.
    Residuals are scored at the original physical x, rather than rounded x.
    """
    nodes = {}
    assignments = []
    for run, measurement in sorted(references.items()):
        own = {}
        for sample in physical_samples(measurement):
            ordinal = math.floor((sample["metric_x_mm"]-anchor_mm)/SPACING_MM+.5)
            node_x = anchor_mm + ordinal*SPACING_MM
            offset = sample["metric_x_mm"]-node_x
            if ordinal not in own or abs(offset) < abs(own[ordinal]["x_offset_mm"]):
                own[ordinal] = dict(run=run, column=sample["column"], node_ordinal=ordinal,
                    node_x_mm=node_x, sample_x_mm=sample["metric_x_mm"], x_offset_mm=offset,
                    height_mm=sample["band_distance_mm"], source_sha256=measurement["source_sha256"])
        for ordinal, sample in own.items():
            nodes.setdefault(ordinal, []).append(sample)
            assignments.append(sample)
    pooled = []
    for ordinal, values in sorted(nodes.items()):
        heights = [row["height_mm"] for row in values]
        accepted = len(values) >= MINIMUM_RUNS_PER_NODE
        pooled.append(dict(node_ordinal=ordinal, metric_x_mm=anchor_mm+ordinal*SPACING_MM,
            contributing_runs=len(values), height_mm=float(np.median(heights)) if accepted else None,
            accepted=accepted, p10_height_mm=float(np.percentile(heights, 10)),
            p90_height_mm=float(np.percentile(heights, 90)), runs=sorted(row["run"] for row in values)))
    accepted = [node for node in pooled if node["accepted"]]
    if not accepted:
        raise ValueError("No 1 cm floor node has three accepted reference runs")
    return dict(anchor_mm=float(anchor_mm), spacing_mm=SPACING_MM,
        minimum_accepted_runs=MINIMUM_RUNS_PER_NODE, nodes=pooled,
        x_mm=[node["metric_x_mm"] for node in accepted],
        height_mm=[node["height_mm"] for node in accepted], assignments=assignments,
        interpolation="linear between pooled nodes with >=3 accepted reference runs; constant beyond both ends",
        assignment="nearest 1 cm common metric-x node, one original accepted sample per run/node; original x retained for residual scoring")


def height_at(pool, x):
    return np.interp(np.asarray(x, float), pool["x_mm"], pool["height_mm"])


def canonical_curve(pool, domain, provenance):
    """Place the pooled height above AC28 reference's bound domain bottom.

    Invert the projective metric transform along each image column. Distance
    remains the original photographic method's Euclidean physical distance.
    """
    domain = np.asarray(domain, bool)
    occupied = domain.any(axis=0)
    bottom = domain.shape[0]-1-np.argmax(domain[::-1], axis=0)
    H = np.asarray(provenance["H_frame_native_to_metric_pixel"], float)
    factor = float(provenance["metric_mm_factor"])
    ox, oy = provenance["native_origin_xy"]
    xs = np.arange(domain.shape[1], dtype=float)+ox
    ys = bottom.astype(float)+oy
    u, v = photo.project(H, xs, ys)
    a, b = photo.project(H, xs, ys-1.)
    length = np.hypot(a-u, b-v)
    if np.any(length[occupied] <= 0):
        raise ValueError("Invalid metric floor-column direction")
    amount = height_at(pool, u*factor)/factor
    tu = u+(a-u)*amount/length
    tv = v+(b-v)*amount/length
    canvas_x, canvas_y = photo.project(np.linalg.inv(H), tu, tv)
    if np.max(np.abs(canvas_x[occupied]-xs[occupied])) > 1e-5:
        raise ValueError("Inverse metric floor curve leaves its source image column")
    curve = canvas_y-oy
    curve[~occupied] = np.nan
    return curve


def score_measurement(measurement, transported_curve):
    provenance = measurement["provenance"]
    H = np.asarray(provenance["H_frame_native_to_metric_pixel"], float)
    factor = float(provenance["metric_mm_factor"])
    ox, oy = provenance["native_origin_xy"]
    scores = []
    for sample in physical_samples(measurement):
        column = int(sample["column"])
        predicted = float(transported_curve[column])
        if not np.isfinite(predicted):
            continue
        x, bottom = column+ox, float(sample["bottom_canvas_y"])+oy
        u, v = photo.project(H, x, bottom)
        a, b = photo.project(H, x, predicted+oy)
        predicted_mm = float(np.hypot(a-u, b-v)*factor)
        residual = sample["band_distance_mm"]-predicted_mm
        scores.append(dict(column=column, metric_x_mm=sample["metric_x_mm"],
            own_height_mm=sample["band_distance_mm"], pooled_transported_height_mm=predicted_mm,
            residual_mm=residual, absolute_residual_mm=abs(residual),
            source_sha256=measurement["source_sha256"]))
    values = [row["absolute_residual_mm"] for row in scores]
    median = float(np.median(values)) if values else None
    return dict(accepted_measurements=len(physical_samples(measurement)), scored_samples=len(scores),
        sample_count=len(measurement["samples"]), unknown_samples=measurement["summary"]["unknown_samples"],
        median_absolute_residual_mm=median, p90_absolute_residual_mm=float(np.percentile(values,90)) if values else None,
        maximum_absolute_residual_mm=max(values) if values else None,
        passes=bool(median is not None and median <= MAXIMUM_MEDIAN_RESIDUAL_MM), samples=scores)
