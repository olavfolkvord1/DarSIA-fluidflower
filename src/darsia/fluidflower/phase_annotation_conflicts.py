"""Exact annotation compatibility on one registered native source frame.

This module reads no files and never changes labels. A phase-presence region
needs a witness somewhere in its area. Partial overlap with an exclusion is
therefore compatible; only complete coverage by the negative union conflicts.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json

import cv2
import numpy as np

PHASES = ("gas", "aqueous", "co2")


def phase_constraints(phase):
    """Return universal positives, universal negatives, existential positives."""
    mapping = {
        "water": ((), PHASES, ()),
        "current_experiment_co2_artifact": ((), PHASES, ()),
        "gas": (("gas", "co2"), (), ()),
        "aqueous": (("aqueous", "co2"), ("gas",), ()),
        "gas_absent_region": ((), ("gas",), ()),
        "aqueous_absent_region": ((), ("aqueous",), ()),
        "co2_absent_region": ((), PHASES, ()),
        "gas_present_region": ((), (), ("gas",)),
        "aqueous_present_region": ((), (), ("aqueous",)),
        "co2_present_region": ((), (), ("co2",)),
        "aqueous_present_gas_absent_region": ((), ("gas",), ("aqueous",)),
        "mixed_gas_aqueous": ((), (), ("gas", "aqueous")),
        "mixed_aqueous_water": ((), ("gas",), ("aqueous",)),
        "mixed_gas_aqueous_water": ((), (), ("gas", "aqueous")),
        "ignored": ((), (), ()),
    }
    if phase not in mapping:
        raise ValueError(f"Unknown phase semantics: {phase}")
    return tuple(frozenset(values) for values in mapping[phase])


@dataclass
class RegionRaster:
    x: int
    y: int
    mask: np.ndarray

    @property
    def box(self):
        return [self.x, self.y, self.mask.shape[1], self.mask.shape[0]]

    @property
    def selection(self):
        return np.s_[self.y:self.y+self.mask.shape[0], self.x:self.x+self.mask.shape[1]]


def rasterize_region(annotation, canvas_wh, content_box, domain=None):
    """Match trainer clipping/rounding/fill exactly, allocating only the ROI."""
    x, y, width, height = content_box
    points = annotation.get("points") or []
    if not points:
        px, py, w, h = (float(annotation[k]) for k in ("x", "y", "width", "height"))
        points = [(px, py), (px+w, py), (px+w, py+h), (px, py+h)]
    polygon = np.asarray([(float(px)*canvas_wh[0]-x, float(py)*canvas_wh[1]-y)
                          for px, py in points], np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError("Annotation must have at least three finite polygon vertices")
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width-1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height-1)
    polygon = np.rint(polygon).astype(np.int32)
    origin = polygon.min(axis=0)
    extent = polygon.max(axis=0)-origin+1
    mask = np.zeros((extent[1], extent[0]), np.uint8)
    cv2.fillPoly(mask, [polygon-origin], 1)
    result = RegionRaster(int(origin[0]), int(origin[1]), mask.astype(bool))
    if domain is not None:
        result.mask &= domain[result.selection]
    return result


def _shared(a, b):
    x, y = max(a.x, b.x), max(a.y, b.y)
    x1 = min(a.x+a.mask.shape[1], b.x+b.mask.shape[1])
    y1 = min(a.y+a.mask.shape[0], b.y+b.mask.shape[0])
    if x >= x1 or y >= y1:
        return None
    sa = np.s_[y-a.y:y1-a.y, x-a.x:x1-a.x]
    sb = np.s_[y-b.y:y1-b.y, x-b.x:x1-b.x]
    return x, y, sa, sb


def raster_intersection(a, b):
    shared = _shared(a, b)
    if shared is None:
        return 0, None
    x, y, sa, sb = shared
    pixels = a.mask[sa] & b.mask[sb]
    count = int(pixels.sum())
    if not count:
        return 0, None
    yy, xx = np.where(pixels)
    return count, [x+int(xx.min()), y+int(yy.min()), int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)]


def _negative_coverage(positive, phase, masks, meanings):
    # No native-frame-sized union: each positive bag allocates its own ROI.
    excluded = np.zeros(positive.mask.shape, bool)
    gas = np.zeros_like(excluded) if phase == "co2" else None
    aqueous = np.zeros_like(excluded) if phase == "co2" else None
    possible = []
    for index, (other, meaning) in enumerate(zip(masks, meanings)):
        negatives = meaning[1]
        if phase not in negatives and not (phase == "co2" and negatives & {"gas", "aqueous"}):
            continue
        shared = _shared(positive, other)
        if shared is None:
            continue
        _, _, sa, sb = shared
        covered = positive.mask[sa] & other.mask[sb]
        if not covered.any():
            continue
        possible.append(index)
        if phase in negatives:
            excluded[sa] |= covered
        if phase == "co2":
            if "gas" in negatives:
                gas[sa] |= covered
            if "aqueous" in negatives:
                aqueous[sa] |= covered
    if phase == "co2":
        excluded |= gas & aqueous
    return excluded, possible


def audit_frame_conflicts(regions, *, canvas_wh, content_box, domain=None):
    """Return ``strict`` contradictions and compatible ``partial`` candidates.

    Regions may be mappings or ReviewedRegion dataclasses. Each needs a unique
    region_id, canonical phase_class and original payload (or annotation).
    The caller supplies one registered frame; present run/frame/source identity
    fields must agree. Optional boolean domain limits all assertions to its
    valid pixels, including the full-coverage denominator.
    """
    if (len(canvas_wh) != 2 or len(content_box) != 4
            or any(type(v) is not int for v in (*canvas_wh, *content_box))
            or min(canvas_wh) <= 0 or min(content_box[:2]) < 0 or min(content_box[2:]) <= 0
            or content_box[0]+content_box[2] > canvas_wh[0]
            or content_box[1]+content_box[3] > canvas_wh[1]):
        raise ValueError("Expected a valid integer native canvas and content crop")
    if domain is not None:
        domain = np.asarray(domain)
        if domain.dtype != bool or domain.shape != (content_box[3], content_box[2]):
            raise ValueError("Domain must be boolean and match the native content crop")
    rows = []
    for region in regions:
        row = asdict(region) if is_dataclass(region) else dict(region)
        if is_dataclass(region) and hasattr(region, "supervision_weight"):
            row["supervision_weight"] = float(region.supervision_weight)
        rows.append(row)
    for field in ("dataset", "run", "frame_index", "elapsed_h", "native_manifest_sha256", "source_sha256"):
        known = {row[field] for row in rows if row.get(field) is not None}
        if len(known) > 1:
            raise ValueError(f"Regions do not identify the same registered source frame: {field}")
    if any(not row.get("region_id") for row in rows) or len({row["region_id"] for row in rows}) != len(rows):
        raise ValueError("Annotation region identifiers must be present and unique")
    for row in rows:
        payload = row.get("payload", row.get("annotation"))
        if not isinstance(payload, Mapping):
            raise ValueError("Original annotation payload is required")
        row["payload"] = payload
    rows = [row for row in rows if row.get("supervision_weight", 1) > 0 and row["phase_class"] != "ignored"]
    meanings = [phase_constraints(row["phase_class"]) for row in rows]
    masks = [rasterize_region(row["payload"], canvas_wh, content_box, domain) for row in rows]
    peers = {row["region_id"]: dict(region_id=row["region_id"], annotation=copy.deepcopy(row["payload"]),
        phase_class=row["phase_class"], **{"class": row["phase_class"]}) for row in rows}
    strict, partial = [], []

    def record(value):
        value["reason_code"] = "annotation_conflict" if value["strict"] else "annotation_partial_overlap"
        value["conflict_id"] = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:16]
        value["conflict_peers"] = [copy.deepcopy(peers[key]) for key in value["region_ids"]]
        (strict if value["strict"] else partial).append(value)

    for i, (row, a) in enumerate(zip(rows, masks)):
        total = int(a.mask.sum())
        if not total:
            continue
        for j in range(i+1, len(rows)):
            opposite = (meanings[i][0] & meanings[j][1]) | (meanings[j][0] & meanings[i][1])
            if not opposite:
                continue
            count, box = raster_intersection(a, masks[j])
            if count:
                record(dict(kind="opposed_universal_pixel_labels", strict=True,
                    region_ids=[row["region_id"],rows[j]["region_id"]],phases=sorted(opposite),
                    overlap_pixels=count,fractions=[count/total,count/int(masks[j].mask.sum())],
                    roi_content_xywh=box,
                    reason=f"Opposite per-pixel assertions for {', '.join(sorted(opposite))} overlap on {count} native pixels."))
        for phase in sorted(meanings[i][2]):
            covered, possible = _negative_coverage(a,phase,masks,meanings)
            count = int(covered.sum())
            if not count:
                continue
            contributors = []
            for j in possible:
                n,_ = raster_intersection(RegionRaster(a.x,a.y,covered),masks[j])
                if n:
                    contributors.append(dict(region_id=rows[j]["region_id"],overlap_pixels=n))
            full = count == total
            record(dict(kind="positive_region_covered_by_negative_union" if full else "partial_positive_negative_overlap",
                strict=full,phase=phase,positive_region_id=row["region_id"],
                region_ids=[row["region_id"],*[r["region_id"] for r in contributors]],
                negative_contributors=contributors,positive_region_pixels=total,
                covered_pixels=count,uncovered_pixels=total-count,covered_fraction=count/total,roi_content_xywh=a.box,
                reason=(f"One of the overlapping annotations requires {phase} somewhere, but the exclusion annotations cover its entire valid area."
                    if full else f"The exclusion annotations overlap {count} of {total} valid pixels; {total-count} pixels remain available for {phase}. Partial overlap alone is compatible.")))
    return dict(strict=strict,partial=partial,regions_checked=len(rows),domain_applied=domain is not None)
