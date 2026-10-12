"""Versioned final domain and disconnected-noise guards for regional maps.

Apply after both phase heads, including inherited gas. Existing candidates opt
in through their own immutable profile; historical runtime identities stay
unchanged. The filter can only remove support, never bridge water channels.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .phase_rendering import two_phase_masks
from .project import sha256_file

VERSION = "regional-final-domain-islands-v1"
PROFILE_KEY = "output_guards"


@dataclass(frozen=True)
class Settings:
    format: str = VERSION
    minimum_island_area_mm2: float = 2.5

    def __post_init__(self):
        if (self.format != VERSION or not np.isfinite(self.minimum_island_area_mm2)
                or not 0 < self.minimum_island_area_mm2 <= 25):
            raise ValueError("Invalid regional output guard settings")


def runtime_identity(inherited, specification):
    settings = Settings(**specification)
    return hashlib.sha256(json.dumps(dict(inherited=inherited, settings=asdict(settings),
        code_sha256=sha256_file(Path(__file__)), opencv=cv2.__version__), sort_keys=True).encode()).hexdigest()


def guard_labels(labels, domain, spacing_m, settings=Settings()):
    labels, domain = np.asarray(labels), np.asarray(domain, bool)
    spacing = np.asarray(spacing_m, float)
    if (labels.ndim != 2 or labels.shape != domain.shape
            or not np.isin(labels, (0, 1, 2)).all()):
        raise ValueError("Guard requires same-grid categorical phase labels and domain")
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Guard requires positive registered y/x spacing")
    valid = (labels > 0) & domain
    # Connectivity is across both phases: gas texture inside a continuous
    # aqueous plume is not mistaken for isolated background specks.
    count, components, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8), connectivity=8)
    area_mm2 = float(np.prod(spacing) * 1e6)
    keep = stats[:, cv2.CC_STAT_AREA] * area_mm2 >= settings.minimum_island_area_mm2
    # A component wholly confined to the outermost pixel row/column is a
    # raster/frame seam. Do not erode the domain or truncate real plumes that
    # touch the edge and extend into the image.
    h, w = labels.shape
    border_seams = (((stats[:, cv2.CC_STAT_WIDTH] == 1)
                    & np.isin(stats[:, cv2.CC_STAT_LEFT], (0, w - 1)))
                   | ((stats[:, cv2.CC_STAT_HEIGHT] == 1)
                      & np.isin(stats[:, cv2.CC_STAT_TOP], (0, h - 1))))
    border_seams[0] = False
    keep[border_seams] = False
    keep[0] = False
    retained = keep[components]
    result = np.where(retained, labels, 0).astype(np.uint8)
    return result, dict(format=VERSION, pixel_area_mm2=area_mm2,
        minimum_island_area_mm2=settings.minimum_island_area_mm2,
        removed_border_seams=int(border_seams.sum()),
        excluded_domain_pixels=int(((labels > 0) & ~domain).sum()),
        removed_islands=int((~keep[1:]).sum()), retained_islands=int(keep.sum()),
        removed_island_pixels=int((valid & ~retained).sum()),
        gas_removed_pixels=int(((labels == 2) & ~retained).sum()),
        aqueous_removed_pixels=int(((labels == 1) & ~retained).sum()),
        added_pixels=0, source_labels_modified=False)


class GuardedRegionSequence:
    def __init__(self, inherited, context, specification):
        self.inherited, self.context = inherited, context
        self.settings = Settings(**specification)
        self.last_diagnostics = {}

    def bind_reference(self, baseline):
        self.inherited.bind_reference(baseline)

    def apply_with_reference(self, predictions, elapsed_h, current):
        result, report = self.inherited.apply_with_reference(predictions, elapsed_h, current)
        masks = two_phase_masks(result)
        labels = np.where(masks["gas"], 2, masks["aqueous"]).astype(np.uint8)
        spacing = (abs(float(self.context.y_m[1]-self.context.y_m[0])),
                   abs(float(self.context.x_m[1]-self.context.x_m[0])))
        guarded, audit = guard_labels(labels, self.context.domain, spacing, self.settings)
        removed = (labels > 0) & (guarded == 0)
        rejected = ~np.asarray(self.context.domain, bool) | removed
        # Preserve scores and classification of retained plume pixels exactly.
        updated = {key: np.where(rejected, 1. if key == "untouched_water_probability" else 0., value)
                   .astype(np.float32) for key, value in result.items()}
        self.last_diagnostics = {**self.inherited.last_diagnostics, "region_guard_removed": removed}
        return updated, {**report, "output_guards": audit}
