"""Expert-knowledge adapter for constraining scalar analysis fields.

Static expert knowledge is a set of ROIs per field. Dynamic expert knowledge is
each image's own phase map: concentration_aq may be nonzero only where it says
CO2, saturation_g only where it says free gas, and both are unknown (NaN) where
the phase map is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import darsia
from darsia.multiphase.phase_rules import PhaseLabel
from darsia.utils.standard_images import roi_to_mask

if TYPE_CHECKING:
    from darsia.presets.workflows.config.expert_knowledge import ExpertKnowledgeConfig
    from darsia.presets.workflows.config.roi import RoiConfig
    from darsia.presets.workflows.config.roi_registry import RoiRegistry

_PHASES = {
    "concentration_aq": (PhaseLabel.DISSOLVED, PhaseLabel.GAS),
    "saturation_g": (PhaseLabel.GAS,),
}
"""Phase-map categories in which each field may be nonzero."""


@dataclass
class ExpertKnowledgeAdapter:
    """Adapter converting expert-knowledge ROI config into reusable image masks."""

    saturation_g_rois: dict[str, "RoiConfig"] = field(default_factory=dict)
    concentration_aq_rois: dict[str, "RoiConfig"] = field(default_factory=dict)
    no_saturation_g_rois: dict[str, "RoiConfig"] = field(default_factory=dict)
    no_concentration_aq_rois: dict[str, "RoiConfig"] = field(default_factory=dict)
    labels_image: darsia.Image | None = None
    """Raw labels image backing shape='label' ROI entries, if any are configured."""
    facies_image: darsia.Image | None = None
    """Resolved facies-ID image backing shape='facies' ROI entries, if any."""
    roi_registry: "RoiRegistry | None" = None
    """Registry used to resolve shape='intersection'/'union' entries' `members`."""
    phase_maps: Path | None = None
    """Folder holding each image's phase map as ``<image stem>.npz``, if any."""
    _mask_cache: dict[tuple[str, tuple], np.ndarray] = field(
        default_factory=dict, init=False
    )
    _last_phase_map: tuple[str | None, darsia.Image | None] = field(
        default=(None, None), init=False
    )

    @classmethod
    def from_config(
        cls,
        config: "ExpertKnowledgeConfig | None",
        roi_registry: "RoiRegistry | None",
        labels_image: darsia.Image | None = None,
        facies_image: darsia.Image | None = None,
    ) -> "ExpertKnowledgeAdapter":
        if config is None:
            return cls()

        saturation_rois = {}
        concentration_rois = {}
        no_saturation_rois = {}
        no_concentration_rois = {}

        if roi_registry is not None:
            if len(config.saturation_g) > 0:
                saturation_rois = roi_registry.resolve_rois(config.saturation_g)
            if len(config.concentration_aq) > 0:
                concentration_rois = roi_registry.resolve_rois(config.concentration_aq)
            if len(config.no_saturation_g) > 0:
                no_saturation_rois = roi_registry.resolve_rois(config.no_saturation_g)
            if len(config.no_concentration_aq) > 0:
                no_concentration_rois = roi_registry.resolve_rois(
                    config.no_concentration_aq
                )

        return cls(
            saturation_g_rois=saturation_rois,
            concentration_aq_rois=concentration_rois,
            no_saturation_g_rois=no_saturation_rois,
            no_concentration_aq_rois=no_concentration_rois,
            labels_image=labels_image,
            facies_image=facies_image,
            roi_registry=roi_registry,
        )

    def _signature(self, image: darsia.Image) -> tuple:
        metadata = image.metadata()
        dimensions = tuple(np.asarray(metadata.get("dimensions", []), dtype=float))
        origin = tuple(np.asarray(metadata.get("origin", []), dtype=float))
        indexing = str(metadata.get("indexing", ""))
        space_dim = int(metadata.get("space_dim", 2))
        return (
            tuple(int(v) for v in image.num_voxels),
            dimensions,
            origin,
            indexing,
            space_dim,
        )

    def _rois_for_mode(self, mode: str) -> dict[str, "RoiConfig"]:
        if mode == "saturation_g":
            return self.saturation_g_rois
        if mode == "concentration_aq":
            return self.concentration_aq_rois
        return {}

    def _negative_rois_for_mode(self, mode: str) -> dict[str, "RoiConfig"]:
        if mode == "saturation_g":
            return self.no_saturation_g_rois
        if mode == "concentration_aq":
            return self.no_concentration_aq_rois
        return {}

    def _check_mask_image(
        self,
        mask_image: darsia.Image | None,
        image: darsia.Image,
        context: str,
        attr_name: str,
    ) -> None:
        if mask_image is None:
            raise ValueError(
                f"ROI '{context}' needs a {attr_name}, but none "
                "was provided to ExpertKnowledgeAdapter."
            )
        if mask_image.img.shape[:2] != image.img.shape[:2]:
            raise ValueError(
                f"ExpertKnowledgeAdapter's {attr_name} shape "
                f"{mask_image.img.shape[:2]} does not match the "
                f"target field's shape {image.img.shape[:2]}."
            )

    def _mask_for_roi(
        self,
        roi: "RoiConfig",
        image: darsia.Image,
        _visiting: frozenset[str] = frozenset(),
    ) -> np.ndarray:
        """Boolean mask for a single ROI entry, dispatching on `.shape`.

        `"box"`/`"polygon"` are masked geometrically via `roi_to_mask`;
        `"label"` is masked against the raw `labels_image`; `"facies"` is
        masked against the resolved `facies_image` (whose pixel values are
        already facies IDs, per `setup_facies`'s `reassign_labels` step);
        `"intersection"`/`"union"` recursively combine their `members`'
        masks (looked up by name via `roi_registry`), so members may
        themselves be any shape, including another intersection/union.
        """
        if roi.shape in ("box", "polygon"):
            return roi_to_mask(roi.roi, image).img.astype(bool)

        if roi.shape == "label":
            self._check_mask_image(self.labels_image, image, roi.name, "labels_image")
            return self.labels_image.img == roi.label_index

        if roi.shape == "facies":
            self._check_mask_image(self.facies_image, image, roi.name, "facies_image")
            return self.facies_image.img == roi.facies_id

        # shape in ("intersection", "union")
        if roi.name in _visiting:
            raise ValueError(f"ROI '{roi.name}' has a circular 'members' reference.")
        if self.roi_registry is None:
            raise ValueError(
                f"ROI '{roi.name}' has shape={roi.shape!r} with members, but "
                "no roi_registry was provided to ExpertKnowledgeAdapter."
            )
        members = self.roi_registry.resolve(roi.members)
        member_masks = [
            self._mask_for_roi(member, image, _visiting | {roi.name})
            for member in members.values()
        ]
        combine = np.logical_and if roi.shape == "intersection" else np.logical_or
        return combine.reduce(member_masks)

    def _field_mask(
        self, rois: dict[str, "RoiConfig"], image: darsia.Image
    ) -> np.ndarray:
        """Boolean union mask across every ROI entry in one field's dict."""
        mask = np.zeros(image.img.shape, dtype=bool)
        for roi in rois.values():
            mask |= self._mask_for_roi(roi, image)
        return mask

    def _static_mask_for(self, image: darsia.Image, mode: str) -> np.ndarray | None:
        """Boolean "keep" mask for `mode` - True where the field may stay non-zero.

        Combines two independent rules: a positive ROI union (only keep
        inside it, if configured) and a negative ROI union (always zero
        inside it, if configured). Where both are configured and overlap,
        the negative rule wins - the overlapping region is zeroed even
        though it's also inside a positive ROI. Each ROI entry's `.shape`
        is resolved independently per entry (see `_mask_for_roi`), so the
        two unions may freely mix shapes.
        """
        positive_rois = self._rois_for_mode(mode)
        negative_rois = self._negative_rois_for_mode(mode)
        if len(positive_rois) == 0 and len(negative_rois) == 0:
            return None

        cache_key = (mode, self._signature(image))
        if cache_key not in self._mask_cache:
            if len(positive_rois) > 0:
                keep_mask = self._field_mask(positive_rois, image)
            else:
                keep_mask = np.ones(image.img.shape, dtype=bool)
            if len(negative_rois) > 0:
                exclude_mask = self._field_mask(negative_rois, image)
                keep_mask = keep_mask & ~exclude_mask
            self._mask_cache[cache_key] = keep_mask
        return self._mask_cache[cache_key]

    def _phase_labels(self, image: darsia.Image) -> np.ndarray:
        """Labels of the image's own phase map, read once per image.

        The map's grid is checked against the image's on every call.
        """
        if image.name is None:
            raise ValueError("A phase map is found by the image's name.")
        if self._last_phase_map[0] != image.name:
            path = self.phase_maps / f"{Path(image.name).stem}.npz"
            self._last_phase_map = (image.name, darsia.imread(path))
        phase = self._last_phase_map[1]
        if (
            phase.img.shape != image.img.shape
            or not np.array_equal(phase.dimensions, image.dimensions)
            or not np.array_equal(phase.origin, image.origin)
        ):
            raise ValueError(f"The phase map of {image.name} has another grid.")
        return phase.img

    def mask_for(self, image: darsia.Image, mode: str) -> np.ndarray | None:
        """Boolean "keep" mask for `mode`: static ROIs and the image's phase map.

        Without a phase-map folder this is the static mask (see
        `_static_mask_for`); with one, it is intersected with the region the
        image's own phase map allows for the field.
        """
        static = self._static_mask_for(image, mode)
        if self.phase_maps is None or mode not in _PHASES:
            return static
        allowed = np.isin(self._phase_labels(image), _PHASES[mode])
        return allowed if static is None else static & allowed

    def apply(self, image: darsia.Image | None, mode: str) -> darsia.Image | None:
        if image is None:
            return None
        mask = self.mask_for(image, mode)
        if mask is None:
            return image

        constrained = image.copy()
        constrained.img[~mask] = 0.0
        if self.phase_maps is not None and mode in _PHASES:
            unknown = self._phase_labels(image) == PhaseLabel.UNKNOWN
            constrained.img[unknown] = np.nan
        return constrained
