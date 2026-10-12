"""Restore an explicitly recorded historical geometric shape registration.

This new-source-only adapter reuses the historical registration operators,
reference, static facies-derived domain, settings, validity bounds and order.
Old source imports and non-registered runs keep their original paths. No
phase labels, colour fit, phase heads or old scientific assets are changed.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tomllib

import cv2
import numpy as np

from .phase_ffac_camera_source_geometry import (
    CameraSourceGeometry, SHAPE, CONTENT_BOX, _asset, _read_native,
    compose_source_geometry, register_static_sand,
)
from .phase_ffac_camera_timeline import canonical_path, file_sha256, _read, _new_json

CONTRACT = "ffac-selected-historical-registered-source-geometry-v1"


def require_frozen_registration(original, run):
    frozen = original.get("correction_contract", {}).get("rig_registration") or {}
    profile = frozen.get("profile") or {}
    if (original.get("run") != run or frozen.get("run") != run
            or frozen.get("series") != "ffac" or frozen.get("active") is not True
            or frozen.get("method") != "sand-texture-residual-registration"
            or frozen.get("stage") != "after shape; before color"
            or frozen.get("baseline_identity") is not True
            or frozen.get("serialized_rig_modified") is not False
            or profile.get("enabled") is not True
            or profile.get("run", run) != run
            or profile.get("series", "ffac") != "ffac"):
        raise ValueError("Source requires its exact active historical shape registration")
    for key in ("reference_sha256", "domain_sha256", "settings", "valid_bounds_xyxy"):
        if profile.get(key) != frozen.get(key):
            raise ValueError("Frozen registration profile differs from its historical contract: " + key)
    return deepcopy(frozen)


def registration_operator_assets():
    from .tools import phase_rig_registration as host
    from darsia_extensions import registration as core
    return dict(host=_asset(host.__file__), core=_asset(core.__file__))


def restore_frozen_registration(shape_corrections, shape_reference, facies, frozen, *, run):
    """Literal historical install, using only its measured static inputs."""
    from .tools import phase_rig_registration as host
    operators = registration_operator_assets()
    if (operators["core"]["sha256"] != frozen.get("implementation_sha256")
            or host.HOST_IMPLEMENTATION_SHA256 != frozen.get("host_implementation_sha256")):
        raise ValueError("Historical registration implementation changed")
    rig = SimpleNamespace(shape_corrections=list(shape_corrections),
        shape_corrected_baseline=shape_reference, facies=facies)
    # install_registration checks both array hashes, settings and fixed bounds,
    # enforces exact baseline identity, and appends the same native operator.
    provenance = host.install_registration(rig, deepcopy(frozen["profile"]), run=run)
    if provenance != frozen:
        raise ValueError("Restored registration differs from the complete historical provenance")
    return rig.shape_corrections, provenance


class HistoricalRegisteredSourceGeometry(CameraSourceGeometry):
    """Original shape chain plus the exact recorded post-shape registration."""
    def __init__(self, parent_native_manifest):
        import darsia
        from darsia.presets.workflows.rig import Rig
        from darsia.fluidflower.tools.ffac_v5_frame_context import (
            _source_photo_grid, project_source_photo_safe_core,
        )
        self.native_path = canonical_path(parent_native_manifest)
        self.native = _read(self.native_path)
        if tuple(self.native["analysis_image_shape_hw"]) != SHAPE or tuple(self.native["content_box_xywh"]) != CONTENT_BOX:
            raise ValueError("Camera geometry requires the unchanged FF_AC native grid")
        self.native_asset = _asset(self.native_path)
        marker = self.native["v5_track1_photo_geometry"]
        self.original_asset = _asset(marker["original_source_manifest_path"],
                                     marker["original_source_manifest_sha256"])
        original = _read(self.original_asset["path"])
        if original["run"] != self.native["run"]:
            raise ValueError("Original shape source belongs to another run")
        old_transport = marker["frames"][0]
        self.transport_asset = _asset(old_transport["receipt_path"], old_transport["receipt_sha256"])
        self.transport = _read(self.transport_asset["path"])
        if (self.transport["original_source_manifest_sha256"] != self.original_asset["sha256"]
            or self.transport["run"] != self.native["run"]):
            raise ValueError("Frozen B transport belongs to another native source")
        self.frozen_b = {key:deepcopy(value) for key,value in self.transport.items() if key.startswith("v5_B_")}
        for prefix in ("v5_B_arw", "v5_B_camera_jpeg", "v5_B_native_jpeg", "v5_B_registration_certificate"):
            _asset(self.frozen_b[prefix+"_path"], self.frozen_b[prefix+"_sha256"])
        _, self.h_ref_b = compose_source_geometry(self.transport["H_ac28_to_frame_native"],
            self.transport["H_v5_B_to_frame_native"], np.eye(3))
        self.baseline = _read_native(self.frozen_b["v5_B_native_jpeg_path"], CONTENT_BOX)
        sand = _source_photo_grid(self.transport["photo_sand_reference_mask_path"],
                                 self.transport["photo_sand_reference_mask_sha256"])
        self.photo_reference_receipt = _asset(self.transport["photo_sand_reference_receipt_path"],
            self.transport["photo_sand_reference_receipt_sha256"])
        self.sand = project_source_photo_safe_core(sand, self.h_ref_b, shape_hw=SHAPE)
        configs = original["correction_contract"]["config_paths"]
        self.config_assets = [_asset(path) for path in configs]
        rig_path = None
        for path in configs:
            config = tomllib.loads(canonical_path(path).read_text(encoding="utf-8-sig"))
            if "path" in config.get("rig",{}):
                rig_path = canonical_path(config["rig"]["path"])
        if rig_path is None:
            raise ValueError("Original source has no serialized shape rig")
        registration = require_frozen_registration(original, self.native["run"])
        paths = sorted(rig_path.glob("shape_correction_*.npz"))
        if not paths:
            raise ValueError("No persisted original shape corrections")
        self.shape_assets = [_asset(path) for path in paths]
        self.shape_corrections = [darsia.read_correction(path) for path in paths]
        if any(not Rig._is_shape_correction(correction) for correction in self.shape_corrections):
            raise ValueError("A non-shape correction entered source geometry")
        self.shape_rig_path = str(rig_path)
        reference_path = rig_path / "shape_corrected_baseline.npz"
        facies_path = rig_path / "facies.npz"
        self.registration_assets = [_asset(reference_path), _asset(facies_path)]
        shape_reference = darsia.imread(reference_path)
        facies = darsia.imread(facies_path)
        self.shape_corrections, provenance = restore_frozen_registration(
            self.shape_corrections, shape_reference, facies, registration,
            run=self.native["run"])
        self.registration_binding = dict(contract=CONTRACT,
            parent_native_manifest=self.native_asset,
            original_native_manifest=self.original_asset,
            frozen_registration=deepcopy(registration),
            registered_shape_assets=deepcopy(self.registration_assets),
            registration_operators=registration_operator_assets(),
            registration_provenance=provenance,
            phase_labels_read=False, photometry_fit=False,
            original_scientific_inputs_changed=False)
        generated_b = self.rectify_camera(self.frozen_b["v5_B_camera_jpeg_path"])
        parity_h, parity = register_static_sand(self.baseline, generated_b, self.sand)
        samples = np.asarray([[[0,0],[5471,0],[2736,1490],[0,2978],[5471,2978]]],np.float32)
        displacements = np.linalg.norm(cv2.perspectiveTransform(samples,parity_h)-samples,axis=2)
        if float(displacements.max()) > 1.:
            raise ValueError("Persisted shape chain differs from the frozen B native geometry")
        self.baseline_geometry_parity = dict(parity, H_frozen_B_to_shape_B=parity_h.tolist(),
            maximum_fivepoint_displacement_native_px=float(displacements.max()),
            required_maximum_native_px=1., colour_pixels_compared=False,
            historical_registered_shape=self.registration_binding)


    def prepare(self, source_row, output):
        # Verify the added static inputs before and after the inherited source
        # operation. The original persisted shape/config verification remains.
        for item in self.registration_assets:
            _asset(item["path"], item["sha256"])
        original = super().prepare(source_row, output)
        for item in self.registration_assets:
            _asset(item["path"], item["sha256"])
        receipt = deepcopy(original)
        receipt.pop("geometry_certificate_path", None)
        receipt.pop("geometry_certificate_sha256", None)
        receipt["binding"]["historical_registered_shape"] = self.registration_binding
        receipt["binding"]["registered_shape_adapter_sha256"] = file_sha256(__file__)
        receipt["source_geometry_adapter"] = dict(
            original_geometry_receipt=_asset(original["geometry_certificate_path"],
                original["geometry_certificate_sha256"]),
            registered_shape=self.registration_binding,
            applied_stage="after persisted shape; before photographic geometry matching",
            existing_sources_changed=False, model_inference=False,
            registration_thresholds_changed=False)
        path = Path(original["geometry_certificate_path"]).with_name("historical_registration.json")
        if path.exists():
            if _read(path) != receipt:
                raise ValueError("Registered source geometry already has another immutable binding")
        else:
            _new_json(path, receipt)
        return dict(receipt, geometry_certificate_path=str(path),
            geometry_certificate_sha256=file_sha256(path))


def source_geometry_for_manifest(parent_native_manifest):
    """Opt in only for the active adapter recorded by this source's manifest."""
    native = _read(parent_native_manifest)
    marker = native["v5_track1_photo_geometry"]
    original_asset = _asset(marker["original_source_manifest_path"],
        marker["original_source_manifest_sha256"])
    original = _read(original_asset["path"])
    registration = original.get("correction_contract", {}).get("rig_registration")
    if not registration:
        return CameraSourceGeometry(parent_native_manifest)
    require_frozen_registration(original, native["run"])
    return HistoricalRegisteredSourceGeometry(parent_native_manifest)


__all__ = ["HistoricalRegisteredSourceGeometry", "source_geometry_for_manifest",
    "restore_frozen_registration", "require_frozen_registration", "CONTRACT"]
