"""Explicit own-run AC59 shape binding missing from its old source manifest.

The old native manifest points to configuration files which implicitly used
data.results/setup/rig. The project migration recorded that same AC59 rig
explicitly. This adapter binds its persisted shape files, then applies the
unchanged frozen-B parity and independent source-texture tests. No parent
source, model, prediction, or configuration is changed.
"""
from copy import deepcopy
from pathlib import Path
import json
import tomllib

import cv2
import numpy as np

from .phase_ffac_camera_source_geometry import (
    CameraSourceGeometry, SHAPE, CONTENT_BOX, _asset, _read_native,
    compose_source_geometry, register_static_sand,
)
from .phase_ffac_camera_timeline import canonical_path, file_sha256, _read, _new_json

CONTRACT = "ffac-selected-ac59-own-persisted-shape-binding-v1"


def validate_recovery_identity(binding, native):
    if (binding.get("contract") != CONTRACT or binding.get("run") != "ac59"
            or native.get("run") != binding["run"]
            or binding.get("borrowed_other_run_rig") is not False):
        raise ValueError("Recovery must bind AC59's own persisted shape rig")
    if binding.get("new_photometry_fit") is not False or binding.get("model_or_labels_used") is not False:
        raise ValueError("Shape recovery cannot learn photometry or model geometry")


class AC59OwnPersistedSourceGeometry(CameraSourceGeometry):
    def __init__(self, parent_native_manifest, recovery_binding):
        import darsia
        from darsia.presets.workflows.rig import Rig
        from darsia.fluidflower.tools.ffac_v5_frame_context import (
            _source_photo_grid, project_source_photo_safe_core,
        )
        self.recovery_asset = _asset(recovery_binding["path"], recovery_binding["sha256"])
        recovery = _read(self.recovery_asset["path"])
        self.native_path = canonical_path(parent_native_manifest)
        self.native = _read(self.native_path)
        validate_recovery_identity(recovery, self.native)
        self.native_asset = _asset(self.native_path)
        if self.native_asset != recovery["parent_native_manifest"]:
            raise ValueError("Recovery binds another AC59 native source")
        if tuple(self.native["analysis_image_shape_hw"]) != SHAPE or tuple(self.native["content_box_xywh"]) != CONTENT_BOX:
            raise ValueError("Own shape recovery changed the native camera grid")
        marker = self.native["v5_track1_photo_geometry"]
        self.original_asset = _asset(marker["original_source_manifest_path"], marker["original_source_manifest_sha256"])
        original = _read(self.original_asset["path"])
        if original["run"] != "ac59" or self.original_asset != recovery["original_source_manifest"]:
            raise ValueError("Recovery changed the original AC59 source binding")
        old_transport = marker["frames"][0]
        self.transport_asset = _asset(old_transport["receipt_path"], old_transport["receipt_sha256"])
        self.transport = _read(self.transport_asset["path"])
        if (self.transport["original_source_manifest_sha256"] != self.original_asset["sha256"]
                or self.transport["run"] != "ac59"):
            raise ValueError("Recovery changed the frozen AC59 baseline transport")
        self.frozen_b = {key: deepcopy(value) for key, value in self.transport.items() if key.startswith("v5_B_")}
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
        self.config_assets = [_asset(path) for path in original["correction_contract"]["config_paths"]]
        if self.config_assets != recovery["original_shape_configs"]:
            raise ValueError("Original AC59 shape configuration changed")
        if original["correction_contract"].get("rig_registration"):
            raise ValueError("Extra original registration needs its own explicit adapter")
        for item in self.config_assets:
            if "path" in tomllib.loads(Path(item["path"]).read_text(encoding="utf-8-sig")).get("rig", {}):
                raise ValueError("Use the original source adapter when its rig path is already explicit")
        project_config = _asset(recovery["project_run_config"]["path"], recovery["project_run_config"]["sha256"])
        configured = tomllib.loads(Path(project_config["path"]).read_text(encoding="utf-8-sig"))["rig"]["path"]
        if canonical_path(configured) != canonical_path(recovery["own_rig_path"]):
            raise ValueError("The migrated AC59 project does not identify this own rig")
        self.config_assets.append(project_config)
        self.shape_assets = [_asset(item["path"], item["sha256"]) for item in recovery["shape_corrections"]]
        actual = sorted(canonical_path(recovery["own_rig_path"]).glob("shape_correction_*.npz"))
        if [Path(item["path"]) for item in self.shape_assets] != actual or not actual:
            raise ValueError("Own persisted AC59 shape file inventory differs")
        self.shape_corrections = [darsia.read_correction(path) for path in actual]
        if any(not Rig._is_shape_correction(item) for item in self.shape_corrections):
            raise ValueError("A colour correction entered the recovered shape chain")
        self.shape_rig_path = str(canonical_path(recovery["own_rig_path"]))
        generated_b = self.rectify_camera(self.frozen_b["v5_B_camera_jpeg_path"])
        parity_h, parity = register_static_sand(self.baseline, generated_b, self.sand)
        samples = np.asarray([[[0,0],[5471,0],[2736,1490],[0,2978],[5471,2978]]], np.float32)
        displacements = np.linalg.norm(cv2.perspectiveTransform(samples, parity_h)-samples, axis=2)
        if float(displacements.max()) > 1.:
            raise ValueError("AC59's own persisted shape chain differs from its frozen B geometry")
        self.baseline_geometry_parity = dict(parity, H_frozen_B_to_shape_B=parity_h.tolist(),
            maximum_fivepoint_displacement_native_px=float(displacements.max()),
            required_maximum_native_px=1., colour_pixels_compared=False,
            own_persisted_shape_recovery=self.recovery_asset,
            recovery_adapter_sha256=file_sha256(__file__))

    def prepare(self, source_row, output):
        original = super().prepare(source_row, output)
        receipt = deepcopy(original)
        receipt.pop("geometry_certificate_path", None)
        receipt.pop("geometry_certificate_sha256", None)
        receipt["binding"]["own_persisted_shape_recovery"] = self.recovery_asset
        receipt["binding"]["recovery_adapter_sha256"] = file_sha256(__file__)
        receipt["source_geometry_recovery"] = dict(source_manifest_changed=False,
            parent_geometry_receipt=_asset(original["geometry_certificate_path"], original["geometry_certificate_sha256"]),
            borrowed_other_run_rig=False, model_or_labels_used=False,
            frozen_B_parity=self.baseline_geometry_parity)
        path = Path(original["geometry_certificate_path"]).with_name("own_shape_recovery.json")
        if path.exists():
            if _read(path) != receipt:
                raise ValueError("AC59 recovered geometry already has another immutable binding")
        else:
            _new_json(path, receipt)
        return dict(receipt, geometry_certificate_path=str(path), geometry_certificate_sha256=file_sha256(path))


__all__ = ["AC59OwnPersistedSourceGeometry", "validate_recovery_identity", "CONTRACT"]
