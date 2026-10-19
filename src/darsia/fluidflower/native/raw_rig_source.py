"""RAW-only acquisition bindings on the existing calibrated native grid.

This producer loads the same frozen photometry and geometry as the historical
producer. It never opens a companion photograph or retains pixels from the
JPEG-positioned reference. New RAW references and inference remain separate
validation gates.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
import json
import re

import numpy as np

from . import extension, photometry, raw_chart, raw_positioning, raw_positioning_port
from . import registration, source
from .io import resolve_asset_path, sha256_file


CONTRACT = "darsia-raw-rig-source-v1"
FRAME_CONTRACT = "darsia-raw-native-frame-v1"


def execution_identity() -> dict:
    """Bind the actual implementation, including unchanged numerical helpers."""
    modules = dict(raw_source=Path(__file__), raw_positioning=Path(raw_positioning.__file__),
                   raw_port=Path(raw_positioning_port.__file__), raw_chart=Path(raw_chart.__file__),
                   registration=Path(registration.__file__), photometry=Path(photometry.__file__),
                   source_helpers=Path(source.__file__), extension=Path(extension.__file__))
    return {name: sha256_file(path) for name, path in modules.items()}


def _keys(value, required, *, where):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError("Explicit RAW-only fields required for " + where)
    return value


def validate_binding(binding: Mapping, parameters: Mapping) -> None:
    """Validate data roles before any camera/model file is opened."""
    _keys(binding, {"contract", "run", "profile_id", "native_shape_hw", "baseline_raw",
                    "baseline_reference_to_native_h", "shape_assets", "image_metadata",
                    "models", "execution_identity"}, where="source binding")
    if binding["contract"] != CONTRACT:
        raise ValueError("Unknown RAW source contract")
    if (not isinstance(binding["run"], str)
            or re.fullmatch(parameters["source"]["source_id_pattern"], binding["run"]) is None):
        raise ValueError("RAW source requires its exact run identity")
    expected_profile = parameters["source"]["profiles"][binding["run"]]
    if binding["profile_id"] != expected_profile:
        raise ValueError("RAW source camera profile differs from its run")
    if tuple(binding["native_shape_hw"]) != tuple(parameters["source"]["native_shape_hw"]):
        raise ValueError("RAW source differs from the unchanged native grid")
    source.source_homography(binding["baseline_reference_to_native_h"], parameters=parameters)
    if not isinstance(binding["shape_assets"], list) or not binding["shape_assets"]:
        raise ValueError("RAW source needs the existing shape correction assets")
    models = _keys(binding["models"], {"fit", "reference", "old_field"}, where="frozen models")
    for name in ("fit", "reference"):
        _keys(models[name], {"data", "sidecar"}, where=name + " model")
    source.verify_execution_identity(execution_identity(), binding["execution_identity"],
                                     tuple(execution_identity()))


def validate_frame(row: Mapping, binding: Mapping, parameters: Mapping) -> tuple:
    _keys(row, {"contract", "run", "acquisition", "reference_to_native_h",
                "baseline_to_native_h"}, where="native frame")
    if row["contract"] != FRAME_CONTRACT or row["run"] != binding["run"]:
        raise ValueError("RAW frame differs from its source contract/run")
    h_ref = source.source_homography(row["reference_to_native_h"], parameters=parameters)
    h_b = source.source_homography(row["baseline_to_native_h"], parameters=parameters)
    h_ref_b = source.source_homography(np.linalg.inv(h_b) @ h_ref, parameters=parameters)
    expected = source.source_homography(binding["baseline_reference_to_native_h"], parameters=parameters)
    if not source._same_h(h_ref_b, expected, parameters=parameters):
        raise ValueError("RAW frame changes the frozen reference-to-baseline geometry")
    return h_ref, h_b, h_ref_b


class RawRigSourceProcessor:
    """Correct RAW acquisitions with unchanged fits, grids and field arithmetic."""

    def __init__(self, binding_path, *, parameters: Mapping):
        from darsia._opencv_runtime import scientific_thread
        scientific_thread("RawRigSourceProcessor:setup")
        self.parameters = parameters
        self.binding_path = self._path(binding_path)
        self.binding_sha256 = sha256_file(self.binding_path)
        self.binding = json.loads(self.binding_path.read_text(encoding="utf-8-sig"))
        validate_binding(self.binding, parameters)
        b = self.binding
        # Both the NPZ and the exact sidecar are inputs. An internally
        # consistent replacement pair cannot substitute for the frozen model.
        loaded = {}
        for name in ("fit", "reference"):
            model = b["models"][name]
            data, sidecar = self._asset(model["data"]), self._asset(model["sidecar"])
            if sidecar != data.with_suffix(".json"):
                raise ValueError("RAW model sidecar differs from the loader's actual input")
            loaded[name] = photometry.load_baseline(data, parameters=parameters["photometry"])
        fit, grid = loaded["fit"]
        reference, ref_grid = loaded["reference"]
        if fit.run != b["run"] or reference.run != "virtual_reference" or grid != ref_grid:
            raise ValueError("RAW source and frozen photometric models/grid differ")
        fit_receipt = json.loads(self._asset(b["models"]["fit"]["sidecar"]).read_text(encoding="utf8"))
        if fit_receipt["source"]["source_arw_sha256"] != b["baseline_raw"]["sha256"]:
            raise ValueError("RAW baseline acquisition differs from the unchanged B fit")
        old_field = None
        if b["models"]["old_field"] is not None:
            with np.load(self._asset(b["models"]["old_field"]), allow_pickle=False) as archive:
                old_field = archive["field"]
            if old_field.dtype != np.float32 or old_field.shape != grid.shape_hw:
                raise ValueError("RAW source old-light field differs from its frozen grid/type")
        self.processor = SimpleNamespace(run=b["run"], fit=fit, reference=reference,
                                         grid=grid, old_field=old_field)
        self.shape_assets = [dict(item, path=str(self._asset(item))) for item in b["shape_assets"]]
        self.shape_chain = raw_positioning.load_shape_chain(self.shape_assets)
        self.image_metadata = dict(b["image_metadata"])
        if isinstance(self.image_metadata.get("date"), str):
            self.image_metadata["date"] = datetime.fromisoformat(self.image_metadata["date"])
        self.baseline_sensor = self._sensor(b["baseline_raw"])
        self.baseline_camera_u = raw_positioning.camera_u(self.baseline_sensor, parameters=parameters["registration"])
        _, self.baseline_renderer = self._render(self.baseline_sensor)
        self.identity = dict(contract=CONTRACT, source_binding=dict(path=str(self.binding_path),
                             sha256=self.binding_sha256), execution_identity=execution_identity(),
                             models=b["models"], profile_id=b["profile_id"],
                             native_shape_hw=b["native_shape_hw"], cpu_only=True,
                             training=False, new_photometric_fit=False, jpeg_inputs=False)

    def _path(self, value):
        return resolve_asset_path(value, self.parameters["source"]["asset_aliases"])

    def _asset(self, descriptor):
        if (not isinstance(descriptor, dict) or set(descriptor) - {"path", "sha256", "size_bytes"}
                or not {"path", "sha256"} <= set(descriptor)):
            raise ValueError("Exact RAW asset descriptor required")
        path = self._path(descriptor["path"])
        digest = descriptor["sha256"]
        if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("Exact RAW asset SHA-256 required")
        if (("size_bytes" in descriptor and path.stat().st_size != descriptor["size_bytes"])
                or sha256_file(path) != digest):
            raise ValueError("RAW source/model input differs from its binding: " + str(path))
        return path

    def _sensor(self, descriptor):
        return raw_positioning.read_sensor(self._asset(descriptor),
                    expected_sha256=descriptor["sha256"], parameters=self.parameters["registration"],
                    profile_id=self.binding["profile_id"])

    def _render(self, sensor):
        linear, audit = raw_positioning.render_native(sensor, shape_assets=self.shape_assets,
                    shape_hw=self.binding["native_shape_hw"], image_metadata=self.image_metadata,
                    parameters=self.parameters["registration"])
        audit.update(source_id=self.processor.run, run=self.processor.run,
                     profile=sensor.profile_id, sources=dict(arw_sha256=sensor.source["sha256"]),
                     white_level=sensor.metadata["white_level"],
                     review_to_camera_h=audit["native_to_camera_u_h"],
                     camera_jpeg_wh=audit["camera_u_wh"], raw_visible_wh=sensor.metadata["raw_visible_wh"],
                     sony_distortion_table=list(sensor.distortion), sony_ca_red_table=list(sensor.ca_red),
                     sony_ca_blue_table=list(sensor.ca_blue), legacy_numeric_grid_names_only=True)
        return linear, audit

    def correct(self, geometry_frame, *, parent_frame_index=None, parent_reference=False):
        from darsia._opencv_runtime import scientific_thread, completed_frame
        before = scientific_thread("RawRigSourceProcessor.correct:before")
        if parent_frame_index is not None or parent_reference:
            raise ValueError("JPEG-positioned parent pixels cannot be reused by the RAW-only source")
        if sha256_file(self.binding_path) != self.binding_sha256:
            raise ValueError("RAW source binding changed after processor setup")
        source.verify_execution_identity(execution_identity(), self.binding["execution_identity"],
                                         tuple(execution_identity()))
        h_ref, h_b, h_ref_b = validate_frame(geometry_frame, self.binding, self.parameters)
        sensor = self._sensor(geometry_frame["acquisition"])
        exact_b = sensor.source["sha256"] == self.baseline_sensor.source["sha256"]
        if exact_b and not source._same_h(h_b, np.eye(3), parameters=self.parameters):
            raise ValueError("Exact RAW B acquisition has a nonidentity baseline transform")
        linear, renderer = self._render(sensor)
        xy = usable = chart = None
        p = self.processor
        if not exact_b:
            xy, usable, chart = raw_chart.register(
                baseline_sensor=self.baseline_sensor, frame_sensor=sensor,
                baseline_camera_u=self.baseline_camera_u,
                frame_camera_u=raw_positioning.camera_u(sensor, parameters=self.parameters["registration"]),
                baseline_chart_xy8=source.chart_baseline_coordinates(p.fit, p.grid, h_ref_b, parameters=self.parameters),
                baseline_chart_log_rgb=p.fit.CB,
                baseline_native_to_u_h=np.asarray(self.baseline_renderer["native_to_camera_u_h"]),
                frame_native_to_u_h=np.asarray(renderer["native_to_camera_u_h"]),
                frame_native_linear_rgb=linear, parameters=self.parameters["registration"])
        corrected, info = photometry.correct_frame_native(
            linear, p.fit, p.reference, p.grid, h_ref,
            np.ones(self.binding["native_shape_hw"], bool), exact_baseline_source=exact_b,
            frame_chart_xy8=xy, frame_chart_usable=usable, old_field_ref8=p.old_field,
            parameters=self.parameters["photometry"])
        return corrected, dict(contract=CONTRACT, processor=self.identity,
            sources=dict(camera_arw=dict(sensor.source)), renderer=renderer, chart=chart,
            photometry=info, exact_B=exact_b,
            **{self.parameters["source"]["symbols"]["reference_to_frame_h_key"]: h_ref.tolist()},
            H_v5_B_to_frame_native=h_b.tolist(), source_identity=dict(kind="raw-acquisition",
            raw_sha256=sensor.source["sha256"]), parent_cache=None, cpu_only=True, training=False,
            jpeg_inputs=False, opencv_ipp=completed_frame(before, "RawRigSourceProcessor.correct:after"))
