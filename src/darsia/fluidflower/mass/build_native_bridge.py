"""Bind a completed Build grid to original RAW photometry, before phase/optics.

Camera-U is the calibrated camera-equivalent sampling grid, not the rectified
rig grid. The saved Build grid and the directly sampled RAW native grid remain
separate: their RGB pixels are not claimed equal. No phase, physical-domain or
training eligibility is inferred here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .native_optical_source import _bound, _same, load_build_physical_context
from .n1_pipeline import DataReader, _stamp, _write_json, _write_npz, asset
from ..job_filesystem import inspect_local_file
from ..native import raw_positioning, raw_rig_source, registration, source
from ..rig_raw_build import _require_quality
from ..rig_raw_write import _no_links


def _fresh(path, forbidden):
    """Reserve one positively local output directory without replacing files."""
    path = Path(path)
    if (not path.is_absolute() or ".." in path.parts or path.exists()
            or not inspect_local_file(path / ".native-bridge-location").allowed):
        raise ValueError("A fresh absolute local native-bridge directory is required")
    _no_links(path.parent)
    actual = path.resolve()
    if any(actual == root or actual.is_relative_to(root) or root.is_relative_to(actual)
           for root in map(lambda value: Path(value).resolve(), forbidden)):
        raise ValueError("Native-bridge output overlaps a protected RAW or saved Build root")
    path.mkdir()
    return path


@dataclass(frozen=True)
class BuildNativeFrame:
    """Actual corrected RAW buffers and original runner inputs, without P1 claims."""
    native_task: dict
    products: dict
    receipt: dict


class _Inputs:
    """Retain consumed producer descriptors for cached late-identity checks."""
    def __init__(self, reader):
        self.reader, self.rows = reader, {}

    def path(self, bound):
        path = self.reader.path(bound)
        self.rows[(str(bound["path"]), bound["sha256"])] = deepcopy(bound)
        return path

    def json(self, bound):
        self.path(bound)
        return self.reader.json(bound)

    def revalidate(self):
        for row in tuple(self.rows.values()):
            self.reader.path(row)


class BuildNativeBridge:
    """Produce original RAW-native inputs joined to an exact completed Build.

    The caller supplies a current, independently bound frozen-photometry source
    binding. Its models are preserved as their own inputs, never relabelled as
    Build products. All output directories must be fresh and positively local;
    a failed preparation is preserved and cannot be reused.
    """

    def __init__(self, operation_result, photometry_binding, output, *, reader=None):
        self.producer_stamp = _stamp(Path(__file__))
        self.producer = asset(Path(__file__))
        self.reader = _Inputs(DataReader() if reader is None else reader)
        self.context = load_build_physical_context(operation_result, reader=self.reader)
        self.operation_result = deepcopy(_bound(operation_result))
        self.native = deepcopy(self.context.provenance["native_configuration"])
        self.parameters = self._document(self.native)
        original = self._document(photometry_binding)
        self.photometry_binding = deepcopy(photometry_binding)
        camera = self.context.provenance["camera_binding"]
        baseline = self.context.provenance["raw_acquisition"]
        if (original["run"] != self.context.run or camera["run_id"] != self.context.run
                or original["profile_id"] != camera["profile_id"]
                or original["baseline_raw"]["sha256"] != baseline["sha256"]
                or list(original["native_shape_hw"]) != self.context.grid["num_voxels"]):
            raise ValueError("RAW photometry and completed Build differ in run/profile/baseline/grid")
        certificate = self._document(self.context.provenance["baseline_renderer"])
        request = self._document(self.context.provenance["raw_build_request"])
        self.forbidden = [*request["output"]["forbidden_raw_roots"],
                          str(Path(certificate["corrected_baseline"]["archive"]["path"]).parent)]
        renderer = certificate["renderer"]
        common = self.parameters["registration"]["common"]
        self.policies = common["review"]
        self.build_h = self._matrix(certificate["H_rig_corrected_native_to_baseline_camera_u"])
        if (not _same(renderer["native_to_camera_u_h"], self.build_h.tolist())
                or renderer["profile_id"] != camera["profile_id"]
                or renderer["camera_u_wh"] != common["sensor"]["camera_wh"]
                or renderer["saved_shape_chain_replay_bit_identical"] is not True
                or not _same(renderer["source_npz"],
                             self.context.provenance["physical_archives"]["shape_corrected_baseline"])):
            raise ValueError("The saved Build lacks its exact camera-U registration and replay")
        _require_quality(renderer["registration"], self.policies)
        assets = renderer["saved_shape_chain_replay"]["correction_assets"]
        if not assets or not _same(assets, renderer["shape_corrections"]):
            raise ValueError("The Build lacks its ordered saved shape assets")
        owned = {row["path"]: row for row in certificate["outputs"]}
        for row in assets:
            if not _same(owned.get(row["path"]), row):
                raise ValueError("A shape asset is not an exact owned Build output")
            self._pin(row)
        old = raw_rig_source.RawRigSourceProcessor(
            self.reader.path(photometry_binding), parameters=self.parameters)
        _require_quality(old.baseline_renderer["registration"], self.policies)
        for row in [original["baseline_raw"], *original["shape_assets"],
                    *(item for model in original["models"].values() if model is not None
                      for item in (model.values() if "data" in model else (model,)))]:
            self._pin(row, resolve=old._path)
        white_range = old.baseline_sensor.metadata["white_level"] - old.baseline_sensor.metadata["black_level"]
        normalization = renderer["saved_shape_chain_replay"]["normalization"]
        if (not np.isfinite(white_range) or white_range <= 0
                or normalization != dict(black_subtracted_dn_divisor=float(white_range),
                                         white_balance_applied=False, gamma_applied=False)
                or old.baseline_renderer["camera_u_wh"] != renderer["camera_u_wh"]):
            raise ValueError("The Build and RAW source do not share the exact calibrated camera-U basis")
        self.review_green = np.asarray(self.context.baseline_rgb[..., 1], np.float32) * np.float32(white_range)
        if not np.isfinite(self.review_green).all():
            raise ValueError("The saved Build green registration input is nonfinite")
        metadata = deepcopy(certificate["inputs"]["uncorrected_grid"]["metadata"])
        metadata.pop("type", None)
        metadata.pop("color_space", None)  # native_review_rgb supplies RGB explicitly.
        if metadata.get("time") is None and isinstance(metadata.get("reference_date"), str):
            raise ValueError("The original RAW binding needs explicit time with serialized Build dates")
        runtime_metadata = deepcopy(metadata)
        for name in ("date", "reference_date"):
            if isinstance(runtime_metadata.get(name), str):
                runtime_metadata[name] = datetime.fromisoformat(runtime_metadata[name])
        _, fresh_renderer = raw_positioning.render_native(old.baseline_sensor,
            shape_assets=assets, shape_hw=original["native_shape_hw"],
            image_metadata=runtime_metadata, parameters=self.parameters["registration"])
        _require_quality(fresh_renderer["registration"], self.policies)
        hn = self._matrix(fresh_renderer["native_to_camera_u_h"])
        ho = self._matrix(old.baseline_renderer["native_to_camera_u_h"])
        frozen_reference = self._matrix(original["baseline_reference_to_native_h"])
        reference = self._matrix(np.linalg.inv(hn) @ ho @ frozen_reference)
        self.build_to_baseline = self._matrix(np.linalg.inv(hn) @ self.build_h)
        binding = deepcopy(original)
        binding.update(baseline_raw=deepcopy(baseline), shape_assets=deepcopy(assets),
                       image_metadata=metadata, baseline_reference_to_native_h=reference.tolist())
        raw_rig_source.validate_binding(binding, self.parameters)
        self._revalidate()
        self.output = _fresh(output, self.forbidden)
        self.binding = _write_json(self.output / "raw_source_binding.json", binding)
        self.processor = raw_rig_source.RawRigSourceProcessor(self.binding["path"], parameters=self.parameters)
        if not source._same_h(hn, self._matrix(self.processor.baseline_renderer["native_to_camera_u_h"]),
                              parameters=self.parameters):
            raise ValueError("The newly bound RAW baseline registration changed")
        self.reference = reference
        self.baseline_sha = baseline["sha256"]
        for value in (self.review_green, self.build_h, self.build_to_baseline, self.reference):
            value.setflags(write=False)
        self._revalidate()
        self.proof = _write_json(self.output / "bridge.json", dict(
            format="darsia-build-native-bridge-v1", operation_result=self.operation_result,
            photometry_source=photometry_binding, native_configuration=self.native,
            raw_source_binding=self.binding, build_to_baseline_native_h=self.build_to_baseline.tolist(),
            frozen_reference_to_baseline_native_h=reference.tolist(),
            producer=dict(callable="darsia.fluidflower.mass.build_native_bridge.BuildNativeBridge",
                          module=self.producer, raw_execution_identity=raw_rig_source.execution_identity()),
            saved_build_to_camera_u_h=self.build_h.tolist(),
            old_raw_baseline_renderer=old.baseline_renderer,
            new_raw_baseline_renderer=self.processor.baseline_renderer,
            input_scope="original RAW photometry and saved Build coordinate join; no phase/domain qualification"))

    def _matrix(self, value):
        return source.source_homography(value, parameters=self.parameters)

    def _pin(self, bound, *, resolve=None):
        """Track the actual consumed bytes, retaining the original declared SHA."""
        path = Path(bound["path"]) if resolve is None else resolve(bound["path"])
        row = dict(path=str(path), sha256=bound["sha256"],
                   size_bytes=bound.get("size_bytes", path.stat().st_size))
        self.reader.path(_bound(row))
        return row

    def _document(self, bound):
        return self.reader.json(_bound(bound))

    def _revalidate(self):
        # Includes every original physical archive, request, owner and plan consumed
        # by the Build reader, without decoding the unchanged archives again.
        if _stamp(Path(__file__)) != self.producer_stamp:
            raise ValueError("The native-bridge producer changed after preparation started")
        self.reader.revalidate()

    def frame(self, acquisition, output):
        """Measure this acquisition's geometry and run the unchanged RAW correction.

        No previous/phase state is invented. The returned native_task is an
        original raw_rig task; phase geometry/history must still be supplied by
        the chronological collector under its own original guards.
        """
        acquisition = self._pin(acquisition)
        self.reader.path(self.binding)
        self.reader.path(self.proof)
        self._revalidate()
        exact_baseline = acquisition["sha256"] == self.baseline_sha
        if exact_baseline:
            baseline_to_frame = np.eye(3)
            build_to_frame = self.build_to_baseline
            measured = dict(exact_baseline=True, registration="original baseline identity")
        else:
            sensor = self.processor._sensor(acquisition)
            view = raw_positioning.camera_u(sensor, parameters=self.parameters["registration"])
            j, quality = registration.review_to_camera(self.review_green, view[1],
                                                       parameters=self.parameters["registration"]["common"])
            _require_quality(quality, self.policies)
            _, renderer = self.processor._render(sensor)
            _require_quality(renderer["registration"], self.policies)
            k = self._matrix(renderer["native_to_camera_u_h"])
            build_to_frame = self._matrix(np.linalg.inv(k) @ self._matrix(j))
            baseline_to_frame = self._matrix(build_to_frame @ np.linalg.inv(self.build_to_baseline))
            measured = dict(exact_baseline=False, build_to_frame_camera_u_h=self._matrix(j).tolist(),
                            registration=quality, frame_native_renderer=renderer)
        geometry = dict(contract=raw_rig_source.FRAME_CONTRACT, run=self.context.run,
            acquisition=acquisition, baseline_to_native_h=baseline_to_frame.tolist(),
            reference_to_native_h=self._matrix(baseline_to_frame @ self.reference).tolist())
        raw_rig_source.validate_frame(geometry, self.processor.binding, self.parameters)
        log, audit = self.processor.correct(geometry)
        expected_h = self._matrix(self.processor.baseline_renderer["native_to_camera_u_h"] if exact_baseline
                                  else measured["frame_native_renderer"]["native_to_camera_u_h"])
        if not source._same_h(expected_h, self._matrix(audit["renderer"]["native_to_camera_u_h"]),
                              parameters=self.parameters):
            raise ValueError("The frame RAW registration changed before correction completed")
        if log.dtype != np.float32 or log.shape != (3, *self.context.grid["num_voxels"]):
            raise ValueError("Original RAW correction returned another native grid or dtype")
        valid = np.isfinite(log).all(0)
        if not valid.any():
            raise ValueError("Original RAW correction has no finite native support")
        self._revalidate()
        target = _fresh(output, self.forbidden)
        frame_binding = _write_json(target / "raw_native_frame.json", geometry)
        products = _write_npz(target / "native.npz", dict(log_linear_rgb=log, valid_mask=valid))
        self._revalidate()
        receipt = _write_json(target / "frame.json", dict(
            format="darsia-build-native-frame-v1", bridge=self.proof,
            raw_source_binding=self.binding, native_frame=frame_binding, products=products,
            build_to_frame_native_h=build_to_frame.tolist(), registration=measured,
            correction=audit, physical_context=self.context.provenance,
            scope="corrected native RAW buffers and coordinate join only; phase/history/domain remain separate"))
        return BuildNativeFrame(dict(kind="raw_rig", geometry=frame_binding,
            processor=dict(raw_binding=deepcopy(self.binding))), products, receipt)
