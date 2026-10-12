"""Frozen V4-R5 full-native RAW radiometry for the explicit rig revision.

Old sidecars deliberately excluded the conservative photo-sand border. This
module decodes the actual ARW to recover scientific values in observed sand;
photographic JPEGs are never inverted into scientific model input. It creates
new artifacts and retains the original finite scientific pixels exactly.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import re

import numpy as np

from .project import sha256_file

CONTRACT = "ffac-rig-full-native-frozen-raw-source-v1"
SHAPE_HW = (2979, 5472)
BOX_XYWH = (416, 496, 5472, 2979)


def read_bound(path, expected):
    path = Path(path)
    if not expected or sha256_file(path) != expected:
        raise ValueError("Source artifact changed: " + str(path))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_calibration_baselines(calibration_manifest_path, expected_manifest_sha256, run):
    """Bind both loaded fits and their sidecars to the sealed calibration.

    The upstream fit loader checks a NPZ against its sidecar. This additional
    check prevents a mutually consistent, but changed, NPZ/sidecar pair from
    silently replacing the frozen run or virtual-reference parameters.
    """
    manifest_path = Path(calibration_manifest_path).resolve()
    calibration = read_bound(manifest_path, expected_manifest_sha256)
    if not isinstance(run, str) or re.fullmatch(r"ac[0-9]{2}", run) is None:
        raise ValueError("Frozen calibration requires an exact FF_AC run name")
    rows = calibration.get("results")
    if not isinstance(rows, list):
        raise ValueError("Frozen calibration has no parameter asset inventory")
    assets = []
    for identity in (run, "virtual_reference"):
        matching = [row for row in rows if isinstance(row, dict) and row.get("run") == identity]
        if len(matching) != 1:
            raise ValueError("Frozen calibration needs one parameter binding for " + identity)
        row = matching[0]
        expected_path = (manifest_path.parent / (identity + "_baseline.npz")).resolve()
        if not row.get("baseline_path"):
            raise ValueError("Frozen calibration lacks its recorded parameter path")
        declared = Path(row["baseline_path"])
        # Historical receipts express paths relative to the workspace. Resolve
        # only against this manifest's ancestors, and require the exact file
        # that the authoritative processor loads; never search other folders.
        candidates = ([declared.resolve()] if declared.is_absolute() else
                      [(base / declared).resolve() for base in (manifest_path.parent, *manifest_path.parent.parents)])
        if expected_path not in candidates:
            raise ValueError("Frozen calibration parameter path differs from the loaded fit")
        for kind, path, key in (("baseline", expected_path, "baseline_sha256"),
                                ("sidecar", expected_path.with_suffix(".json"), "sidecar_sha256")):
            digest = row.get(key)
            if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    or not path.is_file() or sha256_file(path) != digest):
                raise ValueError("Frozen calibration parameter asset changed: " + str(path))
            assets.append(dict(run=identity, kind=kind, path=str(path), sha256=digest))
    return dict(contract="ffac-frozen-calibration-baseline-assets-v1",
                calibration_manifest_path=str(manifest_path),
                calibration_manifest_sha256=expected_manifest_sha256,
                run=run, assets=assets, verified=True)


def retain_parent_scientific_pixels(full, parent, parent_valid):
    """Explicit exact cache reuse, including the previous isolated-pixel repair."""
    full, parent, valid = np.asarray(full), np.asarray(parent), np.asarray(parent_valid)
    if (full.dtype != np.float32 or parent.dtype != np.float32 or
            full.shape != parent.shape or full.shape[0] != 3 or
            valid.dtype != bool or valid.shape != full.shape[1:] or
            not np.isfinite(parent[:, valid]).all()):
        raise ValueError("Parent scientific cache does not share exact native float geometry")
    freshly_finite = np.isfinite(full).all(0) & valid
    differences = np.abs(full[:, freshly_finite].astype(np.float64) - parent[:, freshly_finite])
    audit = dict(parent_supported_pixels=int(valid.sum()),
                 freshly_decoded_max_abs_log_difference=float(np.max(differences, initial=0)),
                 freshly_invalid_parent_supported_pixels=int(np.count_nonzero(valid & ~freshly_finite)),
                 parent_float_bytes_preserved=True)
    full[:, valid] = parent[:, valid]
    return full, audit


class RigRawSourceProcessor:
    def __init__(self, native_manifest_path):
        from .tools.ffac_track1_v4r5_source_processor import Track1V4R5SourceProcessor
        self.native_path = Path(native_manifest_path)
        self.native = json.loads(self.native_path.read_text(encoding="utf-8-sig"))
        binding = self.native["v5_photometry"]
        self.source_path = Path(binding["source_manifest"])
        read_bound(self.source_path, binding["source_manifest_sha256"])
        processor = binding["processor"]
        for path_key, sha_key in (("calibration_manifest_path", "calibration_manifest_sha256"),
                                  ("photo_source_aggregate_path", "photo_source_aggregate_sha256")):
            read_bound(processor[path_key], processor[sha_key])
        self.calibration_baseline_binding = verify_calibration_baselines(
            processor["calibration_manifest_path"], processor["calibration_manifest_sha256"],
            self.native["run"])
        aggregate = read_bound(processor["photo_source_aggregate_path"],processor["photo_source_aggregate_sha256"])
        matching = [row for row in aggregate["rows"] if row["run"] == self.native["run"]
                    and row["source_manifest_sha256"] == binding["source_manifest_sha256"]]
        if len(matching) != 1:
            raise ValueError("Frozen photo transport does not bind this exact source version")
        self.processor = Track1V4R5SourceProcessor(
            matching[0],
            calibration_manifest_path=Path(processor["calibration_manifest_path"]),
            photo_aggregate_path=Path(processor["photo_source_aggregate_path"]),
            enclosed_hole_policy="fill_enclosed_v1", display_mode="natural_modes_v1",
            isolated_invalid_policy="fill_isolated_invalid_v1")
        for key in ("processor_module_sha256", "renderer_module_sha256",
                    "photometry_module_sha256", "pilot_math_module_sha256",
                    "isolated_invalid_module_sha256"):
            if self.processor.identity().get(key) != processor.get(key):
                raise ValueError("Live frozen RAW preprocessing code differs from parent: " + key)
        self.identity = dict(contract=CONTRACT,
            parent_native_manifest_path=str(self.native_path),
            parent_native_manifest_sha256=sha256_file(self.native_path),
            frozen_photometry=self.processor.identity(),
            calibration_baseline_binding=self.calibration_baseline_binding,
            full_native_shape_hw=list(SHAPE_HW), code_sha256=sha256_file(__file__),
            cpu_only=True, training=False, photographic_input=False,
            mass_eligible=False)

    def parent_geometry(self, index=None):
        # index=None requests the real pre-injection B, not timeline frame 0.
        row = dict(self.processor.cohort.frame(0 if index is None else index))
        if index is None:
            row.update(camera_jpeg_path=row["v5_B_camera_jpeg_path"],
                camera_jpeg_sha256=row["v5_B_camera_jpeg_sha256"],
                native_review_jpeg_path=row["v5_B_native_jpeg_path"],
                native_review_jpeg_sha256=row["v5_B_native_jpeg_sha256"],
                camera_arw_path=row["v5_B_arw_path"],
                camera_arw_sha256=row["v5_B_arw_sha256"])
            h = np.linalg.inv(np.asarray(row["H_v5_B_to_frame_native"])) @ np.asarray(row["H_ac28_to_frame_native"])
            row["H_ac28_to_frame_native"] = (h/h[2,2]).tolist()
            row["H_v5_B_to_frame_native"] = np.eye(3).tolist()
        else:
            frame = self.native["frame_sources"][index]
            row["camera_arw_path"], row["camera_arw_sha256"] = frame["v5_raw_path"], frame["v5_raw_sha256"]
        return row

    def correct(self, geometry_frame, *, parent_frame_index=None, parent_reference=False):
        from .tools.ffac_raw_native_ea import registered_edge_aware_raw_linear
        from .tools.ffac_v4r5_native_pilot import correct_frame_native_v4r5
        from .tools.ffac_v5_chart_registration import register_frame_chart_xy8
        from .tools.ffac_track1_v4r5_source_processor import _chart_xy_baseline_native, _h

        row, p = geometry_frame, self.processor
        if parent_reference and parent_frame_index is not None:
            raise ValueError("Reference and timeline-frame reuse are distinct source bindings")
        required = ("camera_jpeg", "camera_arw", "native_review_jpeg")
        sources = {}
        for name in required:
            path, expected = Path(row[name+"_path"]), row[name+"_sha256"]
            if sha256_file(path) != expected:
                raise ValueError("New rig RAW input differs from its source receipt: " + name)
            sources[name] = dict(path=str(path), sha256=expected)
        h_ref = _h(row["H_ac28_to_frame_native"])
        h_b = _h(row["H_v5_B_to_frame_native"])
        h_ref_b = _h(np.linalg.inv(h_b) @ h_ref)
        linear, renderer = registered_edge_aware_raw_linear(
            sources["camera_arw"]["path"], sources["camera_jpeg"]["path"],
            sources["native_review_jpeg"]["path"], content_box=BOX_XYWH,
            expected_source_hashes=dict(arw_sha256=sources["camera_arw"]["sha256"],
                camera_jpeg_sha256=sources["camera_jpeg"]["sha256"],
                review_native_jpeg_sha256=sources["native_review_jpeg"]["sha256"]))
        exact_b = sources["camera_arw"]["sha256"] == row["v5_B_arw_sha256"] and sources["camera_jpeg"]["sha256"] == row["v5_B_camera_jpeg_sha256"]
        xy = usable = chart = None
        if not exact_b:
            certificate_path = Path(row["v5_B_registration_certificate_path"])
            certificate = json.loads(certificate_path.read_text(encoding="utf-8-sig"))
            if certificate["sources"]["review_native_jpeg_sha256"] != row["v5_B_native_jpeg_sha256"]:
                raise ValueError("Frozen B chart registration belongs to another reference")
            xy, usable, chart = register_frame_chart_xy8(
                baseline_camera_jpeg=Path(row["v5_B_camera_jpeg_path"]),
                baseline_camera_sha256=row["v5_B_camera_jpeg_sha256"],
                frame_camera_jpeg=Path(sources["camera_jpeg"]["path"]),
                frame_camera_sha256=sources["camera_jpeg"]["sha256"],
                baseline_chart_xy8=_chart_xy_baseline_native(p.fit,p.grid,h_ref_b),
                baseline_chart_log_rgb=p.fit.CB,
                baseline_review_to_camera_h=np.asarray(certificate["review_to_camera_h"]),
                frame_review_to_camera_h=np.asarray(renderer["review_to_camera_h"]),
                frame_native_linear_rgb=linear)
        corrected, info = correct_frame_native_v4r5(linear,p.fit,p.reference,p.grid,h_ref,
            np.ones(SHAPE_HW,bool), exact_baseline_source=exact_b,
            frame_chart_xy8=xy,frame_chart_usable=usable,old_field_ref8=p.old_field)
        del linear
        cache_audit = None
        source_identity = dict(kind="same-stem-camera-acquisition",
                               camera_member_sha256=sources["camera_jpeg"]["sha256"])
        if parent_frame_index is not None:
            old = self.native["frame_sources"][parent_frame_index]
            if old["source_sha256"] != sources["camera_jpeg"]["sha256"]:
                # Preserve the one already documented prepared-source binding.
                # Rehash it through the original cohort; do not turn its label
                # identity into the member JPEG/ARW identity used by the frozen
                # parent renderer. No other mismatched source is admitted.
                if p.run != "ac61" or parent_frame_index != 0:
                    raise ValueError("Cannot reuse parent pixels for a different acquisition")
                original = p.source["frame_sources"][0]
                transport = p.cohort.frame(0)
                if (original["source_sha256"] != old["source_sha256"] or
                        sha256_file(original["source_path"]) != original["source_sha256"] or
                        transport["camera_jpeg_sha256"] != sources["camera_jpeg"]["sha256"] or
                        old["v5_raw_sha256"] != sources["camera_arw"]["sha256"]):
                    raise ValueError("AC61 prepared source/member binding differs from the frozen parent")
                source_identity = dict(kind="historical-three-ARW-prepared-source",
                    prepared_source_path=original["source_path"],prepared_source_sha256=original["source_sha256"],
                    camera_member=sources["camera_jpeg"],raw_member=sources["camera_arw"],
                    original_transport_receipt_sha256=transport["receipt_sha256"],
                    label_identity_preserved=True,prepared_source_reinterpreted=False,
                    new_radiometry="same SHA-bound member renderer as the frozen parent; no new composite reconstruction")
            for path_key,sha_key in (("v5_log_linear_rgb_path","v5_log_linear_rgb_sha256"),
                                     ("v5_log_linear_valid_mask_path","v5_log_linear_valid_mask_sha256")):
                if sha256_file(old[path_key]) != old[sha_key]:
                    raise ValueError("Parent float cache changed")
            parent = np.load(old["v5_log_linear_rgb_path"],mmap_mode="r",allow_pickle=False)
            valid = np.load(old["v5_log_linear_valid_mask_path"],mmap_mode="r",allow_pickle=False)
            corrected, cache_audit = retain_parent_scientific_pixels(corrected,parent,valid)
        elif parent_reference:
            reference = self.native["reference"]
            state = reference["v5_reference_state"]
            if (not exact_b or state["camera_jpeg_sha256"] != sources["camera_jpeg"]["sha256"] or
                    state["camera_arw_sha256"] != sources["camera_arw"]["sha256"]):
                raise ValueError("Pre-injection reuse differs from its original reference acquisition")
            for path_key,sha_key in (("v5_log_linear_rgb_path","v5_log_linear_rgb_sha256"),
                                     ("v5_log_linear_valid_mask_path","v5_log_linear_valid_mask_sha256"),
                                     ("v5_baseline_receipt_path","v5_baseline_receipt_sha256")):
                if sha256_file(reference[path_key]) != reference[sha_key]:
                    raise ValueError("Parent pre-injection float reference changed")
            parent = np.load(reference["v5_log_linear_rgb_path"],mmap_mode="r",allow_pickle=False)
            valid = np.load(reference["v5_log_linear_valid_mask_path"],mmap_mode="r",allow_pickle=False)
            corrected, cache_audit = retain_parent_scientific_pixels(corrected,parent,valid)
            cache_audit["reference_source_binding_preserved"] = True
        return corrected, dict(contract=CONTRACT, processor=self.identity, sources=sources,
            renderer=renderer, chart=chart, photometry=info, exact_B=exact_b,
            H_ac28_to_frame_native=h_ref.tolist(),H_v5_B_to_frame_native=h_b.tolist(),
            full_native_corrected_finite_pixels=int(np.isfinite(corrected).all(0).sum()),
            source_identity=source_identity,parent_cache=cache_audit, cpu_only=True, training=False)


__all__ = ["RigRawSourceProcessor", "retain_parent_scientific_pixels", "CONTRACT"]
