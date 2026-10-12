"""Prepare the exact AC45 wash using its measured known alternate lens profile.

The intrinsic calibration is chosen from the embedded lens table and camera
EXIF, rather than the experiment number.  This call-local revision preserves
the frozen EA renderer, clearwater photometry and the original 2-pixel gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path

import numpy as np
from PIL import Image

from darsia.fluidflower.project import sha256_file
from . import ffac_common_clearwater_ac45_alt as source_lens
from . import ffac_raw_geometry as geo
from . import ffac_raw_native_ea as renderer
from . import ffac_v5_normalized_clearwater as clearwater


KNOWN_CAMERA_METADATA = dict(make="SONY", model="ZV-1", focal_length_mm=12.15,
    focal_length_35mm=31, lens_specification=[9.4, 25.7, 1.8, 2.8])
KNOWN_ALT_CAMERA_SHA256 = "dbf9525c4cc73c427b086c4739cf9eb851c56d51ae4dc7a0de07ec63bf9a045c"


def camera_metadata(path):
    with Image.open(path) as image:
        exif = image.getexif()
        optics = exif.get_ifd(34665)
        return dict(make=exif.get(271), model=exif.get(272),
            focal_length_mm=float(optics.get(37386)), focal_length_35mm=optics.get(41989),
            lens_specification=list(map(float, optics.get(42034))))


class _KnownAlternateGeo(source_lens._WashGeo):
    # The original renderer calls this slot "late" for AC45.  Its measured
    # alternate-table instrument profile is the already calibrated early one.
    _PROFILE = {**geo._PROFILE, "late": dict(geo._PROFILE["early"])}


def registered_known_alternate_wash(arw_path, camera_jpeg_path, review_native_jpeg_path,
        *, expected_source_hashes, content_box, known_alt_camera_path):
    source_lens._check_parents()
    if dict(expected_source_hashes) != source_lens.EXACT_SOURCES:
        raise ValueError("Known alternate wash rendering requires the exact predeclared source triple")
    if camera_metadata(camera_jpeg_path) != KNOWN_CAMERA_METADATA:
        raise ValueError("The measured wash camera/lens metadata differs from the known alternate calibration")
    # This reads instrument metadata and verifies an existing calibration source;
    # it neither decodes nor analyses the held-out experiment's image pixels.
    known_alt_camera_path = Path(known_alt_camera_path)
    if sha256_file(known_alt_camera_path) != KNOWN_ALT_CAMERA_SHA256:
        raise ValueError("The known alternate lens calibration source changed")
    if camera_metadata(known_alt_camera_path) != KNOWN_CAMERA_METADATA:
        raise ValueError("Known alternate calibration EXIF no longer matches")
    source_lens._exact_tags(arw_path)
    call = source_lens._local_function(renderer.registered_edge_aware_raw_linear, geo=_KnownAlternateGeo())
    linear, certificate = call(arw_path, camera_jpeg_path, review_native_jpeg_path,
        expected_source_hashes=expected_source_hashes, content_box=content_box)
    certificate = dict(certificate)
    certificate["source_exact_alternate_lens"] = dict(
        contract="ffac-common-ac45-clearwater-instrument-profile-v2",
        source_sha256=source_lens.EXACT_SOURCES,
        camera_metadata=KNOWN_CAMERA_METADATA,
        known_alternate_calibration_source=dict(path=str(known_alt_camera_path), sha256=KNOWN_ALT_CAMERA_SHA256),
        intrinsic_parameters=dict(geo._PROFILE["early"]),
        profile_selection="exact measured alternate D/CA table plus matched camera/lens EXIF",
        original_slot_name=certificate["profile"], actual_intrinsics_profile="calibrated early alternate-table instrument profile",
        parent_renderer_sha256=source_lens.PARENT_RENDERER_SHA,
        parent_geometry_sha256=source_lens.PARENT_GEOMETRY_SHA,
        wrapper_module_sha256=sha256_file(Path(__file__)),
        geometry_gate_px=2.0, geometry_gate_unchanged=True,
        numerical_rule="frozen full-resolution EA bytecode with measured source table and existing instrument intrinsics",
        per_run_fitted_parameters=False, module_mutation=False)
    return linear, certificate


def prepare_known_alternate_ac45(task_row, output, *, known_alt_camera_path):
    source_lens._check_parents()
    if task_row["run"] != "ac45" or task_row.get("execution_after_freeze") is True:
        raise ValueError("The exact alternate wash is a calibration-only AC45 source")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Keep earlier probes and references; choose a new output folder")
    args = {key: value if key == "run" else Path(value) for key, value in task_row["prepare_one_args"].items()}
    args["output"] = output
    prepare = source_lens._local_function(clearwater.prepare_one,
        registered_edge_aware_raw_linear=partial(registered_known_alternate_wash, known_alt_camera_path=known_alt_camera_path))
    result = prepare(**args)
    from .ffac_track1_v4r5_source_processor import Track1V4R5SourceProcessor
    native = json.loads(args["native_manifest_path"].read_text(encoding="utf-8-sig"))
    aggregate = json.loads(args["photo_aggregate_path"].read_text(encoding="utf-8-sig"))
    matching = [row for row in aggregate["rows"] if row["run"] == "ac45"
        and row["source_manifest_sha256"] == native["v5_photometry"]["source_manifest_sha256"]]
    if len(matching) != 1:
        raise ValueError("A unique frozen AC45 B geometry is required")
    processor = Track1V4R5SourceProcessor(matching[0],
        calibration_manifest_path=args["calibration_manifest_path"], photo_aggregate_path=args["photo_aggregate_path"])
    baseline = processor.cohort.frame(0)
    matrix = np.linalg.inv(np.asarray(baseline["H_v5_B_to_frame_native"], np.float64)) @ np.asarray(baseline["H_ac28_to_frame_native"], np.float64)
    transport = dict(contract="ffac-common-native-clearwater-transport-v1", run="ac45",
        clearwater_receipt=result["path"], clearwater_receipt_sha256=result["sha256"],
        H_ac28_to_clearwater_grid=matrix.tolist(), baseline_grid="native-B-2979x5472",
        current_frame_transport="H_ac28_to_current_rig @ inv(H_ac28_to_clearwater_grid)",
        missing_support="unquantified; never extrapolated or fitted",
        created_utc=datetime.now(timezone.utc).isoformat(), no_model_fit=True,
        source_generator_sha256=sha256_file(Path(__file__)),
        parent_source_generator_sha256=source_lens.PARENT_CLEARWATER_SHA,
        alternate_lens_source=source_lens.EXACT_SOURCES,
        profile_selection="measured alternate D/CA table and matched instrument EXIF; existing early intrinsics",
        known_alternate_calibration_source=dict(path=str(known_alt_camera_path), sha256=KNOWN_ALT_CAMERA_SHA256),
        geometry_gate_unchanged=True)
    transport_path = output / "common_reference_transport.json"
    with transport_path.open("x", encoding="utf-8") as stream:
        json.dump(transport, stream, indent=2, allow_nan=False)
    return dict(run="ac45", complete=True, reference=result,
        transport=dict(path=str(transport_path), sha256=sha256_file(transport_path)),
        model_fit=False, phase_inference=False, heldout_pixels_computed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--known-alt-camera", type=Path, required=True)
    args = parser.parse_args()
    import cv2
    cv2.setNumThreads(args.threads)
    rows = json.loads(args.tasks.read_text(encoding="utf-8-sig"))["rows"]
    selected = [row for row in rows if row["run"] == "ac45"]
    if len(selected) != 1:
        raise ValueError("Exactly one predeclared AC45 source task is required")
    print(json.dumps(prepare_known_alternate_ac45(selected[0], args.output,
        known_alt_camera_path=args.known_alt_camera), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
