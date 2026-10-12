"""Exact AC45 wash lens metadata, without modifying frozen RAW math.

The five predeclared wash captures all use the known AC44 alternate Sony
distortion/CA table.  AC45's injection baseline uses the standard late table.
This wrapper applies the selected wash's measured embedded table to that
wash only, in a call-local namespace.  Original source functions, intrinsic
parameters, photometry and the 2-pixel wash registration gate stay unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import FunctionType

import numpy as np
import tifffile

from darsia.fluidflower.project import sha256_file
from . import ffac_raw_geometry as geo
from . import ffac_raw_native_ea as original_renderer
from . import ffac_v5_normalized_clearwater as original_clearwater


PARENT_RENDERER_SHA = "041e65a5437ecae4237f518ef08b58e2becfbb4e62d45dc9bb8ead07c194b559"
PARENT_GEOMETRY_SHA = "e200598ee8657339db537859e6c54f76eba82e0d7832a0209b532591e6536695"
PARENT_CLEARWATER_SHA = "6baaa24a661fb1132a54df45e66304e14a3c24a0678d5e120b16e728df6b53b2"
EXACT_SOURCES = dict(arw_sha256="e25306cb83b359cef33d0ae3cb6fe9edfa8d5e459bee97c138f4a7cdf0b78a00",
    camera_jpeg_sha256="490ddc798cea3849be8f35e3e5bd84eab8eb86f19b5c6c08be84b7535a088890",
    review_native_jpeg_sha256="ade47b99105555340a433f05bec1c0eda822d258e3da70cf27859a28bf828525")
ALT_DISTORTION = (704, 684, 615, 497, 331, 126, -110, -359, -600, -824, -1034)
ALT_RED = (0, 640, 512, 512, 512, 640, 640, 640, 768, 768, 768)
ALT_BLUE = (0, 384, 512, 640, 640, 512, 384, 256, 256, 256, 384)


def _check_parents():
    for module, expected in ((original_renderer, PARENT_RENDERER_SHA), (geo, PARENT_GEOMETRY_SHA), (original_clearwater, PARENT_CLEARWATER_SHA)):
        if sha256_file(Path(module.__file__)) != expected:
            raise ValueError("Frozen clearwater source math changed before the exact lens-metadata revision")


def _exact_tags(path):
    path = Path(path)
    if sha256_file(path) != EXACT_SOURCES["arw_sha256"]:
        raise ValueError("The alternate wash lens table is restricted to the exact predeclared AC45 source")
    with tifffile.TiffFile(path) as source:
        tags = source.pages[0].pages[0].tags
        distortion, ca = tuple(map(int, tags[0x7037].value)), tuple(map(int, tags[0x7035].value))
    if distortion[0] != 11 or distortion[1:12] != ALT_DISTORTION or ca[0] != 22 or ca[1:12] != ALT_RED or ca[12:23] != ALT_BLUE:
        raise ValueError("The exact AC45 wash metadata differs from its recorded embedded lens table")
    return ALT_DISTORTION, ALT_RED, ALT_BLUE


class _WashGeo:
    _DISTORTION = {**geo._DISTORTION, "late": ALT_DISTORTION}

    @staticmethod
    def _sony_tags(path):
        return _exact_tags(path)

    def __getattr__(self, name):
        return getattr(geo, name)


def _local_function(function, **overlay):
    namespace = dict(function.__globals__)
    namespace.update(overlay)
    return FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)


def registered_ac45_wash_raw(arw_path, camera_jpeg_path, review_native_jpeg_path, *, expected_source_hashes, content_box):
    _check_parents()
    if dict(expected_source_hashes) != EXACT_SOURCES:
        raise ValueError("Alternate wash rendering cannot be applied to another source triple")
    if not any(part.casefold() == "ac45" for part in Path(arw_path).parts):
        raise ValueError("Alternate clearwater metadata is source-bound to AC45")
    renderer = _local_function(original_renderer.registered_edge_aware_raw_linear, geo=_WashGeo())
    linear, certificate = renderer(arw_path, camera_jpeg_path, review_native_jpeg_path,
        expected_source_hashes=expected_source_hashes, content_box=content_box)
    if tuple(certificate["sony_distortion_table"]) != ALT_DISTORTION or tuple(certificate["sony_ca_red_table"]) != ALT_RED or tuple(certificate["sony_ca_blue_table"]) != ALT_BLUE:
        raise ValueError("Alternate clearwater lens receipt did not preserve the measured source tables")
    certificate = dict(certificate)
    certificate["source_exact_alternate_lens"] = dict(contract="ffac-common-ac45-clearwater-source-lens-v1", source_sha256=EXACT_SOURCES,
        parent_renderer_sha256=PARENT_RENDERER_SHA, parent_geometry_sha256=PARENT_GEOMETRY_SHA,
        wrapper_module_sha256=sha256_file(Path(__file__)), profile_intrinsics="unchanged frozen late FF_AC intrinsics",
        lens_table_source="selected original ARW embedded Sony tags, identical in all five predeclared wash captures",
        numerical_rule="same frozen full-resolution EA bytecode; only measured source lens metadata supplied", parallel_execution="call-local namespace; no module mutation")
    return linear, certificate


def prepare_exact_ac45(task_row, output):
    _check_parents()
    if task_row["run"] != "ac45" or task_row.get("execution_after_freeze") is True:
        raise ValueError("This source-only revision is the calibration run AC45")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve every earlier reference folder; use a new isolated output")
    args = {key: value if key == "run" else Path(value) for key, value in task_row["prepare_one_args"].items()}
    args["output"] = output
    # Call-local preparation uses the original gates/photometry unchanged.
    prepare = _local_function(original_clearwater.prepare_one, registered_edge_aware_raw_linear=registered_ac45_wash_raw)
    result = prepare(**args)
    from .ffac_track1_v4r5_source_processor import Track1V4R5SourceProcessor
    native = json.loads(args["native_manifest_path"].read_text(encoding="utf-8-sig"))
    aggregate = json.loads(args["photo_aggregate_path"].read_text(encoding="utf-8-sig"))
    matching = [row for row in aggregate["rows"] if row["run"] == "ac45" and row["source_manifest_sha256"] == native["v5_photometry"]["source_manifest_sha256"]]
    if len(matching) != 1:
        raise ValueError("A unique frozen AC45 B geometry is required")
    processor = Track1V4R5SourceProcessor(matching[0], calibration_manifest_path=args["calibration_manifest_path"], photo_aggregate_path=args["photo_aggregate_path"])
    baseline = processor.cohort.frame(0)
    matrix = np.linalg.inv(np.asarray(baseline["H_v5_B_to_frame_native"], np.float64)) @ np.asarray(baseline["H_ac28_to_frame_native"], np.float64)
    transport = dict(contract="ffac-common-native-clearwater-transport-v1", run="ac45", clearwater_receipt=result["path"], clearwater_receipt_sha256=result["sha256"],
        H_ac28_to_clearwater_grid=matrix.tolist(), baseline_grid="native-B-2979x5472", current_frame_transport="H_ac28_to_current_rig @ inv(H_ac28_to_clearwater_grid)",
        missing_support="unquantified; never extrapolated or fitted", created_utc=datetime.now(timezone.utc).isoformat(), no_model_fit=True,
        source_generator_sha256=sha256_file(Path(__file__)), parent_source_generator_sha256=PARENT_CLEARWATER_SHA,
        alternate_lens_source=EXACT_SOURCES, profile_intrinsics="unchanged frozen late FF_AC intrinsics", geometry_gate_unchanged=True)
    transport_path = output / "common_reference_transport.json"
    with transport_path.open("x", encoding="utf-8") as stream:
        json.dump(transport, stream, indent=2, allow_nan=False)
    return dict(run="ac45", complete=True, reference=result, transport=dict(path=str(transport_path), sha256=sha256_file(transport_path)),
        model_fit=False, phase_inference=False, heldout_pixels_computed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    import cv2
    cv2.setNumThreads(args.threads)
    rows = json.loads(args.tasks.read_text(encoding="utf-8-sig"))["rows"]
    tasks = [row for row in rows if row["run"] == "ac45"]
    if len(tasks) != 1:
        raise ValueError("One exact AC45 predeclared clearwater task is required")
    print(json.dumps(prepare_exact_ac45(tasks[0], args.output), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
