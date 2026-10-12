"""Stream SHA and bitwise QA of the isolated-invalid-only scientific rerun.

Reads each old/new native float sidecar once, with bounded working memory.
It does not change sources, fits, phase maps, or model eligibility.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from .ffac_track1_isolated_invalid import CONTRACT as FILL_CONTRACT, POLICY

CONTRACT = "ffac-track1-isolated-invalid-rerun-bitwise-qa-v1"


def _read_json(path: Path, expected_sha256: str | None = None):
    data = Path(path).read_bytes()
    digest = sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"JSON source SHA changed: {path}")
    return json.loads(data.decode("utf-8-sig")), digest


class _HashedReader:
    def __init__(self, stream):
        self.stream, self.digest = stream, sha256()

    def read(self, count=-1):
        data = self.stream.read(count)
        self.digest.update(data)
        return data


def _read_header(reader):
    version = np.lib.format.read_magic(reader)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(reader)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(reader)
    else:
        raise ValueError(f"Unexpected native NPY version: {version}")
    if (fortran or len(shape) != 3 or shape[0] != 3 or dtype != np.dtype(np.float32)):
        raise ValueError("Scientific comparison needs native C-order CHW float32, never uint8")
    return shape


def compare_float_sources(old_path: Path, new_path: Path, filled_mask: np.ndarray, *,
        old_sha256: str, new_sha256: str, chunk_pixels: int = 262144) -> dict:
    """Compare exactly retained bits and invalid support in one sequential scan."""
    if not isinstance(filled_mask, np.ndarray) or filled_mask.dtype != np.dtype(bool):
        raise TypeError("Filled-invalid comparison needs a native boolean mask")
    if chunk_pixels < 1:
        raise ValueError("Chunk size must be positive")
    with Path(old_path).open("rb") as old_stream, Path(new_path).open("rb") as new_stream:
        old, new = _HashedReader(old_stream), _HashedReader(new_stream)
        old_shape, new_shape = _read_header(old), _read_header(new)
        if old_shape != new_shape or filled_mask.shape != old_shape[1:]:
            raise ValueError("Scientific comparison sources or fill mask bind different native grids")
        size = int(np.prod(old_shape[1:]))
        filled = filled_mask.reshape(-1)
        old_valid, new_valid = np.ones(size, bool), np.ones(size, bool)
        changed = np.zeros(size, bool)
        changed_old_finite_channels = changed_channels_outside = 0
        for _channel in range(3):
            for start in range(0, size, chunk_pixels):
                stop = min(size, start + chunk_pixels)
                expected_bytes = (stop-start) * 4
                old_bytes, new_bytes = old.read(expected_bytes), new.read(expected_bytes)
                if len(old_bytes) != expected_bytes or len(new_bytes) != expected_bytes:
                    raise ValueError("Truncated native float NPY source")
                a, b = np.frombuffer(old_bytes, np.float32), np.frombuffer(new_bytes, np.float32)
                finite_a, finite_b = np.isfinite(a), np.isfinite(b)
                if np.isinf(a).any() or np.isinf(b).any():
                    raise ValueError("Infinite scientific inputs are not eligible for local repair")
                bits_changed = a.view(np.uint32) != b.view(np.uint32)
                old_valid[start:stop] &= finite_a
                new_valid[start:stop] &= finite_b
                changed[start:stop] |= bits_changed
                changed_channels_outside += int(np.count_nonzero(bits_changed & ~filled[start:stop]))
                changed_old_finite_channels += int(np.count_nonzero(bits_changed & finite_a))
        if old.read(1) or new.read(1):
            raise ValueError("Native float NPY contains unbound trailing bytes")
        actual_old_sha, actual_new_sha = old.digest.hexdigest(), new.digest.hexdigest()
        if actual_old_sha != old_sha256 or actual_new_sha != new_sha256:
            raise ValueError("Scientific sidecar SHA differs from its frozen manifest")
    became_valid = ~old_valid & new_valid
    count_outside = int(np.count_nonzero(changed & ~filled))
    lost = int(np.count_nonzero(old_valid & ~new_valid))
    invalid_before = int(np.count_nonzero(filled & ~old_valid))
    valid_after = int(np.count_nonzero(filled & new_valid))
    exact_fill = bool(np.array_equal(became_valid, filled))
    passed = count_outside == 0 and lost == 0 and changed_old_finite_channels == 0 and exact_fill
    return dict(old_path=str(old_path), old_sha256=actual_old_sha, new_path=str(new_path),
        new_sha256=actual_new_sha, native_shape_chw=list(old_shape), filled_pixel_count=int(filled.sum()),
        changed_pixel_count=int(changed.sum()), changed_pixels_outside_filled_mask=count_outside,
        changed_channels_outside_filled_mask=changed_channels_outside,
        changed_original_finite_channels=changed_old_finite_channels,
        original_valid_pixels_lost=lost, filled_pixels_invalid_before=invalid_before,
        filled_pixels_valid_after=valid_after, actual_validity_gain_matches_fill_mask=exact_fill,
        bitwise_identical_outside_filled_mask=count_outside == 0, passed=passed,
        streams_read_once=True)


def _load_verified_mask(path: Path, expected_sha256: str) -> np.ndarray:
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Filled-invalid mask SHA changed")
    from io import BytesIO
    mask = np.load(BytesIO(data), allow_pickle=False)
    if mask.dtype != np.dtype(bool) or mask.ndim != 2:
        raise ValueError("Filled-invalid mask is not native boolean HW")
    return mask


def _verify_frame_certificates(old_row: dict, new_row: dict, *, reference=False):
    key = "v5_baseline_receipt" if reference else "v5_frame_receipt"
    old_record, _ = _read_json(Path(old_row[f"{key}_path"]), old_row[f"{key}_sha256"])
    new_record, _ = _read_json(Path(new_row[f"{key}_path"]), new_row[f"{key}_sha256"])
    if old_record.get("source") != new_record.get("source"):
        raise ValueError("Old/new materialization receipts identify different camera acquisitions")
    a, b = old_record["certificate"], new_record["certificate"]
    for field in ("transform_sha256", "fit_baseline_sidecar_sha256", "old_field_sha256",
                  "source_photo_receipt_sha256", "source_frame_transport_receipt_sha256",
                  "native_shape_hw"):
        if a.get(field) != b.get(field):
            raise ValueError(f"Old/new correction or geometry binding differs: {field}")
    repair = b.get("isolated_invalid_fill") or {}
    if repair.get("contract") != FILL_CONTRACT:
        raise ValueError("New scientific source lacks isolated-invalid contract")
    for field in ("v5_filled_invalid_mask_path", "v5_filled_invalid_mask_sha256"):
        if new_row.get(field) != new_record.get(field):
            raise ValueError("Manifest repair mask differs from its native frame receipt")
    # Reference receipts call this path log_linear_rgb_path, while native rows use v5_*.
    for row, record in ((old_row, old_record), (new_row, new_record)):
        if (row.get("v5_log_linear_rgb_path") != record.get("log_linear_rgb_path")
                or row.get("v5_log_linear_rgb_sha256") != record.get("log_linear_rgb_sha256")):
            raise ValueError("Manifest scientific sidecar differs from its materialization receipt")
    return repair


def compare_source_revision(old_manifest: Path, new_manifest: Path, *,
        expected_old_sha256: str | None = None, expected_new_sha256: str | None = None,
        chunk_pixels: int = 262144) -> dict:
    """Validate one exact source revision, including all three temporal inputs."""
    old, old_sha = _read_json(old_manifest, expected_old_sha256)
    new, new_sha = _read_json(new_manifest, expected_new_sha256)
    for key in ("run", "frame_count", "analysis_image_shape_hw", "content_box_xywh"):
        if old.get(key) != new.get(key):
            raise ValueError(f"Old/new native manifest binding differs: {key}")
    old_photo, new_photo = old.get("v5_photometry") or {}, new.get("v5_photometry") or {}
    if (not old_photo.get("source_manifest_sha256") or
            old_photo.get("source_manifest_sha256") != new_photo.get("source_manifest_sha256")
            or old.get("v5_track1_photo_geometry") != new.get("v5_track1_photo_geometry")):
        raise ValueError("Old/new source revision or per-frame sand homographies differ")
    new_processor = new.get("correction_contract", {}).get("processor", {})
    if new_processor.get("isolated_invalid_policy") != POLICY:
        raise ValueError("New materialized manifest did not opt in to isolated-invalid repair")
    if old_photo.get("processor", {}).get("isolated_invalid_policy", "keep") != "keep":
        raise ValueError("Old comparison is already a repaired candidate")
    a, b = old.get("frame_sources") or [], new.get("frame_sources") or []
    if not a or len(a) != len(b) or len(b) != new.get("frame_count"):
        raise ValueError("Old/new comparison lacks the complete exact frame sequence")
    outputs = []
    for index, (old_row, new_row) in enumerate(zip(a, b)):
        for key in ("frame_index", "elapsed_h", "v5_raw_sha256", "v5_raw_members"):
            if old_row.get(key) != new_row.get(key):
                raise ValueError(f"Frame {index} binds another time or RAW source: {key}")
        if new_row.get("frame_index") != index:
            raise ValueError("Native frame sequence is incomplete or out of order")
        outputs.append(_compare_row(old_row, new_row, index, False, chunk_pixels))
    old_ref, new_ref = old.get("reference") or {}, new.get("reference") or {}
    if old_ref.get("v5_reference_state") != new_ref.get("v5_reference_state"):
        raise ValueError("Pre-injection reference acquisition changed")
    reference = _compare_row(old_ref, new_ref, "pre_injection", True, chunk_pixels)
    return dict(contract=CONTRACT, run=new.get("run"), old_manifest_path=str(old_manifest),
        old_manifest_sha256=old_sha, new_manifest_path=str(new_manifest), new_manifest_sha256=new_sha,
        source_manifest_sha256=new_photo["source_manifest_sha256"], frame_count=len(outputs),
        frames=outputs, reference=reference, passed=all(row["passed"] for row in [*outputs, reference]),
        scientific_mass_eligible=False, model_fit_changed=False)


def _compare_row(old_row: dict, new_row: dict, index, reference: bool, chunk_pixels: int):
    binding = _verify_frame_certificates(old_row, new_row, reference=reference)
    mask = _load_verified_mask(Path(new_row["v5_filled_invalid_mask_path"]),
                              new_row["v5_filled_invalid_mask_sha256"])
    if (binding.get("filled_pixel_count") != int(mask.sum()) or
            binding.get("filled_mask_packbits_sha256") != sha256(
                np.packbits(mask, bitorder="little").tobytes()).hexdigest()):
        raise ValueError("Native filled-invalid mask differs from correction certificate")
    result = compare_float_sources(Path(old_row["v5_log_linear_rgb_path"]),
        Path(new_row["v5_log_linear_rgb_path"]), mask,
        old_sha256=old_row["v5_log_linear_rgb_sha256"],
        new_sha256=new_row["v5_log_linear_rgb_sha256"], chunk_pixels=chunk_pixels)
    return dict(frame_index=index, elapsed_h=new_row.get("elapsed_h"),
        largest_filled_component_px=binding.get("largest_filled_component_px"),
        filled_mask_path=new_row["v5_filled_invalid_mask_path"],
        filled_mask_sha256=new_row["v5_filled_invalid_mask_sha256"], **result)


def write_report(output: Path, report: dict):
    data = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError("Existing isolated-invalid comparison report differs")
        return
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.pending")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if output.exists():
            if output.read_bytes() != data:
                raise ValueError("Concurrent isolated-invalid report differs")
        else:
            os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-manifest", required=True, type=Path)
    parser.add_argument("--new-manifest", required=True, type=Path)
    parser.add_argument("--old-manifest-sha256")
    parser.add_argument("--new-manifest-sha256")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare_source_revision(args.old_manifest, args.new_manifest,
        expected_old_sha256=args.old_manifest_sha256, expected_new_sha256=args.new_manifest_sha256)
    write_report(args.output, report)
    print(json.dumps(dict(run=report["run"], frame_count=report["frame_count"],
                          passed=report["passed"], output=str(args.output))))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
