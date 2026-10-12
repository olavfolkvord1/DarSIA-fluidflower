"""Source-bound preflight and optical sampling for pooled FF_AC RAW color paths.

This module deliberately stops before fitting a DarSIA ColorPath unless the
physical facies alignment and chemistry grouping have independent evidence.
The returned optical samples are diagnostic only: a large color change alone
does not establish dissolved CO2 or a concentration coordinate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from darsia.fluidflower.phase_training.v5_linear_input import decode_log_linear
from darsia.fluidflower.tools.ffac_v5_frame_context import (
    _source_photo_grid,
    project_source_photo_safe_core,
)
from darsia.fluidflower.tools.ffac_v5_temporal_alignment import (
    warp_reference_to_current,
)


CONTRACT = "ffac-pooled-raw-color-path-preflight-v1"
SUPPORTED_FACIES = (1, 2, 5, 7, 8)
UNQUANTIFIED_FACIES = (3, 4, 6)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bound_json(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    source = Path(path)
    if sha256(source) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {source}")
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {source}")
    return data


def validate_split(
    fit_runs: tuple[str, ...], holdout_run: str,
    chemistry_groups: Mapping[str, str],
) -> str:
    """Never pool provisional chemistry or let a holdout train the path."""
    if not fit_runs or len(set(fit_runs)) != len(fit_runs):
        raise ValueError("Fit runs must be nonempty and unique")
    if holdout_run in fit_runs:
        raise ValueError("Whole-run holdout appears in path-fit runs")
    if set(chemistry_groups) != {*fit_runs, holdout_run}:
        raise ValueError("Every fit and holdout run needs an explicit chemistry group")
    groups = {str(chemistry_groups[run]) for run in fit_runs}
    if len(groups) != 1:
        raise ValueError("Do not pool distinct chemistry groups into one ColorPath")
    group = groups.pop()
    if group != "ac_main_verified":
        raise ValueError(
            "Pooled fitting requires verified main chemistry; provisional runs "
            "belong in a separate sensitivity arm"
        )
    if chemistry_groups[holdout_run] != group:
        raise ValueError("Holdout chemistry differs from the fit group")
    return group


def required_interior_radius(certificate: Mapping[str, Any]) -> int:
    """Use an independent physical-error bound, never sample availability."""
    if certificate.get("status") != "certified" or certificate.get(
        "independent_of_color_response"
    ) is not True:
        raise ValueError("Physical facies alignment is not independently certified")
    for key in ("boundary_error_p95_px", "registration_error_bound_px"):
        value = certificate.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Certificate has no valid {key}")
    radius = certificate.get("interior_radius_px")
    if type(radius) is not int or radius < 0:
        raise ValueError("Certificate has no fixed facies interior radius")
    if radius < certificate["boundary_error_p95_px"] + certificate[
        "registration_error_bound_px"
    ]:
        raise ValueError("Facies interior radius is smaller than measured uncertainty")
    return radius


def validate_certificate(
    certificate: Mapping[str, Any], *, facies_sha256: str,
    geometry_receipts: Mapping[tuple[str, int], str],
) -> int:
    radius = required_interior_radius(certificate)
    if certificate.get("reference_facies_sha256") != facies_sha256:
        raise ValueError("Physical facies certificate binds another facies map")
    rows = certificate.get("verified_frames")
    if not isinstance(rows, list):
        raise ValueError("Physical facies certificate lacks verified frames")
    certified = {
        (str(row["run"]), int(row["frame_index"])): row["geometry_receipt_sha256"]
        for row in rows
    }
    if len(certified) != len(rows):
        raise ValueError("Physical facies certificate repeats a frame")
    if certified != dict(geometry_receipts):
        raise ValueError("Physical facies certificate does not bind every frame")
    return radius


def _bound_signal(
    row: Mapping[str, Any], *, shape: tuple[int, int], invalid_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(row["v5_log_linear_rgb_path"])
    valid_path = Path(row["v5_log_linear_valid_mask_path"])
    rgb = decode_log_linear(
        path, expected_sha256=row["v5_log_linear_rgb_sha256"],
        content_shape_hw=shape, invalid_policy=invalid_policy,
        valid_mask_path=valid_path,
        valid_mask_sha256=row["v5_log_linear_valid_mask_sha256"],
    )
    valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
    if valid.dtype != np.bool_ or valid.shape != shape:
        raise ValueError("Bound scientific valid mask is not native HW bool")
    return rgb, np.asarray(valid)


def diagnostic_samples(
    *, current: np.ndarray, baseline: np.ndarray,
    current_valid: np.ndarray, baseline_valid: np.ndarray,
    facies_reference: np.ndarray, domain_reference: np.ndarray,
    photo_safe_frame: np.ndarray, h_baseline_to_frame: np.ndarray,
    h_facies_to_frame: np.ndarray, radius_px: int, stride_px: int = 8,
    labels: tuple[int, ...] = SUPPORTED_FACIES,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Return baseline-relative log RGB from exact supported native centres.

    The baseline is interpolated with full-footprint support. Facies/domain
    are categorically projected per frame. No image is downsampled: stride
    controls only which native pixel centres enter the diagnostic.
    """
    if current.dtype != np.float32 or baseline.dtype != np.float32:
        raise TypeError("Scientific images must be float32, never uint8 display RGB")
    if current.shape != baseline.shape or current.ndim != 3 or current.shape[2] != 3:
        raise ValueError("Scientific frames must share a native HWC RGB grid")
    shape = current.shape[:2]
    for name, array in (
        ("current_valid", current_valid), ("baseline_valid", baseline_valid),
        ("facies_reference", facies_reference),
        ("domain_reference", domain_reference),
        ("photo_safe_frame", photo_safe_frame),
    ):
        if array.shape != shape:
            raise ValueError(f"{name} does not use the native frame grid")
    for name, mask in (("current_valid", current_valid),
                       ("baseline_valid", baseline_valid),
                       ("domain_reference", domain_reference),
                       ("photo_safe_frame", photo_safe_frame)):
        if mask.dtype != np.bool_:
            raise TypeError(f"{name} must be a native bool support mask")
    if not np.issubdtype(facies_reference.dtype, np.integer):
        raise TypeError("Facies must be categorical integers")
    for name, h in (("baseline", h_baseline_to_frame),
                    ("facies", h_facies_to_frame)):
        h = np.asarray(h)
        if h.shape != (3, 3) or not np.isfinite(h).all() or abs(
            float(np.linalg.det(h))
        ) < 1e-9:
            raise ValueError(f"Invalid {name} homography")
    if type(stride_px) is not int or type(radius_px) is not int or stride_px < 1 or radius_px < 0:
        raise ValueError("Sampling stride and certified radius are invalid")
    baseline_aligned, baseline_support = warp_reference_to_current(
        baseline, h_baseline_to_frame, target_shape_hw=shape,
        source_valid_mask=baseline_valid,
    )
    frame_facies = cv2.warpPerspective(
        np.asarray(facies_reference, np.uint8), h_facies_to_frame,
        (shape[1], shape[0]), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    frame_domain = cv2.warpPerspective(
        np.asarray(domain_reference, np.uint8), h_facies_to_frame,
        (shape[1], shape[0]), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    photo_safe = np.asarray(photo_safe_frame, bool)
    yy, xx = np.mgrid[0:shape[0]:stride_px, 0:shape[1]:stride_px]
    y, x = yy.ravel(), xx.ravel()
    basic = (
        photo_safe[y, x] & frame_domain[y, x] & baseline_support[y, x]
        & current_valid[y, x]
    )
    samples: dict[int, np.ndarray] = {}
    counts: dict[str, int] = {}
    for label in labels:
        facies_mask = np.asarray(frame_facies == label, np.uint8)
        depth = cv2.distanceTransform(facies_mask, cv2.DIST_L2, 5)
        selected = basic & (frame_facies[y, x] == label) & (depth[y, x] >= radius_px)
        b, c = baseline_aligned[y[selected], x[selected]], current[y[selected], x[selected]]
        positive = np.min(np.minimum(b, c), axis=1) > 0
        b, c = b[positive], c[positive]
        samples[int(label)] = (np.log(b) - np.log(c)).astype(np.float32)
        counts[str(label)] = len(samples[int(label)])
    return samples, dict(
        sample_count_by_facies=counts, sampled_native_centres=int(len(y)),
        baseline_support_centres=int(np.count_nonzero(basic)),
        radius_px=radius_px, stride_px=stride_px,
        unsupported_facies=list(UNQUANTIFIED_FACIES),
        optical_evidence_only=True,
    )


def preflight_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Fast SHA-bound gate, before decoding a single full-resolution frame."""
    fit_runs = tuple(request["fit_runs"])
    holdout = str(request["whole_run_holdout"])
    blockers: list[str] = []
    try:
        chemistry = validate_split(fit_runs, holdout, request["chemistry_groups"])
    except (KeyError, ValueError) as exc:
        chemistry = None
        blockers.append(str(exc))
    source_rows = request["sources"]
    if set(source_rows) != {*fit_runs, holdout}:
        raise ValueError("Source manifests must match fit and holdout runs exactly")
    indices = request["frame_indices"]
    if (not isinstance(indices, list) or not indices
            or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("Frame indices must be nonempty unique native indices")
    lineage_spec = request["facies_lineage"]
    lineage = read_bound_json(lineage_spec["path"], lineage_spec["sha256"])
    if lineage.get("contract") != "ffac-v5-source-domain-lineage-52-v1":
        raise ValueError("Unexpected facies lineage contract")
    receipts: dict[tuple[str, int], str] = {}
    source_hashes: dict[str, str] = {}
    facies_hashes: set[str] = set()
    for run, source in sorted(source_rows.items()):
        manifest = read_bound_json(source["manifest_path"], source["manifest_sha256"])
        if manifest.get("run") != run or manifest.get("correction_contract", {}).get(
            "scientific_signal"
        ) != "log-camera-linear-float32-sidecar":
            raise ValueError(f"Wrong Track1 scientific source contract for {run}")
        source_hashes[run] = source["manifest_sha256"]
        facies_path = Path(source["facies_path"])
        facies_sha = source["facies_sha256"]
        if sha256(facies_path) != facies_sha:
            raise ValueError(f"Facies SHA-256 mismatch for {run}")
        lineage_matches = [
            row for row in lineage["rows"]
            if row["run"] == run
            and row["source_manifest_sha256"]
            == manifest["v5_photometry"]["source_manifest_sha256"]
            and row["calibration_domain_npz_sha256"] == facies_sha
            and Path(row["calibration_domain_npz_path"]) == facies_path
            and row["facies_and_domain_arrays_identical"] is True
        ]
        if len(lineage_matches) != 1:
            raise ValueError(f"Facies asset is not source-bound for {run}")
        facies_hashes.add(facies_sha)
        for index in indices:
            binding = manifest["v5_track1_photo_geometry"]["frames"][index]
            receipt = read_bound_json(binding["receipt_path"], binding["receipt_sha256"])
            if receipt.get("frame_index") != index or receipt.get(
                "source_manifest_sha256"
            ) != manifest["v5_photometry"]["source_manifest_sha256"]:
                raise ValueError(f"Per-frame geometry/source mismatch: {run} {index}")
            receipts[(run, index)] = binding["receipt_sha256"]
    if len(facies_hashes) != 1:
        blockers.append("Fit/holdout runs do not share one source-bound facies map")
    certificate = request.get("physical_facies_certificate")
    if certificate is None:
        blockers.append("No independent physical facies boundary certificate")
        radius = None
    else:
        cert = read_bound_json(certificate["path"], certificate["sha256"])
        try:
            radius = validate_certificate(
                cert, facies_sha256=next(iter(facies_hashes)),
                geometry_receipts=receipts,
            )
        except (KeyError, ValueError) as exc:
            radius = None
            blockers.append(str(exc))
    return dict(
        contract=CONTRACT, status="blocked" if blockers else "source-gates-passed",
        scope="diagnostic-only-no-native-path-or-mass-fit",
        fit_runs=list(fit_runs), whole_run_holdout=holdout,
        chemistry_group=chemistry, chemistry_groups=dict(request["chemistry_groups"]),
        source_manifest_sha256=source_hashes,
        geometry_receipt_sha256={f"{run}:{index}": value for (run, index), value in sorted(receipts.items())},
        historical_facies_sha256=sorted(facies_hashes),
        facies_lineage_sha256=lineage_spec["sha256"],
        certified_interior_radius_px=radius,
        scientific_signal="exact SHA-bound full-native float32 log-camera-linear RAW",
        holdout_postinjection_used_in_fit=False,
        physical_facies_verified=radius is not None,
        independent_co2_response_reference_verified=False,
        unquantified_facies=list(UNQUANTIFIED_FACIES),
        mass_eligible=False, blockers=blockers,
    )


def load_bound_frame_samples(
    source: Mapping[str, Any], *, run: str, frame_index: int,
    gate_report: Mapping[str, Any], stride_px: int = 8,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Decode one SHA-bound RAW frame only after independent source gates pass.

    The per-frame geometry receipt determines both the baseline warp and the
    AC28 facies projection. The photo-safe mask is transported independently
    from the same receipt. This does not label optical changes as CO2.
    """
    if gate_report.get("contract") != CONTRACT or gate_report.get(
        "status"
    ) != "source-gates-passed":
        raise ValueError("Pooled color-path source gates have not passed")
    if gate_report.get("source_manifest_sha256", {}).get(run) != source.get(
        "manifest_sha256"
    ):
        raise ValueError("Source request differs from the preflight gate")
    manifest = read_bound_json(source["manifest_path"], source["manifest_sha256"])
    if manifest["run"] != run:
        raise ValueError("Source manifest belongs to another experiment")
    shape = tuple(manifest["analysis_image_shape_hw"])
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Invalid native analysis image shape")
    if not 0 <= frame_index < len(manifest["frame_sources"]):
        raise ValueError("Frame index is outside source manifest")
    binding = manifest["v5_track1_photo_geometry"]["frames"][frame_index]
    if gate_report.get("geometry_receipt_sha256", {}).get(
        f"{run}:{frame_index}"
    ) != binding["receipt_sha256"]:
        raise ValueError("Frame geometry differs from the preflight gate")
    geometry = read_bound_json(binding["receipt_path"], binding["receipt_sha256"])
    source_sha = manifest["v5_photometry"]["source_manifest_sha256"]
    if (geometry.get("source_manifest_sha256") != source_sha
            or geometry.get("frame_index") != frame_index
            or geometry.get("native_shape_hw") != list(shape)
            or geometry.get("photo_sand_transport_qa_pass") is not True):
        raise ValueError("Per-frame geometry/source contract changed")
    if gate_report.get("historical_facies_sha256") != [source["facies_sha256"]]:
        raise ValueError("Facies source differs from the preflight gate")
    facies_path = Path(source["facies_path"])
    if sha256(facies_path) != source["facies_sha256"]:
        raise ValueError("Facies source changed after preflight")
    with np.load(facies_path, allow_pickle=False) as archive:
        facies = np.asarray(archive["facies"], np.uint8)
        domain = np.asarray(archive["domain"], bool)
    if facies.shape != shape or domain.shape != shape:
        raise ValueError("Facies/domain source is not native")
    policy = manifest["v5_float_support_policy"]
    baseline, baseline_valid = _bound_signal(
        manifest["reference"], shape=shape, invalid_policy=policy,
    )
    row = manifest["frame_sources"][frame_index]
    current, current_valid = _bound_signal(
        row, shape=shape, invalid_policy=policy,
    )
    h_baseline = np.asarray(geometry["H_v5_B_to_frame_native"], np.float64)
    h_facies = np.asarray(geometry["H_ac28_to_frame_native"], np.float64)
    grid = _source_photo_grid(
        geometry["photo_sand_reference_mask_path"],
        geometry["photo_sand_reference_mask_sha256"],
    )
    photo_safe = project_source_photo_safe_core(grid, h_facies, shape_hw=shape)
    samples, stats = diagnostic_samples(
        current=current, baseline=baseline, current_valid=current_valid,
        baseline_valid=baseline_valid, facies_reference=facies,
        domain_reference=domain, photo_safe_frame=photo_safe,
        h_baseline_to_frame=h_baseline, h_facies_to_frame=h_facies,
        radius_px=int(gate_report["certified_interior_radius_px"]),
        stride_px=stride_px,
    )
    return samples, dict(
        run=run, frame_index=frame_index, elapsed_h=row["elapsed_h"],
        source_manifest_sha256=source["manifest_sha256"],
        original_source_manifest_sha256=source_sha,
        frame_sidecar_sha256=row["v5_log_linear_rgb_sha256"],
        frame_valid_mask_sha256=row["v5_log_linear_valid_mask_sha256"],
        baseline_sidecar_sha256=manifest["reference"]["v5_log_linear_rgb_sha256"],
        baseline_valid_mask_sha256=manifest["reference"]["v5_log_linear_valid_mask_sha256"],
        geometry_receipt_sha256=binding["receipt_sha256"],
        photo_safe_grid_sha256=geometry["photo_sand_reference_mask_sha256"],
        facies_sha256=source["facies_sha256"],
        **stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="Explicit frozen source/split JSON")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8-sig"))
    if not isinstance(request, dict):
        raise ValueError("Pooled Color Path request must be a JSON object")
    report = preflight_request(request)
    report["request_path"] = str(args.request.resolve())
    report["request_sha256"] = sha256(args.request)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], blockers=report["blockers"],
                          report=str(args.report)), indent=2))


if __name__ == "__main__":
    main()
