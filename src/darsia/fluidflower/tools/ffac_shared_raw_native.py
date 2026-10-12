"""Experimental shared DarSIA Color Paths on FF_AC Track-1 scientific RAW.

This is intentionally separate from the existing JPEG-native and joint-optical
methods.  A path describes *color change*, not a measured CO2 concentration.
The same saved path package and signal response are used for every run; only
the measured physical fields and the registered pre-injection image vary.
There is no post-injection dilution in this native response.  A future dilution
model must have its own named calibration and validation contract.

The functions accept source-bound arrays so that full native images can be
streamed one at a time.  The caller must first obtain the independently
certified per-frame facies/domain/scale certificate and a source-bound
aqueous-positive mask.  An ordinary phase prediction is not that certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

import darsia
from darsia.fluidflower import concentration_native as native
from darsia.fluidflower.tools.ffac_pooled_raw_color_paths import (
    required_interior_radius, validate_split,
)
from darsia.signals.color.color_embedding import ColorEmbeddingBasis


CONTRACT = "ffac-shared-raw-native-path-v1"
POROUS_FACIES = tuple(range(1, 9))
REPRESENTATION = "current-minus-registered-baseline-camera-linear-float32"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _require_readiness(readiness: Mapping[str, Any]) -> None:
    if (readiness.get("contract") != CONTRACT
            or readiness.get("status") != "ready-for-provisional-fit"
            or readiness.get("readiness_sha256") != _json_sha({
                key: value for key, value in readiness.items()
                if key != "readiness_sha256"})):
        raise ValueError("Certified, immutable pooled RAW readiness is required")


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def array_sha256(array: np.ndarray) -> str:
    """Unambiguous in-memory binding for a registered native geometry field."""
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode("ascii"))
    h.update(json.dumps(value.shape).encode("ascii"))
    h.update(value.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class PathSampleBatch:
    """Positive, independently reviewed optical changes from one native frame."""

    run: str
    frame_index: int
    facies: int
    relative_rgb: np.ndarray
    native_manifest_sha256: str
    frame_sidecar_sha256: str
    baseline_sidecar_sha256: str
    geometry_receipt_sha256: str
    facies_mask_sha256: str
    aqueous_label_sha256: str

    def __post_init__(self) -> None:
        if not self.run or type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("A path batch needs an exact run and native frame index")
        if self.facies not in POROUS_FACIES:
            raise ValueError("All path batches need a porous facies ID")
        values = self.relative_rgb
        if (not isinstance(values, np.ndarray) or values.dtype != np.float32
                or values.ndim != 2 or values.shape[1] != 3 or len(values) == 0
                or not np.isfinite(values).all()):
            raise TypeError("Path observations must be nonempty float32 N×3 RAW differences")
        for key in ("native_manifest_sha256", "frame_sidecar_sha256",
                    "baseline_sidecar_sha256", "geometry_receipt_sha256",
                    "facies_mask_sha256", "aqueous_label_sha256"):
            _sha(getattr(self, key), key)

    def provenance(self) -> dict[str, Any]:
        record = {key: getattr(self, key) for key in (
            "run", "frame_index", "facies", "native_manifest_sha256",
            "frame_sidecar_sha256", "baseline_sidecar_sha256",
            "geometry_receipt_sha256", "facies_mask_sha256",
            "aqueous_label_sha256")}
        record["relative_rgb_array_sha256"] = array_sha256(self.relative_rgb)
        record["sample_count"] = len(self.relative_rgb)
        return record


@dataclass(frozen=True)
class PreparedSharedNativeFrame:
    analysis: native.NativeConcentrationAnalysis
    frame: native.PreparedNativeFrame
    receipt: Mapping[str, Any]


def _verify_prepared(item: PreparedSharedNativeFrame,
                     readiness: Mapping[str, Any]) -> None:
    if not isinstance(item, PreparedSharedNativeFrame):
        raise TypeError("Native response fitting requires certified prepared-frame receipts")
    record = item.receipt
    frame_key = f"{item.frame.run_id}:{record.get('frame_index')}"
    physical = readiness.get("verified_frame_bindings", {}).get(frame_key)
    if (physical is None or
            any(record.get(key) != physical.get(key) for key in (
                "frame_geometry_receipt_sha256", "facies_mask_sha256",
                "domain_sha256", "depth_sha256", "porosity_sha256", "pixel_area_m2",
                "pixel_area_m2_sha256"))
            or record.get("physical_certificate_sha256")
            != readiness.get("physical_facies_certificate_sha256")
            or record.get("contract") != CONTRACT
            or record.get("readiness_sha256") != readiness["readiness_sha256"]
            or record.get("receipt_sha256") != _json_sha({
                key: value for key, value in record.items()
                if key != "receipt_sha256"})
            or record.get("run") != item.frame.run_id
            or record.get("native_manifest_sha256")
            != readiness["source_manifest_sha256"].get(item.frame.run_id)):
        raise ValueError("Native prepared frame is not sealed to this RAW source/readiness")


def extract_positive_samples(
    *, run: str, frame_index: int, scientific_rgb: np.ndarray,
    registered_baseline_rgb: np.ndarray, optical_valid: np.ndarray,
    certified_domain: np.ndarray, facies: np.ndarray,
    independent_aqueous_positive: np.ndarray, certificate: Mapping[str, Any],
    source_binding: Mapping[str, str], stride_px: int = 8,
) -> list[PathSampleBatch]:
    """Sample full-resolution float RAW centres within certified facies cores.

    `independent_aqueous_positive` is a reviewed, source-aligned mask; it is
    never inferred from phase colors.  The stride selects native centres and
    never downsamples or smooths a source image.  The certificate is checked
    against the exact frame geometry and facies mask SHA supplied by the caller.
    """
    radius = required_interior_radius(certificate)
    keys = ("native_manifest_sha256", "frame_sidecar_sha256",
            "baseline_sidecar_sha256", "geometry_receipt_sha256",
            "facies_mask_sha256", "aqueous_label_sha256")
    for key in keys:
        _sha(source_binding.get(key), key)
    frame_rows = [row for row in certificate.get("verified_frames", [])
                  if row.get("run") == run and row.get("frame_index") == frame_index
                  and row.get("geometry_receipt_sha256") == source_binding["geometry_receipt_sha256"]]
    if (len(frame_rows) != 1 or frame_rows[0].get("facies_mask_sha256")
            != source_binding["facies_mask_sha256"]):
        raise ValueError("Physical certificate does not bind this facies/frame geometry")
    if (frame_rows[0].get("positive_label_independent") is not True
            or frame_rows[0].get("aqueous_label_sha256") != source_binding["aqueous_label_sha256"]
            or frame_rows[0].get("native_manifest_sha256")
            != source_binding["native_manifest_sha256"]):
        raise ValueError("Independent aqueous label lacks exact native source binding")
    rgb = scientific_rgb
    base = registered_baseline_rgb
    if (not isinstance(rgb, np.ndarray) or not isinstance(base, np.ndarray)
            or rgb.dtype != np.float32 or base.dtype != np.float32
            or rgb.ndim != 3 or rgb.shape[-1] != 3 or base.shape != rgb.shape):
        raise TypeError("Both scientific inputs must be full-native float32 RGB")
    shape = rgb.shape[:2]
    if type(stride_px) is not int or stride_px < 1:
        raise ValueError("Native-centre stride must be positive")
    for key, array in (("optical_valid", optical_valid),
                       ("certified_domain", certified_domain),
                       ("independent_aqueous_positive", independent_aqueous_positive)):
        if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != np.bool_:
            raise TypeError(f"{key} must be a full-native boolean mask")
    if not isinstance(facies, np.ndarray) or facies.shape != shape or not np.issubdtype(facies.dtype, np.integer):
        raise TypeError("Facies must be a full-native categorical image")
    if array_sha256(facies) != source_binding["facies_mask_sha256"]:
        raise ValueError("Facies pixels differ from their certified native array")
    if array_sha256(independent_aqueous_positive) != source_binding["aqueous_label_sha256"]:
        raise ValueError("Aqueous-positive pixels differ from their independent label")
    selected = optical_valid & certified_domain & independent_aqueous_positive
    grid = np.zeros(shape, bool)
    grid[::stride_px, ::stride_px] = True
    selected &= grid
    selected &= np.isfinite(rgb).all(axis=2) & np.isfinite(base).all(axis=2)
    selected &= (np.minimum(rgb.min(axis=2), base.min(axis=2)) > 0)
    selected &= (np.maximum(rgb.max(axis=2), base.max(axis=2)) <= 1)
    result = []
    for label in POROUS_FACIES:
        mask = np.asarray(facies == label, np.uint8)
        interior = cv2.distanceTransform(mask, cv2.DIST_L2, 5) >= radius
        values = (rgb[selected & interior] - base[selected & interior]).astype(np.float32)
        if len(values):
            result.append(PathSampleBatch(run, frame_index, label, values,
                **{key: source_binding[key] for key in keys}))
    return result


def extract_positive_from_reader(
    reader: Any, frame_index: int, *, facies: np.ndarray,
    certified_domain: np.ndarray, independent_aqueous_positive: np.ndarray,
    certificate: Mapping[str, Any], stride_px: int = 8,
) -> list[PathSampleBatch]:
    """Stream an exact V5NativeScientificInputReader frame into path fitting.

    The reader verifies full-native float sidecars, validity masks, JPEG
    display bindings and the per-frame baseline homography before this method
    sees any pixels.  No JPEG RGB values enter the pooled path.
    """
    from darsia.fluidflower.concentration_inputs import V5NativeScientificInputReader

    if not isinstance(reader, V5NativeScientificInputReader):
        raise TypeError("Use the source-bound V5 native scientific input reader")
    frame = reader.load_frame(frame_index)
    run = str(reader.run["run"])
    row = [item for item in certificate.get("verified_frames", [])
           if item.get("run") == run and item.get("frame_index") == frame_index]
    if len(row) != 1 or row[0].get("domain_sha256") != array_sha256(certified_domain):
        raise ValueError("Certified physical domain differs from native frame geometry")
    source = frame.source_reference
    binding = dict(
        native_manifest_sha256=source["native_manifest_sha256"],
        frame_sidecar_sha256=source["frame_scientific_float"]["sidecar_sha256"],
        baseline_sidecar_sha256=source["baseline_scientific_float"]["sidecar_sha256"],
        geometry_receipt_sha256=source["frame_geometry_receipt_sha256"],
        facies_mask_sha256=array_sha256(facies),
        aqueous_label_sha256=array_sha256(independent_aqueous_positive),
    )
    return extract_positive_samples(
        run=run, frame_index=frame_index,
        scientific_rgb=frame.scientific_rgb,
        registered_baseline_rgb=frame.registered_baseline_rgb,
        optical_valid=frame.valid_mask,
        certified_domain=certified_domain,
        facies=facies,
        independent_aqueous_positive=independent_aqueous_positive,
        certificate=certificate, source_binding=binding,
        stride_px=stride_px)


def validate_readiness(
    *, fit_runs: Sequence[str], holdout_run: str,
    chemistry_groups: Mapping[str, str], certificate: Mapping[str, Any],
    physical_binding: Mapping[str, Any], batches: Sequence[PathSampleBatch],
    holdout_native_manifest_sha256: str,
    native_frame_counts: Mapping[str, int],
    required_facies: Sequence[int] = POROUS_FACIES, min_runs_per_facies: int = 2,
    min_samples_per_run_facies: int = 16,
) -> dict[str, Any]:
    """Fail closed before path building *and* before shared mass fitting."""
    fit = tuple(fit_runs)
    group = validate_split(fit, holdout_run, chemistry_groups)
    _sha(holdout_native_manifest_sha256, "holdout_native_manifest_sha256")
    radius = required_interior_radius(certificate)
    if certificate.get("absolute_scale_certified") is not True:
        raise ValueError("Absolute pixel/physical scale is not certified")
    if certificate.get("native_domain_certified") is not True:
        raise ValueError("Per-frame physical sand domain is not certified")
    if physical_binding.get("status") != "certified" or physical_binding.get(
        "independent_of_color_response") is not True:
        raise ValueError("Physical depth, porosity and geometry are not certified")
    for key in ("geometry_sha256", "depth_sha256", "porosity_sha256", "scale_sha256"):
        _sha(physical_binding.get(key), key)
    required = tuple(required_facies)
    if (not required or len(set(required)) != len(required)
            or any(label not in POROUS_FACIES for label in required)):
        raise ValueError("Required facies must be distinct porous IDs")
    if sorted(certificate.get("porous_facies_ids", [])) != sorted(required):
        raise ValueError("Pooled paths must cover every independently certified porous facies")
    receipts = {(str(row["run"]), int(row["frame_index"])): row
                for row in certificate.get("verified_frames", [])}
    if len(receipts) != len(certificate.get("verified_frames", [])):
        raise ValueError("Physical certificate repeats a frame")
    sources: dict[str, str] = {}
    counts: dict[int, dict[str, int]] = {label: {} for label in required}
    bindings = []
    for batch in batches:
        if batch.run not in fit:
            raise ValueError("Holdout or unselected run entered pooled path fitting")
        frame_receipt = receipts.get((batch.run, batch.frame_index))
        if (frame_receipt is None or frame_receipt.get("geometry_receipt_sha256")
                != batch.geometry_receipt_sha256):
            raise ValueError("Path batch has no exact certified frame geometry")
        if frame_receipt.get("facies_mask_sha256") != batch.facies_mask_sha256:
            raise ValueError("Path batch uses a different per-frame facies map")
        if (frame_receipt.get("positive_label_independent") is not True
                or frame_receipt.get("aqueous_label_sha256") != batch.aqueous_label_sha256
                or frame_receipt.get("native_manifest_sha256") != batch.native_manifest_sha256):
            raise ValueError("Path batch has no independent source-aligned aqueous label")
        prior = sources.setdefault(batch.run, batch.native_manifest_sha256)
        if prior != batch.native_manifest_sha256:
            raise ValueError("One run has conflicting native source manifests")
        if batch.facies in counts:
            counts[batch.facies][batch.run] = counts[batch.facies].get(batch.run, 0) + len(batch.relative_rgb)
        bindings.append(batch.provenance())
    if set(sources) != set(fit):
        raise ValueError("Every fit run needs source-bound positive path evidence")
    sources[holdout_run] = holdout_native_manifest_sha256
    if set(run for run, _ in receipts) != set(sources):
        raise ValueError("Physical certificate must cover every fit and holdout run")
    if set(native_frame_counts) != set(sources) or any(
        type(count) is not int or count < 1 or
        {index for (source_run, index) in receipts if source_run == run} != set(range(count))
        for run, count in native_frame_counts.items()
    ):
        raise ValueError("Physical certificate must cover every native frame in every run")
    if any(row.get("native_manifest_sha256") != sources[run]
           for (run, _), row in receipts.items()):
        raise ValueError("Physical certificate uses a different native source revision")
    frame_bindings = {}
    for (run, index), row in receipts.items():
        for key in ("facies_mask_sha256", "domain_sha256", "depth_sha256",
                    "porosity_sha256"):
            _sha(row.get(key), f"{run}:{index}:{key}")
        area = row.get("pixel_area_m2")
        area_sha = row.get("pixel_area_m2_sha256")
        if area_sha is not None:
            _sha(area_sha, f"{run}:{index}:pixel_area_m2_sha256")
        elif not isinstance(area, (int, float)) or not np.isfinite(area) or area <= 0:
            raise ValueError("Every frame needs certified positive physical pixel area")
        frame_bindings[f"{run}:{index}"] = dict(
            frame_geometry_receipt_sha256=row["geometry_receipt_sha256"],
            facies_mask_sha256=row["facies_mask_sha256"],
            domain_sha256=row["domain_sha256"], depth_sha256=row["depth_sha256"],
            porosity_sha256=row["porosity_sha256"],
            pixel_area_m2=None if area_sha is not None else float(area),
            **({"pixel_area_m2_sha256": area_sha} if area_sha is not None else {}))
    for label, by_run in counts.items():
        if sum(count >= min_samples_per_run_facies for count in by_run.values()) < min_runs_per_facies:
            raise ValueError(f"Facies {label} lacks positive evidence in independent runs")
    report = dict(contract=CONTRACT, status="ready-for-provisional-fit",
        representation=REPRESENTATION, fit_runs=list(fit),
        whole_run_holdout=holdout_run, chemistry_group=group,
        facies=list(required), certified_interior_radius_px=radius,
        physical_binding=dict(physical_binding),
        physical_facies_certificate_sha256=_json_sha(certificate),
        verified_frame_bindings=frame_bindings,
        source_manifest_sha256=sources, sample_provenance=bindings,
        native_frame_counts=dict(native_frame_counts),
        sample_counts_by_facies_run=counts,
        holdout_postinjection_used_in_fit=False, full_end_to_end_holdout=False,
        phase_model_mass_eligible=False)
    report["readiness_sha256"] = _json_sha(report)
    return report


def _path_knots(samples_by_run: Mapping[str, np.ndarray], *, min_span: float) -> np.ndarray:
    """Equal-run robust color centres along a shared optical trajectory."""
    per_run: list[np.ndarray] = []
    for run in sorted(samples_by_run):
        values = np.asarray(samples_by_run[run], np.float32)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("Invalid scientific color-difference samples")
        norms = np.linalg.norm(values, axis=1)
        if np.quantile(norms, .95) < min_span:
            raise ValueError(f"Run {run} has no identifiable color-path span")
        anchors = []
        for q in (.25, .55, .85, .98):
            lo, hi = np.quantile(norms, [max(.0, q-.075), min(1., q+.075)])
            band = values[(norms >= lo) & (norms <= hi)]
            anchors.append(np.median(band, axis=0))
        per_run.append(np.asarray(anchors, np.float32))
    centres = np.median(np.stack(per_run), axis=0)
    direction = centres[-1]
    unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
    projected = centres @ unit
    if projected[-1] < min_span or np.any(np.diff(projected) <= min_span * .005):
        raise ValueError("Pooled path is weak, reversed or nonmonotone across optical anchors")
    if max(np.linalg.norm(centres - projected[:, None] * unit, axis=1)) > .65 * projected[-1]:
        raise ValueError("Pooled path turns too strongly for the provisional shared response")
    return np.vstack((np.zeros(3, np.float32), centres)).astype(np.float32)


def fit_pooled_paths(
    batches: Sequence[PathSampleBatch], readiness: Mapping[str, Any],
    *, min_span: float = .005,
) -> dict[int, darsia.ColorPathInterpolation]:
    """Learn one path per facies, with *equal run weight*, on fit runs only."""
    _require_readiness(readiness)
    if not np.isfinite(min_span) or min_span <= 0:
        raise ValueError("Positive minimum optical span required")
    by_facies: dict[int, dict[str, list[np.ndarray]]] = {
        int(label): {} for label in readiness["facies"]}
    if any(batch.run == readiness["whole_run_holdout"] for batch in batches):
        raise ValueError("Holdout samples cannot enter a pooled path")
    expected = {json.dumps(row, sort_keys=True) for row in readiness["sample_provenance"]}
    actual = {json.dumps(batch.provenance(), sort_keys=True) for batch in batches}
    if expected != actual or len(expected) != len(batches):
        raise ValueError("Path samples differ from the sealed readiness inventory")
    for batch in batches:
        if batch.facies in by_facies:
            by_facies[batch.facies].setdefault(batch.run, []).append(batch.relative_rgb)
    paths = {}
    for label, runs in by_facies.items():
        knots = _path_knots({run: np.concatenate(chunks) for run, chunks in runs.items()},
                            min_span=min_span)
        color_path = darsia.ColorPath(colors=[row for row in knots], mode="rgb",
                                      name=f"FF_AC pooled RAW facies {label}")
        paths[label] = darsia.ColorPathInterpolation(
            color_path=color_path, color_mode=darsia.ColorMode.RELATIVE,
            values=np.asarray(color_path.relative_distances))
    return paths


def save_native_path_package(
    folder: Path, paths: Mapping[int, darsia.ColorPathInterpolation],
    readiness: Mapping[str, Any], *, flash_bounds: Mapping[str, float],
    ignored_nonporous_labels: Sequence[int] = (0,),
) -> dict[str, Any]:
    """Write a new isolated native-compatible package; never copy JPEG paths.

    Flash bounds are explicit because a color path alone does not determine
    gas saturation or dissolved concentration.  The same identity signal
    response is saved for every facies; native shared scaling is fitted later.
    """
    _require_readiness(readiness)
    if set(paths) != set(readiness["facies"]):
        raise ValueError("Native path package must cover every certified porous facies")
    names = ("min_value_aq", "max_value_aq", "min_value_g", "max_value_g")
    if set(flash_bounds) != set(names) or not np.isfinite(list(flash_bounds.values())).all():
        raise ValueError("All finite shared flash bounds must be specified")
    if (flash_bounds["max_value_aq"] <= flash_bounds["min_value_aq"]
            or flash_bounds["max_value_g"] <= flash_bounds["min_value_g"]):
        raise ValueError("Shared flash intervals must be positive")
    nonporous = tuple(int(label) for label in ignored_nonporous_labels)
    if len(set(nonporous)) != len(nonporous) or any(label in paths or label < 0 for label in nonporous):
        raise ValueError("Ignored nonporous labels must be distinct from certified sand")
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError("Use an immutable new path-package directory")
    folder.mkdir(parents=True)
    for label, path in sorted(paths.items()):
        if path.color_mode != darsia.ColorMode.RELATIVE:
            raise ValueError("RAW paths must use the relative native ColorMode")
        path.save(folder / "color_path_interpretation" / f"color_path_interpretation_{label}")
        darsia.PWTransformation(supports=[0., 1.], values=[0., 1.]).save(
            folder / "signal_model" / f"signal_model_{label}")
    # DarSIA loads a function for every categorical label in the image,
    # including the frame.  These dummy paths can never contribute mass:
    # NativeConcentrationAnalysis excludes the nonporous domain explicitly.
    for label in nonporous:
        dummy = darsia.ColorPathInterpolation(
            color_path=darsia.ColorPath(colors=[np.zeros(3),
                                                np.full(3, 1e-3)], mode="rgb"),
            color_mode=darsia.ColorMode.RELATIVE, values=[0., 1.])
        dummy.save(folder / "color_path_interpretation" / f"color_path_interpretation_{label}")
        darsia.PWTransformation(supports=[0., 1.], values=[0., 1.]).save(
            folder / "signal_model" / f"signal_model_{label}")
    darsia.SimpleFlash(**flash_bounds).save(folder / "flash" / "flash")
    metadata = dict(basis=ColorEmbeddingBasis.FACIES.value,
        label_ids=sorted((*paths, *nonporous)), ignore_labels=list(nonporous),
        color_mode=darsia.ColorMode.RELATIVE.value, contract=CONTRACT,
        representation=REPRESENTATION, provisional=True,
        shared_signal_response="identity 0..1 for every facies; two shared native mass scales",
        post_injection_dilution="none; separately calibrated method required",
        readiness=readiness)
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    inventory = {str(path.relative_to(folder)).replace("\\", "/"): _digest(path)
                 for path in sorted(folder.rglob("*")) if path.is_file()}
    return dict(path=str(folder.resolve()), contract=CONTRACT, files_sha256=inventory,
                mass_fit_performed=False, calibration_status="provisional-unfitted")


def prepare_verified_raw_native_frame(
    *, run: str, scientific_frame: Any, facies: np.ndarray,
    domain: np.ndarray, depth_m: np.ndarray, porosity: np.ndarray,
    dimensions_m: tuple[float, float], certificate: Mapping[str, Any],
    readiness: Mapping[str, Any], path_folder: Path,
    pressure_bar: float, temperature_c: float,
    pressure_gradient_bar_per_m: float = 0.,
    temperature_gradient_c_per_m: float = 0.,
) -> PreparedSharedNativeFrame:
    """Bridge exact V5 RAW and per-frame physical fields to native DarSIA.

    Transport NaNs outside the certified optical support are set to baseline
    solely to satisfy the native image container.  Any selected CO2 phase on
    invalid/uncertified support is a hard failure, not a silent zero mass.
    """
    _require_readiness(readiness)
    if _json_sha(certificate) != readiness["physical_facies_certificate_sha256"]:
        raise ValueError("Native physical certificate differs from sealed path-fit readiness")
    package = json.loads((Path(path_folder) / "metadata.json").read_text(encoding="utf-8"))
    if (package.get("contract") != CONTRACT or package.get("representation") != REPRESENTATION
            or package.get("readiness", {}).get("readiness_sha256")
            != readiness["readiness_sha256"]):
        raise ValueError("Native path package differs from the certified pooled RAW fit")
    if run not in {*readiness["fit_runs"], readiness["whole_run_holdout"]}:
        raise ValueError("Unselected experiment cannot use pooled calibration")
    provenance = scientific_frame.source_reference
    if (provenance.get("input_domain") != "ffac-v5-camera-linear-float32-raw-v1"
            or provenance.get("native_manifest_sha256")
            != readiness["source_manifest_sha256"].get(run)):
        raise ValueError("Native scientific RAW source differs from sealed path fit")
    frame_rows = [row for row in certificate.get("verified_frames", [])
                  if row.get("run") == run
                  and row.get("frame_index") == scientific_frame.frame_index]
    if len(frame_rows) != 1:
        raise ValueError("Physical certificate lacks this exact native frame")
    row = frame_rows[0]
    if row.get("geometry_receipt_sha256") != provenance.get("frame_geometry_receipt_sha256"):
        raise ValueError("Per-frame sand geometry differs from phase/source binding")
    arrays = dict(facies=facies, domain=domain, depth=depth_m, porosity=porosity)
    shape = scientific_frame.phase_labels.shape
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.shape != shape:
            raise ValueError(f"{name} is not on the exact native frame grid")
        key = "facies_mask_sha256" if name == "facies" else f"{name}_sha256"
        if row.get(key) != array_sha256(array):
            raise ValueError(f"{name} differs from the certified physical array")
    if domain.dtype != np.bool_ or not np.issubdtype(facies.dtype, np.integer):
        raise TypeError("Certified domain/facies need native boolean/integer types")
    if not set(map(int, np.unique(facies[domain]))).issubset(set(readiness["facies"])):
        raise ValueError("Registered sand includes a facies without a pooled RAW path")
    pixel_area = float(dimensions_m[0]) * float(dimensions_m[1]) / domain.size
    area_values = None
    if row.get("pixel_area_m2_sha256"):
        from darsia.fluidflower.phase_pixel_area import validate_pixel_area
        area_values = validate_pixel_area(scientific_frame.context.get("pixel_area_m2"), shape)
        if np.isscalar(area_values) or array_sha256(area_values) != row["pixel_area_m2_sha256"]:
            raise ValueError("Physical pixel-area map differs from certified geometry")
    elif provenance.get("pixel_area") is not None:
        raise ValueError("New rig-plane area requires a certified pixel-area array SHA, not an old scalar certificate")
    elif (not np.isfinite(pixel_area) or pixel_area <= 0
          or not np.isclose(pixel_area, row.get("pixel_area_m2", np.nan), rtol=0, atol=1e-12)):
        raise ValueError("Physical pixel area/scale differs from certified geometry")
    valid = scientific_frame.valid_mask & scientific_frame.supported_mask & domain
    phase = scientific_frame.phase_labels
    if np.any((phase != 0) & ~valid):
        raise ValueError("CO2 phase intersects invalid or uncertified support; mass is unresolved")
    rgb = np.asarray(scientific_frame.scientific_rgb)
    base = np.asarray(scientific_frame.registered_baseline_rgb)
    if (rgb.dtype != np.float32 or base.dtype != np.float32
            or rgb.shape != (*shape, 3) or base.shape != rgb.shape):
        raise TypeError("Native mass input must be exact float32 normalized RAW")
    if (not np.isfinite(rgb[valid]).all() or not np.isfinite(base[valid]).all()
            or np.any(rgb[valid] < 0) or np.any(base[valid] < 0)
            or np.any(rgb[valid] > 1) or np.any(base[valid] > 1)):
        raise ValueError("Certified optical support contains invalid camera-linear RAW")
    baseline_filled = np.where(valid[..., None], base, 0.).astype(np.float32)
    image_filled = np.where(valid[..., None], rgb, baseline_filled).astype(np.float32)
    analysis = native.NativeConcentrationAnalysis(
        path_folder, baseline_rgb=baseline_filled, labels=facies,
        domain=domain, depth_m=depth_m, porosity=porosity,
        dimensions_m=dimensions_m, basis="facies",
        calibration_status="provisional shared RAW Color Paths")
    prepared = analysis.prepare(
        image_filled, phase, run_id=run,
        time_hours=float(scientific_frame.time_hours),
        pressure_bar=pressure_bar, temperature_c=temperature_c,
        nominal_injected_mass_g=scientific_frame.nominal_injected_mass_g,
        pressure_gradient_bar_per_m=pressure_gradient_bar_per_m,
        temperature_gradient_c_per_m=temperature_gradient_c_per_m,
        **({"pixel_area_m2": area_values} if area_values is not None else {}))
    receipt = dict(contract=CONTRACT, readiness_sha256=readiness["readiness_sha256"],
        run=run, frame_index=scientific_frame.frame_index,
        native_manifest_sha256=provenance["native_manifest_sha256"],
        frame_geometry_receipt_sha256=provenance["frame_geometry_receipt_sha256"],
        frame_scientific_source=provenance.get("frame_scientific_float"),
        baseline_scientific_source=provenance.get("baseline_scientific_float"),
        phase_sha256=provenance.get("phase_sha256"),
        physical_certificate_sha256=readiness["physical_facies_certificate_sha256"],
        facies_mask_sha256=row["facies_mask_sha256"],
        domain_sha256=row["domain_sha256"], depth_sha256=row["depth_sha256"],
        porosity_sha256=row["porosity_sha256"], pixel_area_m2=None if area_values is not None else pixel_area,
        **({"pixel_area_m2_sha256": row["pixel_area_m2_sha256"]} if area_values is not None else {}),
        native_path_files_sha256=analysis.source_hashes,
        invalid_pixel_policy="fill outside selected supported phase with baseline; never fit",
        phase_labels_modified=False)
    receipt["receipt_sha256"] = _json_sha(receipt)
    return PreparedSharedNativeFrame(analysis, prepared, receipt)


def fit_shared_native_response(
    prepared_by_run: Mapping[str, Sequence[PreparedSharedNativeFrame]],
    readiness: Mapping[str, Any], *, overflow_references: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """One response for all fit runs, leaving the held-out run untouched.

    Each native analysis may have a different *per-frame* registered facies
    image.  Thus summaries are reduced frame by frame rather than assuming a
    single static analysis object for an entire experiment.  Overflow is
    compare-only and never appears in the optimization residual.
    """
    _require_readiness(readiness)
    if set(prepared_by_run) != set(readiness["fit_runs"]):
        raise ValueError("Native mass fit must use exactly the path-fit runs, no holdout")
    if any(not rows for rows in prepared_by_run.values()):
        raise ValueError("Every fit run requires native prepared frames")
    summaries, provenance = {}, {}
    shared_files = None
    for run, rows in prepared_by_run.items():
        summaries[run] = []
        indices = [item.receipt.get("frame_index") for item in rows]
        if (any(type(index) is not int for index in indices)
                or indices != list(range(readiness["native_frame_counts"][run]))):
            raise ValueError("Each native run must have its complete chronological frame timeline")
        for item in rows:
            _verify_prepared(item, readiness)
            analysis, frame = item.analysis, item.frame
            if frame.run_id != run:
                raise ValueError("Native frame belongs to another fit run")
            metadata = json.loads((Path(analysis.folder) / "metadata.json").read_text(encoding="utf-8"))
            if (metadata.get("contract") != CONTRACT or metadata.get("representation") != REPRESENTATION
                    or metadata.get("readiness", {}).get("readiness_sha256")
                    != readiness["readiness_sha256"]):
                raise ValueError("Native optical path package is not the sealed pooled RAW calibration")
            record = analysis.provenance()
            if record.get("calibration_status") != "provisional shared RAW Color Paths":
                raise ValueError("A historical JPEG-native calibration entered the RAW fit")
            files = record["calibration_files"]
            if shared_files is None:
                shared_files = files
            elif files != shared_files:
                raise ValueError("All runs must use one immutable pooled path package")
            provenance[run] = record
            summaries[run].append(analysis.summarize_frame(frame))
    fit = native.fit_summaries(summaries, provenance, scope="shared",
        overflow_mode="compare" if overflow_references else "off",
        references=overflow_references, unknown_policy="complete_frames")
    fit["method_contract"] = CONTRACT
    fit["readiness"] = dict(readiness)
    fit["whole_run_holdout"] = readiness["whole_run_holdout"]
    fit["path_and_mass_fit_exclude_holdout"] = True
    fit["post_injection_dilution"] = "none"
    fit["mass_eligible"] = False
    return fit


def score_whole_run_holdout(
    frames: Sequence[PreparedSharedNativeFrame],
    readiness: Mapping[str, Any], fit: Mapping[str, Any],
) -> dict[str, Any]:
    """Exact native evaluation on the excluded run, never another fit.

    This is conditional validation: the fixed phase model may have seen that
    experiment during *phase* training.  It must not be described as fully
    independent end-to-end validation.
    """
    _require_readiness(readiness)
    holdout = readiness["whole_run_holdout"]
    if (fit.get("method_contract") != CONTRACT
            or fit.get("readiness", {}).get("readiness_sha256")
            != readiness["readiness_sha256"]
            or fit.get("whole_run_holdout") != holdout
            or holdout in fit.get("runs", ())
            or not fit.get("fit_performed")
            or fit.get("overflow_mode") not in ("off", "compare")):
        raise ValueError("Holdout scoring requires a sealed fit that excluded this run")
    if not frames:
        raise ValueError("Holdout requires native prepared frames")
    indices = [item.receipt.get("frame_index") for item in frames]
    if (any(type(index) is not int for index in indices)
            or indices != list(range(readiness["native_frame_counts"][holdout]))):
        raise ValueError("Holdout needs its complete chronological native timeline")
    parameters = native.CalibrationParameters(**fit["parameters"])
    predictions = []
    for item in frames:
        _verify_prepared(item, readiness)
        analysis, frame = item.analysis, item.frame
        if frame.run_id != holdout:
            raise ValueError("Holdout evaluation received a fit-run frame")
        metadata = json.loads((Path(analysis.folder) / "metadata.json").read_text(encoding="utf-8"))
        if (metadata.get("contract") != CONTRACT
                or metadata.get("readiness", {}).get("readiness_sha256")
                != readiness["readiness_sha256"]):
            raise ValueError("Holdout has a different optical calibration package")
        result = analysis.evaluate(frame, parameters, fields=False)
        result["nominal_mass_residual_g"] = (
            None if frame.nominal_injected_mass_g is None else
            result["total_mass_g"] - frame.nominal_injected_mass_g)
        predictions.append(result)
    return dict(contract=CONTRACT, run=holdout,
        validation_scope="whole-run holdout conditional on fixed phase model",
        path_fit_used_holdout=False, native_mass_fit_used_holdout=False,
        predictions=predictions, post_injection_dilution="none",
        mass_eligible=False)
