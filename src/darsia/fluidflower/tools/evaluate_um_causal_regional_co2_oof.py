"""Independent native-map QA of frozen FF_UM grouped out-of-fold CO2 heads.

This evaluator consumes a completed isolated fit job. It replays all exact
native optical prefixes in run order, but constructs phase maps only on frames
with dense, source-aligned annotations. The published optical and gas heads,
gas confirmation, and both final guards remain frozen. Its result is a
review-only OOF report; it never writes a model, map, or Studio registration.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import cv2
import numpy as np
from PIL import Image
from scipy.special import expit

from darsia.fluidflower import phase_causal_temporal_features as causal
from darsia.fluidflower import phase_free_gas_output_guard as free_gas
from darsia.fluidflower import phase_optical_regions as regional
from darsia.fluidflower import phase_region_output_guards as guards
from darsia.fluidflower import phase_regional_gas as gas
from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import audit_um_temporal_training_cohort as cohort_audit
from darsia.fluidflower.tools import fit_um_causal_regional_co2 as fitted_job
from darsia.fluidflower.tools import phase_training_review_comparison as comparison
from darsia.fluidflower.tools.derive_um_temporal_candidate import _review_evidence


FORMAT = "ffum-causal-regional-co2-native-oof-qa-v1"
PRIMARY_LAMBDA = "0.001"
ARMS = ("static20", "causal30")
ROLES = {"aqueous": "aqueous_positive", "gas": "gas_co2_positive",
         "current_experiment_co2_artifact": "carryover_negative",
         "water": "water_negative"}
EXPECTED_REGIONS = Counter(aqueous_positive=496, gas_co2_positive=32,
                           carryover_negative=45, water_negative=68)


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    return sha256_file(Path(path))


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _crop_rgb(path: Path, box: list[int]) -> np.ndarray:
    x, y, width, height = map(int, box)
    with Image.open(path) as image:
        if image.mode not in ("RGB", "RGBA") or image.size[0] < x + width or image.size[1] < y + height:
            raise ValueError(f"Invalid corrected native RGB photograph: {path}")
        return np.asarray(image.convert("RGB").crop((x, y, x + width, y + height)))


def _fit_values(entry: dict, n_features: int, regularization: float) -> dict:
    """Reject partial, nonconverged, or changed numerical fit certificates."""
    if set(entry) != {"fit", "diagnostics"}:
        raise ValueError("Unexpected fitted head envelope")
    fit, diagnostics = entry["fit"], entry["diagnostics"]
    vectors = [np.asarray(fit[name], dtype=np.float64)
               for name in ("mean", "scale", "coefficients")]
    if (not fit.get("converged") or not diagnostics.get("converged")
            or fit.get("backend") != "cpu" or fit.get("precision") != "float64"
            or float(fit.get("regularization", -1)) != regularization
            or any(v.shape != (n_features,) or not np.isfinite(v).all() for v in vectors)
            or (vectors[1] <= 0).any()
            or not np.isfinite(float(fit["intercept"]))
            or not np.isfinite(float(fit["objective"]))):
        raise ValueError("Nonconverged or incompatible frozen OOF fit")
    certificate = fit.get("convergence_certificate")
    retry = diagnostics.get("pure_irls_retry") or {}
    if (diagnostics.get("mode") != "pure-region-irls"
            or diagnostics.get("weak_bags") != 0
            or not retry.get("final_converged")
            or int(retry.get("final_iterations", -1)) != int(fit["iterations"])
            or int(fit["iterations"]) <= 0
            or int(fit["iterations"]) >= int(retry.get("retry_max_iter") or retry.get("standard_max_iter", 40))):
        raise ValueError("OOF fit lacks matching pure CPU float64 optimizer diagnostics")
    # CPU float64 Newton steps return certificate=None on ordinary tolerance
    # convergence. A separate certificate is only emitted for plateau recovery.
    if certificate is not None and (not isinstance(certificate, dict)
                                    or not certificate.get("certified")):
        raise ValueError("OOF fit has an unverified numerical plateau certificate")
    return fit


def verify_job(job: Path) -> dict:
    """Validate the complete immutable fit artifact chain before any replay."""
    job = Path(job).resolve()
    protocol_path, extraction_path = job / "protocol.json", job / "extraction.json"
    regions_path, fits_path = job / "regions.json", job / "fold_fits.json"
    result_path = job / "result.json"
    protocol, extraction, regions, fits, result = (_read(path) for path in
        (protocol_path, extraction_path, regions_path, fits_path, result_path))
    if (protocol.get("format") != fitted_job.FORMAT
            or extraction.get("format") != fitted_job.FORMAT
            or fits.get("format") != fitted_job.FORMAT
            or regions.get("format") != fitted_job.FORMAT
            or result.get("format") != fitted_job.FORMAT
            or Path(protocol["output"]).resolve() != job
            or result.get("status") != "exploratory_cv_fitted_pending_native_qa"
            or result.get("no_phase_maps") is not True
            or result.get("no_publication") is not True):
        raise ValueError("Incomplete or incompatible isolated FF_UM fit job")
    if (result["protocol_sha256"] != _sha(protocol_path)
            or result["extraction_sha256"] != _sha(extraction_path)
            or result["fold_fits_sha256"] != _sha(fits_path)
            or fits["protocol_sha256"] != _sha(protocol_path)
            or fits["extraction_sha256"] != _sha(extraction_path)
            or fits["regions_sha256"] != _sha(regions_path)
            or fits["samples_sha256"] != extraction["samples_sha256"]
            or extraction["regions_sha256"] != _sha(regions_path)
            or extraction["samples_sha256"] != _sha(Path(extraction["samples"]))
            or Path(extraction["samples"]).resolve() != (job / "samples.npz").resolve()):
        raise ValueError("Frozen fit/extraction payload SHA chain changed")
    parent = Path(protocol["parent"]).resolve()
    model_id, model_sha = _model_identity(parent)
    if (model_id != protocol["parent_model_id"]
            or model_sha != protocol["parent_model_sha256"] != fitted_job.PARENT_SHA256
            or protocol["parent_source_manifest_sha256"] != fitted_job.SOURCE_SHA256
            or protocol["cohort_audit_sha256"] != fitted_job.AUDIT_SHA256
            or _sha(Path(causal.__file__)) != fitted_job.CAUSAL_SHA256
            or _sha(parent / "phase_optical_regions.json") != protocol["parent_profile_sha256"]
            or _sha(parent / "regional_model.json") != protocol["parent_regional_model_sha256"]
            or _sha(parent / "native_source_receipt.json") != protocol["parent_native_receipt_sha256"]
            or _sha(parent / "parent_provenance/training_request.json") != protocol["frozen_optical_request_sha256"]
            or _sha(Path(protocol["comparison"])) != protocol["comparison_sha256"]
            or _sha(Path(protocol["cohort_audit"])) != protocol["cohort_audit_sha256"]
            or _sha(Path(protocol["protocol_document"])) != protocol["protocol_document_sha256"]):
        raise ValueError("Frozen parent, label source, or protocol changed")
    source_receipt = _read(parent / "native_source_receipt.json")
    if (source_receipt["source_manifest_sha256"] != fitted_job.SOURCE_SHA256
            or source_receipt["runs"] != 13 or source_receipt["frames"] != 261):
        raise ValueError("Published source campaign identity changed")
    code_paths = fitted_job._frozen_code_paths()
    if {p.name: _sha(p) for p in code_paths} != protocol["code_sha256"]:
        raise ValueError("Feature extraction or fit code changed after freezing")
    prior = protocol["review_inventory"]
    if (Path(prior["root"]).resolve() != (parent.parent.parent / "reviews").resolve()
            or len(prior["inventory_sha256"]) != 64):
        raise ValueError("Frozen prelaunch review inventory identity changed")
    if (protocol["feature_contract"]["static20"] != list(regional.FEATURE_NAMES)
            or protocol["feature_contract"]["causal30"] !=
               list(regional.FEATURE_NAMES) + list(causal.FEATURE_NAMES)
            or protocol["feature_contract"]["causal_format"] != causal.VERSION
            or protocol["folds"] != {key: list(value) for key, value in fitted_job.FOLDS.items()}
            or protocol["regularization"] != list(fitted_job.REGULARIZATION)):
        raise ValueError("Frozen feature, fold, or regularization contract changed")
    if (len(extraction["frames"]) != 261 or extraction["frames_done"] != 261
            or extraction["runs_done"] != 13 or extraction["sampled_regions"] != 641
            or len(regions["regions"]) != 641
            or len(regions["preinjection_fallback_controls"]) != 4
            or Counter(ROLES[row["phase_class"]] for row in regions["regions"]) != EXPECTED_REGIONS):
        raise ValueError("Frozen 13/261 frame or dense polygon inventory changed")
    if (set(fits["folds"]) != {f"{v:.3f}" for v in fitted_job.REGULARIZATION}
            or fits.get("selected_production_model") is not None):
        raise ValueError("Frozen OOF fit arms or exploratory scope changed")
    by_id = set()
    for row in regions["regions"] + regions["preinjection_fallback_controls"]:
        if (row["region_id"] in by_id or row["dataset"] != "ffum"
                or row["phase_class"] not in ROLES or row["confidence"] == "low"):
            raise ValueError("Invalid dense source-aligned polygon identity")
        by_id.add(row["region_id"])
    audit = _read(Path(protocol["cohort_audit"]))
    audited = {row["region_id"]: row for row in audit["regions"]}
    if len(audited) != 645 or set(audited) != by_id:
        raise ValueError("Frozen dense source-aligned region IDs changed")
    evidence = _read(parent / "evidence.json")
    source_regions = {row["region_id"]: row for row in evidence["regions"]}
    if len(source_regions) != len(evidence["regions"]):
        raise ValueError("Published parent evidence has duplicate region IDs")
    for row in regions["regions"] + regions["preinjection_fallback_controls"]:
        original = audited[row["region_id"]]
        if (original["run"] != row["run"]
                or original["phase_class"] != row["phase_class"]
                or int(original["studio_frame_1based"]) - 1 != int(row["frame_index"])):
            raise ValueError("Frozen dense region class or aligned native frame changed")
        source = source_regions.get(row["region_id"])
        if source is None or row != {**source, "frame_index": row["frame_index"]}:
            raise ValueError("Frozen dense region geometry or reviewer payload changed")
    for lam in fitted_job.REGULARIZATION:
        key = f"{lam:.3f}"
        if set(fits["folds"][key]) != set(fitted_job.FOLDS):
            raise ValueError("Incomplete grouped OOF fold fits")
        for fold, held in fitted_job.FOLDS.items():
            entry = fits["folds"][key][fold]
            if (entry["held_out_runs"] != list(held)
                    or set(entry["train_runs"]) != set(source_receipt["images"]) - set(held)
                    or set(entry["arms"]) != set(ARMS)):
                raise ValueError("OOF fold trained or evaluated on the wrong runs")
            for arm, width in (("static20", 20), ("causal30", 30)):
                _fit_values(entry["arms"][arm], width, lam)
    return dict(job=job, protocol=protocol, extraction=extraction,
                regions=regions, fits=fits, parent=parent,
                source_receipt=source_receipt,
                later_review_activity=_later_review_activity(protocol))


def _later_review_activity(protocol: dict) -> dict:
    """Document later live edits without changing the frozen evaluation cohort."""
    root = Path(protocol["review_inventory"]["root"])
    if not root.is_dir():
        return dict(status="live_review_root_unavailable",
                    later_saved_review_files=None,
                    later_reviews_included=False)
    cutoff = datetime.fromisoformat(protocol["created_utc"].replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        raise ValueError("The isolated fit freeze time lacks a time zone")
    cutoff_ns = int(cutoff.timestamp() * 1_000_000_000)
    later = []
    for path in root.rglob("*"):
        try:
            if path.is_file() and _review_evidence(path) and path.stat().st_mtime_ns > cutoff_ns:
                later.append(path.relative_to(root).as_posix())
        except FileNotFoundError:
            continue  # The live reviewer may replace a file during this optional census.
    return dict(status="observed", later_saved_review_files=len(later),
                examples=sorted(later)[:12], later_reviews_included=False,
                frozen_cohort_source="SHA-bound copied comparison/audit/request/region artifacts")


def _head_probability(features: np.ndarray, fit: dict) -> np.ndarray:
    """Same standardized logistic decision as the regional head, width agnostic."""
    values = np.asarray(features, np.float32)
    mean, scale, coef = (np.asarray(fit[key], np.float32)
                         for key in ("mean", "scale", "coefficients"))
    if (values.ndim != 3 or values.shape[-1] != len(coef)
            or mean.shape != coef.shape or scale.shape != coef.shape
            or (scale <= 0).any()):
        raise ValueError("Invalid complete-frame regional feature contract")
    weights = coef / scale
    return expit(values @ weights + float(fit["intercept"]) - mean @ weights).astype(np.float32)


def final_native_labels(prepared: dict, features: np.ndarray, co2_fit: dict,
                        parent_model: dict, raw_presence: np.ndarray,
                        raw_gas: np.ndarray, domain: np.ndarray,
                        spacing: tuple[float, float], profile: dict,
                        frozen_gas_probability: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct one native final map with only the CO2 fit substituted."""
    settings = regional.Settings(**parent_model["settings"])
    p = _head_probability(features, co2_fit)
    connected = regional.reconstruct(p, prepared, settings)
    field = np.where(connected, p, 0).astype(np.float32)
    support = cv2.resize(field, domain.shape[::-1], interpolation=cv2.INTER_LINEAR) >= settings.support_threshold
    support &= domain
    raw = np.where(raw_presence >= .5, np.where(raw_gas >= .6, 2, 1), 0).astype(np.uint8)
    labels = np.where(raw == 2, 2, support).astype(np.uint8)
    labels[~domain] = raw[~domain]
    conditional = (regional.probability(prepared["features"],
                   parent_model["heads"]["gas_within_co2"])
                   if frozen_gas_probability is None else frozen_gas_probability)
    if (conditional.shape != prepared["domain"].shape or not np.isfinite(conditional).all()
            or (conditional < 0).any() or (conditional > 1).any()):
        raise ValueError("Frozen gas head probability differs from the regional grid")
    labels, _ = gas.confirm_gas(labels, conditional, domain,
                                gas.Settings(**profile["regional_gas"]))
    labels, _ = guards.guard_labels(labels, domain, spacing,
                                    guards.Settings(**profile["output_guards"]))
    labels, _ = free_gas.clean_labels(labels, spacing,
                                      free_gas.Settings(**profile["free_gas_cleanup"]))
    return labels


def _region_mask(region: dict, manifest: dict, shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    if region["payload"].get("coordinate_space") != "normalized_native_review_canvas_16x9":
        raise ValueError(f"Unknown native review annotation geometry: {region['region_id']}")
    bounds, mask = cohort_audit.region_crop(region["payload"], manifest["canvas_wh"],
                                            manifest["content_box_xywh"])
    x0, y0, x1, y1 = bounds
    if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0] and mask.any()):
        raise ValueError(f"Dense polygon outside native crop: {region['region_id']}")
    return bounds, mask


def score_polygon(region: dict, manifest: dict, raw_presence: np.ndarray,
                  static: np.ndarray, causal_map: np.ndarray,
                  parent: np.ndarray, *, fallback: bool = False) -> dict:
    """Score a whole dense native polygon, including optically weak positives."""
    shape = raw_presence.shape
    if any(a.shape != shape for a in (static, causal_map, parent)):
        raise ValueError("Native score and labels must share the exact source grid")
    bounds, mask = _region_mask(region, manifest, shape)
    x0, y0, x1, y1 = bounds
    raw = raw_presence[y0:y1, x0:x1][mask]
    if not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any():
        raise ValueError(f"Frozen optical probability invalid within {region['region_id']}")
    weak = raw < .5
    def arm(labels):
        selected = labels[y0:y1, x0:x1][mask]
        return dict(co2_pixels=int(np.count_nonzero(selected > 0)),
                    aqueous_pixels=int(np.count_nonzero(selected == 1)),
                    gas_pixels=int(np.count_nonzero(selected == 2)),
                    weak_co2_pixels=int(np.count_nonzero((selected > 0) & weak)),
                    weak_aqueous_pixels=int(np.count_nonzero((selected == 1) & weak)))
    return dict(region_id=region["region_id"], run=region["run"],
                frame_index=int(region["frame_index"]),
                phase_class=region["phase_class"], role=ROLES[region["phase_class"]],
                pixels=int(mask.sum()), weak_pixels=int(weak.sum()),
                preinjection_static_fallback=fallback,
                parent=arm(parent), static20=arm(static), causal30=arm(causal_map))


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _stratum(rows: list[dict], *, denominator: str, numerator: str) -> dict:
    eligible = [row for row in rows if row[denominator] > 0]
    if not eligible:
        return dict(status="not_applicable_no_labels", polygons=0, pixels=0,
                    static20=None, causal30=None, parent=None,
                    polygon_macro_static20=None, polygon_macro_causal30=None)
    total = sum(row[denominator] for row in eligible)
    metrics = dict(status="measured", polygons=len(eligible), pixels=total)
    for arm in ("parent", *ARMS):
        metrics[arm] = sum(row[arm][numerator] for row in eligible) / total
        metrics[f"polygon_macro_{arm}"] = float(np.mean([
            row[arm][numerator] / row[denominator] for row in eligible]))
    return metrics


def summarize_rows(rows: list[dict]) -> dict:
    groups = {role: [row for row in rows if row["role"] == role]
              for role in set(ROLES.values())}
    return dict(carryover_false_co2=_stratum(groups["carryover_negative"],
            denominator="pixels", numerator="co2_pixels"),
        weak_aqueous_recall=_stratum(groups["aqueous_positive"],
            denominator="weak_pixels", numerator="weak_aqueous_pixels"),
        aqueous_recall=_stratum(groups["aqueous_positive"],
            denominator="pixels", numerator="aqueous_pixels"),
        aqueous_co2_support=_stratum(groups["aqueous_positive"],
            denominator="pixels", numerator="co2_pixels"),
        water_false_co2=_stratum(groups["water_negative"],
            denominator="pixels", numerator="co2_pixels"),
        gas_co2_support=_stratum(groups["gas_co2_positive"],
            denominator="pixels", numerator="co2_pixels"),
        gas_phase_coverage=_stratum(groups["gas_co2_positive"],
            denominator="pixels", numerator="gas_pixels"))


def _by_run_and_fold(rows: list[dict]) -> dict:
    by_run = {run: summarize_rows([r for r in rows if r["run"] == run])
              for run in sorted(sum((list(v) for v in fitted_job.FOLDS.values()), []))}
    by_fold = {fold: summarize_rows([r for r in rows if r["run"] in held])
               for fold, held in fitted_job.FOLDS.items()}
    return dict(overall=summarize_rows(rows), by_run=by_run, by_fold=by_fold)


def _relative_reduction(before: float, after: float) -> float | None:
    return (before - after) / before if before > 0 else None


def decision_gates(summary: dict, rows: list[dict], gas_parity: list[dict]) -> dict:
    """Apply only predeclared primary-λ screening limits; no threshold search."""
    overall = summary["overall"]
    carry = overall["carryover_false_co2"]
    weak = overall["weak_aqueous_recall"]
    water = overall["water_false_co2"]
    if any(value["status"] != "measured" for value in (carry, weak, water)):
        raise ValueError("Primary screening strata lack source-aligned labels")
    area_reduction = _relative_reduction(carry["static20"], carry["causal30"])
    box_reduction = _relative_reduction(carry["polygon_macro_static20"],
                                        carry["polygon_macro_causal30"])
    fold_wins = []
    for fold in fitted_job.FOLDS:
        metric = summary["by_fold"][fold]["carryover_false_co2"]
        if metric["status"] != "measured":
            raise ValueError(f"Negative-bearing predeclared fold lacks labels: {fold}")
        fold_wins.append(metric["causal30"] <= metric["static20"] and
                         metric["polygon_macro_causal30"] <=
                         metric["polygon_macro_static20"])
    run_losses, run_macro_losses = {}, {}
    for run, metrics in summary["by_run"].items():
        value = metrics["weak_aqueous_recall"]
        if value["status"] == "measured":
            run_losses[run] = value["static20"] - value["causal30"]
            run_macro_losses[run] = (value["polygon_macro_static20"]
                                     - value["polygon_macro_causal30"])
    gas_rows = [row for row in rows if row["role"] == "gas_co2_positive"]
    if len(gas_rows) != 32:
        raise ValueError("The limited RUN6 dense gas support stratum changed")
    checks = dict(carryover_pixel_area_reduction_at_least_20pct=(
            area_reduction is not None and area_reduction >= .20),
        carryover_polygon_macro_reduction_at_least_20pct=(
            box_reduction is not None and box_reduction >= .20),
        carryover_nonworse_in_at_least_two_folds=sum(fold_wins) >= 2,
        weak_aqueous_recall_loss_at_most_2pp=(
            weak["static20"] - weak["causal30"] <= .02 + 1e-12),
        weak_aqueous_polygon_macro_loss_at_most_2pp=(
            weak["polygon_macro_static20"] - weak["polygon_macro_causal30"] <= .02 + 1e-12),
        weak_aqueous_recall_loss_at_most_5pp_each_run=all(
            loss <= .05 + 1e-12 for loss in run_losses.values()),
        weak_aqueous_polygon_macro_loss_at_most_5pp_each_run=all(
            loss <= .05 + 1e-12 for loss in run_macro_losses.values()),
        water_false_co2_nonworse=(water["causal30"] <= water["static20"] + 1e-12
            and water["polygon_macro_causal30"] <=
                water["polygon_macro_static20"] + 1e-12),
        dense_gas_co2_support_nonworse_each_polygon=all(
            row["causal30"]["co2_pixels"] >= row["static20"]["co2_pixels"]
            for row in gas_rows),
        exact_final_gas_parity=bool(gas_parity) and all(
            row["static20_changed_pixels"] == 0 and row["causal30_changed_pixels"] == 0
            for row in gas_parity))
    return dict(status="screening_pass" if all(checks.values()) else "screening_fail",
                checks=checks, carryover_pixel_area_reduction=area_reduction,
                carryover_polygon_macro_reduction=box_reduction,
                fold_nonworse=dict(zip(fitted_job.FOLDS, fold_wins)),
                weak_aqueous_recall_loss_by_run=run_losses,
                weak_aqueous_polygon_macro_loss_by_run=run_macro_losses,
                scope="conditional training-seen optical evidence; no scientific acceptance or publication")


def _map_receipt(source: dict, frame_index: int) -> tuple[Path, str]:
    job = source["job"]
    path = comparison.phase_path(job, frame_index)
    if not path.is_file():
        raise FileNotFoundError(f"Published parent native map unavailable: {path}")
    return path, _sha(path)


def assert_frame_receipt(expected: dict, frame: dict, *, run: str, index: int,
                         manifest_sha256: str, native_jpeg_sha256: str,
                         optical_cache_key: str, optical_payload_sha256: str,
                         temporal_audit: dict) -> None:
    """Reject any changed prefix input before it can influence held-out QA."""
    if (expected["run"] != run or int(expected["frame_index"]) != index
            or expected["native_manifest_sha256"] != manifest_sha256
            or expected["native_jpeg_sha256"] != native_jpeg_sha256
            or expected["source_sha256"] != frame["source_sha256"]
            or expected["source_name"] != frame["source_name"]
            or expected["optical_cache_key"] != optical_cache_key
            or expected["optical_cache_payload_sha256"] != optical_payload_sha256
            or float(expected["elapsed_h"]) != float(frame["elapsed_h"])
            or _digest(expected["temporal"]) != _digest(temporal_audit)):
        raise ValueError(f"Frozen source/cache/feature prefix mismatch: {run} {index}")


def evaluate(job: Path, output: Path) -> dict:
    started = time.perf_counter()
    bound = verify_job(job)
    output = Path(output).resolve()
    if (output.exists() or output.name != "report.json"
            or output.parent.parent != bound["job"]
            or not output.parent.name.startswith("native-oof-qa-")
            or output.parent.exists()):
        raise ValueError("Use a new native-oof-qa-* diagnostic folder under the fit job")
    output.parent.mkdir(parents=False, exist_ok=False)
    protocol, extraction, regions = (bound[key] for key in ("protocol", "extraction", "regions"))
    parent, source_receipt = bound["parent"], bound["source_receipt"]
    cache_root = Path(protocol["optical_cache_root"])
    parent_model = _read(parent / "regional_model.json")
    profile = _read(parent / "phase_optical_regions.json")
    if (parent_model["format"] != regional.VERSION
            or parent_model["feature_names"] != list(regional.FEATURE_NAMES)
            or any(name not in profile for name in ("regional_gas", "output_guards", "free_gas_cleanup"))):
        raise ValueError("Published regional reconstruction/guard contract changed")
    settings = regional.Settings(**parent_model["settings"])
    by_frame = defaultdict(list)
    for region in regions["regions"]:
        by_frame[(region["run"], int(region["frame_index"]))].append(region)
    fallback_by_frame = defaultdict(list)
    for region in regions["preinjection_fallback_controls"]:
        if int(region["frame_index"]) != 0 or region["phase_class"] != "water":
            raise ValueError("Unexpected pre-injection static fallback control")
        fallback_by_frame[(region["run"], 0)].append(region)
    frame_receipts = {(row["run"], int(row["frame_index"])): row
                      for row in extraction["frames"]}
    if len(frame_receipts) != 261:
        raise ValueError("Duplicate or missing frozen native frame receipts")
    fold_of = {run: fold for fold, held in fitted_job.FOLDS.items() for run in held}
    scored = {lam: [] for lam in bound["fits"]["folds"]}
    fallback_scored = {lam: [] for lam in bound["fits"]["folds"]}
    parity = {lam: [] for lam in bound["fits"]["folds"]}
    parent_checks, replay, run_durations = [], [], {}
    for run in sorted(source_receipt["images"]):
        run_started = time.perf_counter()
        source = fitted_job.load_verified_run(parent, run, cache_root, source_receipt)
        manifest = source["manifest"]
        source["job"] = _read(parent / "validation" / "run_packages" / run / "postflight_manifest.json")["jobs"][0]
        domain = np.asarray(source["context"].domain, bool)
        spacing = (abs(float(np.diff(source["context"].y_m)[0])),
                   float(np.diff(source["context"].x_m)[0]))
        box = manifest["content_box_xywh"]
        baseline_path = source["native_dir"] / "frame_000.jpg"
        if _sha(baseline_path) != source["baseline_sha256"]:
            raise ValueError(f"Frozen native baseline changed: {run}")
        baseline = _crop_rgb(baseline_path, box)
        state = causal.CausalTemporalFeatureState(source["context"], run=run,
            native_manifest_sha256=_sha(source["manifest_path"]))
        for index, frame in enumerate(manifest["frame_sources"]):
            photo = source["native_dir"] / f"frame_{index:03d}.jpg"
            if _sha(photo) != source["frame_sha256"][index]:
                raise ValueError(f"Frozen corrected native photo changed: {run} {index}")
            key = source["keys"][index]
            heads, cache_receipt = fitted_job._verified_cached_heads(cache_root, key, domain.shape)
            expected = frame_receipts[(run, index)]
            current = _crop_rgb(photo, box)
            prepared = regional.prepare(current, baseline, heads[..., 0], heads[..., 1],
                                        domain, spacing, settings)
            extra, temporal_audit = state.peek(frame_index=index,
                elapsed_h=float(frame["elapsed_h"]), prepared=prepared,
                raw_presence=heads[..., 0])
            assert_frame_receipt(expected, frame, run=run, index=index,
                manifest_sha256=state.native_manifest_sha256,
                native_jpeg_sha256=source["frame_sha256"][index],
                optical_cache_key=key,
                optical_payload_sha256=cache_receipt["payload_sha256"],
                temporal_audit=temporal_audit)
            records = by_frame.get((run, index), [])
            controls = fallback_by_frame.get((run, index), [])
            if records and extra is None:
                raise ValueError(f"Held-out annotation lacks a valid causal prefix: {run} {index}")
            if records or controls:
                frozen_gas_probability = regional.probability(prepared["features"],
                    parent_model["heads"]["gas_within_co2"])
                parent_labels = final_native_labels(prepared, prepared["features"],
                    parent_model["heads"]["co2_presence"], parent_model,
                    heads[..., 0], heads[..., 1], domain, spacing, profile,
                    frozen_gas_probability)
                map_path, map_sha = _map_receipt(source, index)
                published = comparison.decode_phase(map_path, box)
                if not np.array_equal(parent_labels, published):
                    raise ValueError(f"Evaluator does not reproduce published parent phase map: {run} {index}")
                parent_checks.append(dict(run=run, frame_index=index,
                    published_map_sha256=map_sha,
                    computed_labels_sha256=hashlib.sha256(parent_labels.tobytes()).hexdigest(),
                    exact=True))
                feature30 = (np.concatenate((prepared["features"], extra), axis=-1)
                             if records else None)
                for lam_key, fold_fits in bound["fits"]["folds"].items():
                    fold = fold_fits[fold_of[run]]
                    static_fit = fold["arms"]["static20"]["fit"]
                    static_labels = final_native_labels(prepared, prepared["features"],
                        static_fit, parent_model, heads[..., 0], heads[..., 1], domain,
                        spacing, profile, frozen_gas_probability)
                    causal_fit = fold["arms"]["causal30"]["fit"]
                    causal_labels = (final_native_labels(prepared, feature30, causal_fit,
                        parent_model, heads[..., 0], heads[..., 1], domain, spacing,
                        profile, frozen_gas_probability)
                        if records else static_labels)
                    gas_parent = parent_labels == 2
                    parity[lam_key].append(dict(run=run, frame_index=index,
                        static20_changed_pixels=int(np.count_nonzero((static_labels == 2) != gas_parent)),
                        causal30_changed_pixels=int(np.count_nonzero((causal_labels == 2) != gas_parent)),
                        parent_gas_pixels=int(gas_parent.sum()),
                        preinjection_static_fallback=not records))
                    for region in records:
                        scored[lam_key].append(score_polygon(region, manifest,
                            heads[..., 0], static_labels, causal_labels, parent_labels))
                    for region in controls:
                        fallback_scored[lam_key].append(score_polygon(region, manifest,
                            heads[..., 0], static_labels, static_labels, parent_labels,
                            fallback=True))
                    del static_labels, causal_labels
                del parent_labels, published, feature30, frozen_gas_probability
            state.commit(frame_index=index)
            replay.append(dict(run=run, frame_index=index,
                native_manifest_sha256=state.native_manifest_sha256,
                native_jpeg_sha256=source["frame_sha256"][index],
                optical_cache_key=key, optical_cache_payload_sha256=cache_receipt["payload_sha256"],
                causal_abstention=temporal_audit["abstention"],
                annotated_polygons=len(records), fallback_polygons=len(controls)))
            del heads, current, prepared, extra
            gc.collect()
        del baseline, state, source
        gc.collect()
        run_durations[run] = time.perf_counter() - run_started
    if (len(replay) != 261 or len(parent_checks) == 0
            or any(len(rows) != 641 for rows in scored.values())
            or any(len(rows) != 4 for rows in fallback_scored.values())
            or Counter(row["role"] for row in scored[PRIMARY_LAMBDA]) != EXPECTED_REGIONS):
        raise ValueError("Native replay or exact dense polygon coverage incomplete")
    outcomes = {}
    for lam_key in sorted(scored):
        summary = _by_run_and_fold(scored[lam_key])
        outcomes[lam_key] = dict(summary=summary, regions=scored[lam_key],
            preinjection_static_fallback=dict(summary=summarize_rows(fallback_scored[lam_key]),
                                             regions=fallback_scored[lam_key]),
            gas_parity=dict(scope="annotated held-out native frames and four pre-injection controls",
                frames=parity[lam_key],
                exact=all(row["static20_changed_pixels"] == 0 and
                          row["causal30_changed_pixels"] == 0 for row in parity[lam_key])))
    gates = decision_gates(outcomes[PRIMARY_LAMBDA]["summary"],
                           scored[PRIMARY_LAMBDA], parity[PRIMARY_LAMBDA])
    report = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        scope="exploratory grouped OOF; optical evidence was training-seen; no independent validation",
        job=str(bound["job"]), parent=str(parent), primary_regularization=0.001,
        sensitivity_regularization=0.01,
        artifacts=dict(protocol_sha256=_sha(bound["job"] / "protocol.json"),
            extraction_sha256=_sha(bound["job"] / "extraction.json"),
            fold_fits_sha256=_sha(bound["job"] / "fold_fits.json"),
            regions_sha256=_sha(bound["job"] / "regions.json"),
            parent_model_sha256=protocol["parent_model_sha256"],
            source_manifest_sha256=protocol["parent_source_manifest_sha256"]),
        later_review_activity=bound["later_review_activity"],
        replay=dict(runs=13, frames=len(replay), optical_payloads_sha_verified=len(replay),
            frames_detail=replay, parent_native_map_checks=parent_checks),
        outcomes=outcomes, primary_decision_gates=gates,
        interpretation=dict(run7_residual="qualitative only: no aligned carryover label",
            gas="Only 32 tiny gas boxes in RUN6 frame 19; fold 3 has no dense gas training positives",
            preinjection="Four water boxes use static fallback and are excluded from paired gates",
            publication="No model, phase map, QSettings registration, or production publication"),
        duration_seconds=time.perf_counter() - started,
        duration_seconds_by_run=run_durations,
        phase_maps_written=0, model_published=False)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.job, args.output)
    print(json.dumps(dict(format=report["format"],
        duration_seconds=report["duration_seconds"],
        primary_decision_gates=report["primary_decision_gates"],
        report=str(args.output)), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
