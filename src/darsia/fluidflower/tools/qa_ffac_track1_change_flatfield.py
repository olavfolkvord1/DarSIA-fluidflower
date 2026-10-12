"""Independent read-only acceptance audit for the FF_AC flatfield display.

The new JPEGs and addenda are checked against the immutable Track1 source
manifests.  Earlier phase QA reports provide pre-change SHA-256 values for all
864 native PNG maps.  This command never writes to the source or model tree;
only an explicitly requested, new local JSON report is created.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from darsia.fluidflower.tools.ffac_track1_change_flatfield import (
    CONTRACT, smoothed_sand_luminance_cv,
)
from darsia.fluidflower.tools.materialize_ffac_track1_change_flatfield import (
    DEFAULT_BATCH,
)


DEFAULT_MODEL = Path(
    r"\\UNAS-Pro-8\UiB_FluidFlower\FF_AC\DarSIA\results"
    r"\ffac-track1-display-modes-v2-rerun-20260930"
    r"\phase-model-20260930-track1-natural-lifts-invalidfilled-ffac"
)
DEFAULT_PRIOR_PHASE_QA = Path(
    r"C:\Users\olav_\Documents\FF Albus\outputs"
    r"\ffac-track1-display-modes-v2-20260930\qa_phase"
)
AUDIT_CONTRACT = "ffac-track1-display-change-flatfield-acceptance-v1"
SOURCE_FIELDS = (
    ("v5_log_linear_rgb_path", "v5_log_linear_rgb_sha256", "source_float_sha256"),
    ("v5_log_linear_valid_mask_path", "v5_log_linear_valid_mask_sha256", "source_valid_mask_sha256"),
    ("v5_photo_sand_mask_path", "v5_photo_sand_mask_sha256", "source_photo_sand_sha256"),
)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify(path_text: str, expected: str, label: str) -> tuple[Path, int]:
    path = Path(path_text)
    _check(isinstance(expected, str) and len(expected) == 64,
           f"{label}: missing SHA-256")
    actual = _sha(path)
    _check(actual == expected.lower(), f"{label}: SHA-256 mismatch: {path}")
    return path, path.stat().st_size


def _source_fields(record: dict, change_record: dict, *, label: str,
                   hash_source_bytes: bool) -> tuple[int, int]:
    """Check exact addendum declarations and optionally hash large float data."""
    files = 0
    total_bytes = 0
    for path_field, digest_field, change_field in SOURCE_FIELDS:
        digest = record.get(digest_field)
        _check(change_record.get(change_field) == digest,
               f"{label}: addendum {change_field} differs from sealed source manifest")
        if hash_source_bytes:
            _, size = _verify(str(record.get(path_field) or ""), digest,
                              f"{label} {path_field}")
            files += 1
            total_bytes += size
    lift = (record.get("v5_display_modes") or {}).get("lift1") or {}
    _check(change_record.get("source_lift1_sha256") == lift.get("jpeg_sha256"),
           f"{label}: addendum lift1 SHA differs from sealed source manifest")
    return files, total_bytes


def _source_manifest(model: Path, batch: Path, run: str) -> tuple[dict, Path, str]:
    binding = _read(model / "inference_run_bindings" / f"{run}.json")
    _check(binding.get("run") == run, f"{run}: wrong candidate run binding")
    native_path = Path(str(binding.get("materialized_manifest_path") or ""))
    _check(native_path.name == "manifest.json" and native_path.parent.name == run
           and native_path.parent.parent.name == "native_frames"
           and batch.resolve() in native_path.resolve().parents,
           f"{run}: candidate manifest is outside the sealed prepared batch")
    native_path, _ = _verify(str(native_path), binding.get("materialized_manifest_sha256"),
                             f"{run} sealed native manifest")
    native = _read(native_path)
    _check(native.get("run") == run, f"{run}: native manifest identifies another run")
    return native, native_path, binding["materialized_manifest_sha256"]


def _audit_run(model: Path, batch: Path, run: str, *,
               hash_source_bytes: bool,
               expected_addendum_sha: str | None) -> dict[str, Any]:
    native, native_path, native_sha = _source_manifest(model, batch, run)
    addendum_path = native_path.parent / "change_display_manifest.json"
    if expected_addendum_sha is not None:
        _verify(str(addendum_path), expected_addendum_sha,
                f"{run} change addendum")
    addendum = _read(addendum_path)
    frames = native.get("frame_sources") or []
    changes = addendum.get("frames") or []
    _check(addendum.get("contract") == CONTRACT and addendum.get("run") == run
           and addendum.get("source_manifest_sha256") == native_sha
           and Path(str(addendum.get("source_manifest_path") or "")).resolve()
           == native_path.resolve()
           and addendum.get("frame_count") == len(frames)
           and len(changes) == len(frames),
           f"{run}: change addendum is not bound to the sealed native timeline")
    shape = tuple(native.get("analysis_image_shape_hw") or ())
    box = tuple(native.get("content_box_xywh") or ())
    canvas = tuple(native.get("canvas_wh") or ())
    _check(addendum.get("analysis_image_shape_hw") == list(shape)
           and addendum.get("content_box_xywh") == list(box)
           and addendum.get("canvas_wh") == list(canvas)
           and shape == (box[3], box[2]),
           f"{run}: addendum geometry differs from the sealed native grid")

    sources_hashed = 0
    source_bytes = 0
    jpeg_bytes = 0
    jpeg_count = 0
    reference = native.get("reference") or {}
    changed_reference = addendum.get("reference") or {}
    files, count = _source_fields(reference, changed_reference,
                                  label=f"{run} pre-injection reference",
                                  hash_source_bytes=hash_source_bytes)
    sources_hashed += files
    source_bytes += count
    ref_expected = native_path.parent.parent.parent / "native_references" / run / "pre_injection.change.jpg"
    _check(Path(str(changed_reference.get("jpeg_path") or "")).resolve()
           == ref_expected.resolve(), f"{run}: change reference path escaped its native run")
    _, size = _verify(changed_reference["jpeg_path"], changed_reference["jpeg_sha256"],
                      f"{run} change reference JPEG")
    jpeg_bytes += size
    jpeg_count += 1

    for index, (source, changed) in enumerate(zip(frames, changes, strict=True)):
        _check(source.get("frame_index") == changed.get("frame_index") == index
               and abs(float(source["elapsed_h"]) - float(changed["elapsed_h"])) <= 1e-6,
               f"{run} frame {index}: timeline mismatch")
        files, count = _source_fields(source, changed,
                                      label=f"{run} frame {index}",
                                      hash_source_bytes=hash_source_bytes)
        sources_hashed += files
        source_bytes += count
        expected = native_path.parent / f"frame_{index:03d}.change.jpg"
        _check(Path(str(changed.get("jpeg_path") or "")).resolve() == expected.resolve(),
               f"{run} frame {index}: change JPEG path escaped its native run")
        _, size = _verify(changed["jpeg_path"], changed["jpeg_sha256"],
                          f"{run} frame {index} change JPEG")
        jpeg_bytes += size
        jpeg_count += 1

    x, y, width, height = box
    with Image.open(changes[0]["jpeg_path"]) as image:
        _check(image.size == canvas, f"{run}: frame0 change canvas changed")
        frame0 = np.asarray(image.crop((x, y, x + width, y + height)).convert("RGB"))
    sand_path, _ = _verify(str(frames[0]["v5_photo_sand_mask_path"]),
                           frames[0]["v5_photo_sand_mask_sha256"],
                           f"{run} frame0 photo-sand mask")
    sand = np.load(sand_path, allow_pickle=False)
    _check(sand.dtype == np.bool_ and sand.shape == shape,
           f"{run}: frame0 sand mask has wrong shape or dtype")
    cv = smoothed_sand_luminance_cv(frame0, sand)
    recorded_cv = changes[0].get("sand_luminance_cv_box5")
    _check(isinstance(recorded_cv, (int, float)) and abs(cv - recorded_cv) <= 0.003,
           f"{run}: recomputed frame0 CV differs from addendum ({cv} vs {recorded_cv})")
    _check(cv <= 0.10, f"{run}: frame0 sand luminance CV exceeds 10% ({cv:.4f})")
    return {
        "run": run, "frames": len(frames), "source_manifest_sha256": native_sha,
        "addendum_sha256": _sha(addendum_path),
        "frame0_sand_luminance_cv": cv,
        "frame0_recorded_cv": recorded_cv,
        "change_jpegs_sha_verified": jpeg_count,
        "change_jpegs_bytes_hashed": jpeg_bytes,
        "scientific_source_files_sha_verified": sources_hashed,
        "scientific_source_bytes_hashed": source_bytes,
        "nearest_2p25": addendum.get("nearest_2p25"),
    }


def _audit_phase_maps(model: Path, prior_phase_qa: Path,
                      runs: list[str]) -> dict[str, Any]:
    checked = 0
    total_bytes = 0
    report_sha = {}
    for run in runs:
        report_path = prior_phase_qa / f"{run}_phase_qa.json"
        report_sha[run] = _sha(report_path)
        report = _read(report_path)
        frames = report.get("per_frame") or []
        _check(report.get("run") == run and frames,
               f"{run}: prior phase QA has no bound frame records")
        _check(Path(str(report.get("new_candidate") or "")).resolve() == model.resolve(),
               f"{run}: prior phase QA belongs to another candidate")
        for index, row in enumerate(frames):
            _check(row.get("frame_index") == index,
                   f"{run}: prior phase QA timeline is not indexed")
            path = Path(str(row.get("new_phase_map_path") or ""))
            _check(model.resolve() in path.resolve().parents
                   and path.name == f"frame_{index:03d}.png",
                   f"{run} frame {index}: prior phase map path is outside this candidate")
            _, size = _verify(str(path), row.get("new_phase_map_sha256"),
                              f"{run} frame {index} pre-change phase map")
            checked += 1
            total_bytes += size
    return {"phase_maps_sha_verified": checked,
            "phase_map_bytes_hashed": total_bytes,
            "prior_qa_report_sha256": report_sha}


def audit_change_flatfield(batch: Path, model: Path, prior_phase_qa: Path,
                           materialization_report: Path,
                           *, hash_source_bytes: bool = True,
                           selected_runs: tuple[str, ...] | None = None) -> dict[str, Any]:
    materialized = _read(materialization_report)
    _check(materialized.get("contract") == CONTRACT
           and isinstance(materialized.get("runs"), int)
           and isinstance(materialized.get("frames"), int),
           "Flatfield materialization report has no valid batch count")
    _check(Path(str(materialized.get("batch_root") or "")).resolve() == batch.resolve(),
           "Flatfield materialization report belongs to another source batch")
    rows = materialized.get("per_run") or []
    _check(len(rows) == materialized["runs"]
           and len({row.get("run") for row in rows}) == len(rows),
           "Flatfield report does not identify distinct experiments")
    by_run = {row["run"]: row for row in rows}
    if selected_runs is None:
        _check(len(by_run) == 43 and materialized["frames"] == 864,
               "Flatfield report is not the complete 43/864 batch")
    else:
        _check(set(by_run) == set(selected_runs),
               "Flatfield pilot report does not match the selected experiments")
    runs = sorted(by_run)
    details = []
    for run in runs:
        details.append(_audit_run(model, batch, run,
                                  hash_source_bytes=hash_source_bytes,
                                  expected_addendum_sha=by_run[run].get("addendum_sha256")))
    _check(sum(row["frames"] for row in details) == materialized["frames"],
           "Flatfield addenda do not cover the reported source frames")
    phase = _audit_phase_maps(model, prior_phase_qa, runs)
    _check(phase["phase_maps_sha_verified"] == materialized["frames"],
           "Pre-change phase map hash comparison is incomplete")
    figures = materialized.get("figures") or {}
    _check(set(figures) == {"ac28", "ac56"},
           "AC28 and AC56 nearest-2.25h comparison figures are missing")
    figure_sha = {run: _sha(Path(figures[run])) for run in sorted(figures)}
    return {
        "contract": AUDIT_CONTRACT,
        "source_batch": str(batch), "model": str(model),
        "materialization_report_sha256": _sha(materialization_report),
        "runs": len(details), "frames": sum(row["frames"] for row in details),
        "complete_43_runs_864_frames": len(details) == 43
        and sum(row["frames"] for row in details) == 864,
        "all_frame0_sand_luminance_cv_le_0p10": all(
            row["frame0_sand_luminance_cv"] <= 0.10 for row in details),
        "per_run_frame0_cv": {row["run"]: row["frame0_sand_luminance_cv"] for row in details},
        "source_byte_hashing_complete": hash_source_bytes,
        "scientific_source_files_sha_verified": sum(
            row["scientific_source_files_sha_verified"] for row in details),
        "scientific_source_bytes_hashed": sum(
            row["scientific_source_bytes_hashed"] for row in details),
        "change_jpegs_sha_verified": sum(row["change_jpegs_sha_verified"] for row in details),
        "change_jpeg_bytes_hashed": sum(row["change_jpegs_bytes_hashed"] for row in details),
        **phase,
        "figures": {run: {"path": figures[run], "sha256": figure_sha[run],
                          "nearest_2p25": next(row["nearest_2p25"] for row in details
                                               if row["run"] == run)}
                    for run in sorted(figures)},
        "visual_caveats": [
            "Flatfield change is a photographic display transform, not a gas classifier or new phase map.",
            "Water far from the plume and gas over dissolved CO2 require visual review in AC28/AC56 figures.",
            "Dissolved CO2 itself can brighten substantially; pale pixels alone do not prove free gas.",
        ],
    }


def acceptance_markdown(result: dict[str, Any]) -> str:
    """Reviewable summary without treating visual cues as phase ground truth."""
    complete = result["complete_43_runs_864_frames"]
    lines = [
        "# FF_AC Track1 – uavhengig kontroll av Endring (flatfelt)",
        "",
        f"Omfang: {result['runs']} forsøk, {result['frames']} bilder. "
        + ("Full 43/864-serie." if complete else "Delvis pilot; ingen fullserieaksept."),
        "",
        "Dette er kun en fotografisk visning. Modellen, fasekartene og de vitenskapelige "
        "float-sidecarene er ikke trent eller beregnet på nytt.",
        "",
        f"- Startbilde-CV etter 5×5 utjevning: "
        f"{'alle ≤ 10 %' if result['all_frame0_sand_luminance_cv_le_0p10'] else 'FEIL'}.",
        f"- Verifiserte nye JPEG-filer: {result['change_jpegs_sha_verified']}.",
        f"- Verifiserte lagrede fasekart mot før-endring SHA-256: "
        f"{result['phase_maps_sha_verified']}.",
        f"- Verifiserte vitenskapelige kildefiler (RAW-float, gyldighets- og sandmasker): "
        f"{result['scientific_source_files_sha_verified']}. "
        + ("Alle bytes ble lest og SHA-verifisert."
           if result["source_byte_hashing_complete"]
           else "Bare manifestbinding ble kontrollert; byte-integritet er ikke fastslått."),
        "",
        "| Forsøk | CV startbilde | ≤ 10 % |",
        "|---|---:|:---:|",
    ]
    for run, cv in sorted(result["per_run_frame0_cv"].items()):
        lines.append(f"| {run.upper()} | {cv:.4f} ({cv * 100:.2f} %) | "
                     f"{'ja' if cv <= 0.10 else 'NEI'} |")
    lines.extend(["", "## Bilder ved omtrent 2,25 timer", ""])
    for run, figure in sorted(result["figures"].items()):
        time = figure["nearest_2p25"]["elapsed_h"]
        lines.append(f"- [{run.upper()} ved {time:.3f} t](<{figure['path']}>) "
                     f"(SHA-256 `{figure['sha256']}`).")
    lines.extend([
        "",
        "Visuell vurdering: vann og store blå områder framstår jevnere, og de gule "
        "CO₂-områdene er tydelige. I AC28 og AC56 ved 2,247 t ser fri gass i "
        "de inspiserte fulloppløste utsnittene lysere ut enn mye av det optisk "
        "vannløste området, men framstår fremdeles gul/gullfarget. En tydelig "
        "blek eller hvit gassignatur er derfor **ikke bekreftet** av disse figurene. "
        "Lysstyrke alene er ikke en gassklassifikator; Olav bør vurdere dette "
        "sammen med det separate optiske/fysiske faseoverlegget.",
        "",
        "Ingen masseanalyse eller vitenskapelig faseaksept følger av denne visningskontrollen.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prior-phase-qa", type=Path, default=DEFAULT_PRIOR_PHASE_QA)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path,
                        help="Create a separate human-readable acceptance note")
    parser.add_argument("--runs", nargs="+", metavar="ACNN",
                        help="Audit only this exact pilot cohort; omission requires all 43/864")
    parser.add_argument("--skip-full-source-hash", action="store_true",
                        help="Fast diagnostic only; cannot establish sidecar byte integrity")
    args = parser.parse_args(argv)
    result = audit_change_flatfield(
        args.batch_root, args.model_root, args.prior_phase_qa,
        args.materialization_report,
        hash_source_bytes=not args.skip_full_source_hash,
        selected_runs=tuple(run.lower() for run in args.runs) if args.runs else None)
    data = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists():
        _check(args.report.read_bytes() == data,
               "Refusing to replace a different flatfield acceptance report")
    else:
        with args.report.open("xb") as stream:
            stream.write(data)
    if args.markdown is not None:
        note = acceptance_markdown(result).encode("utf-8")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        if args.markdown.exists():
            _check(args.markdown.read_bytes() == note,
                   "Refusing to replace a different flatfield acceptance note")
        else:
            with args.markdown.open("xb") as stream:
                stream.write(note)
    print(json.dumps({"report": str(args.report), "sha256": _sha(args.report),
                      "runs": result["runs"], "frames": result["frames"],
                      "cv_le_0p10": result["all_frame0_sand_luminance_cv_le_0p10"],
                      "phase_maps_unchanged": result["phase_maps_sha_verified"] == result["frames"],
                      "source_byte_hashing_complete": result["source_byte_hashing_complete"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
