"""Export existing rerun QA numbers without running checks, inference or Studio."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def _load(path: Path):
    if not path.is_file():
        return None, f"MISSING: {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("report is not a JSON object")
        return value, None
    except (OSError, ValueError) as error:
        return None, f"UNREADABLE: {path} ({type(error).__name__}: {error})"


def _fmt(value):
    if value is None:
        return "NOT REPORTED"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    return f"{value:.9g}" if isinstance(value, float) else str(value)


def _total(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))
              and not isinstance(row[key], bool)]
    return sum(values) if values else None


def _range(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))
              and not isinstance(row[key], bool)]
    return (f"{_fmt(min(values))} .. {_fmt(max(values))}" if values else "NOT REPORTED")


def _references(rows):
    return ", ".join(f"{row.get('run', 'AC28').upper()}/frame_{row.get('frame_index', '?')}"
                     for row in rows) or "none"


def _csv_new(path, rows):
    if not rows:
        return None
    preferred = ["run", "frame_index", "elapsed_h"]
    fields = preferred + sorted(set().union(*(row.keys() for row in rows)) - set(preferred))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict)) else value) for key, value in row.items()})
    return path


def write_report(output_dir, candidate) -> Path:
    """Create exclusive plaintext/CSV exports solely from existing receipts.

    Missing or failed QA is reported explicitly; no gate is rerun or repaired.
    This function does not independently verify large payload/weight hashes.
    """
    output, candidate = Path(output_dir).resolve(), Path(candidate).resolve()
    target = output / "rerun_results.txt"
    if target.exists():
        raise FileExistsError(target)
    output.mkdir(parents=True, exist_ok=True)
    paths = dict(sidecar=output / "qa_sidecar.json", display=output / "qa_display.json",
                 phase_maps=output / "remote_bootstrap/qa_phase_maps_remote.json",
                 studio=output / "qa_studio.json")
    if not paths["phase_maps"].is_file() and (output / "qa_phase_maps.json").is_file():
        paths["phase_maps"] = output / "qa_phase_maps.json"
    reports, missing = {}, []
    for name, path in paths.items():
        value, issue = _load(path)
        reports[name] = value or {}
        if issue:
            missing.append(issue)
    inference, inference_issue = _load(candidate / "inference_rerun_receipt.json")
    inference = inference or {}
    if inference_issue:
        missing.append(inference_issue)
    summary, _ = _load(candidate / "summary.json")
    summary = summary or {}
    acceptance, acceptance_issue = _load(output / "acceptance_report.json")
    if acceptance is None and (candidate / "acceptance_report.json").is_file():
        acceptance, acceptance_issue = _load(candidate / "acceptance_report.json")
    acceptance = acceptance or {}
    contacts, _ = _load(output / "contacts_selection.json")
    contacts = contacts or {}

    source = reports["sidecar"]
    display = reports["display"]
    phase = reports["phase_maps"]
    studio = reports["studio"]
    source_rows, phase_rows = source.get("per_frame") or [], phase.get("per_frame") or []
    display_rows = display.get("frames") or []
    checks = dict(sidecar=source.get("passed"),
        display=display.get("display_rendering_qa_pass", display.get("passed")),
        phase_maps=phase.get("passed"), studio=studio.get("passed"))
    frame_counts = dict(sidecar=source.get("frames"), display=display.get("frame_count"),
                        phase_maps=phase.get("frames"))
    complete = not missing and all(value == 864 for value in frame_counts.values())
    if not complete:
        status = "NOT COMPLETED: missing, unreadable or incomplete QA receipts"
    elif all(value is True for value in checks.values()):
        status = "QA REPORTS COMPLETE: Section 4 checks passed; scientific approval is NOT granted"
    else:
        status = "QA REPORTS COMPLETE WITH FAILURES/WARNINGS; scientific approval is NOT granted"
    lines = ["FF_AC Track1: photographic display and filled sand support rerun",
             status, "NOT FOR MASS ANALYSIS", "",
             f"Report created UTC: {datetime.now(timezone.utc).isoformat()}",
             f"Candidate: {candidate}", f"Display name: {summary.get('display_name', candidate.name)}",
             "This is a read-only export of saved QA receipts, not a new check or fit.", ""]
    if missing:
        lines.extend(missing)
        lines.append("")
    lines.extend(["Frozen model and source lineage",
        f"Original candidate: {inference.get('old_candidate', 'NOT REPORTED')}",
        f"Recorded source cohort: {inference.get('run_count', 'NOT REPORTED')} runs / "
            f"{inference.get('frame_count', 'NOT REPORTED')} frames",
        f"Optical checkpoint SHA256: {(inference.get('optical_checkpoint') or {}).get('sha256', 'NOT REPORTED')}",
        f"Original regional fit SHA256: {(inference.get('frozen_regional_model') or {}).get('sha256', 'NOT REPORTED')}",
        f"Preserved candidate regional artifact SHA256: {inference.get('frozen_regional_candidate_sha256', 'NOT REPORTED')}",
        f"Preserved training request SHA256: {inference.get('frozen_training_request_sha256', 'NOT REPORTED')}",
        "Optical/regional training in this revision: "
            f"{inference.get('optical_training_performed', 'NOT REPORTED')} / "
            f"{inference.get('regional_training_performed', 'NOT REPORTED')}",
        f"Final unchanged-fit confirmation recorded: {_fmt(acceptance.get('original_fit_verified_unchanged'))}",
        f"mass_eligible recorded: {inference.get('mass_eligible', 'NOT REPORTED')}",
        f"Scientific acceptance recorded: {acceptance.get('scientific_acceptance', inference.get('scientific_acceptance', 'NOT REPORTED'))}",
        "Weight SHA values above are copied from immutable lineage/acceptance receipts; this exporter does not rehash weights.",
        "Inherited limitations remain: late-yellow normalization gate and 4x4 donor-tile spatial CV warnings.",
        "This support/display rerun does not change annotations or resolve the inherited scientific QA exceptions.", ""])

    sidecar_csv = _csv_new(output / "rerun_sidecar_per_frame.csv", source_rows)
    phase_csv = _csv_new(output / "rerun_phase_maps_per_frame.csv", phase_rows)
    display_csv = _csv_new(output / "rerun_display_per_frame.csv", display_rows)
    lines.extend(["4a. Scientific float sidecars", f"Receipt: {paths['sidecar']}",
        f"Reported QA: {_fmt(checks['sidecar'])}; frames: {_fmt(frame_counts['sidecar'])}",
        f"Rule 1 enclosed-hole additions per frame, expected px min..max: {_range(source_rows, 'expected_enclosed_hole_pixels')}",
        f"Rule 1 finite added px min..max: {_range(source_rows, 'rule1_added_valid_pixels')}; total: {_fmt(_total(source_rows, 'rule1_added_valid_pixels'))}",
        f"Rule 2 vertical-gap additions per frame, expected px min..max: {_range(source_rows, 'expected_vertical_gap_pixels')}",
        f"Rule 2 finite added px min..max: {_range(source_rows, 'rule2_added_valid_pixels')}; total: {_fmt(_total(source_rows, 'rule2_added_valid_pixels'))}",
        "Rule limits: enclosed 8-connected components <=60000 px, no border contact; vertical invalid gaps <=64 px supported above/below.",
        f"Filled rule 1 pixels missing finite signal, total: {_fmt(_total(source_rows, 'filled_hole_pixels_without_finite_sidecar'))}",
        f"Filled rule 2 pixels missing finite signal, total: {_fmt(_total(source_rows, 'filled_vertical_gap_pixels_without_finite_sidecar'))}",
        f"Old-valid max absolute log delta: {_fmt(source.get('max_old_valid_abs_delta'))}; fixed limit: {_fmt(source.get('old_valid_max_abs_delta_limit', 1e-6))}",
        f"Old-valid pixels lost: {_fmt(_total(source_rows, 'old_valid_lost_pixels'))}; added pixels outside allowed rules: {_fmt(_total(source_rows, 'added_outside_allowed_fill_pixels'))}",
        f"Old/new finite-mask mismatch pixels: {_fmt(_total(source_rows, 'old_validity_finite_mismatch_pixels'))} / {_fmt(_total(source_rows, 'new_validity_finite_mismatch_pixels'))}",
        f"Live sidecar payload hashes checked in QA: {_fmt(source.get('payload_hashes_live_verified'))}",
        f"Failed source frames: {_references(source.get('failed_frames') or [])}",
        f"Per-frame CSV: {sidecar_csv or 'NOT AVAILABLE'}", ""])

    chart_pass = sum((row.get("chart_visibility") or {}).get("chart_region_rendered_proxy_pass") is True
                     for row in display_rows)
    chart_fail = [row for row in display_rows
                  if (row.get("chart_visibility") or {}).get("chart_region_rendered_proxy_pass") is False]
    lines.extend(["4b. Photographic display", f"Receipt: {paths['display']}",
        f"Reported QA: {_fmt(checks['display'])}; frames: {_fmt(frame_counts['display'])}",
        f"ColorChecker rendered-region proxy passes: {chart_pass}/{len(display_rows)}; failures: {_references(chart_fail)}",
        "The proxy checks nonblank source chart regions. It does not establish semantic chart visibility, correct chart colours or restoration of a physically occluded chart.",
        f"All photographic/RAW-detail pairs differ: {_fmt(display.get('all_main_and_raw_detail_differ'))}",
        f"Minimum full-frame finite fraction: {_fmt(display.get('minimum_full_frame_finite_fraction'))}",
        f"All43 start contact sheet: {contacts.get('start_contact_path', 'NOT AVAILABLE')}",
        f"All43 nearest24h contact sheet: {contacts.get('nearest24h_contact_path', 'NOT AVAILABLE')}",
        f"Per-frame CSV: {display_csv or 'NOT AVAILABLE'}", ""])

    flagged = [row for row in phase_rows if row.get("outside_changes_flagged") is True]
    continuity = phase.get("early_ac28_bottom_band_continuity") or []
    lines.extend(["4c. Native categorical phase maps", f"Receipt: {paths['phase_maps']}",
        f"Reported QA: {_fmt(checks['phase_maps'])}; runs/frames: {_fmt(phase.get('runs'))}/{_fmt(frame_counts['phase_maps'])}; comparison host: {phase.get('host', 'local')}",
        f"Unresolved rule1 pixels including outer guard: {_fmt(phase.get('total_unresolved_pixels_in_filled_holes', _total(phase_rows, 'unresolved_pixels_in_filled_holes')))}",
        f"Unresolved rule2 pixels including outer guard: {_fmt(phase.get('total_unresolved_pixels_in_filled_vertical_gaps', _total(phase_rows, 'unresolved_pixels_in_filled_vertical_gaps')))}",
        f"Unresolved rule1/rule2 outside outer8px feature guard: {_fmt(phase.get('total_unresolved_filled_pixels_outside_outer_8px_feature_guard'))}",
        f"Unresolved filled pixels confined to outer guard: {_fmt(_total(phase_rows, 'unresolved_filled_pixels_in_outer_8px_feature_guard'))}",
        f"Maximum change outside filled support dilated30px / union plume: {_fmt(phase.get('max_outside_change_fraction_of_plume'))}; fixed limit: {_fmt(phase.get('outside_change_fraction_limit', .001))}",
        f"Outside-change flagged frames: {len(flagged)}; {_references(flagged)}",
        f"All phase QA failed frames: {_references(phase.get('failed_frames') or [])}",
        f"Aqueous px per frame old min..max / new min..max: {_range(phase_rows, 'old_aqueous_pixels')} / {_range(phase_rows, 'new_aqueous_pixels')}",
        f"Free-gas px per frame old min..max / new min..max: {_range(phase_rows, 'old_gas_pixels')} / {_range(phase_rows, 'new_gas_pixels')}",
        "AC28 plume continuity across the former bottom band, 0.25..1h:"])
    if continuity:
        for row in continuity:
            lines.append(f"  frame_{row.get('frame_index')}, t={row.get('elapsed_h')}h: "
                f"crossing components={row.get('crossing_plume_components')}, {_fmt(row.get('passed'))}")
    else:
        lines.append("  NOT REPORTED / NOT EVALUATED")
    lines.extend([f"Per-frame aqueous/free-gas CSV: {phase_csv or 'NOT AVAILABLE'}",
                  "Pixel counts describe categorical maps; they are not CO2 masses or concentrations.", ""])

    lines.extend(["4d. Real DarSIA-dev Studio", f"Receipt: {paths['studio']}",
        f"Reported real-viewer QA: {_fmt(checks['studio'])}; status: {studio.get('status', 'NOT REPORTED')}",
        f"Window/Qt platform: {studio.get('window_class', 'NOT REPORTED')} / {studio.get('qt_platform', 'NOT REPORTED')}",
        f"Run/frame: {studio.get('run', 'NOT REPORTED')}/{studio.get('frame_index', 'NOT REPORTED')}",
        f"Normal/RAW-detail toggle changes actual scene: {_fmt((studio.get('checks') or {}).get('raw_detail_toggle_changes_scene'))}",
        f"Phase overlay unchanged by toggle: {_fmt((studio.get('checks') or {}).get('phase_overlay_unchanged_by_raw_detail_toggle'))}",
        f"Existing user windows touched: {studio.get('existing_user_windows_touched', 'NOT REPORTED')}",
        f"Main screenshot: {studio.get('main_screenshot', 'NOT AVAILABLE')}",
        f"RAW-detail screenshot: {studio.get('raw_detail_screenshot', 'NOT AVAILABLE')}",
        f"Probe error: {studio.get('error', 'none')}", ""])

    batch_path = inference.get("materialization_receipt_path")
    batch, batch_issue = _load(Path(batch_path)) if batch_path else (None, "Batch receipt path unavailable")
    revisions = (batch or {}).get("revisions") or []
    cache_rows = [row["source_verification_cache"] for row in revisions
                  if isinstance(row.get("source_verification_cache"), dict)]
    lines.extend(["Immutable source SHA cache: I/O counts only",
        f"Batch receipt: {batch_path or 'NOT AVAILABLE'}",
        f"Worker cache receipts: {len(cache_rows)}/{len(revisions)}",
        f"Source SHA cache hits: {_fmt(_total(cache_rows, 'cache_hits'))}",
        f"Repeated source-read bytes avoided: {_fmt(_total(cache_rows, 'avoided_source_read_bytes'))}",
        f"Workers sealed by actual SHA rechecks: {sum(row.get('sealed_actual_sha_rechecks') is True for row in cache_rows)}/{len(cache_rows)}",
        "These counters do not measure an end-to-end speedup. Output SHA verification remains separate and complete.",
        f"Cache receipt issue: {batch_issue or 'none'}", "",
        "Acceptance binding (when available)",
        f"Acceptance report: {output / 'acceptance_report.json'}",
        f"Recorded checks: {json.dumps(acceptance.get('acceptance_checks') or {}, sort_keys=True)}",
        f"Acceptance report issue: {acceptance_issue or 'none'}", ""])
    with target.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args(argv)
    print(write_report(args.output_dir, args.candidate))


if __name__ == "__main__":
    main()
