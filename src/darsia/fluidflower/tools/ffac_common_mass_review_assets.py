"""Append saved-output previews to the existing common calibration review.

This companion never loads RAW, concentration arrays or labels, and never
evaluates a calibration. It reads committed frame receipts and their PNGs.
Every output goes into a fresh folder; only the isolated review store's current
pointer changes, with all run statuses remaining pending.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import textwrap
import uuid

CONTRACT = "ffac-common-native-saved-review-assets-v1"
SCOPE = ("Validation is conditional on the locked P1 final 3 phases and fixed v5 "
         "normalisation, not independent end-to-end validation. No map is rescaled "
         "to injected mass. Blocked bottom and below-photo fill remain outside main mass.")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def asset(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha_file(path), size_bytes=path.stat().st_size)


def bound_path(descriptor, *, parent=None):
    path = Path(descriptor["path"])
    if not path.is_absolute() and parent is not None:
        path = Path(parent) / path
    if sha_file(path) != descriptor.get("sha256"):
        raise ValueError(f"Saved review source SHA changed: {path}")
    return path.resolve()


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_context(job, workflow_binding, result_manifest):
    """Verify saved artifact identity without invoking a scientific executor."""
    job = Path(job).resolve()
    binding = read_json(workflow_binding)
    plan_path = bound_path(binding["calibration_plan"])
    plan = read_json(plan_path)
    manifest = read_json(result_manifest)
    if (manifest.get("calibration_plan_id") != plan.get("plan_id") or
            manifest.get("objective_contract_sha256") != binding.get("selection_contract_sha256")):
        raise ValueError("Saved result does not belong to the bound existing calibration plan")
    freeze_path = bound_path(manifest["native_frozen_calibration"], parent=Path(result_manifest).parent)
    freeze = read_json(freeze_path)
    if (freeze.get("frozen") is not True or not freeze.get("frozen_utc") or
            freeze.get("map_mass_rescaling") is not False):
        raise ValueError("Saved review requires an immutable no-rescaling calibration freeze")
    for name, expected in freeze.get("calibration_files", {}).items():
        path = Path(name)
        if not path.is_absolute():
            path = freeze_path.parent / path
        if sha_file(path) != expected:
            raise ValueError(f"Saved frozen calibration artifact changed: {path}")
    input_plan = read_json(bound_path(binding["input_plan"]))
    source_metadata = read_json(bound_path(input_plan["source_metadata"]))
    split = input_plan["split"]
    runs = set(split["production_runs"])
    if (set(freeze.get("calibration_runs", ())) != set(split["calibration_runs"]) or
            set(freeze.get("heldout_runs", ())) != set(split["held_out_runs"]) or
            set(freeze.get("calibration_runs", ())) & set(freeze.get("heldout_runs", ()))):
        raise ValueError("Saved freeze and source-bound run split differ")
    expected = {(row["run"], int(row["frame_index"])): row for row in source_metadata["frames"] if row["run"] in runs}
    check_path = bound_path(manifest["heldout_check"], parent=Path(result_manifest).parent)
    check = read_json(check_path)
    return dict(job=job, binding=binding, plan=plan, manifest=manifest,
        freeze=freeze, freeze_path=freeze_path, freeze_sha256=sha_file(freeze_path),
        check=check, check_path=check_path, expected_frames=expected, expected_runs=runs)


def read_saved_rows(context):
    rows = []
    seen = set()
    for path in sorted((context["job"] / "production").glob("*/frame_*/receipt.json")):
        row = read_json(path)
        key = (row.get("run"), row.get("frame_index"))
        expected = context["expected_frames"].get(key)
        if expected is None or key in seen or row.get("complete") is not True:
            raise ValueError(f"Unexpected or incomplete saved review frame: {path}")
        if (row.get("calibration_freeze_sha256") != context["freeze_sha256"] or
                row.get("source_sha256") != expected["source_sha256"] or
                row.get("phase_sha256") != expected["labels"]["sha256"] or
                row.get("pixel_area_sha256") != expected["area_map"]["sha256"] or
                row.get("elapsed_h") != expected["elapsed_h"]):
            raise ValueError(f"Saved frame differs from its frozen source/time/area identity: {path}")
        if (row.get("map_rescaled_to_injected_mass") is not False or
                row.get("bottom_mass_allowed") is not False):
            raise ValueError("Saved frame violates the no-rescaling/bottom block contract")
        if row.get("created_utc") and datetime.fromisoformat(row["created_utc"].replace("Z", "+00:00")) < datetime.fromisoformat(context["freeze"]["frozen_utc"].replace("Z", "+00:00")):
            raise ValueError("Saved concentration frame predates the common calibration freeze")
        native = row.get("native_result", {})
        for field in ("aqueous_mass_g", "gas_mass_g", "total_mass_g"):
            if not isinstance(native.get(field), (float, int)) or not math.isfinite(native[field]):
                raise ValueError(f"Saved frame lacks finite main native mass: {path}")
        if not math.isclose(native["total_mass_g"], native["aqueous_mass_g"] + native["gas_mass_g"], rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Saved main phase masses do not add to saved main total")
        row["receipt_asset"] = asset(path)
        row["frame_folder"] = str(path.parent)
        rows.append(row)
        seen.add(key)
    if not rows:
        raise ValueError("No saved concentration frames exist; never evaluate them in a display helper")
    context["complete_inventory"] = seen == set(context["expected_frames"])
    if context["check"].get("outcome") != "pass" and context["complete_inventory"]:
        raise ValueError("Failed/unassessable check cannot be labelled as a complete production result")
    return rows


def mass_table_rows(rows):
    """Preserve main, blocked and unknown quantities as distinct columns."""
    output = []
    for row in sorted(rows, key=lambda r: (r["run"], r["elapsed_h"], r["frame_index"])):
        result = row["native_result"]
        item = {key: row.get(key) for key in ("run", "frame_index", "elapsed_h", "heldout", "run_caveat", "phase_caveat", "nominal_injected_mass_g")}
        item.update({key: result.get(key) for key in ("aqueous_mass_g", "gas_mass_g", "total_mass_g",
            "total_mass_lower_g", "total_mass_upper_g", "unknown_mass_upper_g",
            "mass_not_eligible_g", "unknown_physical_main_phase_pixels")})
        item.update(map_rescaled_to_injected_mass=False, bottom_mass_allowed=False,
            conditional_validation_scope=SCOPE, unknown_bounds_are="physical capacity, not statistical confidence")
        for name, value in result.get("excluded_mass", {}).items():
            for field in ("aqueous_mass_g", "gas_mass_g", "total_mass_g", "unknown_mass_upper_g", "unknown_physical_phase_pixels"):
                item[name + "__" + field] = value.get(field)
            item[name + "__physical_capacity_upper_g"] = value.get("phase_mass_capacity", {}).get("capacity_upper_g")
        output.append(item)
    return output


def render_mass_plot(path, rows, *, diagnostic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    timeline = sorted(rows, key=lambda r: (r["elapsed_h"], r["frame_index"]))
    times = [r["elapsed_h"] for r in timeline]
    results = [r["native_result"] for r in timeline]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for field, label, color in (("aqueous_mass_g", "Main aqueous", "#168b78"),
                              ("gas_mass_g", "Main gas", "#bb4f85"),
                              ("total_mass_g", "Main total", "#1e4e8c")):
        axes[0].plot(times, [r[field] for r in results], "o-", ms=3, label=label, color=color)
    axes[0].plot(times, [r.get("nominal_injected_mass_g") for r in timeline], "--", color="#666666", label="Source-bound injected dose (comparison)")
    lower = np.array([r.get("total_mass_lower_g", r["total_mass_g"]) for r in results], float)
    upper = np.array([r.get("total_mass_upper_g") if r.get("total_mass_upper_g") is not None else np.nan for r in results], float)
    if np.any(np.isfinite(upper)):
        axes[0].fill_between(times, lower, upper, where=np.isfinite(upper), color="#1e4e8c", alpha=.16, label="Main unknown-facies physical capacity")
    if any(r.get("total_mass_upper_g") is None for r in results):
        axes[0].text(.01, .97, "Some main physical upper bounds unavailable", transform=axes[0].transAxes, va="top")
    excluded_names = sorted({key for r in results for key in r.get("excluded_mass", {})})
    for name in excluded_names:
        values = [r.get("excluded_mass", {}).get(name, {}) for r in results]
        axes[1].plot(times, [v.get("total_mass_g", 0.) for v in values], "o--", ms=3, label=name + " computed mass (outside main)")
        capacity = [v.get("phase_mass_capacity", {}).get("capacity_upper_g") for v in values]
        if any(v is not None for v in capacity):
            axes[1].plot(times, [v if v is not None else np.nan for v in capacity], ":", label=name + " physical capacity")
        if any(v.get("unknown_physical_phase_pixels", 0) for v in values):
            axes[1].text(.01, .98-.07*excluded_names.index(name), name + ": missing physical geometry, upper bound unbounded", transform=axes[1].transAxes, va="top", fontsize=8)
    if not excluded_names:
        axes[1].text(.03, .6, "No separate excluded contributions in these saved frames", transform=axes[1].transAxes)
    baseline = [r for r in timeline if r["elapsed_h"] <= 0]
    baseline_note = ("Saved pre-injection main mass: " + ", ".join(f"{r['elapsed_h']:.3f} h = {r['native_result']['total_mass_g']:.5g} g" for r in baseline)) if baseline else "No pre-injection output in this diagnostic subset; zero is not imposed."
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_ylabel("CO2 mass (g)")
        if ax.lines:
            ax.legend(fontsize=8, loc="best")
    axes[1].set_xlabel("Camera time relative to injection (h)")
    title = timeline[0]["run"].upper() + (" — diagnostic saved subset" if diagnostic else " — saved native common calibration")
    fig.suptitle(title)
    caveat = timeline[0].get("run_caveat", "") + " " + timeline[0].get("phase_caveat", "")
    footer = "\n".join(textwrap.wrap(baseline_note + " " + SCOPE + " " + caveat + " Capacity bounds are physical limits, not confidence intervals.", 130))
    fig.text(.04, .02, footer, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, .18, 1, .96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def archive_implementation(context, folder):
    """Keep exact producing source files without changing live code or freeze."""
    files = dict(context["freeze"].get("native_implementation_files", {}))
    campaign = context["job"] / "campaign_binding.json"
    if campaign.is_file():
        binding = read_json(campaign)
        files.update(binding.get("code_sha256", {}))
        if binding.get("driver_sha256"):
            files[str(Path(__file__).with_name("run_ffac_common_mass_campaign.py"))] = binding["driver_sha256"]
    archived = {}
    for raw, expected in files.items():
        source = Path(raw)
        if sha_file(source) != expected:
            raise ValueError(f"Cannot archive changed producer implementation: {source}")
        destination = Path(folder) / (expected[:12] + "_" + source.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        if sha_file(destination) != expected:
            raise ValueError("Archived implementation differs from producing source")
        archived[raw] = asset(destination)
    return archived


def publish_pending_review(context, manifest_path, assets):
    from darsia.fluidflower.calibration import CalibrationPlanStore
    from darsia.fluidflower.calibration_review import CalibrationResultReviewStore, default_calibration_review_config
    from darsia.fluidflower.project import FluidFlowerProject
    project = FluidFlowerProject.load(Path(context["binding"]["project_root"]))
    current = CalibrationPlanStore(project).current()
    if current.get("plan_id") != context["plan"].get("plan_id"):
        raise ValueError("Existing workflow changed its calibration plan before review append")
    config = default_calibration_review_config(project)
    config["calibration_result_manifest_path"] = str(manifest_path)
    config["run_reviews"] = {run: dict(status="pending", reviewer="", note="Saved native outputs only; pending human review. " + SCOPE,
        concentration_preview_path=value["concentration_preview"]["path"], mass_plot_path=value["mass_plot"]["path"])
        for run, value in assets.items()}
    revision = CalibrationResultReviewStore(project).save_review(config, actor="Codex",
        reason="Append source-bound concentration previews and mass plots; no run is approved and no production default is changed")
    return dict(review_id=revision["review_id"], project_root=str(project.root), approved_run_count=0,
        default_changed=False, phase_parent_changed=False, status="pending")


def build_review_assets(job, workflow_binding, result_manifest, *, destination=None, publish=True):
    context = verify_context(job, workflow_binding, result_manifest)
    rows = read_saved_rows(context)
    destination = Path(destination) if destination is not None else context["job"] / ("review_assets_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    diagnostic = context["check"].get("outcome") != "pass" or not context["complete_inventory"]
    review_assets = {}
    for run in sorted({r["run"] for r in rows}):
        timeline = [r for r in rows if r["run"] == run]
        selected = min(timeline, key=lambda r: (abs(r["elapsed_h"]-2.25), r["elapsed_h"], r["frame_index"]))
        display = selected["assets"]["c_aq_g_l_display"]
        preview = bound_path(display, parent=selected["frame_folder"])
        mass_plot = destination / (run + "_mass.png")
        render_mass_plot(mass_plot, timeline, diagnostic=diagnostic)
        review_assets[run] = dict(concentration_preview=asset(preview), mass_plot=asset(mass_plot),
            preview_frame_index=selected["frame_index"], preview_camera_time_h=selected["elapsed_h"],
            preview_receipt=selected["receipt_asset"], available_saved_frames=len(timeline),
            all_statuses_pending=True, conditional_validation_scope=SCOPE)
    table = mass_table_rows(rows)
    columns = sorted({key for row in table for key in row})
    with (destination / "saved_mass_inputs.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(table)
    archived = archive_implementation(context, destination / "code_snapshot")
    manifest = dict(context["manifest"], review_assets=review_assets,
        source_result_manifest=asset(result_manifest), review_assets_scope="diagnostic saved subset" if diagnostic else "complete saved inventory",
        available_run_count=len(review_assets), available_frame_count=len(rows),
        expected_run_count=len(context["expected_runs"]), expected_frame_count=len(context["expected_frames"]),
        complete_inventory=context["complete_inventory"], conditional_validation_scope=SCOPE,
        archived_producer_implementations=archived, bottom_mass_allowed=False, phase_parent_changed=False,
        default_changed=False, approved=False, map_mass_rescaling=False)
    manifest["native_frozen_calibration"] = asset(context["freeze_path"])
    manifest["heldout_check"] = asset(context["check_path"])
    manifest["parameter_artifact"] = asset(bound_path(context["manifest"]["parameter_artifact"], parent=Path(result_manifest).parent))
    manifest_path = destination / "calibration_result_with_review_assets.json"
    write_new_json(manifest_path, manifest)
    publication = publish_pending_review(context, manifest_path, review_assets) if publish else dict(status="not published", approved_run_count=0)
    write_new_json(destination / "review_assets_receipt.json", dict(contract=CONTRACT,
        created_utc=datetime.now(timezone.utc).isoformat(), source_result_manifest=asset(result_manifest),
        workflow_binding=asset(workflow_binding), native_frozen_calibration=asset(context["freeze_path"]),
        manifest=asset(manifest_path), review_assets=review_assets, publication=publication,
        available_run_count=len(review_assets), available_frame_count=len(rows),
        complete_inventory=context["complete_inventory"], diagnostic_only=diagnostic,
        scientific_arrays_read=False, raw_images_read=False, scientific_evaluation_performed=False,
        calibration_or_labels_modified=False, conditional_validation_scope=SCOPE,
        code_snapshot=archived))
    (destination / "README.md").write_text("# Saved common calibration review assets\n\n" + SCOPE +
        "\n\nMain and separate excluded quantities are plotted without recomputing or rescaling them. "
        "Grey concentration pixels remain unquantified. Available saved inventory: " + str(len(rows)) + " frames in " + str(len(review_assets)) +
        " runs. The plots do not imply absent frames were computed. Status is pending; no scientific approval or default change.\n\n"
        "Exact producing implementation files are archived for reproducibility. Current existing result validators still check the live native executor SHA; "
        "future executor changes may block this legacy validation route even though these saved outputs are immutable. "
        "A future display-only verifier must validate archived producers and saved output bindings without re-executing or requiring the current engine.\n", encoding="utf-8")
    return dict(folder=str(destination), manifest=str(manifest_path), publication=publication,
        diagnostic_only=diagnostic, available_run_count=len(review_assets), available_frame_count=len(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--workflow-binding", type=Path, required=True)
    parser.add_argument("--result-manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args(argv)
    result = build_review_assets(args.job, args.workflow_binding, args.result_manifest,
        destination=args.destination, publish=not args.no_publish)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
