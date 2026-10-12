"""Report a frozen common mass fit and its saved partial production outputs.

No scientific image, map array, evaluator, or fitter is opened here. Saved scalar masses
and precision-audit rows remain distinct from full production. On a failed or
incomplete check, the report records the stop rather than claiming 39 runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from darsia.fluidflower import phase_ffac_common_mass_production as output
from darsia.fluidflower.tools import ffac_common_mass_review_assets as saved

CONTRACT = "ffac-common-native-frozen-diagnostic-report-v1"
SCOPE = saved.SCOPE


def calibration_tables(context):
    """Read already computed histogram/exact results; never re-evaluate them."""
    fit_path = context["job"] / "calibration" / "shared_fit.json"
    validation_path = context["job"] / "calibration" / "exact_validation" / "validation.json"
    fit = saved.read_json(fit_path)
    audit = saved.read_json(validation_path)
    if (audit.get("complete") is not True or audit.get("heldout_pixels_read") is not False or
            audit.get("summary_precision_bound_verified") is not True or
            audit.get("parameters") != context["freeze"]["shared_scales"] or
            fit.get("parameters") != context["freeze"]["shared_scales"]):
        raise ValueError("Saved calibration audit is incomplete or uses another frozen pair")
    if saved.bound_path(audit["fit"]) != fit_path.resolve():
        raise ValueError("Exact audit describes another calibration fit")
    calibration_runs = set(context["freeze"]["calibration_runs"])
    keys = {(row["run"], int(row["frame_index"])) for row in fit["frames_used"]}
    constraints = {(row["run"], int(row["frame_index"])): row for row in fit.get("per_frame_constraints", [])}
    rows = audit["histogram_rows"]
    if (len(rows) != len(keys) or {(r["run"], int(r["frame_index"])) for r in rows} != keys or
            any(r["run"] not in calibration_runs for r in rows)):
        raise ValueError("Saved histogram rows differ from the calibration-only fit frames")
    histograms = []
    for row in rows:
        dose = row["injected_mass_target_g"]
        if dose <= 0:
            raise ValueError("Saved calibration dose must be positive")
        metadata = context["expected_frames"][(row["run"], int(row["frame_index"]))]
        if metadata["elapsed_h"] != row["elapsed_h"]:
            raise ValueError("Saved histogram row differs from its source camera time")
        constraint = constraints.get((row["run"], int(row["frame_index"])), {})
        histograms.append(dict(row, saved_histogram_main_mass_ratio=row["total_mass_g"] / dose,
            data_kind="calibration histogram summary, not production concentration maps",
            histogram_row_is_production_map=False, heldout=False, map_rescaled_to_injected_mass=False,
            run_caveat=output.CAVEATS.get(row["run"], "No additional protocol caveat in the common-fit specification."),
            phase_caveat=output.PHASE_CAVEAT, conditional_validation_scope=SCOPE,
            source_sha256=metadata["source_sha256"], phase_sha256=metadata["labels"]["sha256"],
            mass_fit_constraint=constraint, calibration_freeze_sha256=context["freeze_sha256"]))
    exact_rows = []
    for row in audit["exact_rows"]:
        if row["run"] not in calibration_runs or (row["run"], int(row["frame_index"])) not in keys:
            raise ValueError("Exact audit contains a non-calibration or unfitted frame")
        exact_rows.append(dict(row, saved_exact_main_mass_ratio=row["exact_main_mass_g"] / row["injected_mass_target_g"],
            data_kind="full-resolution scalar calibration precision audit, not production maps",
            heldout=False, run_caveat=output.CAVEATS.get(row["run"], "No additional protocol caveat in the common-fit specification."),
            phase_caveat=output.PHASE_CAVEAT, conditional_validation_scope=SCOPE,
            calibration_freeze_sha256=context["freeze_sha256"], map_rescaled_to_injected_mass=False))
    return fit, audit, histograms, exact_rows, saved.asset(fit_path), saved.asset(validation_path)


def saved_sensitivities(rows, freeze_sha):
    result = []
    for row in rows:
        path = Path(row["frame_folder"]) / "sensitivities.json"
        if not path.is_file():
            continue
        value = saved.read_json(path)
        if (value.get("calibration_freeze_sha256") != freeze_sha or
                value.get("source_sha256") != row["source_sha256"] or
                value.get("run") != row["run"] or value.get("frame_index") != row["frame_index"]):
            raise ValueError("Saved sensitivity uses another source or common freeze")
        result.append(value)
    return result


def stage_costs(context, audit):
    rows = [dict(stage="exact calibration precision audit", wall_seconds=audit.get("wall_seconds"),
        cpu_seconds=audit.get("cpu_seconds_sum"), workers=audit.get("workers"),
        frames=audit.get("exact_frame_count"), execution="actual new full-resolution calibration-only audit")]
    for run in context["freeze"]["calibration_runs"]:
        path = context["job"] / "work" / "fit_summaries" / run / "receipt.json"
        value = saved.read_json(path)
        rows.append(dict(stage="scale-independent calibration summaries", run=run,
            wall_seconds=value.get("wall_seconds"), cpu_seconds=value.get("cpu_seconds"),
            frames=len(value.get("summaries", [])), execution="preserved original producer timing; imported, not recomputed" if value.get("import_provenance") else "actual producer timing",
            source_receipt=saved.asset(path)))
    for path in sorted(context["job"].glob("*_launch.json")):
        value = saved.read_json(path)
        rows.append(dict(stage=path.stem, execution="launch metadata, not duration estimate", source=saved.asset(path), launch=value))
    for row in rows:
        row["conditional_validation_scope"] = SCOPE
    return rows


def source_input_descriptors(context, input_report):
    binding = context["binding"]
    plan = saved.read_json(saved.bound_path(binding["input_plan"]))
    recipes = saved.read_json(saved.bound_path(plan["recipes"]))
    result = dict(input_plan=binding["input_plan"],
        assets={key: plan.get(key) for key in ("physical_reference", "recipes", "atmosphere", "source_metadata",
            "original_domain_reference", "clearwater_results_folder", "clearwater_overrides")},
        physical_reference_status=plan.get("physical_reference_status"),
        recipe_default_runs=recipes.get("default_runs", []),
        recipe_inputs_by_run={row["run"]: row for row in recipes.get("runs", [])},
        source_metadata_only=True, raw_or_physical_arrays_read=False,
        caveats=["AC32/AC38 use the predeclared common recipe fallback where native chemistry records are missing.",
            "AC41's clear-water source is borrowed from AC42; per-run registration remains a visible provenance assumption.",
            "AC45 uses the source-bound alternate-lens/static-sand transport v3.",
            "Physical V3 geometry is Olav-accepted provisional, with 10–20 pixel boundary uncertainty.",
            "Measured AC29 MFC dose is used; overflow water displacement is compare-only rig gas inventory, not measured CO2 outlet."],
        superseded_policy="The earlier .05–4 shared gain cap and zero-positive-bottom exclusion were implementation restrictions, not requirements in the user's common-calibration specification; archived attempts remain preserved. This positive-gain revision was fixed before held-out computation.")
    if plan.get("atmosphere"):
        atmosphere = saved.read_json(saved.bound_path(plan["atmosphere"]))
        result["pressure_temperature_and_dose_sources_by_run"] = {row["run"]: {
            key: row.get(key) for key in ("injection", "pressure_temperature", "origin_datetime", "injection_source",
                "nominal_final_injected_mass_g", "observations_rewritten", "extrapolation_allowed")}
            for row in atmosphere.get("runs", [])}
    if input_report is not None:
        path = Path(input_report) / "report_receipt.json"
        report = saved.read_json(path)
        for descriptor in report["files"].values():
            saved.bound_path(descriptor)
        result["prior_input_companion"] = saved.asset(path)
        result["prior_input_companion_preserved"] = True
    return result


def expected_check_keys(context):
    keys = set()
    for run in context["freeze"]["heldout_runs"]:
        timeline = [row for row in context["expected_frames"].values() if row["run"] == run]
        for selected in output.nearest_frame_indices(timeline, (1., 2.25), include_last=False):
            keys.add((run, selected["frame_index"]))
    return keys


def build_diagnostic_report(job, workflow_binding, result_manifest, *, input_report=None, destination=None, publish=True):
    context = saved.verify_context(job, workflow_binding, result_manifest)
    evidence = output.FrozenCalibrationEvidence.verify(context["freeze_path"])
    rows = saved.read_saved_rows(context)
    wanted = expected_check_keys(context)
    actual = {(row["run"], row["frame_index"]) for row in rows}
    complete_check = wanted <= actual
    fit, audit, histograms, exact_rows, fit_asset, audit_asset = calibration_tables(context)
    input_descriptors = source_input_descriptors(context, input_report)
    recipes = input_descriptors["recipe_inputs_by_run"]
    for table in (rows, histograms, exact_rows):
        for row in table:
            recipe = recipes.get(row["run"], {})
            row.update(recipe_source_caveat=recipe.get("caveat"), recipe_source_mode=recipe.get("mode"),
                measured_recipe_available=recipe.get("measured_recipe_available"),
                alkalinity_source=recipe.get("alkalinity_source"))
    destination = Path(destination) if destination is not None else context["job"] / ("diagnostic_report_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    sensitivities = saved_sensitivities(rows, context["freeze_sha256"])
    if complete_check:
        gate = output.write_report_tables(destination / "native_saved_tables", rows,
            freeze=evidence, sensitivities=sensitivities)
        if gate != context["check"]:
            raise ValueError("Reproduced saved-output check differs from the completed campaign check")
    else:
        gate = dict(outcome="not_assessable_incomplete_inventory", expected_check_frames=len(wanted),
            available_check_frames=len(wanted & actual), missing=sorted(wanted-actual), calibration_changed=False,
            followup_policy="stop and report incomplete saved check; do not compute or fit missing held-out inputs here")
        partial = destination / "native_saved_tables"
        partial.mkdir(exist_ok=False)
        output.write_csv_new(partial / "mass_per_frame.csv", saved.mass_table_rows(rows))
        output.write_new_json(partial / "heldout_check1.json", gate)
    output.write_csv_new(destination / "calibration_histogram_ratios_not_production.csv", histograms)
    output.write_csv_new(destination / "calibration_exact_audit_rows_not_production.csv", exact_rows)
    output.write_csv_new(destination / "stage_costs.csv", stage_costs(context, audit))
    output.write_new_json(destination / "source_input_descriptors.json", input_descriptors)
    output.write_new_json(destination / "fixed_shared_fit.json", fit)
    available_runs = sorted({row["run"] for row in rows})
    incomplete = not context["complete_inventory"]
    stopped = gate.get("outcome") != "pass"
    identifiability = dict(jacobian_rank=fit.get("jacobian_rank"),
        nonunique_parameter_warning=fit.get("nonunique_parameter_warning"),
        parameter_identifiability_warning=fit.get("parameter_identifiability_warning"),
        identifiable_shared_pair=fit.get("identifiable_shared_pair"),
        calibration_predictions_identified=fit.get("calibration_predictions_identified"),
        gas_plateau=fit.get("gas_plateau"), reviewer_action="Do not present a rank-1 shared pair as uniquely identified.")
    summary = dict(contract=CONTRACT, native_frozen_calibration=saved.asset(context["freeze_path"]),
        fixed_shared_parameters=context["freeze"]["shared_scales"], fit=fit_asset, calibration_exact_audit=audit_asset,
        available_production_runs=available_runs, available_production_run_count=len(available_runs),
        available_production_frame_count=len(rows), expected_production_run_count=len(context["expected_runs"]),
        expected_production_frame_count=len(context["expected_frames"]), complete_production_inventory=not incomplete,
        available_pre_injection_production_frames=sum(row["elapsed_h"] <= 0 for row in rows),
        available_later_than_2_25h_production_frames=sum(row["elapsed_h"] > 2.25 for row in rows),
        complete_heldout_check_inventory=complete_check, check1_outcome=gate.get("outcome"),
        stop_after_check=stopped, calibration_histogram_run_count=len({r["run"] for r in histograms}),
        calibration_histogram_frame_count=len(histograms), exact_calibration_audit_frame_count=len(exact_rows),
        histogram_rows_are_not_production_maps=True, identifiability=identifiability,
        scientific_evaluation_performed_by_report=False, raw_images_or_scientific_arrays_read=False,
        map_mass_rescaling=False, bottom_mass_allowed=False, phase_parent_changed=False, default_changed=False,
        all_scientific_approvals_pending=True, conditional_validation_scope=SCOPE)
    output.write_new_json(destination / "diagnostic_summary.json", summary)
    manifest = dict(context["manifest"], report=str(destination), diagnostic_summary=saved.asset(destination / "diagnostic_summary.json"),
        available_run_count=len(available_runs), available_frame_count=len(rows), complete_inventory=not incomplete,
        scientific_status="Pending human review; frozen diagnostic only" if stopped or incomplete else "Pending human review; saved full production inventory",
        conditional_validation_scope=SCOPE, parameter_identifiability=identifiability,
        native_frozen_calibration=saved.asset(context["freeze_path"]), heldout_check=saved.asset(destination / "native_saved_tables" / "heldout_check1.json"),
        parameter_artifact=saved.asset(saved.bound_path(context["manifest"]["parameter_artifact"], parent=Path(result_manifest).parent)))
    primary = destination / "native_calibration_result_diagnostic.json"
    output.write_new_json(primary, manifest)
    review = saved.build_review_assets(job, workflow_binding, primary, destination=destination / "existing_review_assets", publish=publish)
    outcome = gate.get("outcome")
    description = ("The frozen held-out check failed or is unassessable; full production stops here." if stopped else
        "The frozen held-out check passed; this report still describes only the committed inventory shown below.")
    warning = fit.get("parameter_identifiability_warning") or fit.get("nonunique_parameter_warning") or "No additional parameter-identifiability warning recorded."
    (destination / "README.md").write_text("# FF_AC common native calibration — frozen diagnostic report\n\n"
        + description + "\n\n"
        + f"Available saved concentration/saturation outputs: **{len(rows)} frames in {len(available_runs)} runs**; expected full inventory is {len(context['expected_frames'])} frames in {len(context['expected_runs'])} runs. "
        + "No absent map, figure or sensitivity is claimed. Calibration histograms are not full concentration maps.\n\n"
        + "This early check subset does not establish a pre-injection zero-mass result or late-time mass behaviour; those are not substituted with assumed values.\n\n"
        + f"Check 1: **{outcome}**. Shared pair is frozen at aqueous={context['freeze']['shared_scales']['aqueous_scale']:.9g}, gas={context['freeze']['shared_scales']['gas_scale']:.9g}. "
        + f"Jacobian rank: {fit.get('jacobian_rank')}. {warning}\n\n"
        + f"Freeze SHA-256: `{context['freeze_sha256']}`. No held-out run enters the pooled colour paths or shared mass fit; no map is rescaled to injected mass.\n\n"
        + SCOPE + "\n\n"
        + "`native_saved_tables/mass_per_frame.csv` contains only saved eligible native masses. Blocked bottom/below-photo and unknown contributions remain separate. "
        + "`native_saved_tables/overflow_comparison_only.csv`, when present, compares independent rig gas inventory; water displacement is not escaped CO2. "
        + "`calibration_histogram_ratios_not_production.csv` contains the saved common-fit histogram predictions; "
        + "`calibration_exact_audit_rows_not_production.csv` contains already performed full-resolution scalar precision checks. "
        + "Neither table is substituted for production concentration maps.\n\n"
        + "The saved input descriptors retain recipe fallback for AC32/38, AC41's borrowed AC42 clear-water source, AC45 transport v3, measured AC29 MFC dose and provisional physical V3 boundary uncertainty. "
        + "The earlier hard gain cap and blanket bottom-positive rejection are archived implementation attempts, superseded before any held-out evaluation. "
        + "Physical missing-mass capacity intervals are not statistical confidence intervals. No time dilution or per-frame mass balance rescaling is introduced.\n\n"
        + "`existing_review_assets/` adds available concentration previews and main/separate-blocked mass plots to the existing isolated calibration review. "
        + "Every run remains pending; no default candidate or approved calibration is changed. No full AC28/47/56 figure grid is fabricated from this subset. "
        + "`stage_costs.csv` distinguishes actual execution from imported producer timing; parallel per-run wall times overlap and must not be added as elapsed wall-clock time. Producing code is archived with the review assets.\n", encoding="utf-8")
    files = {str(path.relative_to(destination)): saved.asset(path) for path in destination.rglob("*") if path.is_file()}
    output.write_new_json(destination / "report_receipt.json", dict(contract=CONTRACT, files=files,
        created_utc=datetime.now(timezone.utc).isoformat(), summary=saved.asset(destination / "diagnostic_summary.json"),
        report_tool=saved.asset(__file__),
        source_result_manifest=saved.asset(result_manifest), workflow_binding=saved.asset(workflow_binding),
        native_frozen_calibration=saved.asset(context["freeze_path"]), review=review,
        existing_campaign_artifacts_changed=False, no_scientific_evaluation=True, no_raw_or_map_arrays_read=True))
    return dict(report=str(destination), report_receipt_sha256=output.sha_file(destination / "report_receipt.json"),
        check1_outcome=outcome, stopped=stopped, incomplete_production_inventory=incomplete,
        available_run_count=len(available_runs), available_frame_count=len(rows), review=review)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--workflow-binding", type=Path, required=True)
    parser.add_argument("--result-manifest", type=Path, required=True)
    parser.add_argument("--input-report", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args(argv)
    value = build_diagnostic_report(args.job, args.workflow_binding, args.result_manifest,
        input_report=args.input_report, destination=args.destination, publish=not args.no_publish)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
