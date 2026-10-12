"""Export completed generic MIL fits into a new review candidate without fitting.

The completed development CV and exact fit values are retained. Optical display
thresholds remain CO2=0.5 and conditional gas=0.6. Native review is still pending.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
from darsia.fluidflower import phase_features, phase_runtime
from darsia.fluidflower.phase_learning import HierarchicalPhaseModel, SerializedPhaseHead, inspect_phase_model_bundle
from darsia.fluidflower.phase_training import logistic, trainer as t
from darsia.fluidflower.phase_training.scope import validate_training_series
from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.phase_workflow import prepare_phase_training_request
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.phase_overnight_evaluation import load_frozen_samples
from darsia.fluidflower.tools.prepare_grain_gas_retraining import REVIEWER, _read, _copy, _check_hashes, physics_files


HEADS = ("co2_presence", "gas_within_co2")
GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def validate_fit_report(payload, report, head, runs, width):
    if (report["head"] != head or set(report["eligible_runs"]) != set(runs)
            or report.get("fit_mode") != "mil"):
        raise ValueError("Completed report has a different head/run/training contract")
    for key in ("mean", "scale", "coefficients"):
        values = np.asarray(payload[key], dtype=np.float64)
        if values.shape != (width,) or not np.all(np.isfinite(values)):
            raise ValueError("Fit dimensions or numerical values differ from the generic contract")
    if np.any(np.asarray(payload["scale"]) <= 0) or not np.isfinite(payload["intercept"]):
        raise ValueError("Invalid fit scaling/intercept")
    if (not payload["converged"] or not report["final_fit"]["converged"]
            or payload["regularization"] != report["selected_regularization"]):
        raise ValueError("Final fit did not converge or differs from selected regularization")
    for key in ("backend", "precision", "iterations", "converged", "objective"):
        if payload[key] != report["final_fit"][key]:
            raise ValueError(f"Final fit differs from its report: {key}")
    if not all(report.get(key, {}).get("passed") for key in ("numerical_validation", "cpu64_cuda32_parity")):
        raise ValueError("Completed fit lacks passing CPU/CUDA inference parity")
    grid = report["regularization_grid"]
    if sorted(r["regularization"] for r in grid) != list(GRID):
        raise ValueError("Completed report does not cover the original five regularizations")
    for row in grid:
        if len(row["folds"]) != len(runs) or {f["held_out_run"] for f in row["folds"]} != set(runs):
            raise ValueError("Completed CV has missing or repeated held-out runs")
        if row["regularization"] == payload["regularization"] and not all(f["fit_converged"] for f in row["folds"]):
            raise ValueError("A selected held-out fold did not converge")
    if report["final_fit"]["diagnostics"]["normalization"]["policy"] != logistic.MIL_NORMALIZATION_POLICY:
        raise ValueError("Completed fit used a different normalization policy")


def logistic_fit(payload):
    return logistic.LogisticFit(**{k: np.asarray(v, dtype=np.float64) if k in
        ("mean", "scale", "coefficients") else v for k, v in payload.items()})


def exact_parameters(model, fit):
    for actual, expected in ((model.mean, fit.mean), (model.scale, fit.scale),
                              (model.coefficients, fit.coefficients)):
        if not np.array_equal(actual, expected):
            raise ValueError("Export changed fitted coefficients or normalization values")
    if model.intercept != fit.intercept:
        raise ValueError("Export changed fitted intercept")


def inspect_completed(parent, completed):
    parent, completed = Path(parent).resolve(), Path(completed).resolve()
    data = load_frozen_samples(parent, expected_series="ffac")
    if len(data.regions) != 1925:
        raise ValueError("Expected the complete 1925-region FF_AC evidence")
    protected = dict(data.protected_hashes)
    input_hashes = _read(completed / "ffac/input_hashes.json", protected)
    if input_hashes != data.protected_hashes:
        raise ValueError("Completed fits did not use the exact frozen request/sample/evidence hashes")
    plan = _read(completed / "plan.json", protected)
    if Path(plan["models"]["ffac"]).resolve() != parent or plan["normalization_policy"] != logistic.MIL_NORMALIZATION_POLICY:
        raise ValueError("Completed fit plan differs from the parent or normalized policy")
    for name, expected in plan["code_hashes"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"Completed fitting source changed: {name}")
        protected[str(Path(name).resolve())] = expected
        snapshots = list((completed / "code_snapshot").glob(f"*_{Path(name).name}"))
        if len(snapshots) != 1 or sha256_file(snapshots[0]) != expected:
            raise ValueError("Completed fitting code snapshot is missing or changed")
        protected[str(snapshots[0].resolve())] = expected
    contract = data.contracts["generic"]
    if contract != t.MODEL_VARIANTS["generic"] or len(contract.feature_names) != 30:
        raise ValueError("Expected the shared v4 generic 30-feature pipeline")
    old_head = _read(parent / "models/generic/co2_presence_signal.json", protected)
    if old_head["training_provenance"]["source_sha256"]["optical_features"] != sha256_file(Path(phase_features.__file__)):
        raise ValueError("Optical extraction source changed since the cached v4 feature matrix")
    protected[str(Path(phase_features.__file__).resolve())] = sha256_file(Path(phase_features.__file__))
    heads, fold_audit = {}, []
    runs = data.summary["runs"]
    for head in HEADS:
        directory = completed / "ffac/generic" / head
        payload = _read(directory / "selected_fit.json", protected)
        report = _read(directory / "head_report.json", protected)
        validate_fit_report(payload, report, head, runs, len(contract.feature_names))
        specifications = [(row["regularization"], f["held_out_run"], f)
                          for row in report["regularization_grid"] for f in row["folds"]]
        specifications.append((payload["regularization"], "all_runs", None))
        for regularization, held_out, fold in specifications:
            folder = directory / f"lambda_{regularization:.8g}" / held_out
            receipt = _read(folder / "completed.json", protected)
            trace = _read(folder / "trace.json", protected)
            fitted = _read(folder / "fit.json", protected)
            for name, expected in receipt["hashes"].items():
                path = (folder / name).resolve()
                if path.parent != folder.resolve() or sha256_file(path) != expected:
                    raise ValueError(f"Completed fit artifact changed: {path}")
                protected[str(path)] = expected
            expected_runs = set(runs) - ({held_out} if held_out != "all_runs" else set())
            if (set(trace["runs"]) != expected_runs or trace["head"] != head
                    or trace["regularization"] != regularization or trace["status"] != "completed"):
                raise ValueError("Fold training scope includes held-out evidence or differs from its report")
            if fitted["iterations"] != receipt["iterations"] or fitted["converged"] != receipt["converged"]:
                raise ValueError("Fit receipt differs from completed fit")
            if fold is not None and (fold["fit_iterations"] != fitted["iterations"] or fold["fit_converged"] != fitted["converged"]):
                raise ValueError("CV fold report differs from its completed trace")
            if held_out == "all_runs" and fitted != payload:
                raise ValueError("Selected final fit is not the completed all-run fit")
            fold_audit.append(dict(head=head, regularization=regularization, held_out_run=held_out,
                training_runs=sorted(expected_runs), converged=fitted["converged"],
                iterations=fitted["iterations"], completed=str(folder / "completed.json")))
        heads[head] = dict(fit=logistic_fit(payload), report=report, directory=directory)
    inheritance = _read(parent / "physics_inheritance.json", protected)
    profile = phase_runtime.load_physics_profile(parent)
    if phase_runtime.physics_runtime_identity(profile) != inheritance["physics_runtime"]:
        raise ValueError("Inherited physical runtime changed")
    physics = physics_files(parent, profile)
    old_records = {str(Path(r["target"]).resolve()): r for r in inheritance["copies"]}
    for path in physics:
        row = old_records[str(path.resolve())]
        if sha256_file(path) != row["sha256"] or sha256_file(Path(row["source"])) != row["sha256"]:
            raise ValueError("Physics differs from the exact V12-C3 inherited source")
        protected[str(path.resolve())] = row["sha256"]
    review_source = parent / "review_source/postflight_manifest.json"
    original_source = Path(inheritance["parent_model"]) / "validation/postflight_manifest.json"
    _read(review_source, protected)
    if sha256_file(original_source) != protected[str(review_source.resolve())]:
        raise ValueError("Inherited review source differs from its V12-C3 parent")
    _check_hashes(protected)
    return dict(data=data, protected=protected, plan=plan, contract=contract, heads=heads,
        fold_audit=fold_audit, physics=physics, inheritance=inheritance, profile=profile)


def export(parent, completed, output, *, iteration="V14-BG2"):
    started = time.perf_counter()
    parent, completed, output = (Path(p).resolve() for p in (parent, completed, output))
    if output.exists():
        raise FileExistsError("Use a fresh candidate root; existing models are immutable")
    if output.is_relative_to(parent) or output.is_relative_to(completed):
        raise ValueError("Export must be separate from existing evidence and completed fits")
    checked = inspect_completed(parent, completed)
    data, protected, contract = checked["data"], checked["protected"], checked["contract"]
    protected[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    settings = dict(data.request["settings"], iteration=iteration, parent_model=str(parent),
        model_scope="single-series", series=["ffac"], backend="cuda", mil_max_iters=7680,
        mil_normalization_policy=logistic.MIL_NORMALIZATION_POLICY,
        adapters=["generic"], feature_contracts={"generic": contract.to_dict()},
        fit_heads=[], inherited_heads=[], imported_completed_heads=list(HEADS),
        model_initialization="Export exact completed normalized generic fits; no fitting during export",
        completed_fit_source=str(completed), scientific_acceptance=False,
        manual_approval_required=True, review_label="V14-BG2 generic — awaiting native human review")
    settings.pop("feature_cache_root", None)
    pairs = [(Path(b["decisions"]["snapshot_path"]), Path(b["combined_manifest"]["snapshot_path"]))
             for b in data.request["bundles"]]
    request = prepare_phase_training_request(output_parent=output, cache_parent=output / "unused_cache",
        bundles=pairs, actor=REVIEWER, settings=settings)
    if request.get("locked_review_sessions"):
        raise ValueError("Export unexpectedly locked live reviews")
    root = Path(request["output_root"])
    sources, frozen, _ = t._load_sources_and_regions([t.ReviewBundle(*p) for p in request["frozen_pairs"]])
    validate_training_series(sources, frozen, "ffac")
    if [asdict(r) for r in frozen] != [asdict(r) for r in data.regions]:
        raise ValueError("Export changed frozen evidence")
    copies = []
    def copy_input(source, target):
        expected = protected[str(source.resolve())]
        _copy(source, target, expected)
        copies.append(dict(source=str(source), target=str(target), sha256=expected))
    for source in checked["physics"]:
        copy_input(source, root / source.relative_to(parent))
    copy_input(parent / "review_source/postflight_manifest.json", root / "review_source/postflight_manifest.json")
    for name in ("region_inventory.csv", "training_inventory.json"):
        copy_input(parent / name, root / name)
    copy_input(parent / "training_request.json", root / "fit_source_snapshot/original_training_request.json")
    for source in (completed / "plan.json", completed / "ffac/input_hashes.json"):
        copy_input(source, root / "fit_source_snapshot" / source.name)
    for source in (completed / "code_snapshot").iterdir():
        if source.is_file() and str(source.resolve()) in protected:
            copy_input(source, root / "fit_source_snapshot/code_snapshot" / source.name)
    for head, row in checked["heads"].items():
        for name in ("selected_fit.json", "head_report.json"):
            copy_input(row["directory"] / name, root / "fit_source_snapshot/generic" / head / name)
    matrix = data.matrix("generic")
    # Project the copied sample archive to this candidate's sole feature contract;
    # row/owner identity remains exact, so later audits can reload it normally.
    np.savez_compressed(root / "training_samples.npz", features=matrix,
        region_indices=data.owners, feature_names=np.asarray(contract.feature_names))
    models = {"generic": dict(feature_contract=contract.to_dict(), heads={})}
    policy = dict(schema=1, region_score="90th percentile of pixel probability", models={"generic": {}},
        display_policy=dict(format="two-phase-gas060-v1", co2_presence_threshold=0.5,
                            conditional_gas_threshold=0.6),
        note="OOF presence thresholds describe development CV; Studio retains its fixed display policy")
    source_digests = dict(trainer=sha256_file(Path(t.__file__)),
        training_primitives=sha256_file(Path(logistic.__file__)), optical_features=sha256_file(Path(phase_features.__file__)))
    for head, row in checked["heads"].items():
        report = copy.deepcopy(row["report"])
        provenance = dict(radiometry_profile=data.summary.get("radiometry_profile"),
            radiometry_runtime=data.summary.get("radiometry_runtime"), campaign_id=request["model_id"],
            target_kind=f"visible_{head}_phase_probability", fit_mode="mil", deployment_allowed=True,
            deployment_scope="Optical probability on registered corrected RGB; native human review required",
            runs=data.summary["runs"], head_training_runs=report["eligible_runs"], datasets=["ffac"],
            series_scope="ffac", reviewers=[REVIEWER], training_inventory=data.summary["inventory"],
            cross_validation=report, backend_contract=dict(production_backend=row["fit"].backend,
                production_precision=row["fit"].precision, authoritative_reference="cpu-float64",
                validated_backends=["cpu", "cuda"], parity=report["cpu64_cuda32_parity"]),
            source_sha256=source_digests, created_utc=utc_now(), exported_without_refitting=True,
            completed_fit_source=str(row["directory"]),
            completed_fit_sha256=protected[str((row["directory"] / "selected_fit.json").resolve())],
            fit_input_request_sha256=protected[str((parent / "training_request.json").resolve())],
            evidence_parent_model=str(parent), development_cv_not_independent_acceptance=True,
            human_review_required=True, native_validation_pending=True)
        signal = t._fit_model_object(row["fit"], feature_contract=contract,
            provenance=provenance, texture_sigma=2.0)
        path = signal.save(root / "models/generic" / f"{head}_signal.json")
        report.update(model_path=str(path), model_sha256=signal.to_dict()["model_sha256"], deployment_allowed=True)
        models["generic"]["heads"][head] = report
        policy["models"]["generic"][head] = dict(presence_threshold=report["oof_threshold"],
            model_sha256=report["model_sha256"])
    snapshots = [dict(decisions=str(p[0]), decisions_sha256=sha256_file(p[0]),
        combined_manifest=str(p[1]), combined_manifest_sha256=sha256_file(p[1])) for p in request["frozen_pairs"]]
    summary = dict(schema=1, campaign_id=request["model_id"], model_id=request["model_id"],
        created_utc=utc_now(), iteration=iteration, radiometry_profile=data.summary.get("radiometry_profile"),
        radiometry_runtime=data.summary.get("radiometry_runtime"), backend_requested="cuda",
        cuda_available=True, cuda_probed=False, runs=data.summary["runs"], inventory=data.summary["inventory"],
        source_snapshot=snapshots, models=models, reused_heads=[],
        imported_completed_heads=[f"generic/{h}" for h in HEADS],
        feature_extraction=dict(reused_from=str(parent), original_sample_sha256=protected[str((parent / "training_samples.npz").resolve())],
            projected_feature_names=list(contract.feature_names), region_order_unchanged=True),
        export_seconds=time.perf_counter()-started, seconds=0.0, training_performed_during_export=False,
        development_cv_not_independent_acceptance=True, native_validation_pending=True,
        human_review_required=True, publication_performed=False,
        excluded_adapters={"btb-ratiometric-relative-v2": "Nonconverged final fit; not exported"})
    t._write_json(root / "summary.json", summary)
    t._write_json(root / "cross_validation.json", models)
    t._write_json(root / "phase_decision_policy.json", policy)
    t._write_json(root / "physics_inheritance.json", dict(parent_model=str(parent),
        original_physics_parent=checked["inheritance"]["parent_model"],
        physics_runtime=checked["inheritance"]["physics_runtime"], copies=copies,
        settings_unchanged=True, calibration_unchanged=True))
    if phase_runtime.physics_runtime_identity(phase_runtime.load_physics_profile(root)) != checked["inheritance"]["physics_runtime"]:
        raise ValueError("Export changed inherited physics")
    inspection = inspect_phase_model_bundle(root)
    hierarchical = HierarchicalPhaseModel.load(root, adapter="generic")
    indices = np.unique(np.linspace(0, len(matrix)-1, min(120000, len(matrix)), dtype=np.int64))
    subset = matrix[indices]
    verification = {}
    with threadpool_limits(limits=4):
        for head, row in checked["heads"].items():
            loaded = SerializedPhaseHead.load(root / "models/generic" / f"{head}_signal.json")
            exact_parameters(loaded, row["fit"])
            reference = logistic.predict_logistic(row["fit"], subset, backend="cpu")
            actual = loaded.probability(subset)
            error = np.abs(actual-reference)
            difference = float(np.max(error))
            threshold = .6 if head == "gas_within_co2" else .5
            agreement = float(np.mean((actual >= threshold) == (reference >= threshold)))
            # LogisticFit.predict evaluates float32; SerializedPhaseHead uses
            # float64 then returns float32. Parameters above must remain exact.
            if difference > 1e-5 or agreement < .99999:
                raise ValueError("Reloaded export differs from its exact selected fit")
            verification[head] = dict(rows=len(subset), max_probability_abs_difference=difference,
                p99_probability_abs_difference=float(np.quantile(error, .99)),
                classification_agreement=agreement, display_threshold=threshold,
                reference_precision="float32 fitted-head CPU predictor",
                exported_precision="float64 serialized-head CPU predictor, float32 output",
                fitted_parameters_exact=True, selected_fit_sha256=protected[str((row["directory"] / "selected_fit.json").resolve())])
        scores = hierarchical.predict_features(subset)
        partition = sum(scores[k] for k in ("untouched_water_probability", "aqueous_co2_probability", "free_co2_gas_probability"))
        if not np.allclose(partition, 1.0, atol=1e-6):
            raise ValueError("Reloaded hierarchy does not partition the three phases")
    _check_hashes(protected)
    receipt = dict(model_id=request["model_id"], model_sha256=_model_identity(root)[1],
        series="ffac", iteration=iteration, regions=len(data.regions), adapters=["generic"],
        verification=verification, inspection=inspection, protected_sha256=protected,
        copied_sources=copies, completed_folds=checked["fold_audit"],
        fitted_parameters_preserved=True, training_performed=False, independent_validation=False,
        native_validation_pending=True, publication_performed=False, completed_utc=utc_now())
    t._write_json(root / "export_verification.json", receipt)
    t._write_json(root / "evidence_preflight.json", dict(cumulative_regions=len(data.regions),
        parent_regions=len(data.regions), new_effective_regions=0, parent_model=str(parent),
        live_reviews_modified=False, effective_parent_regions_preserved_exactly=True))
    command = [sys.executable, "-m", "darsia.fluidflower.tools.materialize_phase_iteration", "--model", str(root),
        "--source", str(root / "review_source"), "--work", str(output / "native_validation_staging")]
    result = dict(root=str(root), source=str(root / "review_source"), model_id=request["model_id"],
        iteration=iteration, adapters=["generic"], materialization_command=command,
        export_verification=str(root / "export_verification.json"), training_started=False)
    t._write_json(root / "series_iteration.json", result)
    t._write_json(output / "iteration.json", {"ffac": result})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "completed", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--iteration", default="V14-BG2")
    args = parser.parse_args()
    print(json.dumps(export(args.parent, args.completed, args.output, iteration=args.iteration), indent=2))


if __name__ == "__main__":
    main()
