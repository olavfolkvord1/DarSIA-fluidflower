"""Prepare and fit an unpublished two-head revision from exact frozen samples.

Preparation never trains. Original model files and old cross-validation results
remain immutable; a coupled refinement has training-only status until its own
independent evaluation. The same fitter can finish a new Studio continuation
before that continuation is published.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import uuid


import numpy as np

from darsia.fluidflower import phase_runtime
from darsia.fluidflower.phase_features import OpticalFeatureContract
from darsia.fluidflower.phase_learning import inspect_phase_model_bundle
from darsia.fluidflower.phase_training import logistic, trainer as t
from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.materialize_phase_iteration import relocate
from darsia.fluidflower.tools.publish_fitted_phase_candidate import fitted_files, verify_fitted

HEADS = ("co2_presence", "gas_within_co2")
AQ_CLASSES = {"aqueous_present_region", "aqueous_absent_region"}
SEED = Path("continuation_evidence/coupled_seed")
RECEIPT = Path("continuation_evidence/coupled_revision.json")
SEED_ROOT_FILES = {"summary.json", "training_request.json", "cross_validation.json",
    "phase_decision_policy.json", "change_evidence_ablation.json"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(t._json_safe(value), indent=2, ensure_ascii=False,
                                   allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _check_hash(path, expected):
    if not Path(path).is_file() or sha256_file(path) != expected:
        raise ValueError(f"Frozen source changed: {path}")


def _assert_unpublished(root):
    root = Path(root).resolve()
    if (root / "publication_receipt.json").exists() or (root / "validation").exists():
        raise ValueError("Coupled fitting requires an unpublished model without native validation")
    request = read(root / "training_request.json")
    if Path(request["output_root"]).resolve() != root or request["model_id"] != root.name:
        raise ValueError("The request does not own this unpublished output directory")
    prepared = root / "continuation_preparation.json"
    if prepared.is_file():
        destination = read(prepared).get("destination")
        if (destination and Path(destination).resolve() != root
                and Path(destination).exists()):
            raise ValueError("This model already has a published destination")


def ordered_regions(root):
    """Bind the CSV/sample row order to hashed frozen review payloads."""
    root = Path(root).resolve()
    request = read(root / "training_request.json")
    pairs = []
    for bundle in request["bundles"]:
        pair = []
        for role in ("decisions", "combined_manifest"):
            record = bundle[role]
            path = Path(record["snapshot_path"]).resolve()
            if not path.is_relative_to(root / "input_snapshot"):
                raise ValueError("Frozen training input escaped the new model")
            _check_hash(path, record["sha256"])
            pair.append(path)
        pairs.append(t.ReviewBundle(*pair))
    from darsia.fluidflower.phase_review_retractions import model_retractions

    preserve_revisions = bool(request["settings"].get("preserve_native_revisions", False))
    origin_sources = {} if preserve_revisions else None
    _, loaded, _ = t._load_sources_and_regions(
        pairs, withdrawal_snapshot=model_retractions(root),
        preserve_native_revisions=preserve_revisions, region_sources=origin_sources)
    original_by_id = {region.region_id: region for region in loaded}
    if origin_sources is not None:
        # Extraction concatenates one feature task per original native revision.
        # Replay that order so frozen sample owners still refer to the same labels.
        loaded = [region for _, group in t._native_revision_task_groups(loaded, origin_sources)
                  for region in group]
    by_id = {region.region_id: region for region in loaded}
    if len(by_id) != len(loaded):
        raise ValueError("Effective region IDs are not unique")
    with (root / "region_inventory.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(loaded):
        raise ValueError("Region inventory and frozen evidence counts differ")
    ordered = []
    for index, row in enumerate(rows):
        region = by_id.pop(row["region_id"], None)
        if region is None or int(row["region_index"]) != index:
            raise ValueError("Region inventory has a missing, duplicated, or reordered index")
        original = original_by_id[region.region_id]
        for field in ("dataset", "run", "phase_class", "confidence", "source", "reviewer"):
            if row[field] != str(getattr(region, field)):
                raise ValueError(f"Frozen region semantics changed: {region.region_id}/{field}")
        # The trainer inventories sampled fit rows at their source-resolved
        # frame, while unsampled evaluation rows retain the review's index.
        # _native_revision_task_groups certified any reference insertion by
        # unique elapsed time before either index is accepted here.
        if (int(row["frame_index"]) not in
                {original.frame_index, region.frame_index}
                or not np.isclose(float(row["elapsed_h"]), original.elapsed_h, rtol=0, atol=1e-12)
                or float(row["supervision_weight"]) != original.supervision_weight):
            raise ValueError(f"Frozen region timing/weight changed: {region.region_id}")
        ordered.append(region)
    if by_id:
        raise ValueError("Frozen regions were omitted from the sample inventory")
    return ordered


def sample_contract(root, regions, *, load_features=False):
    """Read each artifact once, validating owners and the exact feature order."""
    root = Path(root)
    with np.load(root / "training_samples.npz", allow_pickle=False) as data:
        owners = data["region_indices"].copy()
        names = tuple(str(value) for value in data["feature_names"])
        features = data["features"].copy() if load_features else None
    if (owners.ndim != 1 or not np.issubdtype(owners.dtype, np.integer) or owners.size == 0
            or owners.min() < 0 or owners.max() >= len(regions)):
        raise ValueError("Invalid region owner indices in frozen samples")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate feature columns in frozen samples")
    if features is not None and (features.shape != (owners.size, len(names))
                                 or not np.all(np.isfinite(features))):
        raise ValueError("Invalid or nonfinite frozen training features")
    counts = np.bincount(owners, minlength=len(regions))
    active, ignored, classes = [], [], Counter()
    for i, region in enumerate(regions):
        applicable = region.phase_class in AQ_CLASSES or any(
            logistic.pixel_target(region.phase_class, head) is not None
            or logistic.regional_presence_target(region.phase_class, head) is not None for head in HEADS)
        if applicable and region.supervision_weight > 0 and counts[i] > 0:
            active.append(region.region_id)
            classes[region.phase_class] += 1
        else:
            ignored.append(dict(region_id=region.region_id, phase_class=region.phase_class,
                sampled_rows=int(counts[i]), supervision_weight=region.supervision_weight,
                recognized_target=applicable))
    contract = dict(regions=len(regions), sampled_rows=int(owners.size),
        active_regions=len(active), active_classes=dict(classes), ignored_regions=ignored,
        ordered_region_ids_sha256=hashlib.sha256(json.dumps([r.region_id for r in regions]).encode()).hexdigest(),
        region_indices_sha256=hashlib.sha256(owners.tobytes()).hexdigest(), feature_names=list(names),
        training_samples_sha256=sha256_file(root / "training_samples.npz"),
        region_inventory_sha256=sha256_file(root / "region_inventory.csv"))
    return features, owners, names, contract


def _require_supported_loss(contract, *, evaluation_only_ids=frozenset()):
    # Low-confidence regions have zero training weight by design. Keep their
    # evidence and sampled rows in the frozen contract, but require every
    # positive-weight region to contribute to a recognized joint objective.
    evaluation_only_ids = set(evaluation_only_ids)
    ignored = {row["region_id"]: row for row in contract["ignored_regions"]}
    if any(identifier not in ignored or ignored[identifier]["sampled_rows"] != 0
           for identifier in evaluation_only_ids):
        raise ValueError("Chemistry evaluation-only review unexpectedly supplied fit rows")
    unhandled = [row for row in contract["ignored_regions"]
                 if row["region_id"] not in evaluation_only_ids
                 and (not row["recognized_target"] or row["supervision_weight"] > 0)]
    if unhandled:
        raise ValueError(f"Some frozen labels still have no active loss: {unhandled}")


def _evaluation_only_ids(settings, regions):
    """Bind all source-review exclusions before reusing optical sample rows."""
    evaluation_only = set()
    scope = settings.get("v5_chemistry_fit_scope")
    if scope is not None:
        from darsia.fluidflower.phase_training.v5_chemistry_fit_scope import verify_scope
        excluded = verify_scope(scope, regions)
        evaluation_only.update(region.region_id for region in regions
                               if region.run.casefold() in excluded)
    partition = settings.get("v5_review_fit_partition")
    if partition is not None:
        from darsia.fluidflower.phase_training.v5_review_fit_partition import (
            verify_partition_rows,
        )
        rows = verify_partition_rows(partition, regions)
        evaluation_only.update(identifier for identifier, row in rows.items()
                               if row["role"] == "evaluation-only")
        if any(rows[identifier]["role"] == "fit" for identifier in evaluation_only):
            raise ValueError("Chemistry exclusion conflicts with frozen Track1 fit role")
    return evaluation_only


def _snapshot_code(root):
    from darsia.fluidflower.phase_training import coupled_aqueous, joint_aqueous
    sources = [Path(__file__), Path(joint_aqueous.__file__), Path(coupled_aqueous.__file__), Path(logistic.__file__)]
    result = {}
    for source in sources:
        target = root / "continuation_evidence/coupled_code" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        result[source.name] = sha256_file(target)
    return result


def _verify_code_snapshot(root, hashes):
    from darsia.fluidflower.phase_training import coupled_aqueous, joint_aqueous
    for source in (Path(__file__), Path(joint_aqueous.__file__), Path(coupled_aqueous.__file__), Path(logistic.__file__)):
        _check_hash(source, hashes[source.name])
        _check_hash(root / "continuation_evidence/coupled_code" / source.name, hashes[source.name])


def _archive_seed(root, source):
    archive = root / SEED
    archive.mkdir(parents=True, exist_ok=False)
    for name in SEED_ROOT_FILES | {p.name for p in source.glob("*_fit_verification.json")}:
        if (source / name).is_file():
            shutil.copy2(source / name, archive / name)
    shutil.copytree(source / "models", archive / "models")
    return {str(p.relative_to(archive)): sha256_file(p) for p in archive.rglob("*") if p.is_file()}


def _new_request(root, seed_request, source, *, backend, max_iter, absence_strength):
    request = relocate(copy.deepcopy(seed_request), source, root)
    request.update(model_id=root.name, created_utc=utc_now(), output_root=str(root),
        cache_root=str(root.parent.parent / "coupled_cache" / root.name), locked_review_sessions=[])
    for key in ("request_path", "frozen_pairs"):
        request.pop(key, None)
    settings = request["settings"]
    baseline_parent = seed_request["settings"].get("parent_model")
    settings.update(iteration="Studio retrain · coupled aqueous", parent_model=str(source),
        comparison_baseline_parent_model=baseline_parent, optical_seed_model=str(source),
        evaluation_mode="coupled-training-only", generalization_validated=False,
        scientific_acceptance=False, human_review_required=True, deployment_allowed=False,
        model_initialization="Joint refinement of both current optical heads on exact frozen samples",
        backend=backend, coupled_max_iters=int(max_iter), aqueous_absence_strength=float(absence_strength),
        coupled_aqueous_refinement=True, inherited_heads=[], fit_heads=list(HEADS))
    write(root / "training_request.json", request)
    return request


def prepare_revision(source, output_parent, *, backend="cuda", max_iter=250,
                     absence_strength=1.0, destination_parent=None):
    """Clone a finished fit into a new unpublished identity; do not fit."""
    source, output_parent = Path(source).resolve(), Path(output_parent).resolve()
    if output_parent == source or output_parent.is_relative_to(source):
        raise ValueError("The new revision must live outside its immutable seed")
    from darsia.fluidflower.phase_review_retractions import assert_reuse_policy_current
    frozen_policy = assert_reuse_policy_current(source)
    checked = verify_fitted(source, allow_nonconverged=True)
    model_id = "phase-model-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    root = output_parent / model_id
    root.mkdir(parents=True, exist_ok=False)
    source_files = fitted_files(source)
    source_files += [p for p in (source / "boundary_code_snapshot").rglob("*") if p.is_file()]
    source_files = sorted(set(source_files))
    hashes = {str(p.relative_to(source)): sha256_file(p) for p in source_files}
    for path in source_files:
        relative = path.relative_to(source)
        if (relative.parts[0] == "models" or relative.as_posix() in SEED_ROOT_FILES
                or path.name.endswith("_fit_verification.json")):
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        _check_hash(target, hashes[str(relative)])
    seed_hashes = _archive_seed(root, source)
    request = _new_request(root, read(source / "training_request.json"), source,
        backend=backend, max_iter=max_iter, absence_strength=absence_strength)
    # This is a new reuse transaction: the live guard above ran before cloning.
    # Freeze even an empty legacy policy, then keep it fixed in its fit worker.
    from darsia.fluidflower import phase_review_retractions as withdrawals
    request["review_retractions"] = withdrawals.write_model_snapshot(root, frozen_policy)
    request["settings"][withdrawals.DOCUMENT_KEY] = frozen_policy
    write(root / "training_request.json", request)
    preparation_path = root / "continuation_preparation.json"
    if preparation_path.is_file():
        preparation = read(preparation_path)
        preparation.update(parent_model=str(source), coupled_seed_model=str(source),
            comparison_baseline_parent_model=request["settings"].get("comparison_baseline_parent_model"))
        if destination_parent is not None:
            preparation["destination"] = str(Path(destination_parent).resolve() / root.name)
        else:
            preparation.pop("destination", None)
        write(preparation_path, preparation)
    regions = ordered_regions(root)
    _, _, _, contract = sample_contract(root, regions)
    _require_supported_loss(contract, evaluation_only_ids=_evaluation_only_ids(
        request["settings"], regions))
    for relative, digest in hashes.items():
        _check_hash(source / relative, digest)
    receipt = dict(format="frozen-coupled-aqueous-revision-v1", status="prepared",
        model_id=root.name, source_model=str(source), source_identity=checked,
        source_files=hashes, seed_files=seed_hashes, sample_contract=contract,
        prepared_utc=utc_now(), training_launched=False,
        physics_runtime=phase_runtime.physics_runtime_identity(phase_runtime.load_physics_profile(root)),
        review_retractions=request["review_retractions"],
        source_models_immutable=True, old_oof_metrics_are_seed_only=True,
        new_head_generalization_validated=False, native_validation_required=True)
    write(root / RECEIPT, receipt)
    request["request_path"] = str(root / "training_request.json")
    return request


def _seed_fit(payload, report):
    final = report["final_fit"]
    return logistic.LogisticFit(mean=np.asarray(payload["feature_mean"], dtype=np.float64),
        scale=np.asarray(payload["feature_scale"], dtype=np.float64),
        coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
        intercept=float(payload["intercept"]), backend=final["backend"], precision=final["precision"],
        iterations=int(final["iterations"]), converged=bool(final["converged"]),
        objective=float(final["objective"]), regularization=float(report["selected_regularization"]))


def _verify_retraction_transaction(root, record):
    from darsia.fluidflower import phase_review_retractions as withdrawals
    if not record:
        # Legacy prepared artifacts have no transaction freeze; check before any
        # new fitting reuses their frozen samples or seed weights.
        withdrawals.assert_reuse_policy_current(root)
        return
    snapshot = withdrawals.model_retractions(root)
    if (snapshot is None or snapshot["digest"] != record.get("digest")
            or sha256_file(Path(root) / withdrawals.MODEL_FILE) != record.get("sha256")):
        raise ValueError("Coupled worker withdrawal policy differs from its frozen transaction")


def fit_revision(root):
    """Fit only a prepared unpublished revision and rebuild all head provenance."""
    from darsia.fluidflower.phase_training import joint_aqueous
    root = Path(root).resolve()
    _assert_unpublished(root)
    receipt = read(root / RECEIPT)
    _verify_retraction_transaction(root, receipt.get("review_retractions"))
    if receipt.get("status") != "prepared":
        raise ValueError("Coupled revision is not in the prepared state")
    if (root / "summary.json").exists() and not receipt.get("in_place_unpublished"):
        raise FileExistsError("The new revision already has a completed fit")
    for relative, digest in receipt["seed_files"].items():
        _check_hash(root / SEED / relative, digest)
    request = read(root / "training_request.json")
    settings = request["settings"]
    regions = ordered_regions(root)
    features, owners, names, contract = sample_contract(root, regions, load_features=True)
    if contract != receipt["sample_contract"]:
        raise ValueError("Prepared sample or supervision identity changed")
    evaluation_only_ids = _evaluation_only_ids(settings, regions)
    _require_supported_loss(contract, evaluation_only_ids=evaluation_only_ids)
    fit_runs = {regions[int(index)].run for index in np.unique(owners)
                if regions[int(index)].region_id not in evaluation_only_ids}
    if len(fit_runs) < 2:
        raise ValueError("Coupled fit has too few source-supported training runs")
    seed_summary = read(root / SEED / "summary.json")
    inventory = read(root / "training_inventory.json")
    if inventory != seed_summary["inventory"] or inventory["regions"] != len(regions):
        raise ValueError("Training inventory changed from the seed")
    profile = phase_runtime.load_physics_profile(root)
    if phase_runtime.physics_runtime_identity(profile) != receipt["physics_runtime"]:
        raise ValueError("Inherited physical runtime changed before refinement")
    code_hashes = _snapshot_code(root)
    started = time.perf_counter()
    work = root / "coupled_fit_work"
    work.mkdir(exist_ok=False)
    reports, policies = {}, {}
    for adapter, variant in seed_summary["models"].items():
        print(f"Joint aqueous refinement: {root.name}/{adapter}", flush=True)
        payloads = {h: read(root / SEED / "models" / adapter / f"{h}_signal.json") for h in HEADS}
        feature_contract = OpticalFeatureContract.from_dict(payloads[HEADS[0]]["feature_contract"])
        if payloads[HEADS[1]]["feature_contract"] != payloads[HEADS[0]]["feature_contract"]:
            raise ValueError("Seed head feature contracts disagree")
        columns = [names.index(name) for name in feature_contract.feature_names]
        matrix = np.asarray(features[:, columns], dtype=np.float32)
        seed_fits = {h: _seed_fit(payloads[h], variant["heads"][h]) for h in HEADS}
        regularization = {h: seed_fits[h].regularization for h in HEADS}
        fits, diagnostics = joint_aqueous.fit_joint_aqueous_mil(matrix, owners, regions,
            seed_fits=seed_fits, regularization=regularization, runs=fit_runs,
            backend=settings["backend"], max_iter=int(settings["coupled_max_iters"]),
            aqueous_absence_strength=float(settings["aqueous_absence_strength"]))
        reports[adapter], policies[adapter] = {"feature_contract": feature_contract.to_dict(), "heads": {}}, {}
        for head in HEADS:
            fit = fits[head]
            if not (np.all(np.isfinite(fit.coefficients)) and np.isfinite(fit.intercept)
                    and np.isfinite(fit.objective) and np.all(np.isfinite(fit.mean))
                    and np.all(np.isfinite(fit.scale)) and np.all(fit.scale > 0)):
                raise ValueError("Coupled fitter returned invalid numerical parameters")
            if not (np.array_equal(fit.mean, seed_fits[head].mean)
                    and np.array_equal(fit.scale, seed_fits[head].scale)):
                raise ValueError("Coupled fitter changed the frozen normalization")
            parity = (t._inference_parity_check(fit, matrix, limit=120_000) if fit.backend == "cuda"
                      else t._cpu_reference_check(fit, matrix, limit=120_000))
            if not parity.get("passed"):
                raise ValueError(f"Coupled numerical parity failed: {adapter}/{head}: {parity}")
            cuda_parity = parity if fit.backend == "cuda" else None
            if settings["backend"] == "cuda" and fit.backend == "cpu":
                # A CPU64 optimizer polish changes the actual fit backend, but
                # the intended CUDA native-map backend still needs validation.
                cuda_parity = t._inference_parity_check(fit, matrix, limit=120_000)
                if not cuda_parity.get("passed"):
                    raise ValueError(f"CPU-polished model failed CUDA prediction parity: {adapter}/{head}")
            report = dict(head=head, eligible_runs=sorted(fit_runs),
                fit_mode="joint-aqueous-mil", evaluation_mode="training-only",
                generalization_validated=False, deployment_allowed=False,
                selected_regularization=regularization[head],
                selection_policy="Seed regularization held fixed; no selection or OOF claim for changed heads",
                numerical_validation=parity, cpu64_cuda32_parity=t._cuda_parity_record(cuda_parity or parity),
                final_fit=dict(backend=fit.backend, precision=fit.precision, iterations=fit.iterations,
                    converged=fit.converged, objective=fit.objective, pixel_rows=len(matrix),
                    objective_scope="joint-two-head-training-only", diagnostics=diagnostics))
            seed_provenance = payloads[head]["training_provenance"]
            provenance = {key: copy.deepcopy(seed_provenance[key]) for key in
                ("chemistry", "datasets", "series_scope", "runs", "head_training_runs", "reviewers",
                 "radiometry_profile", "radiometry_runtime") if key in seed_provenance}
            provenance.update(campaign_id=root.name, target_kind=f"visible_{head}_phase_probability",
                fit_mode="joint-aqueous-mil", deployment_allowed=False, review_candidate_only=True,
                generalization_validated=False, training_inventory=inventory, cross_validation=report,
                coupled_sample_contract=contract, seed_model_sha256=payloads[head]["model_sha256"],
                source_sha256=code_hashes,
                backend_contract=dict(production_backend=fit.backend, production_precision=fit.precision,
                    authoritative_reference="cpu-float64", validated_backends=["cpu", "cuda"]
                    if cuda_parity else ["cpu"], parity=parity, cuda_inference_parity=cuda_parity))
            model = t._fit_model_object(fit, feature_contract=feature_contract,
                provenance=t._json_safe(provenance), texture_sigma=float(payloads[head]["texture_sigma"]))
            model.save(work / "models" / adapter / f"{head}_signal.json")
            report.update(model_path=str(root / "models" / adapter / f"{head}_signal.json"),
                model_sha256=model.to_dict()["model_sha256"])
            reports[adapter]["heads"][head] = report
            policies[adapter][head] = dict(presence_threshold=0.5,
                threshold_basis="fixed display threshold; not validated for coupled revision",
                model_sha256=report["model_sha256"])
        write(work / f"{adapter}_joint_diagnostics.json", diagnostics)
    completed = utc_now()
    _verify_code_snapshot(root, code_hashes)
    summary = dict(schema=1, model_id=root.name, campaign_id=root.name,
        artifact_kind="coupled-aqueous-training-candidate", created_utc=request["created_utc"],
        training_completed_utc=completed, fit_heads=list(HEADS),
        evaluation=dict(kind="training-only", generalization_validated=False,
            independent_experiment_required=True, old_oof_results_archived=True),
        inventory=inventory, runs=seed_summary["runs"], models=reports, source_snapshot=request["bundles"],
        seconds=time.perf_counter() - started, coupled_sample_contract=contract,
        **{k:seed_summary[k] for k in ("radiometry_profile", "radiometry_runtime", "feature_extraction") if k in seed_summary})
    # Old scores and optional ablations refer to the seed heads exclusively.
    # They were archived before fitting; none can remain an active root report.
    for name in ("cross_validation.json", "change_evidence_ablation.json"):
        if (root / name).exists():
            if not (root / SEED / name).exists():
                raise ValueError("Old head evaluation was not archived")
            (root / name).unlink()
    for path in root.glob("*_fit_verification.json"):
        _check_hash(root / SEED / path.name, sha256_file(path))
        path.rename(work / path.name)
    if (root / "models").exists():
        # A new continuation can still have ordinary-fit OOF arrays next to its
        # seed heads. Keep them archived, never expose them as coupled scores.
        for path in (root / "models").rglob("*"):
            if path.is_file():
                archived = root / SEED / path.relative_to(root)
                _check_hash(archived, sha256_file(path))
        original_models = (root / "models").resolve()
        retained_models = (work / "replaced_seed_models").resolve()
        if not original_models.is_relative_to(root) or not retained_models.is_relative_to(root):
            raise ValueError("Seed archive move escaped the unpublished output")
        original_models.rename(retained_models)
    shutil.copytree(work / "models", root / "models")
    write(root / "phase_decision_policy.json", dict(schema=1, models=policies))
    write(root / "summary.json", summary)
    inspect_phase_model_bundle(root)
    checked = verify_fitted(root, allow_nonconverged=True)
    if checked["physics_runtime"] != receipt["physics_runtime"]:
        raise ValueError("Coupled packaging changed inherited physics")
    receipt.update(status="fitted", training_launched=True, training_completed_utc=completed,
        elapsed_seconds=summary["seconds"], fitted_identity=checked,
        code_sha256=code_hashes, source_models_immutable=True)
    write(root / RECEIPT, receipt)
    write(root / "coupled_aqueous_fit_verification.json", dict(**checked,
        training_completed_utc=completed, numerical_verification_passed=True,
        active_constraints_verified=True, generalization_validated=False,
        old_oof_results_archived=True, source_models_immutable=True))
    return summary


def refine_unpublished(request):
    """Finish a new continuation's own ordinary fit before publication."""
    root = Path(request["output_root"]).resolve()
    _assert_unpublished(root)
    _verify_retraction_transaction(root, request.get("review_retractions"))
    regions = ordered_regions(root)
    if not any(r.phase_class in AQ_CLASSES and r.supervision_weight > 0 for r in regions):
        return None
    if (root / RECEIPT).exists():
        raise ValueError("This output already has coupled-refinement provenance")
    checked = verify_fitted(root, allow_nonconverged=True)
    _, _, _, contract = sample_contract(root, regions)
    _require_supported_loss(contract, evaluation_only_ids=_evaluation_only_ids(
        request["settings"], regions))
    seed_hashes = _archive_seed(root, root)
    local_request = read(root / "training_request.json")
    local_request["settings"].update(coupled_aqueous_refinement=True,
        coupled_max_iters=int(local_request["settings"].get("coupled_max_iters", 250)),
        aqueous_absence_strength=float(local_request["settings"].get("aqueous_absence_strength", 1.0)),
        evaluation_mode="coupled-training-only", generalization_validated=False,
        deployment_allowed=False, scientific_acceptance=False)
    write(root / "training_request.json", local_request)
    write(root / RECEIPT, dict(format="frozen-coupled-aqueous-revision-v1", status="prepared",
        model_id=root.name, source_model=str(root), source_identity=checked, seed_files=seed_hashes,
        sample_contract=contract, in_place_unpublished=True, prepared_utc=utc_now(),
        review_retractions=request.get("review_retractions"),
        physics_runtime=checked["physics_runtime"], source_models_immutable=True,
        old_oof_metrics_are_seed_only=True, new_head_generalization_validated=False))
    return fit_revision(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output-parent", type=Path, required=True)
    prepare.add_argument("--destination-parent", type=Path)
    prepare.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    prepare.add_argument("--max-iter", type=int, default=250)
    prepare.add_argument("--absence-strength", type=float, default=1.0)
    fit = sub.add_parser("fit")
    fit.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        request = prepare_revision(args.source, args.output_parent, backend=args.backend,
            max_iter=args.max_iter, absence_strength=args.absence_strength,
            destination_parent=args.destination_parent)
        print(json.dumps(dict(stage="prepared", model_id=request["model_id"],
            output_root=request["output_root"], request_path=request["request_path"])))
    else:
        summary = fit_revision(args.model)
        print(json.dumps(dict(stage="fitted", model_id=summary["model_id"],
            training_completed_utc=summary["training_completed_utc"])))


if __name__ == "__main__":
    main()
