"""CPU joint G3 gas MIL with one standardized, penalized lamp indicator.

Only the immutable T4 scientific dataset is reused. The loss keeps the G3
weighting rule, then balances positive direct and weak weights by light epoch.
No labels or scientific sources are extracted or changed here.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from . import phase_gas_t4_training as t4
from .phase_training import logistic
from .project import sha256_file
from .review_storage import resolve_review_path

CONTRACT = "ffac-g3-t4b-joint-lamp-shift-pixel-gas-refit-v1"
SPEC_SHA256 = "d369bb3f62ea70dfc0f2bf70077fe47f86e04a1ebb3984bf76437bbaaadb2259"
T4_DATASET_SHA256 = "d2cfb26524c5d698923743bcb5b29fcf1d668ab7238093d60c10327e88e1bda1"
LIGHT_EPOCHS_SHA256 = "ac78808f2fceb4429bedd2789497553c4f08064dc571ec49224fb10589a54aea"
COMBINED_LABEL_SHA256 = "5a767bfc4b5062df5d670e4043365c602b4617452655763f58bd242f7877d0ee"
FLATFIELD_NAMES, HEAD = t4.FLATFIELD_NAMES, t4.HEAD
FEATURE_NAMES = (*FLATFIELD_NAMES, "new_lamp_indicator")
G3_SHA256, G3_LABEL_SHA256, CO2_SHA256 = t4.G3_SHA256, t4.G3_LABEL_SHA256, t4.CO2_SHA256
LCAP_ANNOTATIONS_SHA256 = t4.LCAP_ANNOTATIONS_SHA256
read, safe, digest, checked, label_digest = t4.read, t4.safe, t4.digest, t4.checked, t4.label_digest
epoch_for_run, training_inventory, minimum_data_gate = t4.epoch_for_run, t4.training_inventory, t4.minimum_data_gate
load_combined_training_sources = t4.load_combined_training_sources


def write(path, value):
    """Write new immutable evidence; a matching restart leaves it untouched."""
    path = Path(path)
    value = safe(value)
    if path.is_file():
        if read(path) != value:
            raise ValueError(f"Existing immutable T4b receipt differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_light_epochs(path):
    return t4.load_light_epochs(checked(path, LIGHT_EPOCHS_SHA256))


def append_lamp_indicator(features, run, light_epochs):
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 4 or not np.isfinite(matrix).all():
        raise ValueError("T4b requires four finite original G3 features")
    indicator = float(epoch_for_run(light_epochs, run) == "new_lamps")
    return np.column_stack((matrix, np.full(len(matrix), indicator, np.float32)))


def joint_features(features, owners, regions, light_epochs):
    matrix = np.asarray(features, dtype=np.float32)
    owners = np.asarray(owners, dtype=np.int32)
    if (matrix.ndim != 2 or matrix.shape[1] != 4 or owners.shape != matrix.shape[:1]
            or not np.isfinite(matrix).all() or not owners.size
            or owners.min() < 0 or owners.max() >= len(regions)):
        raise ValueError("T4b scientific rows or region owners changed")
    indicators = np.asarray([epoch_for_run(light_epochs, r.run) == "new_lamps" for r in regions], np.float32)
    return np.column_stack((matrix, indicators[owners]))


def direct_weight_ratio(features, owners, regions, *, head=HEAD):
    """Recompute the base ratio from active direct loss, never weak bags."""
    _matrix, targets, weights, selected = logistic.balanced_pixel_view(features, owners, regions, head=head)
    positive, negative = (float(weights[targets == target].sum()) for target in (1, 0))
    if not min(positive, negative) > 0 or not np.isfinite([positive, negative]).all():
        raise ValueError("Base direct loss requires finite positive and negative mass")
    return dict(ratio=positive / negative, positive_direct_weight=positive,
        negative_direct_weight=negative, direct_sample_rows=len(weights),
        positive_direct_regions=len(np.unique(selected[targets == 1])),
        negative_direct_regions=len(np.unique(selected[targets == 0])),
        weak_bags_in_ratio=False,
        policy="confidence-times-source per box; equal (run,target) strata; uniform within box")


def balance_positive_weights(targets, weights, region_indices, regions, light_epochs, base_ratio):
    """Scale direct positives by R*N/P; leave every negative unchanged."""
    targets, weights = np.asarray(targets), np.asarray(weights, np.float64)
    indices = np.asarray(region_indices, np.int32)
    if (targets.shape != weights.shape or indices.shape != weights.shape or weights.ndim != 1
            or not set(np.unique(targets)).issubset({0, 1}) or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not weights.size or indices.min() < 0 or indices.max() >= len(regions)
            or not math.isfinite(float(base_ratio)) or float(base_ratio) <= 0):
        raise ValueError("T4b balancing requires finite positive loss weights and base ratio")
    region_epochs = np.asarray([epoch_for_run(light_epochs, r.run) for r in regions])
    epochs = region_epochs[indices]
    balanced = weights.copy()
    factors, report = {}, {}
    for epoch in light_epochs["epochs"]:
        positive_mask, negative_mask = ( (epochs == epoch) & (targets == target) for target in (1, 0))
        positive, negative = float(weights[positive_mask].sum()), float(weights[negative_mask].sum())
        if min(positive, negative) <= 0:
            raise ValueError(f"T4b epoch lacks active positive or negative direct loss: {epoch}")
        factor = float(base_ratio) * negative / positive
        balanced[positive_mask] *= factor
        factors[epoch] = factor
        report[epoch] = dict(positive_before=positive, negative_before=negative,
            positive_after=float(balanced[positive_mask].sum()), negative_after=float(balanced[negative_mask].sum()),
            ratio_before=positive / negative, ratio_after=float(balanced[positive_mask].sum()) / negative,
            positive_factor=factor)
    return balanced, dict(base_ratio=float(base_ratio), epoch_factors=factors, by_epoch=report,
        negative_weights_unchanged=bool(np.array_equal(weights[targets == 0], balanced[targets == 0])))


def scale_positive_bags(bags, regions, light_epochs, epoch_factors):
    return [(rows, target, weight * (epoch_factors[epoch_for_run(light_epochs, regions[owner].run)]
            if target == 1 else 1.), owner) for rows, target, weight, owner in bags]


def _weight_inventory(targets, weights, balanced, selected, bags, scaled_bags, regions, light_epochs,
                      round1_ids, lcap_ids, weak_strength):
    """Aggregate actual loss weights and active counts without pixel-area bias."""
    report = dict(by_epoch={}, by_source={}, by_run={})
    raw_total, scaled_total = float(weights.sum()), float(balanced.sum())
    raw_bag_total, scaled_bag_total = sum(b[2] for b in bags), sum(b[2] for b in scaled_bags)

    def cells(owner, target):
        region = regions[owner]
        source = t4._source_name(region, round1_ids, lcap_ids)
        for axis, key in (("by_epoch", epoch_for_run(light_epochs, region.run)),
                          ("by_source", source), ("by_run", region.run)):
            group = report[axis].setdefault(key, {})
            yield group.setdefault("positive" if target == 1 else "negative", dict(direct_regions=0,
                sample_rows=0, raw_box_weight=0., direct_weight_before=0., direct_weight_after=0.,
                normalized_direct_loss_before=0., normalized_direct_loss_after=0., weak_bags=0,
                weak_weight_before=0., weak_weight_after=0., weak_loss_before=0., weak_loss_after=0.))

    for target in (0, 1):
        mask = targets == target
        before = np.bincount(selected[mask], weights=weights[mask], minlength=len(regions))
        after = np.bincount(selected[mask], weights=balanced[mask], minlength=len(regions))
        counts = np.bincount(selected[mask], minlength=len(regions))
        for owner in np.flatnonzero(counts):
            for cell in cells(int(owner), target):
                cell["direct_regions"] += 1
                cell["sample_rows"] += int(counts[owner])
                cell["raw_box_weight"] += regions[owner].supervision_weight
                cell["direct_weight_before"] += float(before[owner])
                cell["direct_weight_after"] += float(after[owner])
                cell["normalized_direct_loss_before"] += float(before[owner]) / raw_total
                cell["normalized_direct_loss_after"] += float(after[owner]) / scaled_total
    for (_rows, target, weight, owner), (_rows2, _target2, scaled, _owner2) in zip(bags, scaled_bags, strict=True):
        for cell in cells(owner, target):
            cell["weak_bags"] += 1
            cell["weak_weight_before"] += weight
            cell["weak_weight_after"] += scaled
            cell["weak_loss_before"] += float(weak_strength) * weight / raw_bag_total
            cell["weak_loss_after"] += float(weak_strength) * scaled / scaled_bag_total
    return report


def _weighted_mil(matrix, pure_matrix, pure_labels, pure_weights, bags, regions, *, head,
                  regularization, backend="cpu", weak_strength=.35, positive_top_fraction=.10,
                  max_iter=160, learning_rate=.035, tolerance=2e-5):
    """Use inherited CPU MIL primitives with explicit, already balanced weights."""
    if backend != "cpu":
        raise ValueError("T4b MIL is CPU only")
    logistic.assert_safe_training_regions(regions)
    if (not 0 < float(positive_top_fraction) <= 1 or not math.isfinite(float(weak_strength))
            or float(weak_strength) < 0):
        raise ValueError("Invalid inherited G3 MIL settings")
    selection = logistic._dominance_bag_selection(regions, bags, head=head)
    supervision = logistic._mil_supervision_metadata(regions, bags, selection)
    normalization, normalization_metadata = None, dict(policy="pure-weighted-standardization-v1")
    if bags and weak_strength > 0:
        normalization, normalization_metadata = logistic._mil_normalization(
            pure_matrix, pure_weights, matrix, bags, weak_strength=float(weak_strength))
    initial = logistic.fit_weighted_logistic(pure_matrix, pure_labels, pure_weights,
        regularization=float(regularization), backend="cpu", normalization=normalization, max_iter=40)
    standard_initial = initial
    if not initial.converged:
        initial = logistic.fit_weighted_logistic(pure_matrix, pure_labels, pure_weights,
            regularization=float(regularization), backend="cpu", normalization=normalization, max_iter=120)
    retry = dict(standard_max_iter=40, standard_iterations=int(standard_initial.iterations),
        standard_converged=bool(standard_initial.converged), retry_from_scratch=not standard_initial.converged,
        retry_max_iter=120 if not standard_initial.converged else None,
        final_iterations=int(initial.iterations), final_converged=bool(initial.converged))
    if initial.convergence_certificate is not None:
        retry["convergence_certificate"] = initial.convergence_certificate
    if not bags or weak_strength == 0:
        return initial, dict(mode="pure-region-irls", weak_bags=len(bags), iterations=0,
            converged=bool(initial.converged), pure_irls_retry=retry, normalization=normalization_metadata,
            regional_supervision=supervision)
    fit, diagnostics = logistic._fit_weighted_logistic_mil_cpu(matrix, pure_matrix, pure_labels,
        pure_weights, initial, bags, regularization=float(regularization), weak_strength=float(weak_strength),
        positive_top_fraction=float(positive_top_fraction), max_iter=int(max_iter),
        learning_rate=float(learning_rate), tolerance=float(tolerance), bag_selection=selection)
    diagnostics.update(normalization=normalization_metadata, regional_supervision=supervision, pure_irls_retry=retry)
    return fit, diagnostics


def fit_balanced_mil(features, owners, regions, light_epochs, base_ratio, *, head=HEAD,
                     round1_ids=(), lcap_ids=(), **settings):
    """Shared all-run fitter for pixel or regional inputs, with no rebalancing."""
    matrix = np.asarray(features, np.float32)
    pure, targets, weights, selected = logistic.balanced_pixel_view(matrix, owners, regions, head=head)
    balanced, receipt = balance_positive_weights(targets, weights, selected, regions, light_epochs, base_ratio)
    bags = logistic._weak_region_bags(owners, regions, head=head, runs=None)
    scaled_bags = scale_positive_bags(bags, regions, light_epochs, receipt["epoch_factors"])
    receipt["weights"] = _weight_inventory(targets, weights, balanced, selected, bags, scaled_bags,
        regions, light_epochs, set(round1_ids), set(lcap_ids), settings.get("weak_strength", .35))
    receipt["weak_positive_factors_equal_direct"] = True
    receipt["normalization_scope"] = "all active direct and weak support from all runs together"
    fit, diagnostics = _weighted_mil(matrix, pure, targets, balanced, scaled_bags, regions, head=head, **settings)
    diagnostics["regional_supervision"]["pixel_supervision"] = logistic.pixel_supervision_metadata(
        owners, regions, head=head, runs=None)
    return fit, diagnostics, receipt


def coefficient_report(base_fit, joint_fit):
    def raw(fit):
        values = asdict(fit) if isinstance(fit, logistic.LogisticFit) else fit
        slopes = np.asarray(values["coefficients"], np.float64) / np.asarray(values["scale"], np.float64)
        intercept = float(values["intercept"] - np.dot(np.asarray(values["mean"], np.float64), slopes))
        return dict(standardized_coefficients=safe(values["coefficients"]), raw_coefficients=safe(slopes),
            standardized_intercept=values["intercept"], raw_intercept=intercept)
    old, new = raw(base_fit), raw(joint_fit)
    shift = new["raw_coefficients"][4]
    logit = math.log(.7 / .3) - shift
    threshold = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
    return dict(base=old, joint=new, lamp_standardized_coefficient=new["standardized_coefficients"][4],
        lamp_logit_shift=shift, new_lamps_rig_light_equivalent_pixel_threshold=threshold,
        configured_pixel_threshold=.7, interaction_terms=[])


def validate_pixel_model(model):
    fit = model.get("fit", {})
    if (model.get("variant") != "G3" or model.get("training_revision") != "G3-T4b"
            or tuple(model.get("feature_names", ())) != FEATURE_NAMES or model.get("lamp_shift_only") is not True
            or fit.get("backend") != "cpu" or fit.get("converged") is not True
            or any(np.asarray(fit.get(key, [])).shape != (5,) for key in ("mean", "scale", "coefficients"))
            or not np.isfinite([*fit["mean"], *fit["scale"], *fit["coefficients"], fit["intercept"]]).all()
            or min(fit["scale"]) <= 0):
        raise ValueError("T4b requires one converged five-input CPU lamp-shift head")
    return model


def epoch_pixel_fit(model, run, light_epochs):
    """Fold known e into the intercept for exact inherited four-input scoring."""
    model = validate_pixel_model(read(model) if isinstance(model, (str, Path)) else model)
    if model["training_provenance"]["light_epochs"]["sha256"] != light_epochs["receipt"]["sha256"]:
        raise ValueError("T4b prediction light-epoch table differs from training")
    values = dict(model["fit"])
    indicator = float(epoch_for_run(light_epochs, run) == "new_lamps")
    values["intercept"] += values["coefficients"][4] * (indicator - values["mean"][4]) / values["scale"][4]
    for key in ("mean", "scale", "coefficients"):
        values[key] = np.asarray(values[key][:4], np.float64)
    return logistic.LogisticFit(**values)


def predict_pixel(model, features, run, light_epochs):
    return epoch_pixel_fit(model, run, light_epochs).predict(features)


def load_t4_dataset(dataset_root):
    """Prove exact T4 features, labels, sessions and permitted source inventory."""
    path = checked(Path(dataset_root) / "dataset_receipt.json", T4_DATASET_SHA256)
    receipt = read(path)
    if (receipt.get("contract") != t4.CONTRACT or receipt.get("spec_sha256") != t4.SPEC_SHA256
            or receipt.get("original_g3_regions") != 2849 or receipt.get("original_round1_regions") != 871
            or receipt.get("combined_label_sha256") != COMBINED_LABEL_SHA256
            or receipt.get("old_features_and_owner_indices_bit_identical") is not True
            or receipt.get("t2_bt2_bt3_labels_read") is not False or receipt.get("rounds_2_3_4_read") is not False
            or receipt.get("sealed_scorer_read") is not False
            or receipt.get("lcap", {}).get("annotation_csv_sha256") != LCAP_ANNOTATIONS_SHA256
            or receipt["lcap"].get("box_count") != 156 or receipt["lcap"].get("excluded_aqueous_boxes") != 137):
        raise ValueError("T4b requires the unchanged exact G3 plus156 L-cap gas T4 dataset")
    for source in receipt["training_label_sources"]:
        checked(source["path"], source["sha256"])
        if source.get("session_sha256"):
            checked(Path(source["path"]).parent / "review_session.json", source["session_sha256"])
    for key in ("snapshot_manifest", "annotation_csv", "session"):
        checked(receipt["lcap"][f"{key}_path"], receipt["lcap"][f"{key}_sha256"])
    checked(receipt["g3_extraction_receipt_path"], receipt["g3_extraction_receipt_sha256"])
    for key in ("feature", "owner", "region_inventory"):
        checked(receipt[f"{key}_path"], receipt[f"{key}_sha256"])
    epochs = load_light_epochs(receipt["light_epochs"]["path"])
    if epochs["receipt"] != receipt["light_epochs"]:
        raise ValueError("T4b project light-epoch declaration changed")
    features = np.load(receipt["feature_path"], mmap_mode="r", allow_pickle=False)
    owners = np.load(receipt["owner_path"], mmap_mode="r", allow_pickle=False)
    regions = [logistic.ReviewedRegion(**row) for row in read(receipt["region_inventory_path"])]
    old_rows = receipt["old_sample_rows"]
    if (features.dtype != np.float32 or features.shape[1:] != (4,) or owners.dtype != np.int32
            or owners.shape != features.shape[:1] or not np.isfinite(features).all() or not owners.size
            or owners.min() < 0 or owners.max() >= 3005 or len(regions) != 3005
            or label_digest(regions) != COMBINED_LABEL_SHA256 or label_digest(regions[:2849]) != G3_LABEL_SHA256
            or owners[:old_rows].max() >= 2849 or owners[old_rows:].min() < 2849
            or {r.region_id for r in regions[2849:]} != set(receipt["new_region_ids"])
            or any(r.phase_class != "gas" or r.payload.get("lcap_annotation_sha256") != LCAP_ANNOTATIONS_SHA256
                   for r in regions[2849:])):
        raise ValueError("T4b lost exact G3 scientific prefix or L-cap-only positive suffix")
    logistic.assert_safe_training_regions(regions)
    counts = training_inventory(regions, epochs, owners=owners,
        round1_ids=receipt["original_round1_region_ids"], lcap_ids=receipt["new_region_ids"])
    return dict(features=features, owners=owners, regions=regions, light_epochs=epochs, receipt=receipt,
        receipt_path=path, training_counts=counts, minimum_data_gate=minimum_data_gate(counts))


def _preflight(data, experiment, spec):
    checked(spec, SPEC_SHA256)
    old_head = read(checked(resolve_review_path(experiment) / "variants/G3/gas_head.json", G3_SHA256))
    settings = dict(old_head["training_provenance"]["fit_settings"])
    if settings["backend"] != "cpu" or tuple(old_head["feature_names"]) != FLATFIELD_NAMES:
        raise ValueError("T4b requires the exact original CPU G3 settings")
    old_rows = data["receipt"]["old_sample_rows"]
    ratio = direct_weight_ratio(data["features"][:old_rows], data["owners"][:old_rows], data["regions"][:2849])
    _pure, targets, weights, selected = logistic.balanced_pixel_view(data["features"], data["owners"], data["regions"], head=HEAD)
    balanced, balance = balance_positive_weights(targets, weights, selected, data["regions"], data["light_epochs"], ratio["ratio"])
    bags = logistic._weak_region_bags(data["owners"], data["regions"], head=HEAD, runs=None)
    scaled = scale_positive_bags(bags, data["regions"], data["light_epochs"], balance["epoch_factors"])
    balance["weights"] = _weight_inventory(targets, weights, balanced, selected, bags, scaled, data["regions"],
        data["light_epochs"], set(data["receipt"]["original_round1_region_ids"]),
        set(data["receipt"]["new_region_ids"]), settings["weak_strength"])
    proof = dict(contract=CONTRACT, training_revision="G3-T4b", spec_sha256=SPEC_SHA256,
        dataset_receipt_path=str(data["receipt_path"]), dataset_receipt_sha256=T4_DATASET_SHA256,
        parent_head_sha256=G3_SHA256, combined_label_sha256=COMBINED_LABEL_SHA256,
        original_g3_label_sha256=G3_LABEL_SHA256, light_epochs=data["light_epochs"]["receipt"],
        training_runs=sorted({r.run for r in data["regions"]}), training_counts=data["training_counts"],
        minimum_data_gate=data["minimum_data_gate"], base_direct_weight_ratio=ratio, balancing=balance,
        fit_settings=settings, feature_names=list(FEATURE_NAMES), interaction_terms=[],
        training_label_sources=data["receipt"]["training_label_sources"], lcap=data["receipt"]["lcap"],
        unchanged_co2_head_sha256=CO2_SHA256, cpu_only=True, t2_bt2_bt3_labels_read=False,
        rounds_2_3_4_read=False, sealed_scorer_read=False, fit_started=False)
    return proof, old_head


def preflight(experiment, dataset_root, output, *, spec):
    proof, _head = _preflight(load_t4_dataset(dataset_root), experiment, spec)
    write(Path(output) / "preflight_receipt.json", proof)
    return proof


def fit_pixel_g3_t4b(experiment, dataset_root, output, *, spec, threads=4):
    from threadpoolctl import threadpool_limits
    if not 1 <= int(threads) <= 32:
        raise ValueError("T4b needs a bounded CPU thread allocation")
    data, output = load_t4_dataset(dataset_root), Path(output)
    proof, old_head = _preflight(data, experiment, spec)
    write(output / "preflight_receipt.json", proof)
    if (output / "fit_receipt.json").is_file():
        result = read(output / "fit_receipt.json")
        model = validate_pixel_model(read(checked(result["model_path"], result["model_sha256"])))
        if (result["dataset_receipt_sha256"] != T4_DATASET_SHA256 or not result["converged"]
                or model["training_provenance"]["spec_sha256"] != SPEC_SHA256):
            raise ValueError("Existing T4b fit belongs to another training revision")
        return result
    matrix = joint_features(data["features"], data["owners"], data["regions"], data["light_epochs"])
    cv2.setNumThreads(1)
    started, cpu_started = time.perf_counter(), time.process_time()
    with threadpool_limits(limits=int(threads)):
        fit, diagnostics, balancing = fit_balanced_mil(matrix, data["owners"], data["regions"],
            data["light_epochs"], proof["base_direct_weight_ratio"]["ratio"],
            round1_ids=data["receipt"]["original_round1_region_ids"], lcap_ids=data["receipt"]["new_region_ids"],
            **proof["fit_settings"])
    wall, cpu = time.perf_counter() - started, time.process_time() - cpu_started
    coefficients = coefficient_report(old_head["fit"], fit)
    provenance = dict(**{key: value for key, value in proof.items() if key not in ("fit_started", "balancing")},
        balancing=balancing, threads=int(threads), normalization_scope="joint all-run active MIL support",
        pixel_threshold=.7, regional_threshold=.6, gas_minimum_mm2=2.5, connectivity=4,
        mass_eligible=False, code_sha256=dict(trainer=sha256_file(Path(__file__)),
            logistic=sha256_file(Path(logistic.__file__)), inherited_t4=sha256_file(Path(t4.__file__))))
    model = dict(format=old_head["format"], head=HEAD, variant="G3", training_revision="G3-T4b",
        feature_contract=old_head["feature_contract"], feature_names=list(FEATURE_NAMES), lamp_shift_only=True,
        fit=safe(asdict(fit)), gas_display_threshold=.6, training_provenance=provenance)
    write(output / "gas_head.json", model)
    result = dict(**provenance, converged=bool(fit.converged), fit_wall_seconds=wall, fit_cpu_seconds=cpu,
        old_fit=old_head["fit"], new_fit=safe(asdict(fit)), coefficients=coefficients, diagnostics=diagnostics,
        model_path=str(output / "gas_head.json"), model_sha256=sha256_file(output / "gas_head.json"),
        fit_receipt_path=str(output / "fit_receipt.json"),
        training_region_ids=[r.region_id for r in data["regions"]])
    write(output / "training_report.json", result)
    write(output / "fit_receipt.json", result)
    if not fit.converged:
        raise ValueError("T4b joint gas MIL did not converge; downstream blocked")
    return result
