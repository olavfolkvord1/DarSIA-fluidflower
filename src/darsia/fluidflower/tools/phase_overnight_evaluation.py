"""Auditable sampled optical comparisons for the extended FF_AC experiment.

This module does not fit models, replay physics, or treat weak region labels as
dense ground truth. Paired holdout claims require both training-run exclusions.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from darsia.fluidflower.phase_learning import SerializedPhaseHead
from darsia.fluidflower.phase_training import trainer as t
from darsia.fluidflower.phase_training.logistic import pixel_target, regional_presence_target
from darsia.fluidflower.phase_training.scope import requested_series, validate_training_series
from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.project import sha256_file, utc_now


HEADS = ("co2_presence", "gas_within_co2")
FORMAT = "phase-overnight-sampled-evaluation-v1"


def _read(path: Path, protected: dict[str, str]) -> dict:
    path = path.resolve()
    digest = sha256_file(path)
    if str(path) in protected and protected[str(path)] != digest:
        raise ValueError(f"Dependency changed while reading: {path}")
    protected[str(path)] = digest
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_inventory_rows(rows, regions):
    if len(rows) != len(regions):
        raise ValueError("Cached region count differs from frozen annotation evidence")
    fields = ("region_id", "dataset", "run", "phase_class", "confidence", "source", "reviewer")
    for index, (row, region) in enumerate(zip(rows, regions)):
        if (int(row["region_index"]) != index
                or any(row[key] != str(getattr(region, key)) for key in fields)
                or int(row["frame_index"]) != region.frame_index
                or float(row["elapsed_h"]) != region.elapsed_h
                or float(row["supervision_weight"]) != region.supervision_weight):
            raise ValueError(f"Cached owner {index} differs from frozen region {region.region_id}")


@dataclass
class FrozenSamples:
    model_root: Path
    samples_root: Path
    summary: dict
    request: dict
    protected_hashes: dict[str, str]
    maximal_features: np.ndarray
    feature_names: list[str]
    owners: np.ndarray
    regions: list
    contracts: dict
    model_identity: tuple[str, str]
    source_by_run: dict = field(default_factory=dict)
    _reference_cache: dict = field(default_factory=dict, repr=False)
    _region_feature_cache: list | None = field(default=None, repr=False)
    _owner_groups: tuple | None = field(default=None, repr=False)

    def verify(self):
        for name, digest in self.protected_hashes.items():
            if sha256_file(Path(name)) != digest:
                raise ValueError(f"Frozen comparison dependency changed: {name}")
        return True

    def matrix(self, adapter="generic"):
        contract = self.contracts[adapter]
        return np.asarray(self.maximal_features[:, [self.feature_names.index(n)
                          for n in contract.feature_names]], dtype=np.float32)

    def owner_groups(self):
        if self._owner_groups is None:
            order = np.argsort(self.owners, kind="stable")
            self._owner_groups = (order, np.searchsorted(self.owners[order], np.arange(len(self.regions) + 1)))
        return self._owner_groups

    def reference_predictions(self, adapter="generic"):
        """Return original fitted reference heads, explicitly not holdout scores."""
        if adapter not in self._reference_cache:
            matrix = self.matrix(adapter)
            scores = {}
            with threadpool_limits(limits=2):
                for head in HEADS:
                    model = SerializedPhaseHead.load(self.model_root / "models" / adapter / f"{head}_signal.json")
                    score = np.empty(len(matrix), dtype=np.float32)
                    for first in range(0, len(matrix), 100_000):
                        last = min(first + 100_000, len(matrix))
                        score[first:last] = model.probability(matrix[first:last])
                    scores[head] = score
            self._reference_cache[adapter] = scores
        return self._reference_cache[adapter]

    def region_feature_summaries(self):
        """Descriptive baseline brightness; this does not identify grains as truth."""
        if self._region_feature_cache is None:
            names = [n for n in ("baseline_intensity", "baseline_signed_local_contrast",
                                 "persistent_bright_contrast", "new_bright_contrast") if n in self.feature_names]
            order, cuts = self.owner_groups()
            columns = {n: self.feature_names.index(n) for n in names}
            self._region_feature_cache = []
            for i in range(len(self.regions)):
                index = order[cuts[i]:cuts[i+1]]
                self._region_feature_cache.append({f"{n}_p90": float(np.quantile(self.maximal_features[index, c], .9))
                                                  for n, c in columns.items()})
        return self._region_feature_cache


def load_frozen_samples(model_root, samples_root=None, *, expected_series="ffac"):
    model_root = Path(model_root).resolve()
    samples_root = model_root if samples_root is None else Path(samples_root).resolve()
    protected = {}
    request = _read(model_root / "training_request.json", protected)
    summary = _read(model_root / "summary.json", protected)
    if summary.get("campaign_id") != request["model_id"]:
        raise ValueError("Reference model differs from its frozen training request")
    series = requested_series(request["settings"])
    if series != expected_series or summary["inventory"]["datasets"] != [series]:
        raise ValueError(f"This comparison requires the explicitly {expected_series.upper()} reference")
    bundles = []
    for bundle in request["bundles"]:
        pair = []
        for role in ("decisions", "combined_manifest"):
            record = bundle[role]
            path = Path(record["snapshot_path"]).resolve()
            _read(path, protected)
            if protected[str(path)] != record["sha256"]:
                raise ValueError(f"Frozen annotation checksum mismatch: {path}")
            pair.append(path)
        bundles.append(t.ReviewBundle(*pair))
    sources, regions, _ = t._load_sources_and_regions(bundles)
    validate_training_series(sources, regions, series)
    source_by_run = {}
    for source in sources:
        native = _read(source.asset_manifest_path, protected)
        source_by_run[source.run] = dict(native_manifest_path=str(source.asset_manifest_path),
                                        frame_sources=native.get("frame_sources", []))
    csv_path = samples_root / "region_inventory.csv"
    protected[str(csv_path)] = sha256_file(csv_path)
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    validate_inventory_rows(rows, regions)
    inventory = _read(samples_root / "training_inventory.json", protected)
    if inventory != summary["inventory"]:
        raise ValueError("Cached training inventory differs from the fitted reference")
    archive_path = samples_root / "training_samples.npz"
    protected[str(archive_path)] = sha256_file(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        features = archive["features"]
        owners = archive["region_indices"]
        names = archive["feature_names"].tolist()
    if (not regions or features.dtype != np.float32 or owners.dtype != np.int32
            or features.shape != (inventory["pixel_rows"], len(names))
            or owners.shape != (len(features),) or len(regions) != inventory["regions"]
            or len(names) != len(set(names)) or not np.all(np.isfinite(features))
            or owners.min() < 0 or owners.max() >= len(regions)
            or np.any(np.bincount(owners, minlength=len(regions)) == 0)):
        raise ValueError("Invalid cached matrix dimensions, values, feature names or region owners")
    contracts = {}
    for adapter, report in summary["models"].items():
        if set(report["heads"]) != set(HEADS):
            raise ValueError("Reference adapter does not contain both fitted heads")
        pair = []
        for head in HEADS:
            path = model_root / "models" / adapter / f"{head}_signal.json"
            payload = _read(path, protected)
            provenance = payload["training_provenance"]
            if (provenance["campaign_id"] != request["model_id"]
                    or provenance.get("series_scope") != series
                    or provenance["training_inventory"] != inventory):
                raise ValueError("Reference head does not match the cached training provenance")
            pair.append(SerializedPhaseHead.load(path).feature_contract)
        if pair[0] != pair[1] or any(n not in names for n in pair[0].feature_names):
            raise ValueError("Reference head feature contracts are incompatible with cached columns")
        contracts[adapter] = pair[0]
    maximal = max(contracts.values(), key=lambda contract: len(contract.feature_names))
    if names != list(maximal.feature_names):
        raise ValueError("Cached column order differs from the exact maximal reference contract")
    data = FrozenSamples(model_root, samples_root, summary, request, protected,
                         features, names, owners, regions, contracts, _model_identity(model_root), source_by_run)
    data.verify()
    return data


def _probabilities(values, count, name, scope):
    values = np.asarray(values)
    if values.shape != (count,):
        raise ValueError(f"{name} must contain one probability slot per cached pixel")
    selected = values[scope]
    if not np.all(np.isfinite(selected)) or np.any((selected < 0) | (selected > 1)):
        raise ValueError(f"{name} must be finite and in [0,1] within the evaluated run")
    # Saved fold predictions can contain NaN outside their held-out run. Those
    # pixels are explicitly excluded; substitute zero solely for grouped sums.
    return np.where(scope, values, 0.)


def _kind(kind, training_runs, held_out_run, name):
    if kind not in {"fitted", "held_out"}:
        raise ValueError(f"Unknown {name} comparison kind: {kind}")
    if kind == "held_out" and (held_out_run is None or training_runs is None or held_out_run in training_runs):
        raise ValueError(f"{name} held-out claim requires explicit exclusion of the evaluated run")


def _aggregate(rows):
    def average(name):
        return float(np.mean([r[name] for r in rows])) if rows else None
    controls = {}
    for key, target in (("negative", 0), ("positive", 1)):
        selected = [r for r in rows if r["pixel_target"] == target]
        controls[key] = {"regions": len(selected), **{
            side + "_correct_fraction": (float(np.mean([r[side + "_coverage"] if target else 1-r[side + "_coverage"]
                                                        for r in selected])) if selected else None)
            for side in ("reference", "candidate")}}
    return dict(regions=len(rows), phase_classes=dict(Counter(r["phase_class"] for r in rows)),
        reference_mean_coverage=average("reference_coverage"), candidate_mean_coverage=average("candidate_coverage"),
        mean_coverage_change=average("coverage_change"),
        previously_present_newly_empty=sum(r["reference_coverage"] > 0 and r["candidate_coverage"] == 0 for r in rows),
        pure_pixel_controls=controls)


def evaluate_predictions(data, candidate_probability, *, head, adapter="generic",
                         reference_probability=None, held_out_run=None,
                         reference_kind="fitted", candidate_kind="fitted",
                         reference_training_runs: Sequence[str] | None = None,
                         candidate_training_runs: Sequence[str] | None = None,
                         top_n=30):
    if head not in HEADS:
        raise ValueError("Select co2_presence or gas_within_co2")
    if held_out_run is not None and held_out_run not in {r.run for r in data.regions}:
        raise ValueError("The requested held-out run does not occur in frozen evidence")
    _kind(candidate_kind, candidate_training_runs, held_out_run, "Candidate")
    _kind(reference_kind, reference_training_runs, held_out_run, "Reference")
    if reference_probability is None:
        if reference_kind != "fitted":
            raise ValueError("A held-out reference requires its own explicit fold predictions")
        reference_probability = data.reference_predictions(adapter)[head]
    selected_regions = np.asarray([held_out_run is None or r.run == held_out_run for r in data.regions], dtype=bool)
    scope = selected_regions[data.owners]
    reference = _probabilities(reference_probability, len(data.owners), "Reference predictions", scope)
    candidate = _probabilities(candidate_probability, len(data.owners), "Candidate predictions", scope)
    threshold = .6 if head == "gas_within_co2" else .5
    counts = np.bincount(data.owners, minlength=len(data.regions))
    coverage = {side: np.bincount(data.owners, weights=values >= threshold, minlength=len(data.regions)) / counts
                for side, values in (("reference", reference), ("candidate", candidate))}
    mean_score = {side: np.bincount(data.owners, weights=values, minlength=len(data.regions)) / counts
                  for side, values in (("reference", reference), ("candidate", candidate))}
    order, cuts = data.owner_groups()
    brightness = data.region_feature_summaries()
    rows = []
    for i, region in enumerate(data.regions):
        if held_out_run is not None and region.run != held_out_run:
            continue
        payload = region.payload
        index = order[cuts[i]:cuts[i+1]]
        native = data.source_by_run.get(region.run, {})
        frames = native.get("frame_sources", [])
        frame = next((r for r in frames if int(r.get("frame_index", -1)) == region.frame_index), {})
        row = dict(region_id=region.region_id, dataset=region.dataset, run=region.run,
            frame_index=region.frame_index, elapsed_h=region.elapsed_h, phase_class=region.phase_class,
            confidence=region.confidence, reviewer=region.reviewer, source=region.source,
            category=payload.get("category"), comment=payload.get("comment", ""),
            candidate_id=payload.get("candidate_id"), params_sha256=payload.get("params_sha256"),
            native_manifest_path=native.get("native_manifest_path"), native_frame_source=frame,
            bounds={key: payload[key] for key in ("coordinate_space", "shape", "points", "x", "y", "width", "height") if key in payload},
            pixel_target=pixel_target(region.phase_class, head),
            regional_target=regional_presence_target(region.phase_class, head),
            gas_regional_target=regional_presence_target(region.phase_class, "gas_within_co2"),
            sampled_pixels=int(counts[i]), reference_coverage=float(coverage["reference"][i]),
            candidate_coverage=float(coverage["candidate"][i]),
            coverage_change=float(coverage["candidate"][i] - coverage["reference"][i]),
            reference_mean_probability=float(mean_score["reference"][i]), candidate_mean_probability=float(mean_score["candidate"][i]),
            reference_probability_p90=float(np.quantile(np.asarray(reference[index], dtype=np.float64), .9)),
            candidate_probability_p90=float(np.quantile(np.asarray(candidate[index], dtype=np.float64), .9)),
            baseline_features=brightness[i])
        rows.append(row)
    by_run = {run: _aggregate([r for r in rows if r["run"] == run]) for run in sorted({r["run"] for r in rows})}
    by_phase = {phase: _aggregate([r for r in rows if r["phase_class"] == phase]) for phase in sorted({r["phase_class"] for r in rows})}
    changed = sorted(rows, key=lambda r: (-abs(r["coverage_change"]), r["region_id"]))
    last_quartile = {run: float(np.quantile([r["elapsed_h"] for r in rows if r["run"] == run], .75)) for run in by_run}
    negatives = [r for r in changed if r["gas_regional_target"] == 0]
    bright = sorted([r for r in negatives if r["baseline_features"].get("baseline_signed_local_contrast_p90", 0.) > 0],
                key=lambda r: (-max(0., r["baseline_features"].get("baseline_signed_local_contrast_p90", 0.))
                * r["baseline_features"].get("baseline_intensity_p90", 0.), -abs(r["coverage_change"]), r["region_id"]))
    selected = dict(gas_positive_controls=[r for r in changed if r["gas_regional_target"] == 1][:top_n],
        gas_negative_controls=negatives[:top_n], negative_bright_baseline_candidates=bright[:top_n],
        late_frame_regions=[r for r in changed if r["elapsed_h"] >= last_quartile[r["run"]]][:top_n])
    return dict(format=FORMAT, created_utc=utc_now(), model_root=str(data.model_root),
        baseline_model_id=data.model_identity[0], baseline_model_sha256=data.model_identity[1],
        head=head, adapter=adapter, threshold=threshold, threshold_scope="conditional gas" if head == "gas_within_co2" else "CO2 presence",
        reference_kind=reference_kind, candidate_kind=candidate_kind, held_out_run=held_out_run,
        reference_training_runs=None if reference_training_runs is None else sorted(reference_training_runs),
        candidate_training_runs=None if candidate_training_runs is None else sorted(candidate_training_runs),
        paired_held_out=reference_kind == candidate_kind == "held_out",
        evaluated_pixels=int(np.count_nonzero(scope)),
        aggregate=_aggregate(rows), by_run=by_run, by_phase_class=by_phase,
        regions=rows, top_changed_regions=changed[:top_n], selected_regions=selected,
        selection_semantics=dict(bright_baseline="Negative gas-control regions ranked by positive baseline local-contrast p90 times baseline intensity p90; candidates, not confirmed grains",
            late_frames="Elapsed time at or above the 75th percentile of annotated regions within each evaluated run"),
        dense_ground_truth=False, physics_replayed=False, independent_final_mask_validation=False,
        caveats=["Coverage averages weight regions equally, not by area; weak labels are not pixel ground truth.",
                 "Pure-region correct fractions describe sampled optical head controls, not final phase masks.",
                 "Head probabilities are evaluated before physics and without multiplying the gas head by CO2 presence.",
                 "Holdout exclusions are explicit caller-supplied training provenance; folds used to select settings remain development validation."])


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(t._json_safe(report), stream, indent=2, allow_nan=False)
    return path
