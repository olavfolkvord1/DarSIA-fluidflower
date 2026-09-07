from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from darsia.fluidflower.phase_features import (
    DARSIA_CHANGE_DETECTOR,
    DARSIA_CHANGE_FEATURE_NAMES,
    NO_CHANGE_DETECTOR,
    OpticalFeatureContract,
    darsia_change_signal,
    optical_feature_samples,
    prepare_optical_feature_workspace,
)
from darsia.fluidflower.phase_learning import (
    HierarchicalPhaseModel,
    PHASE_LEARNING_REVISION_FORMAT,
    PhaseActiveLearningStore,
    PhaseLearningPlanStore,
    inspect_phase_model_bundle,
    phase_uncertainty_map,
    phase_learning_plan_from_bundle,
    propose_uncertain_phase_regions,
    validate_phase_learning_plan,
)
from darsia.fluidflower.project import FluidFlowerProject


def _model_bundle(root: Path) -> Path:
    contract = OpticalFeatureContract()
    model_root = root / "models" / "generic"
    model_root.mkdir(parents=True)
    summary = {
        "campaign_id": "cross-series-test",
        "inventory": {
            "datasets": ["ffac", "ffum"],
            "runs": ["ac52", "run16"],
            "regions": 20,
        },
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for head, intercept in (("co2_presence", 0.2), ("gas_within_co2", -0.4)):
        payload = {
            "schema": 2,
            "algorithm": "test-logistic",
            "feature_names": list(contract.feature_names),
            "feature_mean": [0.0] * len(contract.feature_names),
            "feature_scale": [1.0] * len(contract.feature_names),
            "coefficients": [0.0] * len(contract.feature_names),
            "intercept": intercept,
            "feature_contract": contract.to_dict(),
            "training_provenance": {"cross_validation": {"oof_threshold": 0.5}},
        }
        (model_root / f"{head}_signal.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return root


def test_generic_features_expose_darsia_reference_change() -> None:
    baseline = np.zeros((8, 10, 3), dtype=np.float32)
    current = baseline.copy()
    current[3:5, 4:7, 0] = 0.5
    previous = baseline.copy()
    contract = OpticalFeatureContract()
    workspace = prepare_optical_feature_workspace(
        current, baseline, previous_rgb=previous, contract=contract
    )
    signal = darsia_change_signal(workspace)
    features = optical_feature_samples(
        current,
        baseline,
        np.array([0, 3]),
        np.array([0, 4]),
        previous_rgb=previous,
        contract=contract,
    )

    assert signal["detector"] == DARSIA_CHANGE_DETECTOR
    assert signal["baseline_change_norm"].shape == (8, 10)
    assert float(signal["baseline_change_norm"][3, 4]) == 0.5
    assert features.shape == (2, len(contract.feature_names))
    assert features.dtype == np.float32
    delta_r = contract.feature_names.index("delta_r")
    assert features[0, delta_r] == 0.0
    assert features[1, delta_r] == 0.5


def test_change_provider_can_be_ablated_without_changing_image_contract() -> None:
    baseline = np.zeros((8, 10, 3), dtype=np.float32)
    current = baseline.copy()
    current[3:5, 4:7, 0] = 0.5
    contract = OpticalFeatureContract(change_detector=NO_CHANGE_DETECTOR)
    workspace = prepare_optical_feature_workspace(
        current,
        baseline,
        previous_rgb=baseline,
        contract=contract,
    )
    signal = darsia_change_signal(workspace)
    features = optical_feature_samples(
        current,
        baseline,
        np.array([0, 3]),
        np.array([0, 4]),
        previous_rgb=baseline,
        contract=contract,
    )

    assert signal["detector"] == NO_CHANGE_DETECTOR
    assert not set(DARSIA_CHANGE_FEATURE_NAMES).intersection(contract.feature_names)
    assert features.shape == (2, len(contract.feature_names))
    assert features.dtype == np.float32
    assert np.count_nonzero(signal["baseline_change_norm"]) == 0


def test_cross_series_bundle_and_hierarchical_probabilities(tmp_path: Path) -> None:
    bundle_root = _model_bundle(tmp_path / "bundle")
    summary = inspect_phase_model_bundle(bundle_root)
    model = HierarchicalPhaseModel.load(bundle_root)
    baseline = np.zeros((3, 4, 3), dtype=np.float32)
    predictions = model.predict_image(baseline, baseline)

    assert summary["cross_series"]
    assert summary["datasets"] == ["ffac", "ffum"]
    assert summary["change_detector"] == DARSIA_CHANGE_DETECTOR
    assert predictions["aqueous_co2_probability"].shape == (3, 4)
    total = (
        predictions["untouched_water_probability"]
        + predictions["aqueous_co2_probability"]
        + predictions["free_co2_gas_probability"]
    )
    np.testing.assert_allclose(total, 1.0, atol=1e-6)


def test_phase_learning_plan_freezes_cross_series_evidence(tmp_path: Path) -> None:
    bundle_root = _model_bundle(tmp_path / "bundle")
    annotation = tmp_path / "annotations.json"
    annotation.write_text('{"reviewer":"Reviewer"}', encoding="utf-8")
    plan = phase_learning_plan_from_bundle(bundle_root, annotation_files=[annotation])
    report = validate_phase_learning_plan(plan)
    assert report["valid"]
    assert report["summary"]["dataset_count"] == 2
    assert report["summary"]["cross_series"]
    assert report["summary"]["annotation_file_count"] == 1

    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Phase learning",
        series_id="phase-learning",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )
    revision = PhaseLearningPlanStore(project).save_revision(
        plan,
        actor="Reviewer",
        reason="Imported shared FF_AC/FF_UM model",
        validation=report,
    )
    assert revision["format"] == PHASE_LEARNING_REVISION_FORMAT
    assert revision["scientific_scope"]["cross_series_phase_learning"]
    assert not revision["scientific_scope"]["contains_concentration_or_mass"]


def test_active_learning_prefers_uncertainty_and_respects_domain() -> None:
    shape = (72, 96)
    water = np.full(shape, 0.96, dtype=np.float32)
    aqueous = np.full(shape, 0.02, dtype=np.float32)
    gas = np.full(shape, 0.02, dtype=np.float32)
    water[24:56, 32:72] = 0.34
    aqueous[24:56, 32:72] = 0.33
    gas[24:56, 32:72] = 0.33
    predictions = {
        "untouched_water_probability": water,
        "aqueous_co2_probability": aqueous,
        "free_co2_gas_probability": gas,
    }
    domain = np.zeros(shape, dtype=bool)
    domain[8:68, 8:88] = True
    facies = np.zeros(shape, dtype=np.int16)
    facies[:, 48:] = 2

    uncertainty = phase_uncertainty_map(predictions)
    proposals = propose_uncertain_phase_regions(
        predictions,
        run_id="AC52",
        frame_index=17,
        elapsed_h=9.4,
        model_id="model-v1",
        count=5,
        eligible_mask=domain,
        facies_labels=facies,
        box_size_px=16,
    )

    assert float(np.mean(uncertainty[30:50, 40:64])) > 0.99
    assert proposals
    assert proposals[0]["uncertainty_mean"] > 0.9
    assert all(row["analysis_domain"] == "porous_medium_only" for row in proposals)
    assert all(row["domain_fraction"] >= 0.95 for row in proposals)
    assert len({row["proposal_id"] for row in proposals}) == len(proposals)

    excluded = propose_uncertain_phase_regions(
        predictions,
        run_id="AC52",
        frame_index=17,
        elapsed_h=9.4,
        model_id="model-v1",
        count=1,
        eligible_mask=domain,
        box_size_px=16,
        existing_regions=[proposals[0]],
    )
    assert excluded
    assert excluded[0]["proposal_id"] != proposals[0]["proposal_id"]


def test_active_learning_responses_are_append_only(tmp_path: Path) -> None:
    project = FluidFlowerProject.create(
        tmp_path / "project-active-learning",
        name="Active learning",
        series_id="active-learning",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )
    store = PhaseActiveLearningStore(project)
    proposal = {
        "proposal_id": "phase-active-example",
        "run_id": "ac52",
        "frame_index": 4,
        "x": 0.1,
        "y": 0.2,
        "width": 0.1,
        "height": 0.1,
    }
    batch = store.save_proposal_batch([proposal], actor="Reviewer")
    first = store.record_response(
        proposal,
        phase_class="aqueous_co2_dominant",
        reviewer="Reviewer",
    )
    second = store.record_response(
        proposal,
        phase_class="mixed_free_aqueous_co2",
        reviewer="Reviewer",
        comment="Corrected after temporal inspection",
    )

    assert batch["proposal_count"] == 1
    assert len(store.response_history()) == 2
    assert second["supersedes_response_id"] == first["response_id"]
    assert store.current_responses()[proposal["proposal_id"]]["phase_class"] == (
        "mixed_free_aqueous_co2"
    )
