from __future__ import annotations

import json
from pathlib import Path

from darsia.fluidflower.phase_features import OpticalFeatureContract
from darsia.fluidflower.phase_model_history import discover_phase_model_history


def _write_model_head(root: Path, head: str, *, accuracy: float) -> None:
    contract = OpticalFeatureContract()
    payload = {
        "algorithm": "test-logistic",
        "feature_contract": contract.to_dict(),
        "feature_names": list(contract.feature_names),
        "feature_mean": [0.0] * len(contract.feature_names),
        "feature_scale": [1.0] * len(contract.feature_names),
        "coefficients": [0.0] * len(contract.feature_names),
        "intercept": 0.0,
        "training_provenance": {"cross_validation": {"oof_threshold": 0.5}},
    }
    path = root / "models" / "generic" / f"{head}_signal.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_model_history_separates_internal_and_human_validation(tmp_path: Path) -> None:
    model = tmp_path / "campaign" / "phase-model-one"
    model.mkdir(parents=True)
    request = {
        "model_id": "phase-model-one",
        "created_utc": "2026-09-03T21:28:27+00:00",
        "created_by": "Researcher",
        "bundles": [{"bundle_index": 1}],
    }
    (model / "training_request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    summary = {
        "campaign_id": "phase-model-one",
        "inventory": {
            "datasets": ["ffac", "ffum"],
            "runs": ["ac52", "run16"],
            "regions": 25,
            "pixel_rows": 1000,
            "phase_classes": {"water": 5, "gas": 5},
        },
        "models": {
            "generic": {
                "heads": {
                    "co2_presence": {
                        "oof_threshold_balanced_accuracy": 0.98,
                        "oof_threshold": 0.42,
                        "deployment_allowed": True,
                        "final_fit": {
                            "iterations": 160,
                            "converged": True,
                            "backend": "cuda",
                            "precision": "float32",
                        },
                    },
                    "gas_within_co2": {
                        "oof_threshold_balanced_accuracy": 0.84,
                        "oof_threshold": 0.67,
                        "deployment_allowed": True,
                        "final_fit": {
                            "iterations": 160,
                            "converged": False,
                            "backend": "cuda",
                            "precision": "float32",
                        },
                    },
                }
            }
        },
    }
    (model / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _write_model_head(model, "co2_presence", accuracy=0.98)
    _write_model_head(model, "gas_within_co2", accuracy=0.84)

    history = discover_phase_model_history(tmp_path)

    assert len(history) == 1
    entry = history[0]
    assert entry["model_id"] == "phase-model-one"
    assert entry["state"] == "human_validation_required"
    assert entry["internal_holdout_passed"]
    assert entry["co2_balanced_accuracy"] == 0.98
    assert entry["gas_balanced_accuracy"] == 0.84
    assert entry["region_count"] == 25
    assert entry["heads"][1]["iterations"] == 160


def test_model_history_keeps_interrupted_training_visible(tmp_path: Path) -> None:
    model = tmp_path / "phase-model-interrupted"
    model.mkdir()
    (model / "training_request.json").write_text(
        json.dumps(
            {
                "model_id": "phase-model-interrupted",
                "created_utc": "2026-09-03T20:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    history = discover_phase_model_history(tmp_path)

    assert len(history) == 1
    assert history[0]["state"] == "training_incomplete"


def test_model_history_includes_read_only_legacy_siblings(tmp_path: Path) -> None:
    library = tmp_path / "phase_models"
    current = library / "current" / "phase-model-new"
    current.mkdir(parents=True)
    (current / "training_request.json").write_text(
        json.dumps({"model_id": "phase-model-new"}), encoding="utf-8"
    )

    legacy = tmp_path / "crossseries_phase_model_v5"
    legacy.mkdir()
    _write_model_head(legacy, "co2_presence", accuracy=0.98)
    _write_model_head(legacy, "gas_within_co2", accuracy=0.84)
    (legacy / "summary.json").write_text(
        json.dumps(
            {
                "campaign_id": "crossseries_phase_model_v5",
                "models": {
                    "generic": {
                        "heads": {
                            "co2_presence": {},
                            "gas_within_co2": {},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    history = discover_phase_model_history(library)

    assert {row["model_id"] for row in history} == {
        "phase-model-new",
        "crossseries_phase_model_v5",
    }
    historical = next(row for row in history if row["historical_format"])
    assert historical["state"] == "historical"
    assert historical["state_label"] == "Historical model (read-only)"
