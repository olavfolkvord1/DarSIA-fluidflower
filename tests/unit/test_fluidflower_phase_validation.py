from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from darsia.fluidflower.phase_features import OpticalFeatureContract
from darsia.fluidflower.phase_validation import (
    materialize_phase_validation_campaign,
    predict_image_tiled,
)
from darsia.fluidflower.phase_learning import HierarchicalPhaseModel


def _write_model(root: Path) -> Path:
    contract = OpticalFeatureContract()
    model_root = root / "models" / "generic"
    model_root.mkdir(parents=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "model_id": "phase-model-test",
                "campaign_id": "phase-model-test",
                "inventory": {"datasets": ["ffac", "ffum"], "runs": ["ac52"]},
            }
        ),
        encoding="utf-8",
    )
    for name, intercept in (("co2_presence", 0.2), ("gas_within_co2", 0.0)):
        payload = {
            "feature_names": list(contract.feature_names),
            "feature_mean": [0.0] * len(contract.feature_names),
            "feature_scale": [1.0] * len(contract.feature_names),
            "coefficients": [0.0] * len(contract.feature_names),
            "intercept": intercept,
            "feature_contract": contract.to_dict(),
            "training_provenance": {"cross_validation": {"oof_threshold": 0.5}},
        }
        (model_root / f"{name}_signal.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return root


def _write_source_campaign(root: Path) -> Path:
    candidate = root / "postflight" / "old" / "ac52"
    candidate.mkdir(parents=True)
    review = root / "manual_pairwise_review"
    native = review / "native_frames" / "ac52"
    reference = review / "native_references" / "ac52"
    native.mkdir(parents=True)
    reference.mkdir(parents=True)
    shape = (72, 96)
    baseline = np.full((*shape, 3), (30, 70, 130), dtype=np.uint8)
    changed = baseline.copy()
    changed[20:55, 25:75] = (120, 100, 40)
    Image.fromarray(baseline).save(reference / "pre_injection.jpg")
    Image.fromarray(baseline).save(native / "frame_000.jpg")
    Image.fromarray(changed).save(native / "frame_001.jpg")
    (native / "manifest.json").write_text(
        json.dumps(
            {
                "frame_count": 2,
                "canvas_wh": [96, 72],
                "analysis_image_shape_hw": [72, 96],
                "content_box_xywh": [0, 0, 96, 72],
                "frame_sources": [
                    {"frame_index": 0, "elapsed_h": 0.0},
                    {"frame_index": 1, "elapsed_h": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    (candidate / "summary.json").write_text(
        json.dumps(
            {
                "calibration_review_movie": {
                    "calibration_frame_count": 2,
                    "params_sha256": "old-digest",
                    "fps": 5.0,
                }
            }
        ),
        encoding="utf-8",
    )
    (candidate / "calibration_mass_fit.png").write_bytes(b"placeholder")
    (candidate / "calibration_review.mp4").write_bytes(b"placeholder")
    with (candidate / "calibration_review_per_frame.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["elapsed_h"])
        writer.writeheader()
        writer.writerows([{"elapsed_h": 0.0}, {"elapsed_h": 1.0}])
    (root / "postflight_manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": "source-campaign",
                "jobs": [
                    {
                        "run": "ac52",
                        "dataset": "ffac",
                        "role": "reviewed_v5_physics",
                        "variant": "old",
                        "candidate_id": "old-candidate",
                        "candidate_folder": str(candidate),
                        "review_output_dir": str(review),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_tiled_prediction_preserves_full_image_shape(tmp_path: Path) -> None:
    model = HierarchicalPhaseModel.load(_write_model(tmp_path / "model"))
    baseline = np.zeros((83, 121, 3), dtype=np.uint8)
    current = baseline.copy()
    current[10:70, 20:90, 0] = 120

    prediction = predict_image_tiled(
        model, current, baseline, previous_rgb=baseline, tile_height=24
    )

    assert prediction["co2_presence_probability"].shape == (83, 121)
    assert prediction["free_co2_gas_probability"].dtype == np.float32


def test_materialized_campaign_binds_maps_and_proposals_to_selected_model(
    tmp_path: Path,
) -> None:
    model = _write_model(tmp_path / "model")
    source = _write_source_campaign(tmp_path / "source")
    output = tmp_path / "validation"

    binding = materialize_phase_validation_campaign(model, source, output)

    assert binding["model_id"] == "phase-model-test"
    assert binding["source_manifest_sha256"]
    assert binding["selected_candidates"]["ac52"].startswith("phase-model-test__")
    manifest = json.loads(
        (output / "postflight_manifest.json").read_text(encoding="utf-8")
    )
    job = manifest["jobs"][0]
    assert job["params_sha256"] == binding["model_sha256"]
    assert Path(job["review_output_dir"]) == source / "manual_pairwise_review"
    phase_root = Path(job["phase_map_output_dir"])
    fingerprint = binding["model_sha256"][:16]
    assert (
        phase_root / "native_phase_maps" / "ac52" / f"A_{fingerprint}" / "frame_001.png"
    ).is_file()
    proposal_path = (
        phase_root
        / "phase_quiz"
        / "ac52"
        / f"A_{fingerprint}_q2"
        / "proposals.json"
    )
    proposals = json.loads(proposal_path.read_text(encoding="utf-8"))
    assert proposals["candidate_id"] == job["candidate_id"]
    assert proposals["proposal_count"] > 0
