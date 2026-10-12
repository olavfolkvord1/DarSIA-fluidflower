"""Publish only the complete, QA-passed P1-final saved review pointer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from darsia.fluidflower import phase_ffac_p1_final_review as review
from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.review_storage import resolve_review_path
from .register_track1_studio_candidate import FORMAT, registration_path


def register(root, *, results=None):
    root = resolve_review_path(root).resolve()
    seal = review._read(root / review.SEAL)
    identity = list(_model_identity(root))
    native, maps = {}, {}
    for path in (root / "source_alias/native_frames").glob("*/manifest.json"):
        native[path.parent.name] = sha256_file(path)
    for item in seal.get("frames") or []:
        frame = review._read(review._bound(item["receipt"], base=root))
        key = f"{item['run']}/frame_{item['frame_index']:03d}"
        review._require(frame.get("complete") is True
            and frame.get("floor_fill_removed") is True
            and frame.get("floor_rule_applied") is False
            and frame.get("binding", {}).get("pipeline_sha256") == seal.get("pipeline_sha256"),
            "P1-final registered frame lost its complete floor-removal provenance")
        # All dense arrays were checked by §1 producer QA. These descriptors and
        # their sealed receipts bind the exact displayed files without rehashing
        # gigabytes of maps during each subsequent Studio discovery.
        maps[key] = frame["maps"]["combined"]["sha256"]
    review._require(len(native) == 43 and len(maps) == 864, "P1-final registration needs all 43 / 864")
    value = dict(format=FORMAT, registration_kind=review.KIND, model_root=str(root), model_id=identity[0], model_sha256=identity[1],
        placement="P1-final", delta=review._read(root / review.PLAN)["delta"],
        production_receipt_sha256=sha256_file(root / review.SEAL), source_plan_sha256=sha256_file(root / review.PLAN),
        run_count=43, frame_count=864, native_run_manifest_sha256=native, map_sha256=maps,
        metadata_sha256={name: sha256_file(root / name) for name in review.FILES}, publication_scope="studio_validation_only",
        warning=review.WARNING, qa_status="technical_pass_regression_pending", mass_eligible=False, bottom_mass_allowed=False,
        scientific_acceptance=False, deployment_allowed=False, automatic_activation=False,
        physical_scale_approved=True, blind_hold_runs=[], created_utc=utc_now())
    review._validate(root, value)
    path = registration_path(root.name, results=results)
    if path.exists():
        previous = review._read(path)
        value["created_utc"] = previous["created_utc"]
        review._require(previous == value, "Refusing to overwrite a different P1-final registration")
        _index(root, path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    _index(root, path)
    return path


def _index(root, registration):
    from ..phase_model_library import index_registered_model
    try:
        index_registered_model(root, results=registration.parent.parent)
    except (OSError, ValueError, TypeError) as error:
        import warnings
        warnings.warn(f"Phase library index was not updated: {error}", RuntimeWarning)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args(argv)
    print(register(args.candidate, results=args.results))


if __name__ == "__main__":
    main()
