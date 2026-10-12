"""Register only a complete, independently QA-passed P1-final3 review candidate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from darsia.fluidflower import phase_ffac_p1_final3_review as review
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
        review._require(frame.get("complete") is True and frame.get("contract") == review.CONTRACT
            and all(isinstance(frame.get(name), dict) for name in ("below_sand_floor_masks",
                "floor_connectivity_masks", "floor_contact_fill_v2", "gas_envelope_masks", "contact_fill_b_reverted_masks"))
            and set(frame.get("removed_display_layers") or {}) == set(review.OVERLAYS)
            and frame.get("binding", {}).get("pipeline_sha256") == seal.get("pipeline_sha256"),
            "P1-final3 registered frame lost its complete reason-b removal provenance")
        maps[key] = frame["maps"]["combined"]["sha256"]
    review._require(len(native) == 43 and len(maps) == 864, "P1-final3 registration needs all 43 / 864")
    value = dict(format=FORMAT, registration_kind=review.KIND, model_root=str(root), model_id=identity[0], model_sha256=identity[1],
        placement="P1-final3", delta=review._read(root / review.PLAN)["delta"],
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
        review._require(previous == value, "Refusing to overwrite a different P1-final3 registration")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    from .register_ffac_p1_final2_studio_candidate import _index
    _index(root, path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args(argv)
    print(register(args.candidate, results=args.results))


if __name__ == "__main__":
    main()
