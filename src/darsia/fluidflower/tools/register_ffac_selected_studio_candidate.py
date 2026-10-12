"""Register an already sealed selected-camera candidate without activating it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from darsia.fluidflower import phase_ffac_common_mass_inputs as assets
from darsia.fluidflower import phase_ffac_selected_review as review
from darsia.fluidflower.phase_validation import _model_identity
from .build_ffac_selected_review import CONTRACT,SEAL,WARNING
from .register_track1_studio_candidate import FORMAT,registration_path

KIND = "ffac_selected_camera_p1_final3_review"
FILES = (SEAL,review.PROFILE_FILE,"summary.json","training_request.json","studio_preparation.json",
    "review_source/postflight_manifest.json","selected_studio/runtime_manifest.json")
_READER = assets.VerifiedAssetReader()
_METADATA = {}


def _descriptor(path):
    path = Path(path)
    stamp = path.stat()
    key = (str(path.resolve()),stamp.st_size,stamp.st_mtime_ns)
    if key not in _METADATA:
        _METADATA[key] = assets.asset(path)
    return _METADATA[key]


def inspect_candidate(root,*,reader=None):
    root = Path(root)
    reader = reader or _READER
    descriptors = {name:_descriptor(root/name) for name in FILES}
    seal = reader.json(descriptors[SEAL])
    profile = review.load_profile(root,reader=reader)
    identity = list(_model_identity(root))
    preparation = reader.json(descriptors["studio_preparation.json"])
    if (seal.get("contract") != CONTRACT or seal.get("complete") is not True
            or seal.get("model_identity") != identity or preparation.get("model_identity") != identity
            or profile.get("model_identity") != identity
            or preparation.get("status") != "ready" or preparation.get("partial") is not False
            or any(seal.get(key) is not False for key in ("maps_changed","scientific_inputs_changed","labels_changed",
                "sessions_changed","bottom_mass_allowed","mass_eligible","automatic_activation"))):
        raise ValueError("An exact completed selected Studio package is required before registration")
    gate = reader.json(seal["accepted_phase_gate"])
    if (seal["run_count"] != len(gate["runs"]) or seal["selected_frame_count"] != gate["selected_frames"]
            or seal["baseline_frame_count"] != gate["baseline_frames"]
            or seal["timeline_frame_count"] != gate["selected_frames"]+gate["baseline_frames"]
            or preparation["frame_count"] != seal["timeline_frame_count"]
            or set(seal["runs"]) != set(gate["runs"])):
        raise ValueError("Registration must retain the honest selected and baseline counts")
    native,maps={},{}
    for run,state in seal["runs"].items():
        path = reader.path(state["native_manifest"])
        native[run] = state["native_manifest"]["sha256"]
        loaded = review.load_native_source(root/"source_alias",run,reader=reader,verify_geometry=False)
        if (path.resolve() != loaded["native_path"].resolve()
                or loaded["native"]["frame_count"] != state["frame_count"]):
            raise ValueError("Selected native run metadata differs from its package")
        for row in loaded["native"]["frame_sources"]:
            maps[f"{run}/frame_{row['frame_index']:04d}"] = dict(source_sha256=row["source_sha256"],
                source_work_index=row["source_work_index"],map_sha256=row["maps"]["combined"]["sha256"])
    if len(maps) != seal["timeline_frame_count"]:
        raise ValueError("Selected Studio registration omits timeline frames")
    return dict(root=root,seal=seal,identity=identity,metadata=descriptors,native=native,maps=maps)


def register(root,*,results=None):
    inspected = inspect_candidate(root)
    root,seal,identity = inspected["root"],inspected["seal"],inspected["identity"]
    record = dict(format=FORMAT,registration_kind=KIND,model_root=str(root),model_id=identity[0],model_sha256=identity[1],
        run_count=seal["run_count"],frame_count=seal["timeline_frame_count"],selected_frame_count=seal["selected_frame_count"],
        baseline_frame_count=seal["baseline_frame_count"],native_run_manifest_sha256=inspected["native"],
        map_source_binding=inspected["maps"],metadata_sha256={name:value["sha256"] for name,value in inspected["metadata"].items()},
        accepted_phase_gate=seal["accepted_phase_gate"],pipeline_sha256=seal["pipeline_sha256"],
        publication_scope="studio_validation_only",warning=WARNING,qa_status="technical_pass_selected_camera_replay",
        mass_eligible=False,bottom_mass_allowed=False,scientific_acceptance=False,deployment_allowed=False,
        automatic_activation=False,physical_scale_approved=True,blind_hold_runs=[],created_utc=datetime.now(timezone.utc).isoformat())
    path = registration_path(identity[0],results=results)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8-sig"))
        record["created_utc"] = old["created_utc"]
        if old != record:
            raise FileExistsError("A different selected candidate registration already exists")
    else:
        assets.write_new_json(path,record)
    from darsia.fluidflower.phase_model_library import index_registered_model
    index_registered_model(root,results=path.parent.parent)
    return path


def registered_warning(model,*,results=None):
    path = registration_path(Path(model).name,results=results)
    if not path.is_file():return None
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    if record.get("registration_kind") != KIND:return None
    try:
        inspected = inspect_candidate(model)
        if (record.get("model_id") != inspected["identity"][0] or record.get("model_sha256") != inspected["identity"][1]
                or record.get("metadata_sha256") != {name:value["sha256"] for name,value in inspected["metadata"].items()}
                or record.get("native_run_manifest_sha256") != inspected["native"]
                or record.get("map_source_binding") != inspected["maps"]):
            raise ValueError("Registered source metadata changed")
    except (OSError,ValueError,KeyError,TypeError):
        return "QA FAILED: selected-camera Studio registration is incomplete or its source metadata changed."
    return record["warning"]


def selected_saved_review(model,*,results=None):
    """Opt-in readiness predicate used by saved Studio maps, never training."""
    inspected=inspect_candidate(model)
    path=registration_path(inspected["identity"][0],results=results)
    record=json.loads(path.read_text(encoding="utf-8-sig"))
    if (record.get("registration_kind")!=KIND or registered_warning(model,results=results)!=WARNING):
        raise ValueError("Selected saved maps have no intact dedicated review registration")
    return dict(registration=record,pipeline_sha256=inspected["seal"]["pipeline_sha256"],
        model_identity=inspected["identity"],placement="Selected camera P1 final 3",
        blind_hold_runs=[],state_label="Selected-camera grid · Studio review")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",required=True)
    parser.add_argument("--results")
    args=parser.parse_args(argv)
    print(register(args.candidate,results=args.results))


if __name__=="__main__":main()
