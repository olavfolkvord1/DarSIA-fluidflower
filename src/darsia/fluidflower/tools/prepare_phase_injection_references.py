"""Prepare inspection-only injection-end photographs for an existing review.

No timeline, training labels, phase predictions, or model parameters are changed.
Run in a separate process: historical correction settings are process-local.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import tomllib

import numpy as np
from PIL import Image

from darsia.fluidflower.tools import phase_injection_references as refs


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


@contextmanager
def um_reference_source(transfer, work, run):
    """Use verified NAS migrations in a private configuration copy only."""
    if getattr(transfer, "_ffum_project_root", lambda: None)() is not None:
        # DarSIA-format project folder (PHASE_FFUM_PROJECT): nothing to remap.
        with transfer._open_ffum(run, require_clearwater=False) as source:
            yield source
        return
    original_root = transfer.FFUM_WORKFLOW_ROOT
    source_config = original_root/"config"/f"{run}.toml"
    content = source_config.read_text(encoding="utf-8-sig")
    config = tomllib.loads(content)
    folder = Path(config["data"]["folder"])
    if not folder.is_dir():
        locations = {
            "run6": "Run6_220912/220927_CO2_injection",
            "run7": "Run7_221002/221007_CO2_injection",
            "run10": "Run10_221129/221207_CO2_injection",
            "run11": "Run11_221214/221221_CO2_injection",
            "run12": "Run12_230103/230107_CO2_injection",
            "run14": "Run14_230123/230131_CO2_injection",
            "run15": "Run15_230205/230210_CO2_injection",
            "run16": "Run16_230216/230222_CO2_injection",
            "run17": "Run17_230303/230308_CO2_injection",
            "run18": "Run18_230412/230420_CO2_injection",
            "run19": "Run19_230424/230428_CO2_injection",
            "run20": "Run20_230504/230512_CO2_injection",
        }
        if run not in locations:
            raise FileNotFoundError(f"No verified source migration for {run}: {folder}")
        resolved = Path("//UNAS-Pro-8/UiB_FluidFlower/FF_UM/Experiments")/locations[run]
        if not (resolved/config["data"]["baseline"]).is_file():
            raise FileNotFoundError(f"Verified UM baseline is missing: {resolved}")
        old = str(config["data"]["folder"])
        content = content.replace("folder = '"+old+"'", "folder = '"+resolved.as_posix()+"'")
        content = content.replace('"./data/', '"'+original_root.as_posix()+'/data/')
        if Path(tomllib.loads(content)["data"]["folder"]) != resolved:
            raise ValueError("Private UM config path replacement failed")
        mirror = work/"resolved_um_config"
        (mirror/"config").mkdir(parents=True, exist_ok=True)
        (mirror/"config"/f"{run}.toml").write_text(content, encoding="utf-8")
        write(mirror/f"{run}_provenance.json", dict(source=str(source_config),
              source_sha256=sha(source_config), original_folder=old,
              resolved_folder=str(resolved), original_config_modified=False))
        transfer.FFUM_WORKFLOW_ROOT = mirror
    try:
        with transfer._open_ffum(run, require_clearwater=False) as source:
            yield source
    finally:
        transfer.FFUM_WORKFLOW_ROOT = original_root


@contextmanager
def matching_source(dataset, run, native, work):
    from darsia.fluidflower.tools import schema2_phase_transfer_campaign as transfer
    from darsia.fluidflower.tools import phase_rig_registration as registration
    from darsia.fluidflower.tools.phase_geometry_contract import require_matching_geometry

    contract = native.get("correction_contract") or {}
    environment = dict(contract.get("environment") or {})
    environment.setdefault("FFAC_COLORCHECKER_ALGORITHM_REVISION", "legacy-v2")
    original_profile = registration.registration_profile_for_run
    # A newer host registration policy must not change an older review's pixels.
    frozen_registration = contract.get("rig_registration") or {}
    registration.registration_profile_for_run = lambda *args, **kwargs: (
        frozen_registration.get("profile") if frozen_registration.get("active") else None)
    try:
        with transfer.optical_control.rebuilt._temporary_environment(environment):
            manager = (transfer._open_ffac(run, require_clearwater=False)
                       if dataset == "ffac" else
                       um_reference_source(transfer, work, run))
            with manager as source:
                binding = require_matching_geometry(native, source.correction_contract,
                                                    context="injection reference")
                effective = source.correction_contract.get("environment") or {}
                if any(str(effective.get(key)) != str(value)
                       for key, value in (contract.get("environment") or {}).items()):
                    raise ValueError("Current optical profile differs from the frozen Studio environment")
                if tuple(source.baseline.shape[:2]) != tuple(native["analysis_image_shape_hw"]):
                    raise ValueError("Reference source and review geometry differ")
                yield source, binding
    finally:
        registration.registration_profile_for_run = original_profile


def _correct(source, row, native, input_cache):
    import darsia
    from darsia.fluidflower.tools.concentration_movie_compute import encode_review_rgb
    # A retained review can predate the verified NAS folder migration. Resolve
    # the same filename AND acquisition time in this experiment's current data.
    name = row.get("source_name") or Path(row["source_path"]).name
    candidates = [Path(path) for path in source.experiment.data if Path(path).name == name
                  and abs(float(source.experiment.time_since_start(
                      source.experiment.get_datetime(path))) - float(row["elapsed_h"])) < 1e-7]
    if len(candidates) != 1:
        raise ValueError(f"Reference source needs one matching acquisition: {name}")
    path, source_receipt = input_cache.stage(candidates[0])
    if row.get("source_sha256") and row["source_sha256"] != source_receipt["source_sha256"]:
        raise ValueError("Reference raw source differs from the frozen review checksum")
    source_receipt = dict(source_receipt, original_review_source_path=row["source_path"])
    image = darsia.imread(path, transformations=source.rig.corrections,
                         reference_date=source.rig.reference_date, name=path.name,
                         dtype=np.float32)
    pixels, encoded = encode_review_rgb(np.asarray(image.img, np.float32)[..., :3],
                                        native, row.get("run", native["run"]), row["elapsed_h"])
    return pixels, encoded, source_receipt


def _parity(source, native, root, run, input_cache):
    from darsia.fluidflower.tools import calibration_pairwise_review as review
    from darsia.fluidflower.tools.concentration_movie_compute import load_rgb
    # Reproduce an independent, existing post-injection source before publishing.
    rows = native.get("frame_sources") or []
    index = next(i for i, row in enumerate(rows) if float(row.get("elapsed_h", -1)) > 0)
    row = rows[index]
    actual, _, receipt = _correct(source, row, native, input_cache)
    path = review._native_frame_path(root, run, index)
    expected = load_rgb(path, native["content_box_xywh"])
    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    result = dict(frame_index=index, source_name=row.get("source_name"),
                  source_sha256=receipt["source_sha256"], reference_frame_sha256=sha(path),
                  exact=np.array_equal(actual, expected),
                  mean_absolute_dn=float(difference.mean()), q99_dn=float(np.quantile(difference, .99)),
                  max_dn=int(difference.max()))
    # Tiny JPEG rounding is allowed; an optical or registration change is not.
    result["passed"] = result["mean_absolute_dn"] <= .35 and result["q99_dn"] <= 2
    if not result["passed"]:
        raise ValueError(f"Existing Studio photograph could not be reproduced: {result}")
    return result


def _offset(reference, current, native):
    from darsia.fluidflower.tools import calibration_pairwise_review as review
    if (native.get("correction_contract") or {}).get("rig_registration", {}).get("active"):
        return dict(x=0., y=0., method="same_registered_grid")
    with Image.open(reference) as image:
        baseline = np.asarray(image.convert("RGB"))
    with Image.open(current) as image:
        target = np.asarray(image.convert("RGB"))
    dx, dy, response, method = review._estimate_review_translation(
        baseline, target, content_box=native["content_box_xywh"])
    return dict(x=dx/target.shape[1], y=dy/target.shape[0], response=response, method=method)


def prepare(review_root, run, dataset, *, work_root=None):
    from darsia.fluidflower.tools import calibration_pairwise_review as review
    from darsia.fluidflower.tools.concentration_movie_input_cache import MovieInputCache
    root, run = Path(review_root).resolve(), str(run).casefold()
    if dataset not in {"ffac", "ffum", "ffbm"} or not re.fullmatch(r"[a-z]+\d+", run):
        raise ValueError("Invalid experiment identity")
    native_path = root/"native_frames"/run/"manifest.json"
    native, native_sha = read(native_path), sha(native_path)
    if str(native.get("run", run)).casefold() != run:
        raise ValueError("Native manifest experiment mismatch")
    reference_dir = root/"native_references"/run
    reference_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(root).casefold().encode()).hexdigest()[:16]
    work = Path(work_root or "C:/FFAC_cache/phase_reference_images")/dataset/run/key
    work.mkdir(parents=True, exist_ok=True)
    pair = dict(dataset=dataset, run=run)
    with review._exclusive_path_lock(reference_dir/"injection_ends.prepare.lock",
                                     timeout_s=900, stale_after_s=7200):
        initial = refs.injection_end_references(pair, root)
        if initial and all(row.get("available") and not row.get("needs_materialization")
                           and not row.get("needs_inventory") for row in initial):
            for row in initial:
                refs.resolve_reference_asset(row, root, run)
            return initial
        if dataset == "ffbm":
            from darsia.fluidflower.tools.prepare_bm_injection_references import materialize_bm
            inventory = refs.inventory_from_bm_filenames(native)
            write(reference_dir/"acquisition_inventory.json", inventory)
            selected = refs.injection_end_references(pair, root)
            completed = materialize_bm(native, selected, reference_dir)
            parity = None
        else:
            with matching_source(dataset, run, native, work) as (source, geometry):
                inventory = refs.inventory_from_experiment(source.experiment, dataset=dataset, run=run)
                write(reference_dir/"acquisition_inventory.json", inventory)
                selected = refs.injection_end_references(pair, root)
                if not selected or any(row.get("elapsed_h") is None for row in selected):
                    raise ValueError("No complete injection-end source selection")
                cache = MovieInputCache(work, run)
                parity = _parity(source, native, root, run, cache)
                completed = []
                for row in selected:
                    if row.get("frame_index") is not None:
                        completed.append(row)
                        continue
                    print(f"Prepare {dataset}/{run}: {row['label']} {row['source_name']}", flush=True)
                    pixels, encoded, receipt = _correct(source, row, native, cache)
                    image_sha = hashlib.sha256(encoded.tobytes()).hexdigest()
                    path = reference_dir/f"injection_end_{image_sha[:20]}.jpg"
                    if path.exists():
                        if sha(path) != image_sha:
                            raise ValueError("Immutable reference cache has changed")
                    else:
                        temporary = path.with_suffix(f".{os.getpid()}.tmp")
                        temporary.write_bytes(encoded.tobytes())
                        temporary.replace(path)
                    offset = _offset(review._native_reference_path(root, run), path, native)
                    completed.append(dict(row, path=str(path), sha256=image_sha,
                        reference_offset=offset, available=True, needs_materialization=False,
                        source_receipt=receipt, native_corrected_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                        correction_contract=source.correction_contract, geometry_binding=geometry,
                        parity=parity, reason=""))
        if sha(native_path) != native_sha:
            raise ValueError("Review changed while preparing references; reopen and retry")
        write(refs.reference_catalog_path(root, run), dict(
            format="injection-end-references-v1", dataset=dataset, run=run,
            created_utc=datetime.now(timezone.utc).isoformat(),
            native_manifest_sha256=native_sha, selection_policy=refs.POLICY,
            timeline_modified=False, model_modified=False, references=completed))
        return refs.injection_end_references(pair, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--dataset", required=True, choices=["ffac", "ffum", "ffbm"])
    args = parser.parse_args()
    result = prepare(args.review_root, args.run, args.dataset)
    print(json.dumps([dict(label=r['label'], source_name=r.get('source_name'),
                          gap_seconds=r.get('gap_seconds'), available=r['available']) for r in result]))


if __name__ == "__main__":
    main()
