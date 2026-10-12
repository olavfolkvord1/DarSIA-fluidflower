"""Prepare BM comparison references using the native review's frozen contract.

This reproduces prepare_ffbm_studio_pilot's TIFF -> drift -> run-baseline
radiometry -> published curvature -> quality-98 JPEG path. It does not predict
phases or fit a new color target. The caller owns the per-run preparation lock.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

from darsia.fluidflower.storage_paths import resolve_storage_path

METHOD = "bm-native-injection-reference-v1"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_digest(value) -> str:
    return _digest(json.dumps(value,sort_keys=True,separators=(",", ":")).encode())


def _source_bytes(path: Path, expected: str | None = None) -> tuple[bytes, str, str]:
    """Read and hash the same immutable bytes that the TIFF decoder will see."""
    aliases = [resolve_storage_path(path)]
    error = None
    for attempt in range(3):
        for candidate in dict.fromkeys(aliases):
            try:
                data = candidate.read_bytes()
                digest = _digest(data)
                if expected and digest != expected:
                    raise ValueError(f"BM source checksum mismatch: {path}")
                return data,digest,str(candidate)
            except OSError as exc:
                error = exc
        if attempt < 2:
            time.sleep(.5*(attempt+1))
    raise OSError(f"BM reference source is unavailable: {path}") from error


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(temporary,path)


def _context(manifest: Mapping) -> dict:
    """Verify frozen registration/profile provenance before any correction."""
    contract = manifest.get("correction_contract") or {}
    if str(manifest.get("dataset", "ffbm")).casefold() != "ffbm":
        raise ValueError("BM preparation requires a BM native review manifest")
    if manifest.get("analysis_source_review_root"):
        raise ValueError("Derived BM radiometry must be prepared from its explicit source lineage")
    if contract.get("chart_before_geometry") is not True:
        raise ValueError("BM review does not declare chart-before-geometry correction")
    profile = contract.get("radiometry_profile")
    if not isinstance(profile,dict) or not profile.get("sha256"):
        raise ValueError("BM review is missing its frozen radiometry profile")
    width,height = [int(value) for value in manifest["canvas_wh"]]
    if list(manifest.get("content_box_xywh",[])) != [0,0,width,height]:
        raise ValueError("BM native canvas requires a full, unresampled geometry contract")
    domain = manifest.get("domain") or {}
    labels_path = Path(domain.get("source", ""))
    if labels_path.name != "labels_coarse.npy" or labels_path.parent.name != "cache":
        raise ValueError("Cannot resolve BM published registration from domain source")
    original = labels_path.parent.parent
    config_path = original/"phase_segmentation"/"config.json"
    grid_path = original/"phase_segmentation"/"cache"/"curvature_transformation.npy"
    for key in ("registration_config_sha256", "curvature_grid_sha256"):
        if not contract.get(key):
            raise ValueError(f"BM correction contract is missing {key}")
    config_bytes,config_sha,_ = _source_bytes(config_path,contract["registration_config_sha256"])
    grid_bytes,grid_sha,_ = _source_bytes(grid_path,contract["curvature_grid_sha256"])
    baseline = next((row for row in manifest.get("frame_sources",[]) if int(row.get("frame_index",-1)) == 0),None)
    if not baseline or not baseline.get("source_path") or not baseline.get("source_sha256"):
        raise ValueError("BM native frame zero must identify its original baseline TIFF and checksum")
    target = profile.get("payload",{}).get("target_provenance",{})
    if target.get("kind") != "measured-run-baseline" or target.get("sha256") != baseline["source_sha256"]:
        raise ValueError("BM radiometry target does not match the reviewed baseline")
    if not target.get("roi_rc"):
        raise ValueError("BM frozen profile is missing its baseline chart sampling ROI")
    baseline_bytes,baseline_sha,baseline_read_path = _source_bytes(Path(baseline["source_path"]),baseline["source_sha256"])
    return dict(profile=profile,config=json.loads(config_bytes),grid_bytes=grid_bytes,
        config_path=str(config_path),grid_path=str(grid_path),config_sha=config_sha,grid_sha=grid_sha,
        baseline=baseline,baseline_bytes=baseline_bytes,baseline_sha=baseline_sha,
        baseline_read_path=baseline_read_path,shape_hw=(height,width),roi_rc=target["roi_rc"])


def _make_corrector(context: dict, run: str):
    import cv2
    import darsia
    import numpy as np
    from PIL import Image
    from darsia.fluidflower.image_correction import _reference_colorchecker
    from darsia.fluidflower.phase_training.radiometry import core, runtime_identity

    profile = core().validate_profile(context["profile"])
    with Image.open(io.BytesIO(context["baseline_bytes"])) as image:
        baseline_raw = np.asarray(image.convert("RGB"))
    config = context["config"]
    config["curvature"]["use_cache"] = False
    curvature = darsia.CurvatureCorrection(config=config["curvature"])
    curvature.cache = np.load(io.BytesIO(context["grid_bytes"]),allow_pickle=True).item()
    if tuple(curvature.cache["shape"]) != context["shape_hw"]:
        raise ValueError("BM published curvature grid differs from native review geometry")
    roi = np.asarray(context["roi_rc"])
    runtime = runtime_identity()

    def correct(data: bytes, source: Mapping, source_sha: str):
        with Image.open(io.BytesIO(data)) as image:
            rgb8 = np.asarray(image.convert("RGB"))
        # Match pilot setup: chart detection established RNG seed zero before
        # the first frame; every frame then sampled its chart with seed zero.
        cv2.setRNGSeed(0)
        drift = darsia.DriftCorrection(baseline_raw,config=config["drift"])
        registered = rgb8 if source_sha == context["baseline_sha"] else drift.correct_array(rgb8)
        rgb = registered.astype(np.float32)/255
        cv2.setRNGSeed(0)
        swatches,sampled_roi = _reference_colorchecker(rgb,corner="upper_left",roi_rc=roi)
        corrected,receipt = core().prepare_image(rgb,profile,observed_swatches=swatches,
            series_id="ffbm",run_id=run,frame_id=f"injection_reference_{Path(source['source_path']).stem}",
            source={"path":source["source_path"],"sha256":source_sha},
            sampling={"roi_rc":sampled_roi.tolist(),"basis":"detected BM baseline chart ROI after drift registration"})
        corrected = curvature.correct_array(corrected)
        if corrected.shape != (*context["shape_hw"],3):
            raise ValueError("Corrected BM reference differs from native canvas geometry")
        return np.rint(np.clip(corrected,0,1)*255).astype(np.uint8),receipt

    return correct,runtime


def materialize_bm(manifest: Mapping, selected_rows: Sequence[Mapping], reference_dir: Path) -> list[dict]:
    """Return prepared UI rows; write only new reference JPEGs and receipts."""
    from PIL import Image
    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True,exist_ok=True)
    context = _context(manifest)
    run = str(manifest["run"]).casefold()
    binding = {"method":METHOD,"preparer_sha256":_digest(Path(__file__).read_bytes()),
        "correction_contract_sha256":_json_digest(manifest["correction_contract"]),
        "registration_config_sha256":context["config_sha"],"curvature_grid_sha256":context["grid_sha"],
        "baseline_source_sha256":context["baseline_sha"],"shape_hw":list(context["shape_hw"]),
        "jpeg_quality":98,"jpeg_subsampling":0}
    correct = runtime = None
    completed = {}
    result = []
    for original in selected_rows:
        row = dict(original)
        if row.get("frame_index") is not None:
            result.append(row)
            continue
        source = Path(row["source_path"])
        data,source_sha,read_path = _source_bytes(source,row.get("source_sha256"))
        if source_sha in completed:
            result.append({**row,**completed[source_sha]})
            continue
        stem = f"injection_end_{source.stem}_{source_sha[:16]}"
        output = reference_dir/(stem+".jpg")
        receipt_path = reference_dir/(stem+".json")
        receipt = None
        if output.is_file() and receipt_path.is_file():
            saved = json.loads(receipt_path.read_text(encoding="utf-8"))
            if saved.get("binding") == binding and saved.get("source_sha256") == source_sha:
                if _digest(output.read_bytes()) != saved.get("output_sha256"):
                    raise ValueError(f"BM prepared reference checksum mismatch: {output}")
                receipt = saved
        if receipt is None:
            if correct is None:
                correct,runtime = _make_corrector(context,run)
            rgb,radiometry = correct(data,row,source_sha)
            if tuple(rgb.shape) != (*context["shape_hw"],3):
                raise ValueError("BM reference output shape differs from its frozen canvas")
            temporary = output.with_name(output.name+f".{os.getpid()}.tmp")
            Image.fromarray(rgb).save(temporary,format="JPEG",quality=98,subsampling=0)
            output_sha = _digest(temporary.read_bytes())
            os.replace(temporary,output)
            receipt = {"format":"bm-injection-reference-receipt-v1","dataset":"ffbm","run":run,
                "source_path":str(source),"source_read_path":read_path,"source_sha256":source_sha,
                "source_name":source.name,"elapsed_h":row["elapsed_h"],"output_sha256":output_sha,
                "binding":binding,"radiometry":radiometry,"runtime":runtime,
                "baseline_source_path":context["baseline"]["source_path"],
                "baseline_read_path":context["baseline_read_path"],
                "registration_config_path":context["config_path"],"curvature_grid_path":context["grid_path"],
                "reference_offset":{"x":0.0,"y":0.0},"registration":"published drift to original pre-injection baseline"}
            _atomic_json(receipt_path,receipt)
        asset = {"path":str(output),"sha256":receipt["output_sha256"],"source_sha256":source_sha,
            "reference_offset":{"x":0.0,"y":0.0},"receipt_path":str(receipt_path),
            "receipt_sha256":_digest(receipt_path.read_bytes()),"available":True,
            "needs_materialization":False,"reason":""}
        completed[source_sha] = asset
        result.append({**row,**asset})
    return result
