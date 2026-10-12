"""Read-only C1 movie inventory and optical/physical source preflight.

No phase inference or concentration fit is performed by this preflight. The
movie must bind to the subsequently trained BM model before fields are made.
"""
from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.series_paths import local_cache_root, series_path

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = series_path("ffbm", "results") / "concentration_movies_c1"
BM = configured_tool_path('ffbm', 'dataset_root')
MANIFESTS = BM / "Analysis/Updated/manifests"
ORIGINAL = BM / "Analysis/Original/source/fluidflower_benchmark_image_analysis-1.0/image_analysis"
RAW = BM / "Resfys_copy/Original data"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def preflight(root=ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    all_rows = [r for r in read_csv(MANIFESTS / "jpg_frames.csv") if r["run"] == "c1"]
    anchors = [r for r in read_csv(MANIFESTS / "jpg_tif_anchor_map.csv") if r["run"] == "c1"]
    sources = []
    for r in all_rows:
        h = float(r["elapsed_hours"])
        if r["analysis_eligible"] == "true" and 0 <= h <= 72:
            source = RAW / r["relative_path"]
            if not source.is_file():
                raise FileNotFoundError(source)
            sources.append(dict(index=len(sources), source_path=str(source),
                source_name=r["filename"], source_frame_id=r["frame_id"], elapsed_h=h,
                acquired=r["article_timestamp_naive"], injected_mass_g=float(r["mass_total_g"]),
                original_inventory_path=r["image_path"],
                source_bytes=source.stat().st_size, cadence_seconds=float(r["cadence_seconds"]),
                timestamp_source=r["timestamp_source"],
                mass_status=r["mfc_mass_status"], public_tif_filename=r["public_tif_filename"] or None))
    times = np.array([r["elapsed_h"] for r in sources])
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("C1 acquisitions must be strictly chronological")
    baseline_anchor = anchors[0]
    baseline = RAW / baseline_anchor["jpg_relative_path"]
    baseline_row = next(r for r in all_rows if r["frame_id"] == baseline_anchor["jpg_frame_id"])
    record = dict(format="ffbm-c1-concentration-movie-preflight-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        status="prepared_not_computed", dataset="ffbm", run="c1", language="en", source_frames=sources,
        source_frame_count=len(sources), baseline=dict(source_path=str(baseline),
            elapsed_h=float(baseline_row["elapsed_hours"]),
            matching_studio_tif=baseline_anchor["public_tif_path"]),
        time_range_h=[float(times.min()), float(times.max())],
        source_gb=sum(r["source_bytes"] for r in sources) / 1e9,
        cadence_counts=dict(Counter(str(r["cadence_seconds"]) for r in sources)),
        maximum_gap_seconds=float(np.max(np.diff(times)) * 3600),
        nominal_injected_plateau_g=max(r["injected_mass_g"] for r in sources),
        public_tif_anchors_to_72h=sum(0 <= (datetime.fromisoformat(r["article_timestamp_naive"]) -
            datetime(2021, 11, 24, 8, 31)).total_seconds()/3600 <= 72 for r in anchors),
        input_contract="Dense JPG timestamps from released acquisition schedule, not file modification or EXIF time.",
        concentration_status="BM curve not yet fitted; no AC/UM BTB chemistry will be substituted.",
        source_hashes={str(MANIFESTS/name): sha(MANIFESTS/name) for name in
            ("jpg_frames.csv", "jpg_tif_anchor_map.csv", "jpg_acquisition_schedules.csv", "injection_protocol_segments.csv")})
    cfg_path = ORIGINAL / "phase_segmentation/config.json"
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    depth_path, labels_path = ORIGINAL / "cache/depth.npy", ORIGINAL / "cache/labels_coarse.npy"
    depth, labels = np.load(depth_path), np.load(labels_path)
    record["physical_geometry"] = dict(width_m=config["physical_asset"]["dimensions"]["width"],
        height_m=config["physical_asset"]["dimensions"]["height"], porosity=config["physical_asset"]["parameters"]["porosity"],
        depth_shape=list(depth.shape), labels_shape=list(labels.shape), depth_min_max_m=[float(depth.min()),float(depth.max())],
        depth_source=str(depth_path), depth_sha256=sha(depth_path), labels_source=str(labels_path), labels_sha256=sha(labels_path),
        config_source=str(cfg_path), config_sha256=sha(cfg_path),
        property_policy="Released measured depth with published uniform porosity 0.44; no borrowed UM layer properties.")
    clean_dir = RAW / "FF_BM_cycle_1/C1_images/C1_preparations/211027_c0.1_clean_flush"
    candidates = sorted(clean_dir.glob("*.JPG"))
    record["clearwater_candidates"] = dict(folder=str(clean_dir), count=len(candidates),
        status="Unverified clean-flush images; not selected as calibration input", first=str(candidates[0]), last=str(candidates[-1]))
    write(root / "preflight.json", record)
    # Same-capture TIFF/JPG transfer check at baseline, early plume, late plume.
    selected = [anchors[0]] + [min(anchors,key=lambda r:abs((datetime.fromisoformat(r["article_timestamp_naive"])-
        datetime(2021,11,24,8,31)).total_seconds()/3600-h)) for h in (3,72)]
    figures = []
    comparisons = []
    for row in selected:
        jp, tp = RAW / row["jpg_relative_path"], Path(row["public_tif_path"])
        j, t = rgb(jp), rgb(tp)
        if j.shape != t.shape:
            raise ValueError("Same-capture JPG/TIF geometry differs")
        a, b = j[::8,::8].astype(float), t[::8,::8].astype(float)
        delta = a-b
        comparisons.append(dict(jpg=str(jp), tif=str(tp), jpg_sha256=sha(jp), tif_sha256=sha(tp),
            shape=list(j.shape), rgb_mae=np.mean(abs(delta),axis=(0,1)).tolist(),
            rgb_bias=np.mean(delta,axis=(0,1)).tolist(), max_abs_difference=float(abs(delta).max())))
        figures.extend([(f"Dense JPG: {jp.name}",j),(f"Published TIFF: {tp.name}",t)])
    figures.extend([(f"Unverified clean flush: {p.name}",rgb(p)) for p in (candidates[0],candidates[-1])])
    thumb_w, thumb_h = 900, 510
    sheet = Image.new("RGB",(thumb_w*2,thumb_h*4),(240,242,244))
    draw = ImageDraw.Draw(sheet)
    for i,(title,array) in enumerate(figures):
        tile=Image.fromarray(array);tile.thumbnail((thumb_w,thumb_h-30))
        x,y=(i%2)*thumb_w,(i//2)*thumb_h
        sheet.paste(tile,(x,y+26));draw.text((x+5,y+5),title,fill=(0,0,0))
    sheet.save(root/"source_preflight.jpg",quality=94)
    record["same_capture_checks"] = comparisons
    record["source_inspection_image"] = str(root/"source_preflight.jpg")
    write(root/"preflight.json",record)
    print(json.dumps({k:record[k] for k in ("status","source_frame_count","time_range_h","source_gb","cadence_counts","physical_geometry")},indent=2),flush=True)
    return record


def prepare_optical_controls(root=ROOT):
    """Correct paired formats and the candidate wash under one frozen contract."""
    import io
    import time
    from threadpoolctl import threadpool_limits
    from darsia.fluidflower.tools import prepare_bm_injection_references as bm
    source_manifest = resolve_review_path(local_cache_root() / 'ffbm_studio_pilot_20260908/V7-FF_BM-C1-optical-transfer-20260908/validation/review_assets/native_frames/c1/manifest.json')
    root = Path(root)
    inventory = json.loads((root/"preflight.json").read_text(encoding="utf-8"))
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    output = root/"optical_controls"
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(4)
    context = bm._context(manifest)
    correct, runtime = bm._make_corrector(context,"c1")
    candidates = []
    for index, row in enumerate(inventory["same_capture_checks"]):
        candidates.extend([(f"pair_{index}_jpg",row["jpg"]),(f"pair_{index}_tif",row["tif"])])
    candidates.append(("clearwater_candidate",inventory["clearwater_candidates"]["last"]))
    report = dict(source_manifest=str(source_manifest), source_manifest_sha256=sha(source_manifest),
        correction_runtime=runtime, correction_contract=manifest["correction_contract"], images=[],
        decision="Diagnostic inputs only; wash reference and dense-JPG transfer require evaluation.")
    reduced = {}
    with threadpool_limits(limits=4):
        for name,path in candidates:
            started=time.monotonic()
            target=output/(name+".jpg")
            meta=target.with_suffix(".json")
            if target.is_file() and meta.is_file():
                record=json.loads(meta.read_text(encoding="utf-8"))
                if sha(target)!=record["corrected_sha256"]:
                    raise ValueError("Optical control changed")
                image=rgb(target)
            else:
                data,source_sha,read_path=bm._source_bytes(Path(path))
                try:
                    image,receipt=correct(data,dict(source_path=path),source_sha)
                except (ValueError,RuntimeError) as error:
                    report["images"].append(dict(id=name,source_path=path,source_sha256=source_sha,error=str(error)))
                    write(output/"report.json",report)
                    continue
                # The native Studio BM input representation is Pillow JPEG98, no subsampling.
                Image.fromarray(image).save(target,quality=98,subsampling=0)
                record=dict(id=name,source_path=path,source_sha256=source_sha,
                    corrected_path=str(target),corrected_sha256=sha(target),radiometry=receipt,
                    elapsed_seconds=time.monotonic()-started)
                write(meta,record)
                image=rgb(target)
            report["images"].append(record)
            reduced[name]=cv2.resize(image,(1536,round(image.shape[0]*1536/image.shape[1])),interpolation=cv2.INTER_AREA)
            write(output/"report.json",report)
            print(f"Optical control {name}: {time.monotonic()-started:.1f}s",flush=True)
    report["corrected_same_capture_comparisons"]=[]
    for index in range(3):
        a,b=reduced[f"pair_{index}_jpg"].astype(float),reduced[f"pair_{index}_tif"].astype(float)
        d=a-b
        report["corrected_same_capture_comparisons"].append(dict(pair=index,mean_absolute_rgb=np.mean(abs(d),axis=(0,1)).tolist(),
            mean_bias_rgb=np.mean(d,axis=(0,1)).tolist(),p95_rgb=np.percentile(abs(d),95,axis=(0,1)).tolist()))
    np.savez_compressed(output/"reduced_controls.npz",**reduced)
    write(output/"report.json",report)
    return report


def wash_registration_probe(root=ROOT):
    """Measure full-rig feature alignment without altering the Studio contract."""
    from threadpoolctl import threadpool_limits
    root=Path(root)
    inventory=json.loads((root/"preflight.json").read_text(encoding="utf-8"))
    baseline=rgb(inventory["baseline"]["source_path"])
    clear=rgb(inventory["clearwater_candidates"]["last"])
    width=2000
    scale=width/baseline.shape[1]
    size=(width,round(baseline.shape[0]*scale))
    images=[cv2.resize(a,size,interpolation=cv2.INTER_AREA) for a in (baseline,clear)]
    clahe=cv2.createCLAHE(clipLimit=3.,tileGridSize=(16,12))
    gray=[clahe.apply(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)) for a in images]
    cv2.setNumThreads(4);cv2.setRNGSeed(20260914)
    mask=np.zeros(gray[0].shape,np.uint8);mask[int(.20*size[1]):int(.95*size[1]),int(.04*size[0]):int(.96*size[0])]=255
    with threadpool_limits(limits=4):
        detector=cv2.SIFT_create(nfeatures=20000,contrastThreshold=.01)
        kb,db=detector.detectAndCompute(gray[0],mask)
        kc,dc=detector.detectAndCompute(gray[1],mask)
        matches=cv2.BFMatcher().knnMatch(dc,db,k=2)
        good=[a for a,b in matches if a.distance<.7*b.distance]
        src=np.array([kc[m.queryIdx].pt for m in good],np.float32)
        dst=np.array([kb[m.trainIdx].pt for m in good],np.float32)
        train=np.arange(len(src))%4!=0
        transform,inlier=cv2.estimateAffinePartial2D(src[train],dst[train],method=cv2.RANSAC,
            ransacReprojThreshold=2.,maxIters=5000,confidence=.999)
    if transform is None:
        raise ValueError("No supported wash registration")
    predicted=np.column_stack([src,np.ones(len(src))])@transform.T
    residual=np.linalg.norm(predicted-dst,axis=1)
    held=residual[~train]
    report=dict(status="diagnostic_only",method="SIFT CLAHE, ratio0.7, similarity RANSAC; one quarter matches held out",
        matched_features=len(src),training_matches=int(train.sum()),held_matches=int((~train).sum()),
        training_inliers=int(inlier.sum()),fitted_transform_scaled=transform.tolist(),image_scale=scale,
        held_median_residual_native_pixels=float(np.median(held)/scale),
        held_p90_residual_native_pixels=float(np.percentile(held,90)/scale),
        held_inlier_fraction=float(np.mean(held<=2.)),
        source_clearwater=inventory["clearwater_candidates"]["last"],source_baseline=inventory["baseline"]["source_path"],
        applied_to_movie=False)
    write(root/"wash_registration_probe.json",report)
    aligned=cv2.warpAffine(images[1],transform,size)
    # Independent visual comparison: edges of the brighter wash over baseline.
    edges=cv2.Canny(cv2.cvtColor(aligned,cv2.COLOR_RGB2GRAY),35,90)
    overlay=images[0].copy();overlay[edges>0]=[250,0,250]
    Image.fromarray(overlay).save(root/"wash_registration_edges.jpg",quality=94)
    print(json.dumps(report,indent=2),flush=True)
    return report


def prepare_wash_and_physics(root=ROOT):
    """Prepare measured volume, nominal pressure properties, and aligned wash."""
    import io
    import pandas as pd
    import darsia
    from darsia.fluidflower.image_correction import _reference_colorchecker
    from darsia.fluidflower.phase_training.radiometry import core
    from darsia.fluidflower.tools import prepare_bm_injection_references as bm
    from threadpoolctl import threadpool_limits
    root=Path(root);output=root/"optical_controls"
    inventory=json.loads((root/"preflight.json").read_text(encoding="utf-8"))
    probe=json.loads((root/"wash_registration_probe.json").read_text(encoding="utf-8"))
    if probe["held_inlier_fraction"]<.85 or probe["held_median_residual_native_pixels"]>3.:
        raise ValueError("Whole-rig wash registration failed predeclared support guard")
    manifest_path=resolve_review_path(local_cache_root() / 'ffbm_studio_pilot_20260908/V7-FF_BM-C1-optical-transfer-20260908/validation/review_assets/native_frames/c1/manifest.json')
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    context=bm._context(manifest)
    matrix=np.array(probe["fitted_transform_scaled"]);matrix[:,2]/=probe["image_scale"]
    data,source_sha,_=bm._source_bytes(Path(probe["source_clearwater"]))
    with Image.open(io.BytesIO(data)) as image:clear_raw=np.asarray(image.convert("RGB"))
    cv2.setNumThreads(4)
    with threadpool_limits(limits=4):
        registered=cv2.warpAffine(clear_raw,matrix,(clear_raw.shape[1],clear_raw.shape[0]))
        normalized=registered.astype(np.float32)/255
        cv2.setRNGSeed(0)
        swatches,roi=_reference_colorchecker(normalized,corner="upper_left",roi_rc=np.asarray(context["roi_rc"]))
        corrected,receipt=core().prepare_image(normalized,context["profile"],observed_swatches=swatches,
            series_id="ffbm",run_id="c1",frame_id="movie_wash_reference",source={"path":probe["source_clearwater"],"sha256":source_sha},
            sampling={"roi_rc":roi.tolist(),"basis":"Full-rig supported similarity registration then frozen baseline chart ROI"})
        config=context["config"];config["curvature"]["use_cache"]=False
        curvature=darsia.CurvatureCorrection(config=config["curvature"])
        curvature.cache=np.load(io.BytesIO(context["grid_bytes"]),allow_pickle=True).item()
        corrected=curvature.correct_array(corrected)
        target=output/"clearwater_registered.jpg"
        Image.fromarray(np.rint(np.clip(corrected,0,1)*255).astype(np.uint8)).save(target,quality=98,subsampling=0)
    width=1536;native_shape=tuple(inventory["physical_geometry"]["depth_shape"])
    shape=(round(native_shape[0]*width/native_shape[1]),width);size=shape[::-1]
    depth=np.load(inventory["physical_geometry"]["depth_source"])
    labels=np.load(inventory["physical_geometry"]["labels_source"])
    pixel_area=2.8*1.5/(shape[0]*shape[1])
    reduced_depth=cv2.resize(depth,size,interpolation=cv2.INTER_AREA)
    domain=labels!=0
    volume=cv2.resize(.44*depth*domain,size,interpolation=cv2.INTER_AREA)*pixel_area
    facies=cv2.resize(labels.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)
    clear=cv2.resize(rgb(target),size,interpolation=cv2.INTER_AREA).astype(np.float32)/255
    baseline=cv2.resize(rgb(output/"pair_0_tif.jpg"),size,interpolation=cv2.INTER_AREA).astype(np.float32)/255
    pressure_source=ORIGINAL/"measurements/Florida_2021-11-24_2022-02-02_1669032742.xlsx"
    pressure=pd.read_excel(pressure_source)
    dates=pd.to_datetime(pressure["Dato"].astype(str)+" "+pressure["Tid"].astype(str),format="%Y-%m-%d %H:%M")
    hours=(dates-datetime(2021,11,24,8,31)).dt.total_seconds().to_numpy()/3600
    selected=(hours>=-1)&(hours<=73)
    hours=hours[selected]
    bars=(pressure["Lufttrykk"].to_numpy()[selected]+3.125)*.001
    if len(hours)<3 or np.any(np.diff(hours)<=0) or hours[0]>0 or hours[-1]<72:
        raise ValueError("Published atmospheric pressure does not span the movie")
    np.savez_compressed(root/"physical_references.npz",baseline=baseline,clearwater=clear,facies=facies,
        pore_volume=volume.astype(np.float32),depth_m=reduced_depth.astype(np.float32),
        pressure_elapsed_h=hours,pressure_bar=bars)
    record=dict(status="prepared_candidate_reference",clearwater_path=str(target),clearwater_sha256=sha(target),
        clearwater_original_sha256=source_sha,registration_probe_sha256=sha(root/"wash_registration_probe.json"),
        registration_native_transform=matrix.tolist(),radiometry=receipt,
        analysis_shape_hw=list(shape),analysis_spacing_mm=[1500/shape[0],2800/shape[1]],
        total_porous_volume_l=float(volume.sum()*1000),
        pressure_source=str(pressure_source),pressure_sha256=sha(pressure_source),
        atmospheric_pressure_range_bar=[float(bars.min()),float(bars.max())],
        pressure_policy="Published original BM analysis: Florida pressure plus3.125hPa height correction, chronological interpolation.",
        solubility_g_l=1.8,solubility_source="Original BM mass_analysis.py dissolution_limit=1.8 kg/m3",
        gas_density_policy="Published original BM NIST23C linear density:1.805726990977443*(pbar+0.101325*depth_from_top_m)-0.009218969932330845 kg/m3",
        concentration_calibrated=False)
    write(root/"physical_references.json",record)
    print(json.dumps(record,indent=2),flush=True)
    return record


def prepare_calibration_photographs(root=ROOT):
    import time
    from darsia.fluidflower.tools import prepare_bm_injection_references as bm
    from threadpoolctl import threadpool_limits
    root=Path(root);out=root/"calibration_inputs";out.mkdir(parents=True,exist_ok=True)
    manifest_path=resolve_review_path(local_cache_root() / 'ffbm_studio_pilot_20260908/V7-FF_BM-C1-optical-transfer-20260908/validation/review_assets/native_frames/c1/manifest.json')
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    all_rows=[r for r in read_csv(MANIFESTS/"jpg_frames.csv") if r["run"]=="c1"]
    anchors={r["public_tif_filename"]:r for r in read_csv(MANIFESTS/"jpg_tif_anchor_map.csv") if r["run"]=="c1"}
    by_id={r["frame_id"]:r for r in all_rows}
    rows=[]
    for frame in manifest["frame_sources"]:
        if frame["elapsed_h"]>72:continue
        anchor=anchors[Path(frame["source_path"]).name]
        row=by_id[anchor["jpg_frame_id"]]
        rows.append(dict(id=f"frame_{frame['frame_index']:03d}",source_path=str(RAW/row["relative_path"]),
            source_frame_id=row["frame_id"],elapsed_h=float(row["elapsed_hours"]),acquired=row["article_timestamp_naive"],
            injected_mass_g=float(row["mass_total_g"]),matching_tif=anchor["public_tif_path"],
            matching_studio_frame_index=frame["frame_index"],kind="calibration_candidate"))
    negative=[r for r in all_rows if rows[0]["elapsed_h"]<float(r["elapsed_hours"])<0]
    for n,target_h in enumerate((-.05,-.033333,-.016667)):
        row=min(negative,key=lambda r:abs(float(r["elapsed_hours"])-target_h))
        rows.append(dict(id=f"null_{n:03d}",source_path=str(RAW/row["relative_path"]),source_frame_id=row["frame_id"],
            elapsed_h=float(row["elapsed_hours"]),acquired=row["article_timestamp_naive"],injected_mass_g=0.,kind="independent_preinjection_control"))
    cv2.setNumThreads(4)
    context=bm._context(manifest);correct,runtime=bm._make_corrector(context,"c1")
    result=dict(format="ffbm-c1-calibration-photographs-v1",status="preparing",model_bound=False,
        source_manifest=str(manifest_path),source_manifest_sha256=sha(manifest_path),radiometry_runtime=runtime,
        calibration_split="Defined later by fixed chronological temporal_split, before any concentration fitting.",frames=[])
    with threadpool_limits(limits=4):
        for i,row in enumerate(rows):
            start=time.monotonic();target=out/(row["id"]+".jpg");meta=target.with_suffix(".json")
            if target.is_file() and meta.is_file():
                record=json.loads(meta.read_text(encoding="utf-8"))
                if record["source_frame_id"]!=row["source_frame_id"] or sha(target)!=record["corrected_sha256"]:
                    raise ValueError("Calibration photo cache changed")
            else:
                data,digest,_=bm._source_bytes(Path(row["source_path"]))
                image,receipt=correct(data,row,digest)
                Image.fromarray(image).save(target,quality=98,subsampling=0)
                record={**row,"source_sha256":digest,"corrected_path":str(target),"corrected_sha256":sha(target),"radiometry":receipt}
                write(meta,record)
            result["frames"].append({k:v for k,v in record.items() if k!="radiometry"})
            write(out/"manifest.json",result)
            print(f"Calibration photograph {i+1}/{len(rows)} {row['id']}: {time.monotonic()-start:.1f}s",flush=True)
    result["status"]="photographs_ready_model_pending"
    write(out/"manifest.json",result)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--references",action="store_true")
    p.add_argument("--wash-registration",action="store_true")
    p.add_argument("--physics",action="store_true")
    p.add_argument("--calibration-images",action="store_true")
    args=p.parse_args()
    if args.calibration_images: prepare_calibration_photographs(args.root)
    elif args.physics: prepare_wash_and_physics(args.root)
    elif args.wash_registration: wash_registration_probe(args.root)
    elif args.references: prepare_optical_controls(args.root)
    else: preflight(args.root)
