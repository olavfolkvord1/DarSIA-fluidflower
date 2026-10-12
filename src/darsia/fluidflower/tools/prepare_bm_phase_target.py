"""Prepare a BM experiment's native photographs and context, without a model."""
from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from darsia.fluidflower.tools import phase_injection_references as refs
from darsia.fluidflower.tools import prepare_bm_injection_references as bm

ROOT = configured_tool_path('ffbm', 'dataset_root')
ORIGINAL = ROOT/"Analysis/Original/source/fluidflower_benchmark_image_analysis-1.0/image_analysis"
PROTOCOL = ROOT/"Analysis/Updated/manifests/injection_protocol_segments.csv"
TARGET_HOURS = [-200/3600,0,5/60,10/60,20/60,.5,1,1.5,2,130/60,140/60,
                2.5,3,4,290/60,5,310/60,5.5,6,8,12,24,48,72,96,120]


def elapsed(path, origin):
    return (datetime.strptime(Path(path).name[:17],"%y%m%d_time%H%M%S")-origin).total_seconds()/3600


def protocol_stages(rows, run):
    selected=[row for row in rows if row["run"].casefold()==run]
    if not selected:
        raise ValueError(f"No published injection protocol for {run}")
    origin=min(datetime.fromisoformat(row["start_timestamp"]) for row in selected)
    ports={"port1":("ip1",.9,.3),"port2":("ip2",1.7,.7)}
    stages=[]
    for port in sorted({row["port"] for row in selected}):
        identity,x,y=ports[port]
        segments=[]
        for row in selected:
            if row["port"]!=port:continue
            start=float(row["start_offset_minutes"])/60
            end=float(row["end_offset_minutes"])/60
            if abs((datetime.fromisoformat(row["start_timestamp"])-origin).total_seconds()-start*3600)>1e-5:
                raise ValueError("BM protocol offset and absolute start timestamps disagree")
            if abs((datetime.fromisoformat(row["end_timestamp"])-origin).total_seconds()-end*3600)>1e-5:
                raise ValueError("BM protocol offset and absolute end timestamps disagree")
            segments.append(dict(start_h=start,end_h=end,rate_kg_s=float(row["rate_g_min"])/60000))
        stages.append(dict(id=identity,x_m=x,y_m=y,start_h=min(row["start_h"] for row in segments),
            end_h=max(row["end_h"] for row in segments),rate_segments=segments,
            coordinate_source="Benchmark description, Figure 7, 100 mm grid; nominal positions",
            timing_source=str(PROTOCOL)))
    return origin,stages


def prepare(run: str, output: Path):
    from darsia.fluidflower.phase_training.radiometry import core,detect_chart
    output=Path(output).resolve()
    marker=output/"postflight_manifest.json"
    if marker.exists():
        raise FileExistsError(f"Completed BM target already exists: {output}")
    output.mkdir(parents=True,exist_ok=True)
    snapshot=output/"source_snapshot/image_analysis"
    raw_folder=ROOT/"DarSIA Dataset/raw"/run
    protocol_bytes,protocol_sha,_=bm._source_bytes(PROTOCOL)
    origin,stages=protocol_stages(list(csv.DictReader(io.StringIO(protocol_bytes.decode("utf-8-sig")))),run)
    sources=sorted(raw_folder.glob("*.TIF"),key=lambda path:elapsed(path,origin))
    if not sources or elapsed(sources[0],origin)>=0:
        raise ValueError("BM target requires an original pre-injection TIFF")
    selected={min(sources,key=lambda path:abs(elapsed(path,origin)-time)) for time in TARGET_HOURS}
    for stage in stages:
        selected.add(next(path for path in sources if elapsed(path,origin)>=stage["end_h"]))
    selected=sorted(selected,key=lambda path:elapsed(path,origin))
    if selected[0]!=sources[0]:
        raise ValueError("BM pilot baseline policy requires the first published pre-injection TIFF")
    inputs={}
    for relative in ("phase_segmentation/config.json","phase_segmentation/cache/curvature_transformation.npy","cache/labels_coarse.npy"):
        source=ORIGINAL/relative
        data,digest,read_path=bm._source_bytes(source)
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and bm._digest(target.read_bytes())!=digest:
            raise ValueError(f"BM frozen input changed: {target}")
        if not target.exists():target.write_bytes(data)
        inputs[str(source)]={"sha256":digest,"snapshot":str(target),"read_path":read_path}
    protocol_snapshot=output/"source_snapshot/injection_protocol_segments.csv"
    protocol_snapshot.write_bytes(protocol_bytes)
    config=json.loads((snapshot/"phase_segmentation/config.json").read_text())
    labels=np.load(snapshot/"cache/labels_coarse.npy",allow_pickle=False)
    if not np.issubdtype(labels.dtype,np.integer) or labels.min()<0 or labels.max()>255:
        raise ValueError("BM registered coarse labels are not supported categorical IDs")
    height,width=labels.shape
    width_m=float(config["physical_asset"]["dimensions"]["width"])
    height_m=float(config["physical_asset"]["dimensions"]["height"])
    context_path=output/"target_context.npz"
    np.savez_compressed(context_path,facies=labels.astype(np.uint8),domain=labels!=0,
        x_m=(np.arange(width)+.5)*width_m/width,
        y_m=height_m-(np.arange(height)+.5)*height_m/height)
    baseline_data,baseline_sha,_=bm._source_bytes(selected[0])
    with Image.open(io.BytesIO(baseline_data)) as image:
        baseline_raw=np.asarray(image.convert("RGB"))
    sampling_source="automatic baseline chart detection"
    try:
        swatches,roi,_=detect_chart(baseline_raw.astype(np.float32)/255)
    except ValueError:
        from darsia.fluidflower.image_correction import _reference_colorchecker,_checker_orientation_score
        # Released BM config uses XY corners; current DarSIA takes row/column.
        # Sample the target's own chart at that documented geometry, retaining
        # the same orientation guard as the automatic detector.
        roi=np.fliplr(np.asarray(config["color"]["roi"],dtype=int))
        swatches,roi=_reference_colorchecker(baseline_raw.astype(np.float32)/255,corner="upper_left",roi_rc=roi)
        if _checker_orientation_score(swatches)>.20:
            raise ValueError("Released BM chart ROI fails the existing orientation guard")
        sampling_source="released phase_segmentation/config.json color.roi, XY converted to row/column; automatic detection failed"
    profile=core().make_profile(swatches,target_provenance={"kind":"measured-run-baseline",
        "path":str(selected[0]),"sha256":baseline_sha,"roi_rc":roi.tolist(),
        "absolute_cross_series_standardization":False,
        "sampling_source":sampling_source,
        "baseline_policy":"first published pre-injection TIFF, matching Studio BM pilot policy"})
    review=output/"review_assets"
    frame_dir=review/"native_frames"/run;frame_dir.mkdir(parents=True,exist_ok=True)
    reference_dir=review/"native_references"/run;reference_dir.mkdir(parents=True,exist_ok=True)
    correction_contract=dict(radiometry_profile=profile,chart_before_geometry=True,
        registration_config_sha256=inputs[str(ORIGINAL/"phase_segmentation/config.json")]["sha256"],
        curvature_grid_sha256=inputs[str(ORIGINAL/"phase_segmentation/cache/curvature_transformation.npy")]["sha256"],
        model_input="full-resolution registered 8-bit RGB JPEG quality 98")
    frames=[dict(frame_index=index,source_path=str(path),source_name=path.name,elapsed_h=elapsed(path,origin),
        timestamp=datetime.strptime(path.name[:17],"%y%m%d_time%H%M%S").isoformat()) for index,path in enumerate(selected)]
    frames[0].update(source_sha256=baseline_sha,selection_status="pre_injection_negative_control")
    points=[dict(id=stage["id"],label=f"Injection Point {index+1}",x_m=stage["x_m"],y_m=stage["y_m"],
        x=stage["x_m"]/width_m,y=1-stage["y_m"]/height_m,active_windows_h=[[stage["start_h"],stage["end_h"]]],
        coordinate_source=stage["coordinate_source"]) for index,stage in enumerate(stages)]
    native=dict(schema=1,dataset="ffbm",run=run,frame_count=len(frames),canvas_wh=[width,height],
        content_box_xywh=[0,0,width,height],analysis_image_shape_hw=[height,width],
        kind="native_resolution_corrected_review_frames",full_analysis_resolution=True,
        additional_spatial_resampling=False,jpeg_quality=98,frame_sources=frames,injection_points=points,
        phase_topology={"injection_stages":stages},correction_contract=correction_contract,
        reference=dict(included_in_timeline=True,timeline_frame_index=0,baseline_policy="first published pre-injection TIFF"),
        reference_registration=[dict(frame_index=i,dx_canvas_fraction=0.,dy_canvas_fraction=0.,
            method="frames_already_drift_registered_to_baseline") for i in range(len(frames))],
        timeline_revision=dict(kind="prepend-evaluated-baseline-v1",new_prepared_target=True,
            existing_predictions_recomputed=False,previous_predictions_used=False),
        domain=dict(path=str(context_path),sha256=bm._digest(context_path.read_bytes()),
            source=str(snapshot/"cache/labels_coarse.npy"),source_sha256=inputs[str(ORIGINAL/"cache/labels_coarse.npy")]["sha256"],
            excluded_labels=[0]))
    context=bm._context(native)
    correct,runtime=bm._make_corrector(context,run)
    for index,row in enumerate(frames):
        data,source_sha,read_path=bm._source_bytes(Path(row["source_path"]),row.get("source_sha256"))
        output_path=frame_dir/f"frame_{index:03d}.jpg"
        receipt_path=review/"radiometry"/run/f"frame_{index:03d}.json"
        receipt_path.parent.mkdir(parents=True,exist_ok=True)
        if output_path.exists() and receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if (receipt.get("source_sha256")!=source_sha or receipt.get("output_sha256")!=bm._digest(output_path.read_bytes())
                or receipt.get("correction_contract")!=correction_contract):
                raise ValueError("BM target cached image identity changed")
        else:
            rgb,radiometry=correct(data,row,source_sha)
            Image.fromarray(rgb).save(output_path,quality=98,subsampling=0)
            receipt=dict(radiometry=radiometry,source_path=row["source_path"],source_read_path=read_path,
                source_sha256=source_sha,output_sha256=bm._digest(output_path.read_bytes()),
                correction_contract=correction_contract,runtime=runtime,shape_hw=[height,width])
            bm._atomic_json(receipt_path,receipt)
        row.update(source_sha256=source_sha,native_jpeg_sha256=receipt["output_sha256"],radiometry_receipt=str(receipt_path))
        print(f"{run}: corrected native frame {index+1}/{len(frames)} {row['source_name']}",flush=True)
    shutil.copy2(frame_dir/"frame_000.jpg",reference_dir/"pre_injection.jpg")
    native_path=frame_dir/"manifest.json"
    bm._atomic_json(native_path,native)
    native_sha=bm._digest(native_path.read_bytes())
    descriptor=dict(format="phase-transfer-target-context-v1",dataset="ffbm",run=run,
        native_manifest_sha256=native_sha,path=str(context_path),sha256=bm._digest(context_path.read_bytes()),
        injection_stages=stages,facies_identity="same-series-categorical-labels",
        provenance=dict(target_binding=dict(dataset="ffbm",run=run,native_manifest_sha256=native_sha,source_manifest=str(marker)),
            source_inputs=inputs,protocol=dict(path=str(PROTOCOL),sha256=protocol_sha,snapshot=str(protocol_snapshot)),
            geometry=dict(width_m=width_m,height_m=height_m,indexing="ij",pixel_centers=True,
                physical_dimensions_source="released BM benchmark, same registered shared sand pack",
                categorical_identity="unaltered released labels_coarse.npy; 0 excluded; uint8 lossless encoding"),
            permeability="unverified; categorical boundaries only, no permeability invented",
            correction_contract=correction_contract,baseline_policy="first published pre-injection TIFF",
            phase_model_loaded=False,training_labels_used=False,mass_calibration_used=False))
    candidate=output/"prepared_inputs"/run;candidate.mkdir(parents=True,exist_ok=True)
    postflight=dict(schema=1,dataset="ffbm",kind="label-free-prepared-phase-transfer-target",include_in_training=False,
        selection_id=output.name,created_utc=datetime.now(timezone.utc).isoformat(),
        jobs=[dict(job_id=f"ffbm_{run}_prepared_target",dataset="ffbm",run=run,candidate_folder=str(candidate),
            review_output_dir=str(review),analysis_source_review_root=str(review),independent_reference_frames=[0],
            phase_transfer_context=descriptor)])
    inventory=refs.inventory_from_bm_filenames(native)
    bm._atomic_json(reference_dir/"acquisition_inventory.json",inventory)
    references=refs.injection_end_references(dict(dataset="ffbm",run=run),review)
    bm._atomic_json(reference_dir/"injection_ends.json",dict(format="injection-end-references-v1",dataset="ffbm",run=run,
        native_manifest_sha256=native_sha,selection_policy=refs.POLICY,references=references))
    for original,record in inputs.items():
        if bm._digest(Path(record["snapshot"]).read_bytes())!=record["sha256"]:
            raise ValueError("Frozen BM target input changed")
    report=dict(status="prepared_target_inputs_ready",dataset="ffbm",run=run,manifest=str(marker),
        native_manifest=str(native_path),native_manifest_sha256=native_sha,frame_count=len(frames),
        native_shape=[height,width],phase_inference_run=False,training_run=False,
        baseline_elapsed_h=frames[0]["elapsed_h"],baseline_jpeg_identical_to_frame_zero=True,
        input_snapshot=inputs,actual_end_references=[{k:r.get(k) for k in ["id","source_name","elapsed_h","gap_seconds","frame_index"]} for r in references])
    bm._atomic_json(output/"preparation_report.json",report)
    bm._atomic_json(marker,postflight)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,choices=["c1","c2","c3","c4","c5"])
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    cv2.setNumThreads(2)
    with threadpool_limits(limits=2):
        prepare(args.run,args.output)
