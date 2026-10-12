"""Prepare source-bound FF_AC selected-image manifests with DarSIA selectors.

Only selection, RAW light screening and append-only project declarations are
performed. No phase/mass inference, film or transport solver is launched.
All project/source/output locations are supplied by the caller.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import tomllib

import numpy as np

from darsia.fluidflower.frame_quality import (
    classify_chart_pillar_light, measure_raw_chart_pillar, raw_chart_pillar_reference,
)
from darsia.fluidflower.image_selection_plan import resolve_quality_time_grid
from darsia.fluidflower.mass_analysis import synchronized_image_selection_configuration
from darsia.fluidflower.phase_ffac_camera_timeline import camera_elapsed_hours, read_capture_time

INJECTION_HOURS = [i/8 for i in range(1,20)]
REST_HOURS = [2.5,2.75,3.,3.5,4.,5.,6.,7.,8.,10.,12.,14.,16.,20.,24.,28.,32.,36.,40.,44.,48.]
EXTENSION_HOURS = [56.,64.,72.,96.]
TARGET_HOURS = INJECTION_HOURS + REST_HOURS + EXTENSION_HOURS
CONTRACT = "ffac-selected-camera-images-v1"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def descriptor(path):
    p=Path(path)
    return dict(path=str(p),sha256=digest(p),size_bytes=p.stat().st_size)


def read_bound(binding):
    p=Path(binding["path"])
    if digest(p) != binding["sha256"]:
        raise ValueError(f"Bound source changed: {p}")
    return json.loads(p.read_text(encoding="utf-8-sig"))


def new_json(path, value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write("\n")
    return descriptor(p)


def copy_exact(source, target):
    source,target=Path(source),Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(target)!=digest(source):
            raise FileExistsError(target)
    else:
        with source.open("rb") as src,target.open("xb") as dst:
            shutil.copyfileobj(src,dst,4*1024*1024)
    return descriptor(target)


def append_toml_sections(path, text, *, backup_folder, marker):
    """Append only, prove old TOML values survive, retain exact original bytes."""
    path=Path(path);before=path.read_bytes();old=tomllib.loads(before.decode("utf-8-sig"))
    if marker.encode() in before:
        raise ValueError(f"Named selection already exists; no rewrite allowed: {path}")
    after=before+b"\n"+text.encode("utf-8")
    merged=tomllib.loads(after.decode("utf-8-sig"))
    for key,value in old.items():
        current=merged[key]
        if isinstance(value,list) and isinstance(current,list):
            if current[:len(value)]!=value:
                raise ValueError(f"Existing TOML array changed: {key}")
        elif current!=value:
            raise ValueError(f"Existing TOML section changed: {key}")
    backup=Path(backup_folder)/path.name
    backup_binding=copy_exact(path,backup)
    if path.read_bytes()!=before:
        raise ValueError("Project TOML changed during preparation")
    with path.open("ab") as stream:
        stream.write(after[len(before):]);stream.flush();os.fsync(stream.fileno())
    if path.read_bytes()!=after:
        raise ValueError("Append-only TOML write failed")
    return dict(path=str(path),before_sha256=hashlib.sha256(before).hexdigest(),
        after_sha256=digest(path),backup=backup_binding,existing_sections_unchanged=True,
        bytes_only_appended=True)


def _raw_pair(source):
    p=Path(source)
    for suffix in (".ARW",".arw"):
        raw=p.with_suffix(suffix)
        if raw.is_file():return raw
    raise FileNotFoundError(f"Missing paired RAW: {p}")


def _prepare_run(task):
    start=time.monotonic();run=task["run"];output=Path(task["output"])/"runs"/run
    output.mkdir(parents=True,exist_ok=True)
    camera=read_bound(task["camera_run"])
    reference_state=read_bound(task["frame_list"])["reference_state"][run]
    cache_meta=read_bound(task["cache_metadata"])
    if digest(task["cache_arrays"]["path"])!=task["cache_arrays"]["sha256"]:
        raise ValueError("Frozen RAW reference arrays changed")
    with np.load(task["cache_arrays"]["path"],allow_pickle=False) as archive:
        reference=raw_chart_pillar_reference(cache_meta,{k:archive[k] for k in archive.files})
    # Original rule compares clipping to pipeline frame0, even if the light
    # reference was the extra-light-off floor. Keep that exact convention.
    old_rows=read_bound(task["frame_list"])["frames"]
    old0=next(row for row in old_rows if row["run"]==run and row["frame"]==0)
    clipping_reference=int(old0.get("clipped",0))
    measurements={}
    frames=[]
    for capture in camera["camera_inventory"]:
        elapsed=camera_elapsed_hours(capture["camera_datetime"],camera["anchor"])
        if elapsed <= 0:continue
        frames.append(dict(capture,elapsed_h=elapsed,frame_index=capture["source_index"]))

    def quality(frame):
        key=str(Path(frame["source_path"]).with_suffix(".ARW"))
        if key in measurements:return measurements[key]
        source=Path(frame["source_path"]);raw=_raw_pair(source)
        raw_bytes=raw.read_bytes();raw_sha=hashlib.sha256(raw_bytes).hexdigest()
        try:
            measurement=measure_raw_chart_pillar(raw_bytes,reference)
            dc=None if measurement["c"] is None else measurement["c"]-reference_state["c"]
            dp=measurement["p"]-reference_state["p"]
            cls=classify_chart_pillar_light(np.nan if dc is None else dc,dp,
                clipped_delta=measurement["clipped"]-clipping_reference)
            value=dict(measurement,dc=dc,dp=dp,dd=None if dc is None else dc-dp,
                       **{"class":cls})
        except (ValueError,RuntimeError) as error:
            value=dict(**{"class":"reject"},error=str(error))
        value.update(raw_path=str(raw),raw_sha256=raw_sha,
            raw_size_bytes=len(raw_bytes),raw_mtime_ns=raw.stat().st_mtime_ns,
            frozen_reference=task["cache_arrays"],reference_state=reference_state,
            rule=task["quality_spec"])
        measurements[key]=value
        return value

    selection=resolve_quality_time_grid(frames,TARGET_HOURS,quality=quality)
    studio=read_bound(task["studio_manifest"])
    studio_by_path={str(Path(f["source_path"])).casefold():f for f in studio["frame_sources"]}
    for row in selection["selected"]:
        source=Path(row["source_path"]);s=source.stat()
        # The acquisition clock is re-used only for the exact immutable file.
        stat_changed=s.st_size!=row["source_bytes"] or s.st_mtime_ns!=row["source_mtime_ns"]
        if stat_changed:
            capture=read_capture_time(source)
            if capture["camera_datetime"]!=row["camera_datetime"]:
                raise ValueError(f"4-Oct acquisition camera timestamp changed: {source}")
            row["source_stat_drift_audit"]=dict(old_size_bytes=row["source_bytes"],
                old_mtime_ns=row["source_mtime_ns"],current_size_bytes=s.st_size,
                current_mtime_ns=s.st_mtime_ns,camera_datetime_reverified=True)
        row["source_sha256"]=digest(source)
        row["raw_path"]=row["quality"]["raw_path"]
        row["raw_sha256"]=row["quality"]["raw_sha256"]
        row["v5_raw_path"]=row["raw_path"]
        row["v5_raw_sha256"]=row["raw_sha256"]
        previous=studio_by_path.get(str(source).casefold())
        if previous is not None and previous["source_sha256"]!=row["source_sha256"]:
            raise ValueError("Selected JPEG differs from locked Studio source")
        row["studio_frame_index"]=None if previous is None else previous["frame_index"]
        row["studio_source_sha256"]=None if previous is None else previous["source_sha256"]
        row["source_alias_status"]="existing_locked_v5_raw" if previous is not None else "new_v5_raw_preparation_required"
        if previous is not None:
            row["existing_source_alias"]={k:v for k,v in previous.items() if k.startswith("v5_")}
            # A locked composite/member representation remains explicit;
            # photographic acquisition identity is not substituted for it.
            for key in ("v5_raw_path","v5_raw_sha256","v5_raw_members","v5_historical_composite_reproduction"):
                if key in previous:row[key]=previous[key]
    by_index={row["selection_index"]:row for row in selection["selected"]}
    for match in selection["target_matches"]:
        if not match["selected"]:continue
        frame=by_index[match["selection_index"]]
        for key in ("source_sha256","raw_path","raw_sha256","v5_raw_path","v5_raw_sha256",
                    "studio_frame_index","studio_source_sha256","source_alias_status"):
            match[key]=frame[key]
        match["camera_datetime"]=frame["camera_datetime"]
    baseline=studio["frame_sources"][0]
    result=dict(contract=CONTRACT,run=run,grid_version="comparison_grid_v1",
        target_hours=TARGET_HOURS,camera_run=task["camera_run"],camera_origin=camera["anchor"],
        studio_manifest=task["studio_manifest"],parent_native_manifest=task["parent_native_manifest"],baseline=baseline,
        clearwater=read_bound(task["clearwater_result"]),quality_rule=task["quality_spec"],
        source_representation="v5 scientific linear float32 RAW; JPEG identifies acquisition only",
        frame_sources=selection["selected"],target_matches=selection["target_matches"],
        missing_target_count=selection["missing_target_count"],
        selected_frame_count=selection["selected_frame_count"],
        quality_assessment_count=selection["quality_assessment_count"],
        model_inference=False,mass_computation=False,wall_seconds=time.monotonic()-start)
    measurement_binding=new_json(output/"raw_light_measurements.json",dict(run=run,
        measurements=list(measurements.values()),scientific_model_input_changed=False,
        classification_source_only=True))
    result["measurements"]=measurement_binding
    binding=new_json(output/"manifest.json",result)
    print(json.dumps(dict(run=run,selected=result["selected_frame_count"],
        missing=result["missing_target_count"],seconds=result["wall_seconds"])),flush=True)
    return run,binding


def prepare(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if (output/"selection.json").exists():raise FileExistsError(output/"selection.json")
    if args.execute_preparation:
        preparation=json.loads((output/"preparation.json").read_text(encoding="utf-8"))
        if digest(args.spec)!=preparation["spec"]["sha256"]:
            raise ValueError("Selection specification changed after preparation")
        return execute_preparation(preparation,args)
    source_plan=json.loads(Path(args.source_plan).read_text(encoding="utf-8-sig"))
    source_metadata=json.loads(Path(args.source_metadata).read_text(encoding="utf-8-sig"))
    runs=source_metadata["split"]["production_runs"]
    if len(runs)!=39 or set(runs)&{"ac20","ac24","ac27","ac51"}:
        raise ValueError("The selected comparison cohort must be the exact39 approved runs")
    stage=output/"inputs";quality_root=Path(args.quality_reference)
    quality_spec=copy_exact(quality_root/"frame_list_20260928/spec.md",stage/"quality/spec.md")
    frame_list=copy_exact(quality_root/"frame_list_20260928/frame_list.json",stage/"quality/frame_list.json")
    for filename in ("classify.py","measure_frames.py"):
        copy_exact(quality_root/"frame_list_20260928"/filename,stage/"quality/reference_only"/filename)
    tasks=[]
    source_runs={row["run"]:row for row in source_plan["source_runs"]}
    candidate=Path(source_metadata["candidate_root"])
    for run in runs:
        source=source_runs[run]["camera_run"];read_bound(source)
        camera=copy_exact(source["path"],stage/"camera_runs"/f"{run}.json")
        cache_meta=copy_exact(quality_root/"cache"/f"{run}.json",stage/"quality/cache"/f"{run}.json")
        cache_arrays=copy_exact(quality_root/"cache"/f"{run}.npz",stage/"quality/cache"/f"{run}.npz")
        tasks.append(dict(run=run,output=str(output),camera_run=camera,cache_metadata=cache_meta,
            cache_arrays=cache_arrays,quality_spec=quality_spec,frame_list=frame_list,
            studio_manifest=descriptor(candidate/"source_alias/native_frames"/run/"manifest.json"),
            parent_native_manifest=source_runs[run]["parent_native_manifest"],
            clearwater_result=descriptor(Path(args.clearwater_results)/f"{run}.json")))
    preparation=dict(contract=CONTRACT,tasks=tasks,spec=descriptor(args.spec),
        source_plan=descriptor(args.source_plan),source_metadata=descriptor(args.source_metadata),
        quality_spec=quality_spec,target_hours=TARGET_HOURS,runs=runs,
        created_utc=datetime.now(timezone.utc).isoformat(),only_sections=[1],
        current_code={str(Path(__file__)):digest(__file__)})
    new_json(output/"preparation.json",preparation)
    if args.prepare_only:return
    return execute_preparation(preparation,args)


def execute_preparation(preparation,args):
    output=Path(args.output);tasks=preparation["tasks"];runs=preparation["runs"]
    if (output/"selection.json").exists():raise FileExistsError(output/"selection.json")
    new_json(output/"execution_binding.json",dict(preparation=descriptor(output/"preparation.json"),
        spec=preparation["spec"],workers=args.workers,internal_threads=1,
        code={"orchestrator":descriptor(__file__),
              "selector":descriptor(Path(__file__).parents[1]/"image_selection_plan.py"),
              "frame_quality":descriptor(Path(__file__).parents[1]/"frame_quality.py"),
              "mass_analysis":descriptor(Path(__file__).parents[1]/"mass_analysis.py")}))
    manifests={}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(_prepare_run,task) for task in tasks]):
            run,binding=future.result();manifests[run]=binding
    table=[];all_results={run:read_bound(binding) for run,binding in manifests.items()}
    for target in TARGET_HOURS:
        rows=[next(row for row in all_results[run]["target_matches"] if row["target_hour"]==target) for run in runs]
        table.append(dict(target_hour=target,selected_run_count=sum(r["selected"] for r in rows),
            missing_runs=[run for run,r in zip(runs,rows) if not r["selected"]],
            replacement_count=sum(r["replacement"] is not None for r in rows),
            minor_count=sum(r.get("frame_quality_class")=="minor" for r in rows)))
    selection=dict(contract=CONTRACT,complete=True,runs=runs,manifests=manifests,
        target_hours=TARGET_HOURS,coverage=table,
        total_selected_images=sum(d["selected_frame_count"] for d in all_results.values()),
        total_matches=sum(row["selected_run_count"] for row in table),
        phase_inference=False,mass_computation=False,defaults_changed=False)
    selection_binding=new_json(output/"selection.json",selection)
    config=synchronized_image_selection_configuration(TARGET_HOURS,manifest_path=selection_binding["path"],
        manifest_sha256=selection_binding["sha256"])
    new_json(output/"mass_analysis_selection_plan.json",dict(image_selection=config,
        calibration_waits_for_round3_and_Olav_approval=True,production_authorized=False))
    lines=["[image_selection]",'policy = "synchronized_representative_times"',
        "target_hours = "+json.dumps(TARGET_HOURS),"chronological = true",
        "require_same_target_times = true","maximum_images_per_run = 44",
        "drift_tolerance_seconds = 150.0",'manifest = '+json.dumps(selection_binding["path"]),
        'manifest_sha256 = '+json.dumps(selection_binding["sha256"]),"production_authorized = false"]
    with (output/"mass_analysis_plan.toml").open("x",encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")
    if args.apply_project_config:
        records=apply_project_declarations(args.series_root,output,selection_binding,all_results)
        new_json(output/"project_config_changes.json",dict(files=records,append_only=True,
            existing_sections_unchanged=True,old_results_unchanged=True))
    print(json.dumps(dict(complete=True,selection=selection_binding,coverage=table)),flush=True)


def apply_project_declarations(series_root,output,selection_binding,results):
    root=Path(series_root);output=Path(output);records=[]
    def hms(value):
        seconds=round(value*3600);return f"{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}"
    marker="# comparison_grid_v1 source-bound selected image grid"
    text=marker+"\n"
    for name,times,tol in (("injection",INJECTION_HOURS,"00:00:15"),
        ("rest",REST_HOURS+EXTENSION_HOURS,"00:02:30")):
        text+='\n[[data_time]]\nname = "comparison_grid_v1_'+name+'"\ntimes = '+json.dumps([hms(t) for t in times])+'\ntol = '+json.dumps(tol)+'\n'
    records.append(append_toml_sections(root/"config/common.toml",text,
        backup_folder=output/"config_backups/common",marker=marker))
    for run,result in sorted(results.items()):
        binding=descriptor(output/"runs"/run/"manifest.json")
        native=result["studio_manifest"]
        text=marker+'\n\n[comparison_grid_v1]\nselection_manifest = '+json.dumps(binding["path"])+\
            '\nselection_manifest_sha256 = '+json.dumps(binding["sha256"])+\
            '\n\n[comparison_grid_v1.source_alias]\nrepresentation = "v5_linear_float32_raw"\n'+\
            'source_manifest = '+json.dumps(native["path"])+\
            '\nsource_manifest_sha256 = '+json.dumps(native["sha256"])+\
            '\nnew_sources = "selection_manifest.frame_sources.v5_raw_path"\n'
        records.append(append_toml_sections(root/"config/runs"/f"{run}.toml",text,
            backup_folder=output/"config_backups/runs",marker=marker))
    return records


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("series-root","output","spec","source-plan","source-metadata","quality-reference","clearwater-results"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--execute-preparation",action="store_true")
    parser.add_argument("--apply-project-config",action="store_true")
    args=parser.parse_args(argv)
    if args.workers<1:parser.error("workers must be positive")
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name]="1"
    prepare(args)


if __name__=="__main__":main()
