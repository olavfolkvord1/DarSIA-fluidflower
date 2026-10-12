"""Resume source-bound selected-image P1-final3 inference after a strict replay."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time

from darsia.fluidflower.phase_ffac_selected_phase import (
    SPEC_SHA256, asset, bound_json, area_time_series,
)
from darsia.fluidflower.tools.refit_ffac_g3_downstream import (
    atomic_json, canonical_sha256, utc_now, worker_initializer,
)

CONTRACT="ffac-selected-phase-campaign-corrected-v2"
_STATE={}
MODULES=(
    "darsia.fluidflower.tools.run_ffac_selected_phase",
    "darsia.fluidflower.tools.run_ffac_selected_phase_corrected",
    "darsia.fluidflower.phase_ffac_selected_inference_corrected",
    "darsia.fluidflower.phase_ffac_selected_native_inputs_corrected",
    "darsia.fluidflower.phase_ffac_selected_inference",
    "darsia.fluidflower.phase_ffac_selected_phase",
    "darsia.fluidflower.phase_ffac_selected_sources",
    "darsia.fluidflower.phase_ffac_selected_native_centre",
    "darsia.fluidflower.tools.run_ffac_rig_camera_inference",
    "darsia.fluidflower.phase_rig_domain_inference",
    "darsia.fluidflower.phase_retained_source_inference",
    "darsia.fluidflower.phase_ffac_old_domain_seam",
    "darsia.fluidflower.phase_gas_barrier_rule",
    "darsia.fluidflower.phase_ffac_barrier_extension",
    "darsia.fluidflower.phase_gas_t4c_placements",
    "darsia.fluidflower.phase_gas_t2_regional_inputs",
    "darsia.fluidflower.phase_gas_t4_regional_inputs",
    "darsia.fluidflower.phase_gas_t4b_regional_inputs",
    "darsia.fluidflower.phase_l1_neighbourhood_gas",
    "darsia.fluidflower.phase_ffac_p1_final",
    "darsia.fluidflower.phase_ffac_p1_final2",
    "darsia.fluidflower.phase_ffac_p1_final3",
    "darsia.fluidflower.tools.build_ffac_p1_final2_candidate",
    "darsia.fluidflower.phase_gas_flatfield",
    "darsia.fluidflower.phase_gas_regional_context",
    "darsia.fluidflower.phase_g3_production",
    "darsia.fluidflower.phase_optical_regions",
    "darsia.fluidflower.phase_features",
    "darsia.fluidflower.phase_learning",
    "darsia.fluidflower.tools.phase_inference_benchmark_backends",
    "darsia.fluidflower.tools.train_ffac_gas_flatfield_variants",
    "darsia.fluidflower.tools.ffac_v5_temporal_alignment",
    "darsia.fluidflower.tools.ffac_v5_frame_context",
    "darsia.fluidflower.tools.ffac_v5_extended_reference",
    "darsia.fluidflower.phase_ffac_rig_geometry",
    "darsia.fluidflower.phase_pixel_area",
    "darsia.fluidflower.phase_ffac_rig_camera_metadata",
    "darsia.fluidflower.phase_ffac_rig_camera_runtime",
)


def bind(source_plan_path, destination, *, bounded_pilot=False, scientific_source_root=None):
    source_descriptor=asset(source_plan_path)
    source=bound_json(source_descriptor)
    spec_sha=source.get("spec_sha256",(source.get("binding",{}).get("spec") or
        source.get("binding",{}).get("approved_spec") or {}).get("sha256"))
    if spec_sha != SPEC_SHA256 or (len(source["runs"])!=39 and not bounded_pilot):
        raise ValueError("Selected phase plan must bind the approved all-39 source selection")
    pilot_producers={}
    if bounded_pilot:
        for run,state in source["runs"].items():
            if state.get("original_producer_receipts"):continue
            producers={}
            for row in bound_json(state["source_inventory"])["sources"]:
                if not row.get("studio_anchor"):continue
                path=Path(source["parent_chain"]["base_candidate"])/"rig_camera_frames"/run/f"frame_{row['studio_frame_index']:03d}.json"
                parent=bound_json(asset(path))
                if parent["source_sha256"]!=row["source_sha256"] or parent["elapsed_h"]!=row["elapsed_h"]:
                    raise ValueError("Pilot original area belongs to a different acquisition")
                producers[row["source_sha256"]]=parent["source_producer_receipt"]
            state["original_producer_receipts"]=producers
            pilot_producers[run]=producers
    codes={name:{"sha256":asset(importlib.import_module(name).__file__)["sha256"]}
           for name in MODULES}
    binding=dict(contract=CONTRACT,spec_sha256=SPEC_SHA256,source_preparation=source_descriptor,
        code=codes,history_policy="baseline and selected sources only; unselected Studio anchors are probes",
        strict_replay=dict(co2_changed_area_fraction_max=.001,gas_changed_area_fraction_max=.01,
            zero_parent_phase_allows_zero_changed_pixels_only=True),training=False,
        bulk_requires_all_studio_replays=True,existing_maps_used_as_inference=False,
        bounded_pilot=bool(bounded_pilot),production_run_count=39,
        native_input_correction=dict(contract="ffac-selected-historical-native-input-correction-v2",
            historical_masked_native=True, exact_exported_anchor_photo_mask=True,
            no_undeclared_temporal_ring=True, full_rig_context_unchanged=True,
            frozen_weights_and_postprocessing_unchanged=True),
        pilot_original_producer_receipts=pilot_producers,scientific_source_root=scientific_source_root)
    plan=dict(source,phase_campaign_binding=binding,pipeline_sha256=canonical_sha256(binding))
    if scientific_source_root is not None:
        plan["scientific_source_root"]=str(scientific_source_root)
    destination=Path(destination)
    if destination.exists():
        if bound_json(asset(destination))!=plan:
            raise FileExistsError("Selected phase campaign binding already exists with different code")
    else:
        atomic_json(destination,plan)
    return asset(destination)


def verify_code(plan):
    for name,descriptor in plan["phase_campaign_binding"]["code"].items():
        if asset(importlib.import_module(name).__file__)["sha256"]!=descriptor["sha256"]:
            raise ValueError("Selected phase numerical code changed: "+name)


def initialise(plan_asset,root,backend):
    worker_initializer(1)
    plan=bound_json(plan_asset);verify_code(plan)
    _STATE.update(plan=plan,plan_asset=plan_asset,root=Path(root),backend=backend)


def select_rows(inventory,stage,source_sha=None):
    rows=sorted(inventory["sources"],key=lambda r:(float(r["elapsed_h"]),r["source_sha256"]))
    if stage=="pilot":
        if source_sha is None:
            rows=[min((r for r in rows if r.get("studio_anchor") and float(r["elapsed_h"])>0),
                      key=lambda r:abs(float(r["elapsed_h"])-2.25))]
        else:
            rows=[r for r in rows if r["source_sha256"]==source_sha]
            if len(rows)!=1:raise ValueError("Pilot source is absent from the bound inventory")
    elif stage=="replay":
        rows=[r for r in rows if r.get("studio_anchor")]
    elif stage=="production":
        rows=[r for r in rows if r.get("selected") or r.get("baseline")]
    else:raise ValueError("Unknown selected phase stage")
    return rows


def run_one(run,stage,source_sha=None):
    from darsia.fluidflower.phase_ffac_selected_inference_corrected import SelectedPhaseEvaluator
    started=time.perf_counter()
    evaluator=SelectedPhaseEvaluator(_STATE["root"],_STATE["plan"],run,backend=_STATE["backend"])
    rows=select_rows(evaluator.inventory,stage,source_sha)
    receipts=[]
    for source in rows:
        digest=source["source_sha256"]
        existing=evaluator.saved(digest)
        result_asset=existing if existing is not None else evaluator.save(evaluator.evaluate(digest))
        saved=bound_json(result_asset)
        replay=saved["qa"].get("strict_studio_replay")
        receipts.append(result_asset)
        print(json.dumps(dict(event="frame_complete",run=run,stage=stage,
            source_sha256=digest,elapsed_h=source["elapsed_h"],resume=existing is not None,
            seconds=saved["qa"]["seconds"],replay_pass=None if replay is None else replay["all_pass"])),flush=True)
        if stage in ("pilot","replay") and replay is not None and not replay["all_pass"]:
            return dict(run=run,complete=False,stopped="strict Studio replay gate failed",
                receipts=receipts,failed_replay=replay,seconds=time.perf_counter()-started)
    return dict(run=run,complete=True,receipts=receipts,seconds=time.perf_counter()-started)


def strict_gate(root,plan):
    """All original Studio sources, including sources absent from the targets."""
    rows=[];missing=[]
    for run,state in sorted(plan["runs"].items()):
        inventory=bound_json(state["source_inventory"])
        for source in select_rows(inventory,"replay"):
            path=Path(root)/"selected_phase_frames"/run/(source["source_sha256"]+".json")
            if not path.is_file():
                missing.append(dict(run=run,source_sha256=source["source_sha256"],
                    studio_frame_index=source["studio_frame_index"]));continue
            descriptor=asset(path);saved=bound_json(descriptor)
            if (saved.get("complete") is not True or saved["binding"]["source"]!=source
                    or saved["binding"]["pipeline_sha256"]!=plan["pipeline_sha256"]):
                raise ValueError("Studio replay receipt changed source/history/numerical pipeline")
            replay=saved["qa"]["strict_studio_replay"]
            if replay is None:raise ValueError("An inherited map is not an inference replay")
            rows.append(dict(replay,receipt=descriptor))
    complete=not missing
    return dict(contract=CONTRACT,pipeline_sha256=plan["pipeline_sha256"],complete=complete,
        all_pass=complete and all(row["all_pass"] for row in rows),runs=len(plan["runs"]),
        replayed_frames=len(rows),missing=missing,failed=[r for r in rows if not r["all_pass"]],
        rows=rows,training=False,heldout_mass_computed=False,parameters_changed=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan");parser.add_argument("--plan")
    parser.add_argument("--bind-output");parser.add_argument("--root",required=True)
    parser.add_argument("--bounded-pilot",action="store_true")
    parser.add_argument("--source-root",help="Bound prepared RAW root, separate from the new candidate outputs")
    parser.add_argument("--stage",choices=("pilot","replay","gate","production"),default="pilot")
    parser.add_argument("--runs",nargs="+");parser.add_argument("--source-sha")
    parser.add_argument("--workers",type=int,default=12)
    parser.add_argument("--backend",choices=("auto","cpu","gpu"),default="auto")
    parser.add_argument("--stage-receipt",required=True)
    args=parser.parse_args(argv)
    if args.bind_output:
        print(json.dumps(bind(args.source_plan,args.bind_output,bounded_pilot=args.bounded_pilot,
            scientific_source_root=args.source_root)));return 0
    if not args.plan:raise ValueError("Use an immutable bound selected phase plan")
    plan_asset=asset(args.plan);plan=bound_json(plan_asset);verify_code(plan)
    destination=Path(args.stage_receipt)
    if destination.exists():raise FileExistsError("Operational stage receipt must be a new file")
    if args.stage in ("gate","production"):
        if plan["phase_campaign_binding"].get("bounded_pilot") or len(plan["runs"])!=39:
            raise ValueError("A bounded diagnostic pilot cannot authorize bulk or all-39 acceptance")
        gate=strict_gate(args.root,plan)
        if args.stage=="gate":
            atomic_json(destination,dict(gate,created_utc=utc_now(),plan=plan_asset))
            return 0 if gate["all_pass"] else 2
        if not gate["all_pass"]:raise ValueError("Bulk selected phase inference is blocked by the strict all-Studio replay gate")
    runs=args.runs or sorted(plan["runs"])
    if len(set(runs))!=len(runs) or any(run not in plan["runs"] for run in runs):
        raise ValueError("Run shard escapes the immutable all-39 phase plan")
    if args.workers<1:raise ValueError("At least one bounded worker is required")
    started=utc_now();results=[];errors=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(runs)),
        mp_context=multiprocessing.get_context("spawn"),initializer=initialise,
        initargs=(plan_asset,args.root,args.backend)) as executor:
        future_runs={executor.submit(run_one,run,args.stage,args.source_sha):run for run in runs}
        for future in as_completed(future_runs):
            try:results.append(future.result())
            except Exception as error:
                errors.append(dict(run=future_runs[future],error=repr(error)))
    complete=not errors and all(r["complete"] for r in results)
    atomic_json(destination,dict(contract=CONTRACT,complete=complete,stage=args.stage,
        plan=plan_asset,pipeline_sha256=plan["pipeline_sha256"],host=socket.gethostname(),
        pid=os.getpid(),started_utc=started,finished_utc=utc_now(),runs=runs,
        workers=min(args.workers,len(runs)),internal_threads=1,backend=args.backend,
        results=results,errors=errors,training=False,parent_changed=False,
        full_production_started=args.stage=="production"))
    return 0 if complete else 2


if __name__=="__main__":
    raise SystemExit(main())
