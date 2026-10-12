"""Seal completed selected maps, scientific sources and the strict Studio replay.

This metadata bridge never changes or computes a phase map. It writes the
mass-loader admission flags only after the real all-39 inventory is complete.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from darsia.fluidflower.phase_ffac_selected_phase import asset,bound_json
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json,utc_now
from darsia.fluidflower.tools.run_ffac_selected_phase import strict_gate,verify_code

CONTRACT="ffac-selected-p1-final3-completion-v1"
ALL_RUNS={f"ac{i}" for i in range(17,62)}-{"ac20","ac24","ac27","ac36","ac37","ac51"}


def verify_source(scientific,phase,source,plan,inventory_descriptor,descriptor):
    binding=scientific.get("binding",{})
    if (scientific.get("contract")!="ffac-selected-frozen-scientific-source-v1"
            or scientific.get("complete") is not True or binding.get("source")!=source
            or binding.get("frozen_parent")!=plan["parent_chain"]["parent_seal"]
            or scientific.get("run")!=source["run"]
            or scientific.get("source_sha256")!=source["source_sha256"]
            or scientific.get("elapsed_h")!=source["elapsed_h"]
            or scientific["native_scientific"]["source_sha256"]!=source["source_sha256"]
            or scientific["full_extended_scientific"]!=phase["full_extended_scientific"]
            or scientific["baseline_extended_scientific"]!=phase["baseline_extended_scientific"]):
        raise ValueError("Selected source lacks exact frozen-parent/radiometry/source/time binding")
    current=(binding.get("job_sha256")==plan["job_sha256"]
        and binding.get("source_inventory")==inventory_descriptor)
    imported=(plan.get("source_imports",{}).get(source["run"],{}).get(source["source_sha256"])==descriptor)
    if not current and not imported:
        raise ValueError("Scientific source producer/import is not declared by the frozen phase plan")
    audit=scientific["source_audit"]
    if audit.get("reused_exact_scientific_source") is True:
        if audit.get("native_centre_changed") is not False or audit.get("rig_geometry_changed") is not False:
            raise ValueError("Reused source changed its scientific centre or geometry")
    elif not (audit.get("new_raw_source_corrected") is True and audit.get("frozen_photometry") is True
            and audit.get("photometry_fit") is False and audit.get("parent_scientific_source_changed") is False):
        raise ValueError("New source has no frozen photometry/no-fit producer evidence")
    return dict(current_producer=current,declared_import=imported,
        frozen_parent=binding["frozen_parent"],source=descriptor)


def collect_completed(root,plan):
    if set(plan["runs"])!=ALL_RUNS or plan["phase_campaign_binding"].get("bounded_pilot"):
        raise ValueError("Only the full exact 39-run phase campaign can be sealed")
    entries=[];baselines=[];manifests={};missing=[];time_parity=[]
    source_root=Path(plan["scientific_source_root"])
    for run,state in sorted(plan["runs"].items()):
        inventory=bound_json(state["source_inventory"])
        manifest=bound_json(state["source_manifest"])
        if manifest.get("run")!=run or manifest.get("contract")!="ffac-selected-camera-images-v1":
            raise ValueError("Completed phase source manifest changed run/selection identity")
        manifests[run]=state["source_manifest"]
        lookup={row["source_sha256"]:row for row in inventory["sources"]}
        selected={row["source_sha256"]:row for row in manifest["frame_sources"]}
        if len(selected)!=len(manifest["frame_sources"]):
            raise ValueError("Selected manifest repeats an acquisition")
        targets={}
        for target in manifest["target_matches"]:
            if target.get("selected") is not True:continue
            targets.setdefault(target["source_sha256"],[]).append(target)
        for digest in [inventory["baseline_source_sha256"],*selected]:
            source=lookup[digest]
            is_baseline=digest==inventory["baseline_source_sha256"]
            if is_baseline:
                if digest!=manifest["baseline"]["source_sha256"] or not source.get("baseline"):
                    raise ValueError("Completed run uses a different pre-injection baseline")
            else:
                accepted=selected[digest]
                if (not source.get("selected") or source["selected_index"]!=accepted["selection_index"]
                        or source["elapsed_h"]!=accepted["elapsed_h"] or not targets.get(digest)
                        or any(t["selection_index"]!=accepted["selection_index"]
                            or t["elapsed_h"]!=accepted["elapsed_h"] for t in targets[digest])):
                    raise ValueError("Selected source/target camera time or index does not match")
            phase_path=Path(root)/"selected_phase_frames"/run/(digest+".json")
            source_path=source_root/"selected_sources"/run/digest/"receipt.json"
            absent=[str(p) for p in (phase_path,source_path) if not p.is_file()]
            if absent:
                missing.append(dict(run=run,source_sha256=digest,baseline=is_baseline,paths=absent));continue
            phase_descriptor,source_descriptor=asset(phase_path),asset(source_path)
            phase,science=bound_json(phase_descriptor),bound_json(source_descriptor)
            if (phase.get("contract")!="ffac-selected-p1-final3-phase-v1" or phase.get("complete") is not True
                    or phase["binding"]["source"]!=source
                    or phase["binding"]["pipeline_sha256"]!=plan["pipeline_sha256"]
                    or phase["run"]!=run or phase["source_sha256"]!=digest
                    or phase["elapsed_h"]!=source["elapsed_h"] or phase["frame_index"]!=source["work_index"]
                    or phase.get("bottom_mass_allowed") is not False or phase.get("inference_performed") is not True):
                raise ValueError("Completed selected phase differs from its numerical source/time/pipeline")
            proof=verify_source(science,phase,source,plan,state["source_inventory"],source_descriptor)
            entry=dict(run=run,frame_index=source["work_index"],elapsed_h=source["elapsed_h"],
                source_sha256=digest,selection_index=source.get("selected_index"),source_binding=source,
                receipt=phase_descriptor,scientific_receipt=source_descriptor,run_manifest=state["source_manifest"],
                source_binding_proof=proof,phase_area=phase["qa"]["phase_area"],
                target_hours=[] if is_baseline else [t["target_hour"] for t in targets[digest]])
            (baselines if is_baseline else entries).append(entry)
            if source.get("studio_anchor"):
                replay=phase["qa"].get("strict_studio_replay")
                if replay is None or replay["source_sha256"]!=digest or replay["elapsed_h"]!=source["elapsed_h"]:
                    raise ValueError("Studio source/time parity lacks an actual inference replay")
                time_parity.append(dict(run=run,source_sha256=digest,elapsed_h=source["elapsed_h"],
                    studio_frame_index=source["studio_frame_index"],receipt=phase_descriptor))
    return dict(entries=entries,baselines=baselines,run_manifests=manifests,missing=missing,
        source_time_parity=time_parity,selected_expected=sum(len(bound_json(s["source_manifest"])["frame_sources"])
            for s in plan["runs"].values()))


def seal(plan_path,root,destination,report_path):
    plan_descriptor=asset(plan_path);plan=bound_json(plan_descriptor);verify_code(plan)
    gate=strict_gate(root,plan)
    if not gate["complete"] or not gate["all_pass"]:
        raise ValueError("The fixed all-Studio replay gate has not passed; no completion flags are written")
    complete=collect_completed(root,plan)
    if (complete["missing"] or len(complete["baselines"])!=39
            or len(complete["entries"])!=complete["selected_expected"]):
        raise ValueError("Selected production or baseline source inventory is incomplete; no completion flags are written")
    report_descriptor=asset(report_path);report=bound_json(report_descriptor)
    if (report.get("pipeline_sha256")!=plan["pipeline_sha256"]
            or report.get("all_studio_replays_pass") is not True or report.get("production_complete") is not True):
        raise ValueError("Completed phase seal needs the actual complete area/time/replay report")
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    strict_path=destination/"strict_studio_replay.json"
    atomic_json(strict_path,dict(gate,created_utc=utc_now(),plan=plan_descriptor))
    inventory_path=destination/"selected_completion_inventory.json"
    atomic_json(inventory_path,dict(contract=CONTRACT,complete=True,pipeline_sha256=plan["pipeline_sha256"],
        plan=plan_descriptor,**complete))
    record=dict(contract=CONTRACT,complete=True,created_utc=utc_now(),all_39_complete=True,
        studio_time_parity_passed=True,runs=sorted(ALL_RUNS),run_manifests=complete["run_manifests"],
        pipeline_sha256=plan["pipeline_sha256"],phase_plan=plan_descriptor,
        strict_replay_gate=asset(strict_path),selected_completion_seal=asset(inventory_path),
        area_time_report=report_descriptor,selected_frames=len(complete["entries"]),baseline_frames=39,
        original_studio_replays=gate["replayed_frames"],bottom_mass_allowed=False,
        maps_changed_by_seal=False,training=False,mass_computed=False,default_changed=False,
        scientific_acceptance=False,automatic_activation=False)
    atomic_json(destination/"accepted_phase_gate.json",record)
    return asset(destination/"accepted_phase_gate.json")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("plan","root","destination","report"):parser.add_argument("--"+name,required=True)
    args=parser.parse_args(argv)
    print(seal(args.plan,args.root,args.destination,args.report));return 0


if __name__=="__main__":raise SystemExit(main())
