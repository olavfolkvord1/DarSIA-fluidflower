"""Report saved selected-phase receipts without computing or changing a map."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from darsia.fluidflower.phase_ffac_selected_phase import asset,bound_json,area_time_series
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json,utc_now

CONTRACT="ffac-selected-phase-saved-report-v1"


def flatten_replay(replay,receipt):
    row={key:replay[key] for key in ("run","frame_index","source_sha256","elapsed_h",
        "categorical_changed_pixels","all_pass","bit_identical","support_changed_pixels")}
    for phase in ("co2","gas"):
        row.update({phase+"_"+key:value for key,value in replay[phase].items()})
    row["receipt_sha256"]=receipt["sha256"]
    return row


def write_csv(path,rows):
    stream=io.StringIO(newline="")
    columns=sorted({key for row in rows for key in row})
    if columns:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    with Path(path).open("x",encoding="utf-8",newline="") as target:target.write(stream.getvalue())
    return asset(path)


def build(plan_path,root,destination):
    root,destination=Path(root),Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    plan_descriptor=asset(plan_path);plan=bound_json(plan_descriptor)
    replay_rows=[];phase_rows=[];steps=[];inventories=[];missing=[]
    for run,state in sorted(plan["runs"].items()):
        inventory=bound_json(state["source_inventory"])
        inventories.append(state["source_inventory"])
        original=[];selected=[]
        for source in inventory["sources"]:
            descriptor_path=root/"selected_phase_frames"/run/(source["source_sha256"]+".json")
            if not descriptor_path.is_file():
                missing.append(dict(run=run,source_sha256=source["source_sha256"],
                    selected=bool(source.get("selected")),studio_anchor=bool(source.get("studio_anchor"))))
                continue
            descriptor=asset(descriptor_path);saved=bound_json(descriptor)
            if (saved.get("complete") is not True or saved["binding"]["source"]!=source
                    or saved["binding"]["pipeline_sha256"]!=plan["pipeline_sha256"]):
                raise ValueError("Saved report found a changed source/history/pipeline binding")
            area=saved["qa"]["phase_area"]
            row=dict(run=run,source_sha256=source["source_sha256"],elapsed_h=source["elapsed_h"],
                selected=bool(source.get("selected")),baseline=bool(source.get("baseline")),
                selected_index=source.get("selected_index"),studio_frame_index=source.get("studio_frame_index"),
                previous_source_sha256=source["previous_source_sha256"],receipt_sha256=descriptor["sha256"],
                **{key:area[key] for key in ("co2_pixels","gas_pixels","aqueous_pixels", "co2_cm2","gas_cm2","aqueous_cm2")})
            phase_rows.append(row)
            replay=saved["qa"].get("strict_studio_replay")
            if source.get("studio_anchor"):
                if replay is None:raise ValueError("Copied labels cannot be presented as a replay")
                replay_rows.append(flatten_replay(replay,descriptor))
                original.append(dict(elapsed_h=source["elapsed_h"],co2_cm2=replay["co2"]["parent_cm2"],
                    gas_cm2=replay["gas"]["parent_cm2"]))
            if source.get("selected") or source.get("baseline"):selected.append(row)
        steps.extend(area_time_series(selected,original))
    files=dict(replay=write_csv(destination/"studio_replay.csv",replay_rows),
        phase_areas=write_csv(destination/"phase_areas.csv",phase_rows),
        selected_area_steps=write_csv(destination/"selected_area_steps.csv",steps))
    expected_replays=sum(sum(bool(r.get("studio_anchor")) for r in bound_json(s["source_inventory"])["sources"])
        for s in plan["runs"].values())
    expected_production=sum(sum(bool(r.get("selected") or r.get("baseline")) for r in bound_json(s["source_inventory"])["sources"])
        for s in plan["runs"].values())
    all_pass=len(replay_rows)==expected_replays and all(r["all_pass"] for r in replay_rows)
    record=dict(contract=CONTRACT,created_utc=utc_now(),plan=plan_descriptor,
        pipeline_sha256=plan["pipeline_sha256"],run_count=len(plan["runs"]),
        replay_frames=len(replay_rows),expected_replay_frames=expected_replays,
        production_frames=sum(bool(r["selected"] or r["baseline"]) for r in phase_rows),
        expected_production_frames=expected_production,all_studio_replays_pass=all_pass,
        production_complete=sum(bool(r["selected"] or r["baseline"]) for r in phase_rows)==expected_production,
        missing=missing,files=files,source_inventories=inventories,
        parameters_changed=False,maps_changed=False,training=False,mass_computed=False,
        bottom_mass_allowed=False,default_changed=False,registered_by_this_report=False)
    atomic_json(destination/"report.json",record)
    status="Complete strict Studio replay gate passed." if all_pass else "Partial or failed strict Studio replay; production remains blocked."
    text=("Selected P1 final 3 phase report\n\n"+status+"\n\n"
        f"Saved original Studio replays: {len(replay_rows)}/{expected_replays}; "
        f"selected production frames including baselines: {record['production_frames']}/{expected_production}.\n\n"
        "CO2 changes use the sum of added and removed measured pixel area divided by the parent CO2 area. "
        "The fixed limits are 0.1% for CO2 and 1% for gas. A zero-area parent requires zero changed pixels. "
        "Selected area-step flags compare each measured-area change with the surrounding old Studio interval; they are descriptive, not a changed gate.\n\n"
        "Policy A uses the baseline and accepted selected sources only. Replay-only Studio sources are probes and do not become predecessors. "
        "Frozen heads, post-processing, labels and the original candidate are unchanged. The bottom strip remains blocked for mass. "
        "This report computes no concentration or mass and does not register or activate a candidate.\n")
    with (destination/"README.txt").open("x",encoding="utf-8") as stream:stream.write(text)
    return asset(destination/"report.json")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",required=True);parser.add_argument("--root",required=True)
    parser.add_argument("--destination",required=True)
    args=parser.parse_args(argv);print(build(args.plan,args.root,args.destination));return 0


if __name__=="__main__":raise SystemExit(main())
