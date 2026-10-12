"""Step A only: reuse exact bound photo measurements, pool and score the floor.

No domains or labels are changed. A failed gate cannot authorize later steps.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from .. import phase_ffac_final_round_floor_measurement as pooling
from .. import phase_ffac_p1_final_bottom_report as photo
from . import report_ffac_p1_final as original

SPEC_SHA256 = "e843e61751b795984550cfe853548ea7ca4e50364a5cb44681621d485abd2c0b"


def compute_gate(candidate, source_report, spec, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.perf_counter(), time.process_time()
    candidate, source_report, spec = Path(candidate).resolve(), Path(source_report).resolve(), Path(spec).resolve()
    spec_binding = original.artifact(spec)
    if spec_binding["sha256"] != SPEC_SHA256:
        raise ValueError("The locked final-round spec SHA changed")
    saved = original.read(source_report)
    seal_asset = original.artifact(candidate/"rig_camera_candidate_receipt.json")
    seal = original.read(seal_asset["path"])
    if (saved.get("complete") is not True or saved.get("candidate") != seal_asset
            or seal.get("complete") is not True or (seal.get("run_count"), seal.get("frame_count")) != (43,864)
            or saved.get("measurement_method") != photo.method_document()
            or saved["report_code"]["measurement"]["sha256"] != original.artifact(photo.__file__)["sha256"]):
        raise ValueError("Stored measurement method or exact P1-final candidate binding changed")
    measurements = {row["run"].casefold(): row for row in saved["per_run"]}
    runs = sorted({row["run"].casefold() for row in seal["frames"]})
    if len(measurements) != 43 or sorted(measurements) != runs:
        raise ValueError("Gate needs the same 43 source-bound runs, not a subset")
    for run, record in measurements.items():
        manifest_path = original.bound(record["source_manifest"])
        manifest = original.read(manifest_path)
        input_path = original.bound(record["input_manifest"], base=manifest_path.parent)
        inputs = original.read(input_path)
        for image, source in (("reference", manifest["reference"]), ("last", manifest["frame_sources"][-1])):
            measurement = record[image]
            if measurement["source_sha256"] != source["source_sha256"] or measurement["elapsed_h"] != source["elapsed_h"]:
                raise ValueError("Stored photographic measurement belongs to another source or camera time")
            key = "reference" if image == "reference" else "frames"
            input_row = inputs[key] if image == "reference" else inputs[key][-1]
            if measurement["provenance"]["H_ac28_to_frame_native"] != input_row["H_ac28_to_frame_native"]:
                raise ValueError("Stored photographic registration differs from exact input manifest")
        print(f"Verified stored photographic measurement {run.upper()}", flush=True)
    canonical = measurements["ac28"]["reference"]
    Hcanon = np.asarray(canonical["provenance"]["H_ac28_to_frame_native"],float)
    prov = canonical["provenance"]
    first = min(canonical["samples"], key=lambda sample: sample["column"])
    u,_ = photo.project(prov["H_frame_native_to_metric_pixel"],first["column"]+prov["native_origin_xy"][0],first["bottom_canvas_y"]+prov["native_origin_xy"][1])
    anchor = float(u*prov["metric_mm_factor"])
    pool = pooling.pool_line({run:record["reference"] for run,record in measurements.items()},anchor_mm=anchor)
    domain,_ = original.arrays({"classes":prov["classes"],"area_map":prov["area_map"]})
    curve = pooling.canonical_curve(pool,domain,prov)
    np.save(output/"pooled_floor_AC28_reference_canvas_y.npy",curve,allow_pickle=False)
    rows=[]
    for run in runs:
        values=dict(run=run.upper())
        for image in ("reference","last"):
            measurement=measurements[run][image]
            target=measurement["provenance"]
            transform=np.asarray(target["H_ac28_to_frame_native"],float)@np.linalg.inv(Hcanon)
            transported=photo.transport_curve(curve,H_source_to_target=transform,
                source_origin_xy=prov["native_origin_xy"],target_origin_xy=target["native_origin_xy"],
                target_width=len(measurement["curve_canvas_y"]))
            values[image]=pooling.score_measurement(measurement,transported)
            values[image]["source_sha256"]=measurement["source_sha256"]
            values[image]["transport_H_AC28_reference_to_source"]=transform.tolist()
        rows.append(values)
    ref_pass=sum(row["reference"]["passes"] for row in rows)
    last_pass=sum(row["last"]["passes"] for row in rows)
    gate=dict(reference_passing_runs=ref_pass,last_passing_runs=last_pass,run_count=43,
        required_passing_runs=pooling.MINIMUM_PASSING_RUNS,maximum_median_absolute_residual_mm=pooling.MAXIMUM_MEDIAN_RESIDUAL_MM,
        reference_passes=ref_pass>=pooling.MINIMUM_PASSING_RUNS,last_passes=last_pass>=pooling.MINIMUM_PASSING_RUNS,
        passes=ref_pass>=pooling.MINIMUM_PASSING_RUNS and last_pass>=pooling.MINIMUM_PASSING_RUNS,
        reference_failed_runs=[row["run"] for row in rows if not row["reference"]["passes"]],
        last_failed_runs=[row["run"] for row in rows if not row["last"]["passes"]],
        decision="continue to separately authorized A application and B-D" if ref_pass>=39 and last_pass>=39 else "STOP; no A application or B-D candidate is authorized by this gate")
    code_dir=output/"code";code_dir.mkdir()
    code={}
    for name,path in (("pooling",pooling.__file__),("driver",__file__)):
        snapshot=code_dir/Path(path).name
        with snapshot.open("xb") as f:f.write(Path(path).read_bytes())
        code[name]=original.artifact(snapshot)
    result=dict(contract=pooling.CONTRACT,complete=True,step="A measurement and gate only",spec=spec_binding,
        candidate=seal_asset,source_report=original.artifact(source_report),measurement_reuse=True,
        reuse_policy="The complete source-bound prior report's exact unchanged photographic method, 43 references and 43 last measurements are reused. Source manifests, camera times, source hashes and per-frame homographies are verified again. Dense RAW assets are not re-read or re-hashed; their exact descriptors remain bound through the immutable report and source manifests.",
        photographic_method=photo.method_document(),pool=pool,gate=gate,per_run=rows,
        canonical_curve=dict(asset=original.artifact(output/"pooled_floor_AC28_reference_canvas_y.npy"),reference_source_sha256=canonical["source_sha256"],
            native_origin_xy=prov["native_origin_xy"],H_ac28_to_frame_native=prov["H_ac28_to_frame_native"],
            frame_native_to_metric_pixel=prov["H_frame_native_to_metric_pixel"],metric_mm_factor=prov["metric_mm_factor"]),
        code=code,code_origins={"pooling":str(pooling.__file__),"driver":str(__file__)},
        limitations=["Original photographic acceptance thresholds and metric conversion are retained exactly; no gate thresholds are tuned.",
            "Shared 1 cm nodes are anchored to the first AC28-reference sample's common metric x; an accepted measurement is assigned to its nearest node and its original x is retained for residual scoring.",
            "Common metric homography already contains the width revision's scale and the frozen photographic method additionally multiplies metric_mm_factor by that revision's linear_factor. This gate retains the original method convention rather than silently changing its millimetres.",
            "Pooled interpolation/constant continuation supplies a common proposed line at unobserved columns; it does not create new photographic evidence.",
            "Unknown photographic samples do not contribute to the pool or become passing residual measurements; residual counts are explicit.",
            "No domain-area or CO2-removal totals are computed or applied in this measurement-only step."],
        timing=dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,gpu_seconds=0.,backend="cpu-metadata-measurement-pooling"),
        scientific_assets_changed=False,labels_changed=False,candidate_created=False,mass_eligible=False,bottom_mass_allowed=False)
    original.write_new(output/"floor_measurement_gate.json",result)
    print("STEP_A_GATE "+str(gate),flush=True)
    return result,measurements


def make_figures(result,measurements,output):
    output=Path(output);pool=result["pool"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,4.3))
    for run,row in sorted(measurements.items()):
        samples=pooling.physical_samples(row["reference"])
        ax.scatter([s["metric_x_mm"] for s in samples],[s["band_distance_mm"] for s in samples],s=7,alpha=.13,color="#3073a3")
    ax.plot(pool["x_mm"],pool["height_mm"],color="#bd2b28",lw=2,label="Pooled median at >=3 accepted runs; linear/constant continuation")
    ax.set(xlabel="Inherited common metric x (mm; original photographic metric convention)",ylabel="Height above previous rig-domain bottom (mm)",title=f"Locked Step A: references {result['gate']['reference_passing_runs']}/43, last {result['gate']['last_passing_runs']}/43 pass <=1.0 mm")
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/"pooled_sand_floor_line.png",dpi=160);plt.close(fig)
    rows=[]
    for row in result["per_run"]:
        for image in ("reference","last"):
            own=row[image]
            rows.append(dict(run=row["run"],image=image,accepted_measurements=own["accepted_measurements"],scored_samples=own["scored_samples"],unknown_samples=own["unknown_samples"],
                median_absolute_residual_mm=own["median_absolute_residual_mm"],p90_absolute_residual_mm=own["p90_absolute_residual_mm"],maximum_absolute_residual_mm=own["maximum_absolute_residual_mm"],passes=own["passes"]))
    with (output/"floor_residuals_reference_last.csv").open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    cellw,cellh,cols=700,155,2
    montage=Image.new("RGB",(cellw*cols,70+math.ceil(len(measurements)/cols)*cellh),"#16191f");draw=ImageDraw.Draw(montage)
    draw.text((15,12),"Step A photographic reference evidence: red old domain bottom; green accepted sand/bright-band transition",fill="white")
    draw.text((15,35),"Images are immutable prior-measurement photographs, reused after binding checks. Residuals use the common pooled line and exact per-frame transport.",fill="#c7d1da")
    scores={row["run"]:row for row in result["per_run"]}
    for n,(run,measurement) in enumerate(sorted(measurements.items())):
        path=original.bound(measurement["reference"]["figures"][0]);image=Image.open(path).convert("RGB");image.thumbnail((cellw-12,cellh-30),Image.Resampling.LANCZOS)
        x=(n%cols)*cellw;y=70+(n//cols)*cellh;own=scores[run.upper()]
        def display_residual(image):
            value=own[image]['median_absolute_residual_mm']
            return "unknown" if value is None else f"{value:.3f} mm"
        draw.text((x+8,y),f"{run.upper()}  ref median|res|={display_residual('reference')}  last={display_residual('last')}",fill="white")
        montage.paste(image,(x+6,y+25))
    montage.save(output/"sand_floor_references_all_43.png")
    original.write_new(output/"floor_measurement_figures.json",dict(contract=pooling.CONTRACT,gate_report=original.artifact(output/"floor_measurement_gate.json"),
        figures=[original.artifact(output/name) for name in ("pooled_sand_floor_line.png","sand_floor_references_all_43.png")],
        residual_table=original.artifact(output/"floor_residuals_reference_last.csv"),scientific_assets_changed=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("candidate","source-report","spec","output"):parser.add_argument("--"+key,required=True)
    args=parser.parse_args(argv)
    result,measurements=compute_gate(args.candidate,args.source_report,args.spec,args.output)
    make_figures(result,measurements,args.output)


if __name__=="__main__":main()
