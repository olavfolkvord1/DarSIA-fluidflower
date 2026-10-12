"""Post-check, early-only R1 objective control with exact imported native optics.

The original two gains, 27 runs, early frame list, doses and no-dilution gas
treatment remain unchanged. Only the objective becomes per-dose, two-sided and
finite. Known missing capacity is imported from its frozen producer; only the
previously unbounded geometry capacity is replayed. No RAW image is opened.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import json
import gc
import uuid
import importlib.util
from collections import Counter

import numpy as np
from scipy.optimize import least_squares
from PIL import Image
import darsia

from darsia.fluidflower.tools import run_ffac_common_mass_round2 as driver
from darsia.fluidflower import phase_ffac_common_mass as original
from darsia.fluidflower import phase_ffac_common_mass_inputs as inputs
from darsia.fluidflower import phase_ffac_common_mass_round2 as native
from darsia.fluidflower import phase_ffac_common_mass_round2_production as out

CONTRACT="ffac-round2-legacy-objective-only-early-control-v1"
STATE={}

# The runtime companion folder contains the current unbound scripts, while
# PYTHONPATH deliberately points to immutable snapshot3 for all core maths.
# Do not accidentally import an older helper copy from that snapshot package.
_gate_path=Path(__file__).with_name("sensitivity_ffac_common_mass_round2.py")
_gate_spec=importlib.util.spec_from_file_location("ffac_round2_runtime_sensitivity_gate",_gate_path)
gate=importlib.util.module_from_spec(_gate_spec);_gate_spec.loader.exec_module(gate)


def imported_rows(job):
    rows=[];assets=[]
    for path in sorted((Path(job) / "work/fit_summaries").glob("*/receipt.json")):
        record=out.read_json(path)
        if record.get("complete") is not True or record.get("heldout_pixels_read") is not False:
            raise ValueError("Original control summaries must be complete calibration-only producers")
        for row in record["summaries"]:
            if row["run"] not in original.CALIBRATION_RUNS or row["contract"]!=original.CONTRACT or row["elapsed_h"]>2.25:
                raise ValueError("Original control imports only the exact old27 early frames")
            rows.append(row)
        assets.append(inputs.asset(path))
    keys=[(r["run"],r["frame_index"]) for r in rows]
    if len(keys)!=len(set(keys)) or {r["run"] for r in rows}!=set(original.CALIBRATION_RUNS):
        raise ValueError("Original early control requires all27 unique source-bound runs")
    return rows,assets


def setup(job,destination,round1_job):
    checked=gate.context(job,Path(destination) / "passed_check_gate")
    rows,assets=imported_rows(round1_job)
    original_freeze,original_sha=native.verify_original_round1_freeze(checked["campaign"]["round1_freeze"]["path"])
    plan=checked["plan"];base=inputs.read_bound_json(plan["base_input_plan"])
    if original_freeze["input_plan_sha256"]!=plan["base_input_plan"]["sha256"]:
        raise ValueError("Legacy physical/recipe/mask input plan differs from its original freeze")
    atmospheres=inputs.read_bound_json(base["atmosphere"])
    source_index={ (r["run"],r["frame_index"]):r for r in inputs.read_bound_json(plan["source_metadata"])["frames"]}
    binding=dict(contract=CONTRACT,campaign_sha256=out.digest(checked["campaign"]),round2_freeze=inputs.asset(checked["freeze"].path),
        final_check=checked["check"],original_freeze_sha256=original_sha,original_summary_producers=assets,
        original_rows_sha256=out.digest(rows),helper_sha256=inputs.sha256(__file__),heldout_pixels_read=False,
        scope="Early calibration-only control; no new held-out prediction or all-frame conservation claim")
    driver.unchanged_or_new(Path(destination) / "binding.json",binding)
    return dict(destination=Path(destination),binding=out.digest(binding),rows=rows,original_freeze=original_freeze,
        plan=plan,base=base,index=source_index,atmosphere={r["run"]:r for r in atmospheres["runs"]})


def init(job,destination,round1_job):
    import cv2
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(1)
    STATE.update(setup(job,destination,round1_job),reader=inputs.VerifiedAssetReader(),threads=threadpool_limits(limits=1))


def validate_sources(row,metadata,base,transport):
    provenance=row["source_binding"]
    if (row["elapsed_h"]!=metadata["elapsed_h"] or provenance["source_receipt"]!=metadata["receipt"]
            or provenance["physical_reference"]["sha256"]!=base["physical_reference"]["sha256"]
            or provenance["measured_area"]["sha256"]!=metadata["area_map"]["sha256"]
            or provenance["clear_water_reference"]["receipt"]["sha256"]!=transport["clearwater_receipt_sha256"]):
        raise ValueError("Legacy optical support/source/time/area/physical/clearwater binding differs")
    if provenance.get("phase_labels_modified") is not False or provenance.get("bottom_mass_allowed") is not False:
        raise ValueError("Legacy control must preserve original labels and main support")


def finite_capacity(row,*,unknown_pixels,unknown_area_cm2,unknown_capacity_g):
    """Keep archived old quantifiability; add only unmeasured-geometry capacity."""
    if (unknown_pixels!=row["unknown_physical_phase_pixels"]
            or not np.isclose(unknown_area_cm2,row["unknown_physical_phase_area_cm2"],rtol=1e-12,atol=1e-10)):
        raise ValueError("Legacy unknown-geometry pixelcount/area does not reproduce its producer")
    known_main=float(row["unknown_mass_upper_g"])
    known_excluded=sum(float(part["known_physical_capacity_upper_g"]) for part in row["excluded_strip_capacities"].values())
    upper=known_main+known_excluded+float(unknown_capacity_g)
    previous=row["mass_balance_missing_capacity_upper_g"]
    if previous is not None and not np.isclose(previous,upper,rtol=1e-12,atol=1e-10):
        raise ValueError("Legacy finite capacity changed rather than importing its exact producer")
    return dict(row,mass_balance_missing_capacity_upper_g=upper,
        finite_capacity_audit=dict(known_main_missing_path_capacity_g=known_main,
            known_excluded_capacity_g=known_excluded,unknown_geometry_only_capacity_g=float(unknown_capacity_g),
            unknown_geometry_phase_pixels=unknown_pixels,unknown_geometry_area_cm2=unknown_area_cm2,
            old_optical_quantifiability_imported=True,R2_fixed_gas_support_used=False,main_geometry_imputed=False,
            finite_limit_depth_m=.0155,finite_limit_porosity=.45,RAW_read=False))


def capacity_run(run):
    c=STATE;reader=c["reader"];folder=c["destination"] / "capacity" / run
    if driver.resume(folder,c["binding"]):return dict(run=run,cached=True)
    physical=inputs.load_physical_reference(c["base"]["physical_reference"],reader)
    transport=inputs.read_bound_json(c["plan"]["clearwater_transports"][run]);rows=[]
    for old in (r for r in c["rows"] if r["run"]==run):
        metadata=c["index"][(run,old["frame_index"])];validate_sources(old,metadata,c["base"],transport)
        # The original native response and known-capacity histogram are imported,
        # not recalculated from R2's physical gas support. Only missing geometry
        # needs a new finite bound. Exact archived count/area is the replay gate.
        if old["unknown_physical_phase_pixels"]==0:
            rows.append(finite_capacity(old,unknown_pixels=0,unknown_area_cm2=0.,unknown_capacity_g=0.));continue
        receipt=reader.json(metadata["receipt"]);revision=receipt["rig_domain_revision"]
        shape=tuple(revision["phase_shape_hw"])
        matrix=inputs.matrix_ac28_to_canvas(receipt["full_extended_scientific"]["binding"]["geometry"],revision["native_origin_xy"])
        fields=inputs.transport_physical_reference(physical,matrix,shape)
        with Image.open(reader.path(receipt["labels_map"])) as image:labels=np.asarray(image).copy()
        with np.load(reader.path(revision["classes"]),allow_pickle=False) as archive:domain=archive["rig_domain"].copy()
        if not np.array_equal(domain,labels!=255):raise ValueError("Legacy replay locked phase/domain differs")
        with np.load(reader.path(revision["area_map"]),allow_pickle=False) as archive:
            area=np.asarray(archive[revision["area_map"].get("array_key","pixel_area_mm2")],np.float64)/1e6
        unknown=domain & np.isin(labels,(1,2)) & ~fields["physical_supported_mask"]
        atmosphere=inputs.atmosphere_at(c["atmosphere"][run],old["elapsed_h"])
        physics=darsia.CO2MassAnalysis(darsia.ScalarImage(np.zeros(shape,np.float32),dimensions=[.49,.90]),
            atmospheric_pressure=atmosphere["pressure_bar"],atmospheric_temperature=atmosphere["temperature_c"],
            atmospheric_pressure_gradient=atmosphere["pressure_gradient_bar_per_m"],atmospheric_temperature_gradient=atmosphere["temperature_gradient_c_per_m"])
        maximum=np.where(labels==2,np.maximum(physics.solubility_co2,physics.density_gaseous_co2),physics.solubility_co2)
        upper=float(np.sum(area[unknown]*.0155*.45*maximum[unknown],dtype=np.float64))*1000
        rows.append(finite_capacity(old,unknown_pixels=int(unknown.sum()),unknown_area_cm2=float(area[unknown].sum())*1e4,unknown_capacity_g=upper))
        del labels,domain,area,unknown,physics,fields;gc.collect()
    driver.commit(folder,dict(contract=CONTRACT,campaign_binding_sha256=c["binding"],rows=rows,
        main_native_histograms_replaced=False,RAW_read=False,heldout_pixels_read=False))
    return dict(run=run,frames=len(rows),cached=False)


def fit_pair(rows):
    """Only the objective changes: old dose/frame/cohort/native gas stay fixed."""
    counts=Counter(r["run"] for r in rows)
    for row in rows:
        if row["run"] not in original.CALIBRATION_RUNS or row["nominal_injected_mass_g"]<=0 or row["mass_balance_missing_capacity_upper_g"] is None:
            raise ValueError("Legacy objective control accepts only finite calibration intervals")
    def pair(theta):
        values=np.maximum(np.exp(np.minimum(theta,np.log(np.finfo(float).max))),np.finfo(float).tiny)
        return dict(aqueous_scale=float(values[0]),gas_scale=float(values[1]))
    def residual(theta):
        p=pair(theta);output=[]
        for row in rows:
            mass=original.evaluate_common_summary(row,p)["total_mass_g"]
            dose=row["nominal_injected_mass_g"];upper=row["mass_balance_missing_capacity_upper_g"]
            signed=max(mass-dose,0.)+min(mass+upper-dose,0.)
            output.append(signed/dose/np.sqrt(counts[row["run"]]))
        return np.asarray(output)
    fits=[least_squares(residual,np.log(start),max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-10) for start in ((1.,1.),(20.,1.))]
    usable=[i for i,f in enumerate(fits) if f.success]
    if not usable:raise ValueError("Legacy objective control optimizer failed")
    selected=min(usable,key=lambda i:(fits[i].cost,i))
    if len(usable)==2 and abs(fits[0].cost-fits[1].cost)<=1e-12+1e-8*max(f.cost for f in fits):selected=0
    fit=fits[selected];singular=np.linalg.svd(fit.jac,compute_uv=False)
    rank=int(np.count_nonzero(singular>max(float(singular.max(initial=0))*1e-6,1e-10)))
    return dict(parameters=pair(fit.x),calibration_runs=sorted(counts),frames_used=[dict(run=r["run"],frame_index=r["frame_index"]) for r in rows],
        source_rows_sha256=out.digest(rows),objective="signed finite [M,M+U] interval distance / own dose; equal run and equal frame weights",
        original_fit_frames_retained=True,U_five_percent_filter_applied=False,dilution="none",native_gas_gain_refitted=True,
        gas_gain_not_fixed_to_round1=True,jacobian_rank=rank,nonunique_parameter_warning=rank<2,
        optimizer_cost=float(fit.cost),optimizer_multistart=[dict(parameters=pair(f.x),cost=float(f.cost),success=bool(f.success)) for f in fits],
        heldout_pixels_read=False,selection_changed=False)


def predict(rows,pair,leftout,kind):
    results=[]
    for row in rows:
        native_result=original.evaluate_common_summary(row,pair)
        mass=native_result["total_mass_g"];dose=row["nominal_injected_mass_g"];gas=native_result["gas_mass_g"]
        results.append(dict(out.mass_row(dict(native_result,run=row["run"],frame_index=row["frame_index"],elapsed_h=row["elapsed_h"],
            injected_mass_g=dose,missing_capacity_upper_g=row["mass_balance_missing_capacity_upper_g"],gas_mass_share=gas/mass if mass else 0.)),
            control=kind,leftout_run=leftout,parameters=pair,source_binding=row["source_binding"],
            scope="Early source-bound calibration-only control; no later-frame conservation or held-out claim",
            gas_treatment="Original native response with a fitted common gas gain and equilibrium gas-pixel water",dilution="none"))
    return results


def fits(c):
    rows=[]
    for run in original.CALIBRATION_RUNS:
        record=driver.resume(c["destination"] / "capacity" / run,c["binding"])
        if record is None:raise ValueError("All27 capacity imports/replays must finish first")
        rows.extend(record["rows"])
    if {(r["run"],r["frame_index"]) for r in rows}!={(r["run"],r["frame_index"]) for r in c["rows"]}:
        raise ValueError("Legacy early frame list changed")
    predictions=[]
    for leftout in original.CALIBRATION_RUNS:
        folder=c["destination"] / "loro" / leftout
        record=driver.resume(folder,c["binding"])
        if record is None:
            fit=fit_pair([r for r in rows if r["run"]!=leftout])
            gate.parameter_freeze(folder.with_name(leftout+".parameters.json"),fit)
            values=predict([r for r in rows if r["run"]==leftout],fit["parameters"],leftout,"objective-only LORO")
            driver.commit(folder,dict(contract=CONTRACT,campaign_binding_sha256=c["binding"],fit=fit,predictions=values))
            record=driver.resume(folder,c["binding"])
        predictions.extend(record["predictions"])
    baseline=predict(rows,c["original_freeze"]["parameters"],None,"Original frozen R1 baseline; in-fit, not LORO")
    report=c["destination"] / ("report_"+uuid.uuid4().hex[:8]);report.mkdir()
    driver.saved.write_csv_new(report / "legacy_objective_early_loro.csv",predictions)
    driver.saved.write_csv_new(report / "original_frozen_baseline_in_fit.csv",baseline)
    driver.new_json(report / "report.json",dict(contract=CONTRACT,binding=inputs.asset(c["destination"] / "binding.json"),
        original_early_frame_count=len(rows),calibration_run_count=27,scope="Early only; no held-out pixels, no full Studio-frame prediction",
        original_histograms_unchanged=True,objective_only_new_fit=True,main_round2_changed=False,
        known_support_capacity_imported=True,unknown_geometry_capacity_replayed=True,RAW_read=False,
        saved_tables={p.name:inputs.asset(p) for p in report.glob("*.csv")},
        fixed_colour_paths_scope="Original pooled paths include all27 calibration runs; LORO is conditional on these fixed paths",
        baseline_scope="Original frozen R1 pair is shown as an in-fit baseline, never labelled an external validation"))
    return dict(report=str(report),prediction_frames=len(predictions))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",type=Path,required=True);parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--round1-job",type=Path,required=True);parser.add_argument("--stage",choices=("capacity","fits"),required=True)
    parser.add_argument("--workers",type=int,default=12);args=parser.parse_args(argv)
    if not 1<=args.workers<=24:parser.error("Use1–24 workers, one internal thread")
    c=setup(args.job,args.destination,args.round1_job)
    if args.stage=="fits":result=fits(c)
    else:
        result=[]
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(str(args.job),str(args.destination),str(args.round1_job))) as pool:
            for future in as_completed([pool.submit(capacity_run,r) for r in original.CALIBRATION_RUNS]):
                value=future.result();result.append(value);print(json.dumps(value),flush=True)
    print(json.dumps(driver.saved._jsonable(result),allow_nan=False),flush=True)


if __name__=="__main__":main()
