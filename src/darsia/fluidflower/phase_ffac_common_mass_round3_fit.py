"""Selected-image N1 gain/beta fit and literal grid audits for FF_AC round 3.

No final held-out image enters this module. The beta sample set is frozen at
the fit's stage-one gain and round-two beta before optimization. Acceptance
later uses the actual interval width of each evaluated calibration.
"""
from __future__ import annotations

from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import least_squares

from . import phase_ffac_common_mass_round3 as r3


def _gate(value):
    policy_a = isinstance(value,dict) and value.get("contract") == "ffac-round3-policy-a-claude-pass-admission-v1"
    if (not isinstance(value,dict) or value.get("all_39_complete") is not True
            or not (value.get("round3_allowed") is True and value.get("claude_pass_verified") is True
                if policy_a else value.get("studio_time_parity_passed") is True)
            or not value.get("source_binding")):
        raise ValueError("Fitting starts only after all 39 selected-image phase manifests and Studio-time parity are sealed")


def _validate_rows(rows):
    if not rows or any(row.get("run") not in r3.FIT_RUNS for row in rows):
        raise ValueError("Only the 26 pre-registered calibration runs enter the selected-image fit")
    identifiers = [(row["run"],row["frame_index"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate source frames cannot receive repeated calibration weight")
    if any(row.get("selected_target_h") is None or not row.get("source_binding") for row in rows):
        raise ValueError("Each selected image needs its manifest target and exact source binding")


def anchors_by_run(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["run"]].append(row)
    anchors = {}
    for run,values in groups.items():
        # Preserve the round-two end-of-injection definition. The selected
        # target binds the source; camera elapsed time remains the dilution time.
        anchor = min(values,key=lambda row:(abs(row["elapsed_h"]-2.247),row["frame_index"]))
        if abs(anchor["elapsed_h"]-2.247)>.13 or anchor["affected_pore_volume_m3"]<=0:
            raise ValueError(f"No source-bound injection-end volume anchor for {run}")
        anchors[run] = {key:anchor[key] for key in ("elapsed_h","affected_pore_volume_m3","frame_index","source_binding")}
    return anchors


def post_time_bin(target_h):
    if target_h<2.5:
        return None
    return "2.5-8h" if target_h<=8 else "8-48h" if target_h<=48 else "after48h"


def frame_weights(rows, *, post=False):
    counts = Counter(row["run"] for row in rows)
    if not counts:
        return np.zeros(0)
    if not post:
        return np.array([1/np.sqrt(len(counts)*counts[row["run"]]) for row in rows],np.float64)
    bins = defaultdict(set)
    bin_counts = Counter((row["run"],post_time_bin(row["selected_target_h"])) for row in rows)
    for run,time_bin in bin_counts:
        bins[run].add(time_bin)
    return np.array([1/np.sqrt(len(counts)*len(bins[row["run"]])*
        bin_counts[(row["run"],post_time_bin(row["selected_target_h"]))]) for row in rows],np.float64)


def select_round3_fit_frames(rows, *, excluded_run=None, stage1_gain=None, anchors=None):
    """Use exact selected target times and source-recorded rejection/no-loss."""
    _validate_rows(rows)
    if excluded_run is not None and excluded_run not in r3.FIT_RUNS:
        raise ValueError("A leave-one-run-out fold must omit one of the 26 fit runs")
    stages,decisions = {"stage1":[],"stage2":[]},[]
    for row in rows:
        stage = "stage1" if .5<=row["selected_target_h"]<=2.375 else "stage2" if row["selected_target_h"]>=2.5 else None
        if stage is None:
            continue
        reasons = []
        if row["run"]==excluded_run:
            reasons.append("leave_one_run_out")
        frame_list = row.get("frame_list_evidence",{})
        if frame_list.get("accepted") is not True or not frame_list.get("source_binding"):
            reasons.append("frame_list_rejected_or_unbound")
        no_loss = row.get("no_outflow_evidence",{})
        if no_loss.get("no_co2_left_rig") is not True or not (no_loss.get("source") or no_loss.get("source_binding")):
            reasons.append("no_loss_proxy_not_passed_or_unbound")
        dose = row.get("injected_mass_g")
        if dose is None or not np.isfinite(dose) or dose<=0:
            reasons.append("no_positive_dose")
        if row.get("urest_g") is None:
            reasons.append("estimate_geometry_undefined")
        baseline = None
        if stage=="stage2":
            if stage1_gain is None or anchors is None:
                reasons.append("stage1_not_yet_frozen")
            else:
                compiled = r3.prepare_round3_beta_summary(row,stage1_gain)
                baseline = r3.evaluate_round3_compiled(compiled,r3.B0_BETA,anchors[row["run"]])
                if baseline["W_over_d"] is None or baseline["W_over_d"]>.05:
                    reasons.append("baseline_width_exceeds_five_percent")
        if not reasons:
            stages[stage].append(row)
        decisions.append(dict(run=row["run"],frame_index=row["frame_index"],selected_target_h=row["selected_target_h"],
            elapsed_h=row["elapsed_h"],stage=stage,included=not reasons,omitted_reasons=reasons,
            baseline_eligibility_gain=stage1_gain if stage=="stage2" else None,
            baseline_eligibility_beta=r3.B0_BETA if stage=="stage2" else None,
            baseline_W_over_d=None if baseline is None else baseline["W_over_d"],source_binding=row["source_binding"]))
    return stages,decisions


def _multistart(function, starts, bounds):
    results = [least_squares(function,[start],bounds=bounds,max_nfev=600,ftol=1e-10,xtol=1e-10,gtol=1e-10) for start in starts]
    finite = [result for result in results if result.success and np.isfinite(result.cost)]
    if not finite:
        raise ValueError("Native common least-squares fit failed")
    minimum = min(result.cost for result in finite)
    chosen = next(result for result in finite if result.cost<=minimum+max(1e-12,abs(minimum)*1e-8))
    return chosen,dict(multistart=[dict(initial=float(start),solution=result.x.tolist(),cost=float(result.cost),
        success=bool(result.success),nfev=int(result.nfev)) for start,result in zip(starts,results)],
        jacobian_rank=int(np.linalg.matrix_rank(chosen.jac)),jacobian_singular_values=np.linalg.svd(chosen.jac,compute_uv=False).tolist())


def audit_grid(function, fitted_coordinate, *, kind):
    """A wide profile plus a fine local profile, using the identical residual."""
    if kind=="gain":
        values = np.unique(np.r_[np.geomspace(1e-4,1e5,901),
            np.geomspace(max(fitted_coordinate*.5,1e-8),fitted_coordinate*2,401),fitted_coordinate])
        coordinates = np.log(values)
    elif kind=="beta":
        values = np.unique(np.r_[np.linspace(0,4,801),np.linspace(4,16,241),np.linspace(16,64,97),
            [128.,256.],np.linspace(max(0,fitted_coordinate-.25),fitted_coordinate+.25,401),fitted_coordinate])
        coordinates = values
    else:
        raise ValueError("Only the pre-registered gain/beta profile is defined")
    objectives = np.array([.5*float(np.dot(value:=function([coordinate]),value)) for coordinate in coordinates])
    fitted_value = np.log(fitted_coordinate) if kind=="gain" else fitted_coordinate
    residual = function([fitted_value])
    fitted_objective = .5*float(np.dot(residual,residual))
    index = int(np.argmin(objectives))
    tolerance = max(1e-12,abs(fitted_objective)*1e-8)
    superior = objectives[index]<fitted_objective-tolerance
    return dict(parameter=kind,values=values.tolist(),objectives=objectives.tolist(),
        fitted_value=float(fitted_coordinate),fitted_objective=fitted_objective,
        grid_minimum_value=float(values[index]),grid_minimum_objective=float(objectives[index]),
        grid_finds_lower_objective=bool(superior),stop_required=bool(superior),
        numerical_comparison_tolerance=tolerance,
        same_residual_and_weights=True,shape_profile_wide_and_local=True)


def fit_round3_candidate(rows, *, selected_phase_gate, excluded_run=None):
    """Fit N1 gain, freeze the late sample set, then beta only; no held-out."""
    _gate(selected_phase_gate)
    _validate_rows(rows)
    anchors = anchors_by_run(rows)
    first,_ = select_round3_fit_frames(rows,excluded_run=excluded_run)
    first = first["stage1"]
    if not first:
        raise ValueError("No source-bound selected injection frames are eligible")
    weights1 = frame_weights(first)
    def residual1(vector):
        gain = float(np.exp(vector[0]))
        values = [r3.evaluate_round3_compiled(r3.prepare_round3_beta_summary(row,gain),0.,anchors[row["run"]])["signed_interval_residual"] for row in first]
        return weights1*np.array(values,np.float64)
    result1,audit1 = _multistart(residual1,[0.,np.log(20.),np.log(r3.GAIN)],([-700.],[700.]))
    gain = float(np.exp(result1.x[0]))
    gain_grid = audit_grid(residual1,gain,kind="gain")
    stages,decisions = select_round3_fit_frames(rows,excluded_run=excluded_run,stage1_gain=gain,anchors=anchors)
    second = stages["stage2"]
    if gain_grid["stop_required"]:
        return dict(contract=r3.CONTRACT,stop_required=True,reason="gain_grid_lower_objective",
            parameters=dict(candidate="N1",gain=gain,beta=r3.B0_BETA),grid_audits=dict(stage1=gain_grid),
            optimizer_audits=dict(stage1=audit1),frame_decisions=decisions,anchors_by_run=anchors)
    if not second:
        return dict(contract=r3.CONTRACT,stop_required=True,reason="no_eligible_selected_late_frames",
            parameters=dict(candidate="N1",gain=gain,beta=r3.B0_BETA),grid_audits=dict(stage1=gain_grid),
            optimizer_audits=dict(stage1=audit1),frame_decisions=decisions,anchors_by_run=anchors)
    compiled = [r3.prepare_round3_beta_summary(row,gain) for row in second]
    weights2 = frame_weights(second,post=True)
    def residual2(vector):
        beta = float(vector[0])
        return weights2*np.array([r3.evaluate_round3_compiled(value,beta,anchors[value["row"]["run"]])["signed_interval_residual"] for value in compiled],np.float64)
    result2,audit2 = _multistart(residual2,[r3.B0_BETA,0.,1.,3.],([0.],[np.inf]))
    beta = float(result2.x[0])
    beta_grid = audit_grid(residual2,beta,kind="beta")
    for decision in decisions:
        if decision["included"] and decision["stage"]=="stage2":
            row = next(row for row in second if row["run"]==decision["run"] and row["frame_index"]==decision["frame_index"])
            final = r3.evaluate_round3_compiled(r3.prepare_round3_beta_summary(row,gain),beta,anchors[row["run"]])
            decision["fitted_beta_W_over_d"] = final["W_over_d"]
    return dict(contract=r3.CONTRACT,parameters=dict(candidate="N1",gain=gain,beta=beta),
        calibration_runs=[run for run in r3.FIT_RUNS if run!=excluded_run],heldout_fit_run=excluded_run,
        anchors_by_run=anchors,frame_decisions=decisions,
        stages={name:dict(frames=len(values),runs=sorted({row["run"] for row in values}),
            targets=dict(Counter(str(row["selected_target_h"]) for row in values))) for name,values in stages.items()},
        grid_audits=dict(stage1=gain_grid,stage2=beta_grid),optimizer_audits=dict(stage1=audit1,stage2=audit2),
        stop_required=beta_grid["stop_required"],reason="beta_grid_lower_objective" if beta_grid["stop_required"] else None,
        stage2_eligibility_policy="freeze at this fit's stage-one gain and B0 beta before beta optimization",
        equal_run_weight=True,equal_occupied_time_bin_weight=True,equal_frame_weight_within_bin=True,
        post_time_bins=["[2.5,8]","(8,48]","(48,last selected time]"],
        gas_branch_fitted=False,colour_paths_refitted=False,map_mass_rescaling=False,overflow_in_objective=False,
        final_heldout_used=False,selected_phase_gate=selected_phase_gate)
