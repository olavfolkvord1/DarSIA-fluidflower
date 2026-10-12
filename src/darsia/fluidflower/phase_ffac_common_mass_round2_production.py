"""Saved outputs and the pre-registered FF_AC round-2 selection rules.

The functions in this module never fit a response or change a phase map. Scores
operate only on saved, source-bound predictions. Scientific maps require the
new 26/12/AC38 freeze, and final production requires both held-out checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import uuid

import numpy as np

from . import phase_ffac_common_mass_production as saved

CONTRACT = "ffac-common-mass-round2-production-v1"
FREEZE_CONTRACT = "ffac-common-mass-round2-frozen-v1"
HELDOUT = saved.HELDOUT
EXCLUDED = saved.EXCLUDED
DESCRIPTIVE = frozenset({"ac38"})
CALIBRATION = frozenset(f"ac{i}" for i in range(17, 62) if i != 36 and i != 37) - HELDOUT - EXCLUDED - DESCRIPTIVE
CANDIDATE_PARAMETERS = {"N1": 2, "N2": 2, "N3": 3, "J": 4}
POST_TIMES = (4., 8., 12., 24.)
SCOPE_CAVEAT = ("Conditional on locked P1 final 3 phases, fixed v5 normalisation and the frozen pooled colour paths; "
    "not independent end-to-end validation. Round-1 early held-out ratios were already known but were not used for selection.")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(saved._jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def light_group(run, *, final=False, supplied=None):
    run = str(run).lower()
    if final and run in {"ac21", "ac22", "ac32", "ac44"}:
        return "caveat"
    if supplied in {"rig_light", "new_lamps"}:
        return supplied
    return "new_lamps" if int(run[2:]) >= 50 else "rig_light"


def mass_row(result, metadata=None):
    """Keep the main mass separate from every excluded/unknown contribution."""
    meta = metadata or {}
    native = result.get("native_result", result)
    run = str(native.get("run", meta.get("run"))).lower()
    mass = float(native.get("total_mass_g", native.get("main_mass_g", native.get("M_g", 0.))))
    dose = float(native.get("injected_mass_g", native.get("nominal_injected_mass_g", native.get("dose_g", meta.get("injected_mass_g", 0.)))))
    capacity = native.get("missing_capacity_upper_g", native.get("mass_balance_missing_capacity_upper_g", native.get("U_g")))
    if capacity is None or not np.isfinite(capacity) or capacity < 0:
        raise ValueError("Round 2 requires a finite nonnegative physical capacity bound, including unknown geometry")
    if not np.isfinite(mass) or not np.isfinite(dose) or mass < 0 or dose < 0:
        raise ValueError("Finite nonnegative main mass and dose required")
    ratio = mass / dose if dose > 0 else None
    delta = max(mass - dose, dose - mass - float(capacity), 0.) / dose if dose > 0 else None
    return {**{k: v for k, v in native.items() if k != "fields"}, **meta,
        "run": run, "frame_index": int(native.get("frame_index", meta.get("frame_index", 0))),
        "elapsed_h": float(native.get("elapsed_h", meta.get("elapsed_h", 0.))),
        "main_mass_g": mass, "total_mass_g": mass, "injected_mass_g": dose,
        "missing_capacity_upper_g": float(capacity), "ratio": ratio, "delta": delta,
        "U_over_d": float(capacity) / dose if dose > 0 else None,
        "no_map_mass_rescaling": True, "bottom_mass_allowed": False,
        "conditional_validation_scope": SCOPE_CAVEAT}


def nearest(rows, target):
    if not rows:
        raise ValueError("Missing Studio frames")
    return min(rows, key=lambda r: (abs(float(r["elapsed_h"]) - target), float(r["elapsed_h"]), int(r["frame_index"])))


def _prediction_inventory(rows, expected_runs):
    by_run = {}
    seen = set()
    for row in rows:
        run = row["run"].lower()
        key = run, int(row["frame_index"])
        if key in seen:
            raise ValueError("Duplicate saved prediction")
        seen.add(key)
        if run not in expected_runs:
            raise ValueError("A selection/check prediction comes from the wrong cohort")
        by_run.setdefault(run, []).append(row)
    if set(by_run) != set(expected_runs):
        raise ValueError("Selection/check needs the exact complete run cohort")
    return by_run


def score_predictions(rows, *, final=False):
    """Apply §3 or §4 literally to saved predictions; no parameter adjustment."""
    expected = HELDOUT if final else CALIBRATION
    by_run = _prediction_inventory(rows, expected)
    injection = []
    for target in (1., 2.25):
        chosen = [nearest(run_rows, target) for run_rows in by_run.values()]
        if any(r.get("ratio") is None for r in chosen):
            raise ValueError("A required injection check has no positive source-bound dose")
        medians = {}
        for group in (("rig_light", "new_lamps", "caveat") if final else ("rig_light", "new_lamps")):
            values = [r["ratio"] for r in chosen if light_group(r["run"], final=final, supplied=r.get("light_group")) == group]
            if not values:
                raise ValueError("A pre-registered light group is absent")
            medians[group] = statistics.median(values)
        error = statistics.median(abs(r["ratio"] - 1.) for r in chosen)
        injection.append(dict(target_h=target, run_count=len(chosen), median_absolute_ratio_error=error,
            median_ratio_by_group=medians, pass_gate=error <= .10 and all(.90 <= x <= 1.10 for x in medians.values())))
    post = []
    minimum = 7 if final else 13
    for target in POST_TIMES:
        chosen = [nearest(run_rows, target) for run_rows in by_run.values()]
        eligible = [r for r in chosen if r.get("U_over_d") is not None and r["U_over_d"] <= .05]
        assessable = len(eligible) >= minimum
        value = statistics.median(r["delta"] for r in eligible) if assessable else None
        post.append(dict(target_h=target, required_run_count=minimum, eligible_run_count=len(eligible),
            eligible_runs=sorted(r["run"] for r in eligible), assessable=assessable,
            median_interval_distance=value, pass_gate=(value <= .10) if assessable else None))
    e_inj = max(r["median_absolute_ratio_error"] for r in injection)
    assessable_values = [r["median_interval_distance"] for r in post if r["assessable"]]
    e_post = max(assessable_values) if assessable_values else None
    injection_pass = all(r["pass_gate"] for r in injection)
    post_pass = all(r["pass_gate"] for r in post if r["assessable"])
    passes = injection_pass and post_pass
    outcome = "fail" if not passes else "requires_Olav_decision" if final and not assessable_values else "pass"
    return dict(contract=CONTRACT, final=final, run_count=len(by_run), E_inj=e_inj, E_post=e_post,
        S=max(e_inj, e_post) if e_post is not None else e_inj, injection=injection, conservation=post,
        post_assessable=bool(assessable_values), pass_gate=passes, outcome=outcome,
        full_production_allowed=final and outcome == "pass", parameters_changed=False,
        conditional_validation_scope=SCOPE_CAVEAT)


def choose_candidate(scores):
    """§3 fixed native preference/tie/J-stop policy."""
    by_name = {row["candidate"]: row for row in scores}
    if not {"N1", "N2", "J"} <= set(by_name) or set(by_name) - set(CANDIDATE_PARAMETERS):
        raise ValueError("Selection requires N1, N2 and J; N3 only if native blend exists")
    native = [r for name, r in by_name.items() if name != "J" and r["pass_gate"]]
    chosen = None
    if native:
        best = min(float(r["S"]) for r in native)
        tied = [r for r in native if float(r["S"]) <= best + .005 + 1e-12]
        chosen = min(tied, key=lambda r: (CANDIDATE_PARAMETERS[r["candidate"]], r["candidate"] != "N1", r["candidate"]))
    joint = by_name["J"]
    if joint["pass_gate"] and (chosen is None or joint["S"] < chosen["S"] - .03):
        return dict(outcome="stop_for_Olav_J_choice", chosen_candidate=None, proposed_native=chosen["candidate"] if chosen else None,
            reason="J passes and no native passes or J is more than 0.03 better", freeze_allowed=False)
    if chosen is None:
        return dict(outcome="stop_no_candidate_passes", chosen_candidate=None, freeze_allowed=False)
    return dict(outcome="native_selected", chosen_candidate=chosen["candidate"], S=chosen["S"],
        freeze_allowed=True, selection_uses_heldout=False, tie_tolerance=.005)


@dataclass(frozen=True)
class FrozenEvidence:
    path: Path
    sha256: str
    record: dict

    @classmethod
    def verify(cls, path):
        path = Path(path).resolve()
        data = read_json(path)
        if data.get("contract") != FREEZE_CONTRACT or data.get("frozen") is not True or not data.get("frozen_utc"):
            raise ValueError("New round-2 freeze required before out-of-fit pixels")
        if set(data.get("calibration_runs", ())) != CALIBRATION or set(data.get("heldout_runs", ())) != HELDOUT or set(data.get("descriptive_runs", ())) != DESCRIPTIVE:
            raise ValueError("Round-2 freeze needs the exact 26/12/AC38 split")
        if data.get("map_mass_rescaling") is not False or data.get("overflow_in_objective") is not False:
            raise ValueError("A freeze cannot permit map rescaling or overflow fitting")
        files = data.get("calibration_files", {})
        if not files:
            raise ValueError("Frozen numerical source inventory required")
        for name, sha in files.items():
            asset = Path(name)
            if not asset.is_absolute():
                asset = path.parent / asset
            if saved.sha_file(asset) != sha:
                raise ValueError(f"Frozen calibration bytes changed: {asset}")
        return cls(path, saved.sha_file(path), data)

    def check_frame_run(self, run):
        run = str(run).lower()
        if run not in CALIBRATION | HELDOUT | DESCRIPTIVE:
            raise ValueError("Scientific output outside the frozen 39-run cohort")
        return run


def save_frame(folder, result, metadata, freeze, *, render=True):
    """Write full float32 maps once, with fixed support and SHA provenance."""
    from PIL import Image
    run = freeze.check_frame_run(metadata["run"])
    fields = result["fields"]
    caq = np.asarray(fields["c_aq_g_l"], np.float32)
    sg = np.asarray(fields["s_g"], np.float32)
    phase = np.asarray(fields["phase_labels"])
    unknown = np.asarray(fields["unquantified_phase_mask"], bool)
    if caq.ndim != 2 or not all(x.shape == caq.shape for x in (sg, phase, unknown)):
        raise ValueError("Full-resolution field geometry differs")
    if not np.isin(phase,(0,1,2,255)).all():
        raise ValueError("Scientific fields require the frozen categorical phase contract")
    if np.isinf(caq).any() or np.isinf(sg).any() or np.any(caq[(phase == 0) | (phase == 255)] != 0) or np.any(sg[phase != 2] != 0):
        raise ValueError("Scientific fields escaped unchanged phase support")
    if np.any((~np.isfinite(caq) | ~np.isfinite(sg)) & ~unknown) or np.any(caq[np.isfinite(caq)] < 0) or np.any((sg[np.isfinite(sg)] < 0) | (sg[np.isfinite(sg)] > 1)):
        raise ValueError("Invalid concentration/saturation outside declared unknown support")
    target = Path(folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    staging.mkdir()
    assets = {"c_aq_g_l": saved._save_array(staging / "c_aq_g_l.npy", caq), "s_g": saved._save_array(staging / "s_g.npy", sg),
        "unquantified_phase_mask": saved._save_array(staging / "unquantified_phase_mask.packbits.npy", np.packbits(unknown.ravel()))}
    if render:
        for name, values, maximum in (("c_aq_g_l", caq, 2.), ("s_g", sg, 1.)):
            display = staging / f"{name}.png"
            Image.fromarray(saved._display_rgb(values, maximum=maximum, unquantified=unknown)).save(display, compress_level=1)
            assets[name + "_display"] = dict(path=display.name, sha256=saved.sha_file(display), display_range=[0., maximum])
    row = mass_row(result, metadata)
    receipt = dict(row, contract=CONTRACT, run=run, complete=True, assets=assets, shape_hw=list(caq.shape),
        created_utc=datetime.now(timezone.utc).isoformat(), calibration_freeze_path=str(freeze.path),
        calibration_freeze_sha256=freeze.sha256, phase_parent_changed=False, default_changed=False,
        producer_sha256=saved.sha_file(__file__), backend="CPU native DarSIA; float64 physics, float32 saved maps",
        native_result={k: v for k, v in result.items() if k != "fields"})
    saved.write_new_json(staging / "receipt.json", receipt)
    staging.rename(target)
    return receipt


def write_tables(folder, rows, *, sensitivities=(), stage="selection"):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    saved.write_csv_new(folder / "mass_per_frame.csv", rows)
    for run in sorted({r["run"] for r in rows}):
        saved.write_csv_new(folder / f"{run}_mass.csv", [r for r in rows if r["run"] == run])
    saved.write_csv_new(folder / "sensitivities.csv", sensitivities)
    saved.write_new_json(folder / "report.json", dict(contract=CONTRACT, stage=stage, saved_frame_count=len(rows),
        saved_run_count=len({r["run"] for r in rows}), bottom_mass_allowed=False, no_map_mass_rescaling=True,
        approved=False, default_changed=False, conditional_validation_scope=SCOPE_CAVEAT))
    return folder
