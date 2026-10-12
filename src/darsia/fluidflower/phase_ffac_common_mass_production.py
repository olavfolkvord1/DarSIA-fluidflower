"""Immutable full-resolution outputs and reports for the common FF_AC fit.

This module applies no fitting, mass normalisation, phase rule or time dilution.
The native evaluator owns the physics. A verified frozen calibration is required
before this module invokes it, including for its diagnostic sensitivities.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Mapping
import uuid

import numpy as np

CONTRACT = "ffac-common-native-mass-production-v1"
EXCLUDED = frozenset({"ac20", "ac24", "ac27", "ac51"})
HELDOUT = frozenset({"ac19", "ac25", "ac35", "ac41", "ac47", "ac52", "ac56", "ac60", "ac21", "ac22", "ac32", "ac44"})
SENSITIVITY_TIMES = (1., 2.25, 4., 12., 24.)
CAVEATS = {
    "ac21": "Lower pH or acid residue after AC20.",
    "ac22": "Lower pH or acid residue after AC20; acid at the bottom documented.",
    "ac32": "Protocol BTB and NaOH amounts are probably swapped; no independent weighing log.",
    "ac44": "Protocol gives 58% BTB, while the photographs look standard.",
    "ac18": "Wall carryover can be labelled CO2 by the locked phase map.",
    "ac52": "Flickering new lamp; corrected images are measured inputs.",
}
PHASE_CAVEAT = ("Locked P1 final 3 phase uncertainty retained: L1 port false gas and missed gas, "
    "overflow-calibrated new-lamp gas shift, AC18 wall carryover and carryover-coloured floor gaps. "
    "Bottom strip and pixels filled below the photograph are excluded from the main mass. "
    "Unquantified facies are a physical capacity bound, not a confidence interval.")


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True)
class FrozenCalibrationEvidence:
    path: Path
    sha256: str
    record: dict

    @classmethod
    def verify(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data.get("frozen") is not True or not data.get("frozen_utc"):
            raise ValueError("Common calibration must be frozen before any production evaluation")
        files = data.get("calibration_files")
        calibration = {str(x).lower() for x in data.get("calibration_runs", ())}
        heldout = {str(x).lower() for x in data.get("heldout_runs", ())}
        if not files or len(calibration) != 27 or heldout != HELDOUT or calibration & heldout or (calibration | heldout) & EXCLUDED:
            raise ValueError("Frozen calibration lacks the exact 27/12 independent run split")
        for name, expected in files.items():
            asset = Path(name)
            if not asset.is_absolute():
                asset = path.parent / asset
            if sha_file(asset) != expected:
                raise ValueError(f"Frozen common calibration file changed: {asset}")
        scales = data.get("shared_scales") or {}
        if set(scales) != {"aqueous_scale", "gas_scale"} or any(not np.isfinite(x) or x <= 0 for x in scales.values()):
            raise ValueError("Exactly one positive common aqueous/gas scale pair is required")
        return cls(path, sha_file(path), data)

    def check_frame_run(self, run):
        run = str(run).lower()
        allowed = set(self.record["calibration_runs"]) | set(self.record["heldout_runs"])
        allowed = {str(x).lower() for x in allowed}
        if run not in allowed or run in EXCLUDED:
            raise ValueError(f"Run is outside the frozen common-calibration cohort: {run}")
        return run


def report_identity(metadata, freeze):
    run = freeze.check_frame_run(metadata["run"])
    elapsed = float(metadata["elapsed_h"])
    if not np.isfinite(elapsed):
        raise ValueError("Finite source-bound camera time required")
    return dict(contract=CONTRACT, run=run, frame_index=int(metadata["frame_index"]),
        elapsed_h=elapsed, heldout=run in HELDOUT,
        light_group=("caveat" if run in {"ac21", "ac22", "ac32", "ac44"} else "new_lamps" if int(run[2:]) >= 50 else "rig_light"),
        era=("AC17–38" if int(run[2:]) <= 38 else "AC39–49" if int(run[2:]) <= 49 else "AC50–61"),
        run_caveat=CAVEATS.get(run, "No additional protocol caveat in the common-fit specification."),
        phase_caveat=PHASE_CAVEAT, calibration_freeze_sha256=freeze.sha256,
        calibration_freeze_path=str(freeze.path), map_rescaled_to_injected_mass=False,
        bottom_mass_allowed=False, default_candidate=False, **{
            key: metadata.get(key) for key in ("source_sha256", "phase_sha256", "pixel_area_sha256", "nominal_injected_mass_g", "pressure_bar", "temperature_c", "overflow")})


class _HashedOutput:
    """Hash NumPy writes in memory; do not re-read large arrays from the NAS."""
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
    def write(self, values):
        self.digest.update(values)
        return self.stream.write(values)
    def flush(self):
        return self.stream.flush()
    def tell(self):
        return self.stream.tell()


def _save_array(path, values):
    values = np.asarray(values)
    with Path(path).open("xb") as stream:
        hashed = _HashedOutput(stream)
        np.save(hashed, values, allow_pickle=False)
        hashed.flush()
    return dict(path=Path(path).name, sha256=hashed.digest.hexdigest(), shape=list(values.shape), dtype=str(values.dtype))


def _display_rgb(values, *, maximum, unquantified=None):
    """A fixed scale for every frame; this never edits the scientific array."""
    import matplotlib
    array = np.asarray(values, np.float32)
    masked = ~np.isfinite(array)
    if unquantified is not None:
        masked |= np.asarray(unquantified, bool)
    palette = (matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[..., :3] * 255).astype(np.uint8)
    codes = np.rint(np.clip(np.nan_to_num(array) / maximum, 0, 1) * 255).astype(np.uint8)
    rgb = palette[codes]
    rgb[masked] = (160, 160, 160)
    return rgb


def save_frame_result(folder, result, metadata, freeze, *, render=True):
    """Publish a complete frame once; refuse replacement of any existing frame."""
    from PIL import Image
    started = time.perf_counter()
    identity = report_identity(metadata, freeze)
    fields = result.get("fields") or {}
    caq = np.asarray(fields.get("c_aq_g_l", fields.get("concentration_g_l")), np.float32)
    sg = np.asarray(fields.get("s_g", fields.get("gas_saturation")), np.float32)
    phase = np.asarray(fields.get("phase_labels", metadata.get("phase_labels")))
    unknown = np.asarray(fields.get("unquantified_phase_mask", np.zeros(caq.shape, bool)), bool)
    if caq.ndim != 2 or sg.shape != caq.shape or phase.shape != caq.shape or unknown.shape != caq.shape:
        raise ValueError("Full-resolution concentration, saturation and phase grids must agree")
    if not np.isin(phase, [0, 1, 2, 255]).all() or np.isinf(caq).any() or np.isinf(sg).any():
        raise ValueError("Invalid phase labels or infinite native concentration")
    if np.any(caq[(phase == 0) | (phase == 255)] != 0) or np.any(sg[phase != 2] != 0):
        raise ValueError("Concentration or gas saturation escaped the fixed phase support")
    if np.any((sg[np.isfinite(sg)] < 0) | (sg[np.isfinite(sg)] > 1)) or np.any(caq[np.isfinite(caq)] < 0):
        raise ValueError("Native fields violate concentration/saturation bounds")
    if np.any((~np.isfinite(caq) | ~np.isfinite(sg)) & ~unknown):
        raise ValueError("Non-finite concentration outside explicitly unquantified facies")
    target = Path(folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Never overwrite a production frame: {target}")
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    staging.mkdir(exist_ok=False)
    assets = {"c_aq_g_l": _save_array(staging / "c_aq_g_l.npy", caq), "s_g": _save_array(staging / "s_g.npy", sg)}
    assets["unquantified_phase_mask"] = _save_array(staging / "unquantified_phase_mask.packbits.npy", np.packbits(unknown.ravel()))
    if render:
        for name, values, maximum in (("c_aq_g_l", caq, 2.), ("s_g", sg, 1.)):
            image_path = staging / (name + ".png")
            Image.fromarray(_display_rgb(values, maximum=maximum, unquantified=unknown)).save(image_path, compress_level=1)
            assets[name + "_display"] = dict(path=image_path.name, sha256=sha_file(image_path), display_range=[0., maximum], scientific_values_unchanged=True)
    summaries = {k: v for k, v in result.items() if k != "fields"}
    receipt = {**identity, "native_result": summaries, "assets": assets, "shape_hw": list(caq.shape),
        "frame_folder": str(target.resolve()),
        "producer_sha256": sha_file(__file__),
        "complete": True, "backend": "native DarSIA / CPU", "concentration_units": "g/L", "mass_units": "g",
        "created_utc": datetime.now(timezone.utc).isoformat(), "write_seconds": time.perf_counter() - started}
    write_new_json(staging / "receipt.json", receipt)
    staging.rename(target)
    return receipt


def evaluate_sensitivities(frame, frozen_package, evaluator: Callable, *, heldout=False, injector_verified=False,
                          provisional_l1_source_bound=False):
    """Evaluate only fixed alternatives, never refit or reassign production phases."""
    variants = (("titration_equilibrium", {}), ("linear_equilibrium", {"flash_kind": "linear"}),
        ("titration_native_gas_water", {"gas_equilibrium": False}))
    if heldout:
        variants += (("titration_unconstrained", {"phase_constraint": False}),)
    if float(frame.time_hours) >= 4. and injector_verified:
        variants += (("l1_gas_as_aqueous", {"port_as_aqueous": True}),)
    elif float(frame.time_hours) >= 4. and provisional_l1_source_bound:
        variants += (("l1_protocol_proxy_gas_as_aqueous", {"port_as_aqueous": True, "allow_provisional_l1_sensitivity": True}),)
    output = {name: {k: v for k, v in evaluator(frame, frozen_package, fields=False, **kwargs).items() if k != "fields"}
              for name, kwargs in variants}
    output["l1_port_sensitivity_status"] = ("not_applicable_before_4h" if float(frame.time_hours) < 4. else "evaluated" if injector_verified else "not_evaluated_injector_unverified")
    if provisional_l1_source_bound and not injector_verified and float(frame.time_hours) >= 4.:
        output["l1_protocol_proxy_sensitivity_status"] = "evaluated_provisional_source_bound_protocol_proxy_not_verified_port"
        output["provisional_l1_location"] = True
        output["l1_location_evidence"] = {key: value for key, value in getattr(frame, "provenance", {}).items() if key.startswith("l1_")}
    output["all_variants_share_frozen_calibration"] = True
    output["map_rescaled_to_injected_mass"] = False
    return output


def nearest_frame_indices(rows, targets=SENSITIVITY_TIMES, *, include_last=True):
    ordered = sorted(rows, key=lambda row: (float(row["elapsed_h"]), int(row["frame_index"])))
    if not ordered:
        return []
    selected = []
    for target in targets:
        row = min(ordered, key=lambda row: (abs(float(row["elapsed_h"]) - target), float(row["elapsed_h"]), int(row["frame_index"])))
        selected.append(dict(target_h=target, frame_index=int(row["frame_index"]), elapsed_h=float(row["elapsed_h"]), selection="nearest_camera_time_earlier_tie"))
    if include_last:
        selected.append(dict(target_h="last", frame_index=int(ordered[-1]["frame_index"]), elapsed_h=float(ordered[-1]["elapsed_h"]), selection="last_camera_time"))
    return selected


def _mass(row):
    result = row.get("native_result", row)
    for name in ("total_mass_g", "main_total_mass_g"):
        if name in result:
            return float(result[name])
    main = result.get("main_mass") or result.get("main")
    if main:
        return float(main.get("total_mass_g", main.get("total_g")))
    raise ValueError("Frame lacks a main eligible native mass")


def heldout_check1(rows):
    """Pre-registered descriptive gate; never derives a correction or fit."""
    runs = {}
    for row in rows:
        runs.setdefault(str(row["run"]).lower(), []).append(row)
    if not HELDOUT <= set(runs):
        raise ValueError("Check 1 needs all twelve frozen held-out runs")
    inputs = []
    for run in sorted(HELDOUT):
        for selection in nearest_frame_indices(runs[run], (1., 2.25), include_last=False):
            row = next(r for r in runs[run] if int(r["frame_index"]) == selection["frame_index"])
            nominal = row.get("nominal_injected_mass_g")
            if nominal is None or not np.isfinite(nominal) or nominal <= 0:
                raise ValueError(f"Missing positive measured injected mass: {run} {selection}")
            overflow = row.get("overflow") or {}
            needs = overflow.get("co2_has_left_rig")
            outlet = overflow.get("cumulative_co2_outlet_volume_ml")
            density = overflow.get("outlet_gas_density_g_l")
            source_bound = bool(overflow.get("source_sha256"))
            correction = 0.
            no_loss_status = overflow.get("no_loss_status", "explicit_no_outlet_statement")
            status = ("no_outlet_correction_" + no_loss_status) if needs is False else "outlet_status_unknown"
            valid = needs is False
            if needs is True:
                if outlet is not None and density is not None and source_bound and min(outlet, density) >= 0:
                    correction = float(outlet) / 1000 * float(density)
                    valid, status = True, "measured_cumulative_CO2_outlet_added"
                else:
                    status = "CO2_left_rig_but_source_bound_outlet_volume_or_density_missing"
            # Overflow displacement of water is an independent rig gas-volume
            # observation, not automatically CO2 which has escaped the rig.
            main = _mass(row)
            ratio = (main + correction) / float(nominal)
            inputs.append({**{k: row.get(k) for k in ("run", "frame_index", "elapsed_h", "light_group", "era", "run_caveat", "phase_caveat", "pressure_bar", "temperature_c", "source_sha256", "phase_sha256")},
                **selection, "main_total_mass_g": main, "nominal_injected_mass_g": float(nominal),
                "overflow_outlet_correction_g": correction, "overflow_status": status,
                "no_loss_status": no_loss_status, "no_loss_source": overflow.get("no_loss_source"),
                "outlet_source_sha256": overflow.get("source_sha256"), "ratio": ratio,
                "ratio_valid_for_preregistered_gate": valid})
    by_time = []
    complete = all(row["ratio_valid_for_preregistered_gate"] for row in inputs)
    for target in (1., 2.25):
        selected = [row for row in inputs if row["target_h"] == target]
        mederr = float(np.median([abs(row["ratio"] - 1) for row in selected]))
        groups = {name: float(np.median([row["ratio"] for row in selected if row["light_group"] == name])) for name in ("rig_light", "new_lamps", "caveat")}
        passed = mederr <= .10 and all(.9 <= v <= 1.1 for v in groups.values())
        by_time.append(dict(target_h=target, median_absolute_ratio_error=mederr, median_ratio_by_group=groups, numerical_pass=passed))
    return dict(check="specification_section5_check1", inputs=inputs, per_time=by_time,
        gate_inputs_complete=complete, all_pass=complete and all(x["numerical_pass"] for x in by_time),
        outcome="pass" if complete and all(x["numerical_pass"] for x in by_time) else "fail" if complete else "not_assessable_missing_outlet_evidence",
        calibration_changed=False, followup_policy="stop_and_report_on_failure_no_per_run_or_group_correction",
        interpretation="Conditional on locked P1 final 3 phase support; Claude independently scores these frozen inputs once.")


def write_csv_new(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(_jsonable(v), sort_keys=True) if isinstance(v, (dict, list, tuple)) else _jsonable(v) for k, v in row.items()})


def comparison_ratios(rows):
    """Same untouched 1/2.25 h main-mass ratios for both cohorts, by era."""
    by_run = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row)
    selected = []
    for run, run_rows in sorted(by_run.items()):
        for choice in nearest_frame_indices(run_rows, (1., 2.25), include_last=False):
            row = next(r for r in run_rows if r["frame_index"] == choice["frame_index"])
            dose = row.get("nominal_injected_mass_g")
            selected.append({**{key: row.get(key) for key in ("run", "frame_index", "elapsed_h", "era", "heldout", "light_group", "run_caveat", "phase_caveat", "pressure_bar", "temperature_c")},
                **choice, "main_total_mass_g": _mass(row), "nominal_injected_mass_g": dose,
                "raw_main_mass_ratio": _mass(row) / dose if dose is not None and dose > 0 else None,
                "overflow": row.get("overflow"), "map_rescaled_to_injected_mass": False})
    aggregates = []
    for target in (1., 2.25):
        for heldout in (False, True):
            for era in ("AC17–38", "AC39–49", "AC50–61"):
                group = [r for r in selected if r["target_h"] == target and r["heldout"] is heldout and r["era"] == era and r["raw_main_mass_ratio"] is not None]
                if group:
                    aggregates.append(dict(target_h=target, heldout=heldout, era=era, run_count=len(group),
                        median_raw_main_mass_ratio=float(np.median([r["raw_main_mass_ratio"] for r in group])),
                        median_absolute_raw_ratio_error=float(np.median([abs(r["raw_main_mass_ratio"] - 1) for r in group])),
                        run_caveats={r["run"]: r["run_caveat"] for r in group}, phase_caveat=PHASE_CAVEAT,
                        interpretation="Raw main-mass comparison; source-bound outlet additions are shown separately in check 1"))
    return selected, aggregates


def sanity_diagnostics(rows):
    by_run = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row)
    diagnostics = []
    for run, timeline in sorted(by_run.items()):
        timeline = sorted(timeline, key=lambda r: r["elapsed_h"])
        anchor = min(timeline, key=lambda r: (abs(r["elapsed_h"] - 2.389), r["elapsed_h"]))
        later = [row for row in timeline if row["elapsed_h"] > 2.389]
        diagnostics.append(dict(run=run, run_caveat=timeline[0]["run_caveat"], phase_caveat=PHASE_CAVEAT,
            injection_end_target_h=2.389, anchor_frame_index=anchor["frame_index"], anchor_actual_camera_h=anchor["elapsed_h"],
            anchor_main_mass_g=_mass(anchor), late_main_mass_max_g=max((_mass(row) for row in later), default=None),
            late_mass_growth_from_nearest_anchor_g=max((_mass(row) - _mass(anchor) for row in later), default=None),
            solubility_exceedance_policy="Native relative aqueous response clipped to [0,1]; gas water equals local solubility",
            late_mass_uses_no_balance_rescaling=True, dilution="none", map_rescaled_to_injected_mass=False,
            caveat="No noise threshold is invented here; outlet loss and phase/support uncertainty must be considered"))
    return diagnostics


def overflow_comparisons(rows):
    result = []
    for row in rows:
        metadata = row.get("overflow") or {}
        native = row.get("native_result") or {}
        volume = metadata.get("gas_in_rig_volume_ml")
        density = metadata.get("gas_density_g_l", native.get("overflow_gas_density_g_l", native.get("gas_density_pore_volume_mean_g_l")))
        available = volume is not None and density is not None and bool(metadata.get("source_sha256"))
        mass = float(volume) / 1000 * float(density) if available else None
        result.append(dict(run=row["run"], frame_index=row["frame_index"], elapsed_h=row["elapsed_h"],
            run_caveat=row["run_caveat"], phase_caveat=row["phase_caveat"],
            main_gas_mass_g=native.get("gas_mass_g"), overflow_based_rig_gas_mass_g=mass,
            independent_gas_volume_ml=volume, gas_density_g_l=density, source_sha256=metadata.get("source_sha256"),
            measurement_kind=metadata.get("measurement_kind"), measured_vs_imputed=metadata.get("measured_vs_imputed"),
            feed_density_assumptions=metadata.get("assumptions"), source_available=available,
            comparison_only=True, used_in_objective=False, interpretation="Rig gas-volume observation; not automatically CO2 outlet mass"))
    return result


def write_report_tables(folder, rows, *, freeze, sensitivities=()):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    rows = list(rows)
    table = []
    for row in rows:
        result = row.get("native_result", {})
        table.append({**{key: value for key, value in row.items() if key not in {"assets", "native_result"}}, **result})
    write_csv_new(folder / "mass_per_frame.csv", table)
    for run in sorted({row["run"] for row in rows}):
        write_csv_new(folder / f"{run}_mass.csv", (row for row in table if row["run"] == run))
    identity_by_key = {(row["run"], int(row["frame_index"])): row for row in rows}
    sensitivity_rows = []
    for row in sensitivities:
        source = identity_by_key.get((row.get("run"), int(row.get("frame_index", -1))), {})
        sensitivity_rows.append({**{key: source.get(key) for key in ("run_caveat", "phase_caveat", "calibration_freeze_sha256", "heldout", "light_group", "era")}, **row})
    write_csv_new(folder / "sensitivities.csv", sensitivity_rows)
    check = heldout_check1(rows)
    write_new_json(folder / "heldout_check1.json", check)
    write_csv_new(folder / "heldout_check1_inputs.csv", check["inputs"])
    selected, aggregates = comparison_ratios(rows)
    write_csv_new(folder / "heldout_vs_calibration_ratios.csv", selected)
    write_csv_new(folder / "ratios_by_era.csv", aggregates)
    write_csv_new(folder / "sanity_diagnostics.csv", sanity_diagnostics(rows))
    write_csv_new(folder / "overflow_comparison_only.csv", overflow_comparisons(rows))
    write_new_json(folder / "report_binding.json", dict(contract=CONTRACT, calibration_freeze_sha256=freeze.sha256,
        calibration_freeze_path=str(freeze.path), run_count=len({row["run"] for row in rows}), frame_count=len(rows),
        map_rescaled_to_injected_mass=False, backend="native DarSIA / CPU", bottom_mass_allowed=False,
        main_total_excludes_noneligible=True, default_candidate=False, phase_caveat=PHASE_CAVEAT))
    with (folder / "README.md").open("x", encoding="utf-8") as stream:
        stream.write("# Common FF_AC native concentration and mass\n\n"
            f"Frozen calibration: `{freeze.path}`\n\nSHA-256: `{freeze.sha256}`.\n\n"
            "The fixed P1 final 3 phase map is the support. No frame is rescaled to the injected mass. "
            "One common path per quantified facies and one shared aqueous/gas scale pair are used. "
            "Gas water has relative aqueous concentration 1, i.e. local solubility in g/L.\n\n"
            f"{PHASE_CAVEAT}\n\n"
            "`mass_per_frame.csv` and one file per run contain eligible aqueous/gas/total mass and separate blocked/unknown contributions. "
            "`sensitivities.csv` contains the pre-declared alternatives only. Targets use the nearest actual camera time, with the earlier frame for ties.\n\n"
            f"Check 1 outcome: **{check['outcome']}**. Claude independently scores `heldout_check1_inputs.csv`. "
            "Overflow water displacement is not treated as CO2 leaving the rig. A correction is added only for source-bound cumulative CO2 outlet evidence. "
            "Any inferred no-loss assumption is labelled in the inputs. No per-run or group correction is fitted if the check fails.\n\n"
            "`heldout_vs_calibration_ratios.csv` and `ratios_by_era.csv` retain raw main-mass ratios; outlet additions are shown in check 1. "
            "`overflow_comparison_only.csv` is descriptive and never enters an objective. "
            "`sanity_diagnostics.csv` reports late mass growth without inventing a post-hoc noise tolerance.\n\n"
            "Scientific arrays are full-resolution float32. Review PNGs use fixed 0–2 g/L and 0–1 saturation display scales; grey is unquantified. "
            "Display-only colour limits never alter scientific values. Native DarSIA runs on CPU and needs no dedicated GPU.\n")
    return check


def render_report_figures(folder, rows, *, photo_resolver: Callable, phase_resolver: Callable):
    """The specified three runs/times; reductions are display-only, not science."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=False)
    rows = list(rows)
    figures = []
    for run in ("ac28", "ac47", "ac56"):
        run_rows = [row for row in rows if row["run"] == run]
        for selection in nearest_frame_indices(run_rows, (2.25, 6., 24.), include_last=False):
            row = next(r for r in run_rows if int(r["frame_index"]) == selection["frame_index"])
            frame_folder = Path(row["frame_folder"])
            photo_path = Path(photo_resolver(row))
            phase_path = Path(phase_resolver(row))
            with Image.open(photo_path) as image:
                photograph = np.asarray(image.convert("RGB"))
            with Image.open(phase_path) as image:
                phase = np.asarray(image)
            if phase.ndim == 2:
                colours = np.zeros((*phase.shape, 3), np.uint8)
                colours[phase == 1] = (40, 190, 95)
                colours[phase == 2] = (235, 80, 130)
                colours[phase == 255] = (40, 200, 220)
                phase = colours
            caq = np.load(frame_folder / "c_aq_g_l.npy", mmap_mode="r", allow_pickle=False)
            sg = np.load(frame_folder / "s_g.npy", mmap_mode="r", allow_pickle=False)
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.3), constrained_layout=True)
            axes[0].imshow(photograph); axes[0].set_title("Photograph · saved RAW detail view")
            axes[1].imshow(phase); axes[1].set_title("Locked P1 final 3 phases")
            cplot = axes[2].imshow(caq, cmap="viridis", vmin=0, vmax=2)
            axes[2].set_title("Aqueous concentration (g/L)")
            splot = axes[3].imshow(sg, cmap="viridis", vmin=0, vmax=1)
            axes[3].set_title("Gas saturation")
            fig.colorbar(cplot, ax=axes[2], fraction=.03)
            fig.colorbar(splot, ax=axes[3], fraction=.03)
            for ax in axes:
                ax.set_axis_off()
            fig.suptitle(f"{run.upper()} · camera time {row['elapsed_h']:.4f} h · nearest {selection['target_h']} h · "
                "common frozen native calibration; bottom excluded from mass")
            filename = f"{run}_target_{str(selection['target_h']).replace('.', 'p')}_frame_{row['frame_index']:03d}.png"
            fig.savefig(destination / filename, dpi=150)
            plt.close(fig)
            figures.append(dict(run=run, **selection, path=str(destination / filename),
                photo_sha256=sha_file(photo_path), phase_sha256=sha_file(phase_path),
                calibration_freeze_sha256=row["calibration_freeze_sha256"],
                run_caveat=row.get("run_caveat"), phase_caveat=row.get("phase_caveat"),
                display_only_reduction=True, scientific_arrays_full_resolution=True))
    write_new_json(destination / "figures.json", dict(contract=CONTRACT, figures=figures))
    return figures


def register_concentration_review(series_root, production_root, rows, *, freeze, candidate_id, candidate_sha256):
    """A new pointer to concentration files; no phase registration is modified."""
    rows = list(rows)
    if len({row["run"] for row in rows}) != 39:
        raise ValueError("Studio concentration registration requires all 39 included runs")
    keys = {(row["run"], int(row["frame_index"])) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("Duplicate concentration frame in Studio inventory")
    production_root = Path(production_root).resolve()
    frames = []
    for row in rows:
        folder = Path(row["frame_folder"]).resolve()
        if not folder.is_relative_to(production_root):
            raise ValueError("Concentration registration escapes its new production folder")
        if row.get("calibration_freeze_sha256") != freeze.sha256 or row.get("complete") is not True:
            raise ValueError("Concentration frame not complete or belongs to another freeze")
        receipt = folder / "receipt.json"
        frames.append(dict(run=row["run"], frame_index=int(row["frame_index"]), elapsed_h=float(row["elapsed_h"]),
            receipt=dict(path=str(receipt), sha256=sha_file(receipt)), frame_folder=str(folder)))
    name = production_root.name
    target = Path(series_root) / "results" / "concentration_layer_registrations" / (name + ".json")
    record = dict(contract="ffac-common-native-concentration-review-v1", candidate_id=str(candidate_id),
        candidate_sha256=str(candidate_sha256), calibration_freeze=dict(path=str(freeze.path), sha256=freeze.sha256),
        production_root=str(production_root), complete=True, default=False, run_count=39,
        frame_count=len(frames), frames=frames, mass_policy="Main total excludes bottom strip, below-photo fill and unquantified facies",
        no_per_run_mass_fit=True, map_rescaled_to_injected_mass=False, scientific_status="Common frozen fit; held-out and phase uncertainties reported")
    write_new_json(target, record)
    return target
