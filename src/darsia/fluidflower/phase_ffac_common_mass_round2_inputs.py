"""Immutable round-2 inputs over the unchanged P1-final3 scientific adapter.

Discovery and dose integration read controller/protocol metadata only. The
12 final-check runs and descriptive AC38 require a round-2 freeze before any
scientific pixel loading. Measured depth/porosity maxima bound missing mass;
they never replace unknown geometry in the main concentration calculation.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time as clock_time
import csv
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from . import phase_ffac_common_mass_inputs as r1

CONTRACT = 'ffac-common-mass-round2-inputs-v1'
FREEZE_CONTRACT = 'ffac-common-mass-round2-frozen-v1'
HELD_OUT = r1.HELD_OUT
DESCRIPTIVE = ('ac38',)
PRODUCTION_RUNS = tuple(f'ac{n}' for n in range(17, 62) if n not in (20, 24, 27, 36, 37, 51))
FIT_RUNS = tuple(run for run in PRODUCTION_RUNS if run not in (*HELD_OUT, *DESCRIPTIVE))
EXCLUDED = r1.EXCLUDED
VerifiedAssetReader = r1.VerifiedAssetReader
to_native_frame = r1.to_native_frame


def validate_split(split):
    expected = dict(production_runs=PRODUCTION_RUNS, fit_runs=FIT_RUNS,
                    held_out_runs=HELD_OUT, descriptive_runs=DESCRIPTIVE,
                    excluded_runs=EXCLUDED)
    for key, runs in expected.items():
        values = split.get(key, ())
        if len(values) != len(runs) or set(values) != set(runs):
            raise ValueError(f'Round-2 fixed cohort differs: {key}')


def require_evaluation_access(run, frozen_calibration=None, *, reader=None):
    run = str(run).casefold()
    if run not in PRODUCTION_RUNS:
        raise ValueError('Excluded/unknown run has no round-2 scientific evaluation')
    if run in FIT_RUNS:
        return
    if (not frozen_calibration or frozen_calibration.get('frozen') is not True
            or frozen_calibration.get('contract') != FREEZE_CONTRACT
            or set(frozen_calibration.get('calibration_runs', ())) != set(FIT_RUNS)
            or set(frozen_calibration.get('heldout_runs', ())) != set(HELD_OUT)
            or set(frozen_calibration.get('descriptive_runs', ())) != set(DESCRIPTIVE)):
        raise ValueError('Freeze round 2 before final-check or descriptive pixels')
    files = frozen_calibration.get('calibration_files') or {}
    if not files:
        raise ValueError('Round-2 freeze has no immutable calibration files')
    reader = reader or VerifiedAssetReader()
    for path, value in files.items():
        reader.path(dict(path=path, sha256=value) if isinstance(value, str) else value)


def physical_limits(depth_path, facies_path):
    """Audit the spec's physical maxima and the measured facies Swi table."""
    with Path(depth_path).open(encoding='utf-8-sig', newline='') as stream:
        depth = list(csv.DictReader(stream))
    with Path(facies_path).open(encoding='utf-8-sig', newline='') as stream:
        facies = list(csv.DictReader(stream))
    depths = np.asarray([float(row['depth']) for row in depth], np.float64)
    if not len(depths) or not np.isfinite(depths).all() or np.any(depths <= 0):
        raise ValueError('Measured depth CSV contains invalid observations')
    porous = [row for row in facies if int(row['id']) != 0]
    porosities = np.asarray([float(row['porosity']) for row in porous], np.float64)
    if not len(porosities) or not np.isfinite(porosities).all() or np.any((porosities <= 0) | (porosities > 1)):
        raise ValueError('Facies porosity table is invalid')
    max_depth, max_porosity = float(depths.max()), float(porosities.max())
    if not np.isclose(max_depth, .0155, rtol=0, atol=1e-12) or not np.isclose(max_porosity, .45, rtol=0, atol=1e-12):
        raise ValueError('Measured physical maxima differ from fixed round-2 specification')
    swi = {str(int(row['id'])): float(row['swi CO2']) for row in porous}
    expected = {'1': .32, '2': .14, **{str(n): .12 for n in range(3, 9)}}
    if swi != expected:
        raise ValueError('Measured Swi table differs from the predeclared fixed gas rule')
    return dict(contract='ffac-round2-measured-physical-upper-bounds-v1',
        max_depth_m=max_depth, max_porosity=max_porosity, swi_by_facies=swi,
        facies_names={row['id']: row['name'] for row in porous},
        depth_source=r1.asset(depth_path), facies_source=r1.asset(facies_path),
        unknown_geometry_main_imputed=False, bounds_only=True,
        unknown_facies_rule='unquantified; capacity only, not assigned a Swi or a colour path')


def script_clock(protocol, origin_datetime):
    """Read the original script-start cell; never infer clock from phase data."""
    import openpyxl
    workbook = openpyxl.load_workbook(protocol['path'], read_only=True, data_only=True)
    try:
        value = workbook['Sheet1']['P23'].value
    finally:
        workbook.close()
    if isinstance(value, datetime):
        clock = value.time()
    elif isinstance(value, clock_time):
        clock = value
    elif isinstance(value, str):
        clock = clock_time.fromisoformat(value.strip())
    else:
        raise ValueError('Original Sheet1!P23 does not contain a script-start clock')
    origin = datetime.fromisoformat(origin_datetime)
    stamp = datetime.combine(origin.date(), clock).isoformat()
    return dict(protocol=dict(protocol), cell='Sheet1!P23', value=str(value),
        script_start_datetime=stamp,
        timestamp_precision='protocol clock, commonly recorded to the minute',
        timing_sensitivity_seconds=30.,
        independent_subsecond_clock_verified=False)


def parse_mfc(path):
    """Read the six original controller columns without altering readings."""
    raw = Path(path).read_text(encoding='utf-8-sig')
    first = next((line for line in raw.splitlines() if line.strip()), '')
    delimiter = ';' if ';' in first else ',' if ',' in first else None
    data = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 6 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError('MFC injection log must have six finite original columns')
    if any(np.any(np.diff(data[:, col]) <= 0) for col in (0, 2, 4)):
        raise ValueError('MFC stream clock reset, duplicate, or nonchronological sample')
    if np.any(data[:, 1] < 0) or np.any(data[:, 3] < 0):
        raise ValueError('Negative original MFC flow/setpoint cannot be silently clipped')
    return data


def integrate_mfc(data, atmosphere, script_start_datetime, *, clock_gate_s=2.):
    """QA anchored controller seconds and integrate measured flow in float64.

    The fixed controller offset comes from the original protocol clock. The
    setpoint transitions only validate it; they never fit a clock correction.
    Observed spikes are retained and no unobserved final sample is fabricated.
    """
    stages = atmosphere['injection_stages']
    with Path(atmosphere['injection']['path']).open(encoding='utf-8-sig', newline='') as stream:
        original = list(csv.DictReader(stream))
    factors = [float(row['rate_sccm'])/float(row['flow_percent']) for row in original if float(row['flow_percent']) > 0]
    if not factors or not np.allclose(factors, factors[0], rtol=0, atol=1e-12):
        raise ValueError('MFC percent-to-sccm calibration is not source-consistent')
    densities = [float(row['density_kg_m3']) for row in stages]
    if not np.allclose(densities, densities[0], rtol=0, atol=1e-12):
        raise ValueError('Variable injection density needs an explicitly time-resolved measurement')
    offset = (datetime.fromisoformat(atmosphere['origin_datetime'])-
              datetime.fromisoformat(script_start_datetime)).total_seconds()
    times = data[:, 0]-offset
    begin = min(float(row['start_h']) for row in stages)*3600
    end = max(float(row['end_h']) for row in stages)*3600
    gap = float(np.diff(data[:, 0]).max())
    if times[0] > begin or times[-1] < end-clock_gate_s or gap > clock_gate_s:
        raise ValueError('MFC injection stream lacks complete acquired time coverage')
    changes = np.r_[True, np.abs(np.diff(data[:, 3])) > .001]
    change_times, change_values = times[changes], data[changes, 3]
    checks = []
    previous = None
    for row in original:
        target = float(row['flow_percent'])
        when = (datetime.fromisoformat(row['start'])-datetime.fromisoformat(atmosphere['origin_datetime'])).total_seconds()
        if previous is not None and np.isclose(target, previous, rtol=0, atol=1e-10):
            continue
        previous = target
        candidates = change_times[np.isclose(change_values, target, rtol=0, atol=.001)]
        if not len(candidates):
            raise ValueError(f'MFC setpoint {target:g}% has no sourced controller transition')
        observed = float(candidates[np.argmin(np.abs(candidates-when))])
        checks.append(dict(expected_s=when, observed_s=observed, setpoint_percent=target,
                           offset_error_s=observed-when))
    error = max(abs(row['offset_error_s']) for row in checks)
    if error > clock_gate_s:
        raise ValueError(f'MFC/protocol anchored transition differs by {error:.6g}s (gate {clock_gate_s}s)')
    selected = times[(times >= begin) & (times <= end)]
    if not len(selected):
        raise ValueError('No acquired MFC injection samples')
    if selected[0] > begin:
        selected = np.r_[begin, selected]
    if selected[-1] < end and end <= times[-1]:
        selected = np.r_[selected, end]
    rate = np.interp(selected, times, data[:, 1])*factors[0]*densities[0]*1e-3/60
    cumulative = np.r_[0., np.cumsum(np.diff(selected)*(rate[:-1]+rate[1:])*.5)]
    good = (times >= begin) & (times <= end)
    nominal = r1.nominal_mass_at(atmosphere, end/3600)
    return dict(elapsed_h=(selected/3600).tolist(), cumulative_mass_g=cumulative.tolist(),
        measured_final_mass_g=float(cumulative[-1]), nominal_final_mass_g=nominal,
        measured_over_nominal=float(cumulative[-1]/nominal),
        gas_start_offset_s=offset, percent_to_sccm=float(factors[0]), density_kg_m3=float(densities[0]),
        clock_qa=dict(passed=True, gate_seconds=clock_gate_s, maximum_transition_error_s=error,
                      counter_resets=0, transitions=checks, clock_adjustment_fitted=False),
        units_qa=dict(passed=True, time='seconds', measured_column=1, setpoint_column=3,
                      measured_unit='percent of MFC full scale', rate_conversion='sccm=percent*source calibration',
                      density_conversion='g/s=sccm*density_kg_m3*1e-3/60'),
        readings_above_full_scale=int(np.count_nonzero(data[good, 1]>100)),
        max_measured_percent=float(data[good, 1].max()), maximum_acquisition_gap_s=gap,
        last_observed_h=float(selected[-1]/3600), protocol_end_h=end/3600,
        source_values_clamped=False, integrated_warmup=False, extrapolated_final_sample=False,
        assumptions=['Original protocol script clock anchors elapsed controller seconds; no fitted clock shift.',
            'Controller setpoint schedule confirms the declared injection clock within the recorded sampling interval.',
            'MFC flow is assumed delivered to the connected rig; no independent downstream delivery meter.',
            'Protocol standard gas density converts standard measured flow to mass.',
            'Measurements above full scale are retained unchanged.'])


def dose_at(record: Mapping, time_hours, *, nominal=False):
    if not np.isfinite(time_hours):
        raise ValueError('Dose needs finite camera-relative time')
    if nominal or record['dose_kind'] != 'measured_mfc':
        return r1.nominal_mass_at(record, float(time_hours))
    measured = record['measured_mfc_dose']
    return 0. if time_hours <= 0 else float(np.interp(time_hours, measured['elapsed_h'], measured['cumulative_mass_g']))


def load_common_frame(metadata, input_plan, *, clearwater_transport=None,
                      session_reader=None, frozen_calibration=None):
    reader = session_reader or VerifiedAssetReader()
    require_evaluation_access(metadata['run'], frozen_calibration, reader=reader)
    validate_split(input_plan['split'])
    base = reader.json(input_plan['base_input_plan'])
    base['clearwater_results_folder'] = input_plan['clearwater_results_folder']
    records = reader.json(input_plan['dose_inventory'])['runs']
    dose = next(row for row in records if row['run'] == metadata['run'])
    # R1's held-out adapter guard only verifies immutable file bindings, while
    # the stronger R2 cohort/contract guard above controls access to pixels.
    frame = r1.load_common_frame(metadata, base, clearwater_transport=clearwater_transport,
        session_reader=reader, frozen_calibration=frozen_calibration)
    epochs = reader.json(input_plan['light_epochs'])
    epoch = next(row['epoch'] for row in epochs['runs'] if row['run'] == frame.run)
    return replace(frame, nominal_injected_mass_g=dose_at(dose, metadata['elapsed_h']),
        provenance=dict(frame.provenance, round2_input_contract=CONTRACT,
            round2_dose_inventory=input_plan['dose_inventory'],
            dose_source=dose['dose_kind'], dose_qa=dose.get('qa'),
            dose_nominal_fallback_reasons=dose.get('fallback_reasons', []),
            dose_clock_source=dose.get('clock_source'), measured_mfc_source=dose.get('measured_source'),
            nominal_setpoint_mass_g=dose_at(dose, metadata['elapsed_h'], nominal=True),
            light_epoch=epoch, physical_limits=input_plan['physical_limits'],
            descriptive_out_of_fit=frame.run in DESCRIPTIVE))


def metadata_index(input_plan, *, reader=None):
    reader = reader or VerifiedAssetReader()
    base = reader.json(input_plan['base_input_plan'])
    return reader.json(base['source_metadata'])
