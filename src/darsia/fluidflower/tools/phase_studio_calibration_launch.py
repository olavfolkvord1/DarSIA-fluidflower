"""Build explicit Studio calibration requests on one series' NAS storage.

The UI may browse results from another series, but a new calculation always
uses the currently selected, registered phase model.  This module does not
prepare images or fit a model on the GUI thread.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from darsia.fluidflower import concentration_configuration as configuration


UNCONFIRMED_CHEMISTRY = frozenset({"uncertainty", "provisional", "unconfirmed"})
FFUM_DOCUMENTED_PROFILE = "ffum-allframes-physical-assumptions-20260922"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def documented_ffum_profile(catalog):
    """Verify the optional, model-specific RUN9/RUN11 physical assumptions."""
    if str(catalog.get("dataset", "")).casefold() != "ffum":
        raise ValueError("The documented physical profile belongs to FF_UM")
    root = Path(catalog["results_root"]).resolve().parent.parent
    folder = root / "calibration" / "profiles"
    path = folder / (FFUM_DOCUMENTED_PROFILE + ".toml")
    receipt_path = folder / (FFUM_DOCUMENTED_PROFILE + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if (receipt.get("record_type") != "frozen_ffum_joint_physical_profile"
            or receipt.get("selected_phase_model_id") != catalog.get("model_id")
            or Path(receipt.get("profile_path", "")).resolve() != path.resolve()
            or receipt.get("profile_sha256") != _sha256(path)
            or receipt.get("source_request_sha256") != _sha256(receipt["source_request_path"])):
        raise ValueError("The documented FF_UM physical profile or its source changed")
    assumptions = receipt.get("clearwater_reference_assumptions")
    if not isinstance(assumptions, dict) or set(assumptions) != {"run9"}:
        raise ValueError("The documented RUN9 clear-water source is missing")
    return path, receipt


def frozen_gas_study_request(result_path, results_root):
    """Find a selected FF_UM result's immutable gas/local study request."""
    root = Path(results_root).resolve()
    result = Path(result_path).resolve()
    if not result.is_relative_to(root):
        for ancestor in result.parents:
            try:
                if ancestor.samefile(root):
                    break
            except OSError:
                continue
        else:
            return None
    for parent in result.parents:
        if parent.parent.name != "jobs":
            continue
        request = parent / "request.json"
        if request.is_file():
            try:
                value = json.loads(request.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                return None
            if value.get("format") in {"local-optical-dilution-study-v1",
                                       "gas-history-sensitivity-v1"}:
                return request
        return None
    return None


def calibration_cohorts(catalog, *, physical_profile="registered"):
    """Offer shared fits only for confirmed compatible chemistry.

    Every ready run is also reachable through an individual-only entry if it
    cannot join a shared cohort.  The joint worker fits an individual response
    for each run in a shared cohort as well.
    """
    profile_path = (documented_ffum_profile(catalog)[0]
                    if physical_profile == "documented_ffum" else None)
    if physical_profile not in {"registered", "documented_ffum"}:
        raise ValueError("Unknown physical profile")
    groups: dict[str, list[str]] = {}
    individual: list[dict] = []
    unavailable: list[dict] = []
    for job in catalog.get("jobs", []):
        run = str(job["run"]).casefold()
        selected_job = (dict(job, concentration_profile_path=str(profile_path))
                        if profile_path else job)
        detail = configuration.describe_run(selected_job, purpose="joint_reference")
        if not detail["ready"]:
            unavailable.append(dict(run=run, reasons=detail["reasons"]))
            continue
        chemistry = str(detail["chemistry_group"])
        if detail["shared_fit_eligible"] and chemistry.casefold() not in UNCONFIRMED_CHEMISTRY:
            groups.setdefault(chemistry, []).append(run)
        else:
            individual.append(dict(id="individual:" + run,
                label=f"{run.upper()} · individual only ({chemistry})",
                runs=[run], chemistry_group=chemistry, shared=False))
    cohorts = []
    for chemistry, runs in sorted(groups.items()):
        if len(runs) >= 2:
            cohorts.append(dict(id="shared:" + chemistry,
                label=f"{chemistry} · {len(runs)} runs · shared + every individual fit",
                runs=runs, chemistry_group=chemistry, shared=True))
        else:
            run = runs[0]
            individual.append(dict(id="individual:" + run,
                label=f"{run.upper()} · individual only ({chemistry}; one available run)",
                runs=[run], chemistry_group=chemistry, shared=False))
    return cohorts + sorted(individual, key=lambda row: row["id"]), unavailable


def calibration_request(catalog, cohort, output, *, method, overflow_mode="off",
                        overflow=None, overflow_weight=3.0, workers=4,
                        analysis_width=1536, photometry_variants=None,
                        physical_profile="registered",
                        image_selection_receipt_path=None):
    """Freeze a real joint or correction-comparison worker request.

    Requested photometry branches reprocess the raw photographs and reference;
    the original phase labels remain fixed.  A registered-photometry fit uses
    exactly the radiometry attached to the selected phase model.
    """
    if method not in {"registered", "photometry"}:
        raise ValueError("Unknown calibration method")
    if not catalog or not catalog.get("model_root") or not catalog.get("model_id"):
        raise ValueError("Load a registered phase model before calibrating")
    dataset = str(catalog.get("dataset") or "").casefold()
    if dataset not in {"ffum", "ffac", "ffbm"}:
        raise ValueError("Unknown experiment series")
    selected = [str(run).casefold() for run in cohort.get("runs", [])]
    available = {str(job["run"]).casefold() for job in catalog.get("jobs", [])}
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= available:
        raise ValueError("The selected cohort differs from this phase model")
    if method == "photometry" and len(selected) < 2:
        raise ValueError("A correction comparison requires a shared cohort of two or more runs")
    if type(workers) is not int or workers < 1 or type(analysis_width) is not int or analysis_width < 1:
        raise ValueError("Workers and analysis width must be positive integers")
    output = Path(output).resolve()
    results_root = Path(catalog["results_root"]).resolve()
    if not output.is_relative_to(results_root) or output.exists():
        raise ValueError("Use a new output directory inside the selected series' NAS results")
    documented = None
    if physical_profile == "documented_ffum":
        profile_path, documented = documented_ffum_profile(catalog)
    elif physical_profile == "registered":
        profile_path = results_root.parent.parent / "concentration.toml"
    else:
        raise ValueError("Unknown physical profile")
    request = dict(model_root=str(Path(catalog["model_root"]).resolve()),
                   model_id=catalog["model_id"], dataset=dataset,
                   runs=selected, output=str(output), workers=workers,
                   analysis_width=analysis_width,
                   concentration_profile_path=str(profile_path),
                   gas_bounds=[.5, 1.],
                   optimization_settings=dict(start_count=12, maxiter=360,
                                              maxfev=20000, xtol=1e-6, ftol=1e-8))
    if profile_path.is_file():
        request["physical_profile_file_sha256"] = _sha256(profile_path)
    if documented is not None:
        request.update(physical_profile_receipt_path=str(
            profile_path.with_name(FFUM_DOCUMENTED_PROFILE + ".receipt.json")),
            physical_profile_receipt_sha256=_sha256(
                profile_path.with_name(FFUM_DOCUMENTED_PROFILE + ".receipt.json")),
            physical_assumptions_note=documented["physical_assumptions_note"],
            clearwater_reference_assumptions=documented["clearwater_reference_assumptions"])
        if method == "photometry" and analysis_width == 1536:
            request["physical_prepared_root"] = documented["physical_prepared_root"]
    if image_selection_receipt_path is not None:
        selection = Path(image_selection_receipt_path).resolve()
        series_root = results_root.parent.parent
        if not selection.is_relative_to(series_root) or not selection.is_file():
            raise ValueError("Image-selection receipt must exist under the selected series' NAS DarSIA folder")
        request.update(image_selection_receipt_path=str(selection),
                       image_selection_receipt_sha256=_sha256(selection))
    if method == "photometry":
        if overflow_mode != "off" or overflow:
            raise ValueError("The correction comparison does not fit overflow targets")
        from darsia.fluidflower.tools.phase_studio_photometry_policy import (
            load_policy,
        )
        policy = load_policy(dataset, results_root)
        defaults = (["none_cc_on", "facies_cc_on"] if dataset == "ffac"
                    else policy["allowed_variants"] if dataset == "ffum"
                    else ["none_cc_off", "facies_cc_off"])
        requested = (list(defaults) if photometry_variants is None
                     else list(photometry_variants))
        from darsia.fluidflower.tools.phase_studio_photometry_compare import _selected_variants
        _selected_variants({"requested_variants": requested,
                            "photometry_policy": policy}, dataset)
        request.update(requested_variants=requested, overflow_mode="off")
        if policy is not None:
            request["photometry_policy"] = policy
    else:
        if overflow_mode not in {"off", "compare", "constrain", "hard"}:
            raise ValueError("Unknown overflow policy")
        attached = {run: str(Path(path).resolve()) for run, path in (overflow or {}).items()
                    if run in selected}
        if overflow_mode == "off":
            attached = {}
        elif not attached:
            raise ValueError("Attach measured, time-linked overflow targets for this cohort")
        if not 0 < float(overflow_weight) <= 10:
            raise ValueError("Overflow weight must be in (0, 10]")
        request.update(overflow_mode=overflow_mode, overflow=attached,
                       overflow_weight=float(overflow_weight), require_all_runs=True)
    return request
