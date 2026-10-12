"""Bind deferred Studio references or explicitly prepare their exact maps.

Only the readiness receipt and disposable reference caches are written. Native
evidence, model parameters and the final validation binding remain unchanged.
"""
from __future__ import annotations

from pathlib import Path
import time

from darsia.fluidflower.tools import phase_reference_maps as maps
from darsia.fluidflower.tools.phase_injection_references import injection_end_references
from darsia.fluidflower.tools.phase_reference_batch import resolve_layers
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools._paths import canonical_metadata_paths

RECEIPT = "phase_reference_readiness.json"
FORMAT = "phase-reference-readiness-v1"


def _same_bound_asset(recorded, current):
    recorded = canonical_metadata_paths(recorded)
    if recorded == current:
        return True
    # The project share can be opened through its server name or IP address.
    # Keep the frozen receipt intact and accept that spelling difference only
    # when the digest agrees and the two paths identify the same filesystem file.
    if (not isinstance(recorded, dict) or not isinstance(current, dict)
            or set(recorded) != {"path", "sha256"}
            or set(current) != {"path", "sha256"}
            or recorded["sha256"] != current["sha256"]):
        return False
    try:
        return Path(recorded["path"]).samefile(current["path"])
    except (OSError, TypeError, ValueError):
        return False


def require_reference_readiness(validation, *, allow_pending=False):
    """Accept deferred references only where ordinary timeline review permits it."""
    validation = resolve_review_path(validation)
    path = validation / RECEIPT
    if not path.is_file():
        return
    receipt = maps._read(path)
    allowed = {"ready", "preparing", "deferred"} if allow_pending else {"ready"}
    if receipt.get("format") != FORMAT or receipt.get("status") not in allowed:
        raise ValueError("Reference phase maps are not ready: " + str(receipt.get("error") or receipt.get("status")))
    if any(key not in receipt for key in ('binding', 'campaign')):
        raise ValueError('Validation package is not ready: no bound reference receipt')
    for key, filename in (("binding", "validation_binding.json"), ("campaign", "postflight_manifest.json")):
        if not _same_bound_asset(receipt.get(key), maps._asset(validation / filename)):
            raise ValueError("Reference readiness identifies another validation package")


def mark_references_pending(validation):
    maps._write(Path(validation) / RECEIPT, {"format": FORMAT, "status": "preparing"})


def defer_validation_references(validation):
    """Publish ordinary maps without computing optional reference overlays.

    The reference button still resolves the exact acquisition against this
    frozen binding, with chronological replay when an endpoint is off timeline.
    No image, inference, or replay is needed merely to make a run reviewable.
    """
    validation = Path(validation).resolve()
    binding_path = validation / "validation_binding.json"
    campaign_path = validation / "postflight_manifest.json"
    binding, campaign = maps._read(binding_path), maps._read(campaign_path)
    mode = binding.get("output_mode", "review")
    if mode not in ("review", "production") or campaign.get("output_mode", "review") != mode:
        raise ValueError("Reference preparation requires matching phase output modes")
    jobs, selected = campaign["jobs"], binding["selected_candidates"]
    if (not jobs or len(jobs) != len(selected)
            or {job["run"]: job["candidate_id"] for job in jobs} != selected):
        raise ValueError("Reference preparation requires the complete selected validation campaign")
    if any(job["dataset"] not in {"ffac", "ffum", "ffbm"} for job in jobs):
        raise ValueError("Unsupported reference experiment series")
    existing_path = validation / RECEIPT
    if existing_path.is_file():
        existing = maps._read(existing_path)
        if existing.get("format") != FORMAT:
            raise ValueError("Unknown reference readiness receipt")
        if existing.get("status") in {"ready", "deferred"}:
            require_reference_readiness(validation, allow_pending=True)
            return existing
        # A previous failed preparation is evidence of an error, not permission
        # to hide it by replacing the receipt with a deferred one.
        if existing.get("status") == "failed":
            require_reference_readiness(validation)
        if existing.get("status") != "preparing":
            raise ValueError("Unknown reference readiness status")
        if "binding" in existing or "campaign" in existing:
            require_reference_readiness(validation, allow_pending=True)
    receipt = {"format": FORMAT, "status": "deferred", "output_mode": mode,
               "binding": maps._asset(binding_path), "campaign": maps._asset(campaign_path),
               "materialization_policy": "on-demand-exact-reference",
               "reference_coverage": "not-requested", "references": [],
               "unavailable_references": []}
    maps._write(existing_path, receipt)
    return receipt


def prepare_validation_references(validation, progress=None, *, allow_inference=True):
    """Resolve baseline and every injection end against final published paths.

    The resolver validates the frozen candidate and reuses verified cached maps.
    Missing source acquisitions are recorded separately from phase-map failures.
    """
    from darsia.fluidflower.phase_training.radiometry import analysis_review_root

    validation = Path(validation).resolve()
    binding_path = validation / "validation_binding.json"
    campaign_path = validation / "postflight_manifest.json"
    binding, campaign = maps._read(binding_path), maps._read(campaign_path)
    output_mode = binding.get("output_mode", "review")
    if output_mode not in ("review", "production") or campaign.get("output_mode", "review") != output_mode:
        raise ValueError("Reference preparation requires matching phase output modes")
    selected = binding["selected_candidates"]
    jobs = campaign["jobs"]
    if (not jobs or len(jobs) != len(selected)
            or {j["run"]: j["candidate_id"] for j in jobs} != selected):
        raise ValueError("Reference preparation requires the complete selected validation campaign")
    receipt = {"format": FORMAT, "status": "preparing", "output_mode": output_mode, "binding": maps._asset(binding_path),
               "campaign": maps._asset(campaign_path), "references": [],
               "unavailable_references": []}
    maps._write(validation / RECEIPT, receipt)
    try:
        for job in jobs:
            from darsia.fluidflower.phase_review_basis import is_regional, REGIONAL_VERSION
            regional = is_regional(job) or (campaign.get('physics_profile') or {}).get('format') == REGIONAL_VERSION
            required_layers = ({"all", "aqueous", "gas"} if output_mode == "production"
                               else {"all", "aqueous", "gas", "raw"} if regional
                               else maps.LAYERS)
            run, dataset = job["run"], job["dataset"]
            if dataset not in {"ffac", "ffum", "ffbm"}:
                raise ValueError(f"Unsupported reference experiment series: {dataset}")
            root = analysis_review_root(Path(job.get("analysis_source_review_root") or job["review_output_dir"]), run)
            pair = {"run": run, "dataset": dataset}
            references = injection_end_references(pair, root)
            # A reference the selector could not resolve is not automatically a
            # dead end. Without acquisition_inventory.json the selector sees only
            # the sparse reviewed timeline, so it reports no elapsed time for any
            # row -- which is the very state the completeness test below rejects,
            # so the runs that needed the inventory were the ones never given it.
            # Building it is what decides the question: attempt it whenever a row
            # is unusable and the selector read no complete inventory. A source
            # acquisition that genuinely does not exist still lands in
            # unavailable_references, and a failure here is recorded rather than
            # failing the evaluation: missing acquisitions stay separate from
            # phase-map failures.
            unusable = [row for row in references
                        if not row.get("available") or row.get("needs_inventory")
                        or row.get("needs_materialization")]
            complete_selection = bool(references) and all(
                row.get("elapsed_h") is not None for row in references)
            no_inventory = any("inventory" in row
                               and not (row.get("inventory") or {}).get("complete")
                               for row in unusable)
            if unusable and (complete_selection or no_inventory):
                from darsia.fluidflower.tools.prepare_phase_injection_references import prepare
                try:
                    prepare(root, run, dataset)
                except Exception as error:  # noqa: BLE001 - reported, never fatal
                    receipt.setdefault("reference_preparation_errors", []).append(
                        {"dataset": dataset, "run": run,
                         "reason": f"{type(error).__name__}: {error}"})
                    maps._write(validation / RECEIPT, receipt)
                else:
                    references = injection_end_references(pair, root)
            available = [row for row in references if row.get("available")
                         and not row.get("needs_inventory")
                         and not row.get("needs_materialization")]
            for row in references:
                if row not in available:
                    receipt["unavailable_references"].append({
                        "dataset": dataset, "run": run, "reference_id": row["id"],
                        "reason": str(row.get("reason") or "Exact reference image is unavailable")})
            for reference in [None, *available]:
                label = "Before injection" if reference is None else reference["label"]
                if progress:
                    progress(f"{run.upper()}: preparing reference phase map — {label}")
                started = time.perf_counter()
                layers = resolve_layers(reference, root, run, job, required_layers, progress=progress,
                                        allow_inference=allow_inference)
                receipt["references"].append({"dataset": dataset, "run": run,
                    "reference_id": "baseline" if reference is None else reference["id"],
                    "layers": layers, "preparation_seconds": time.perf_counter() - started})
                maps._write(validation / RECEIPT, receipt)
        for name, path in (("binding", binding_path), ("campaign", campaign_path)):
            if maps._asset(path) != receipt[name]:
                raise ValueError("Validation binding changed during reference preparation")
        receipt["reference_coverage"] = (
            "partial" if receipt["unavailable_references"] else "complete")
        receipt["status"] = "ready"
        maps._write(validation / RECEIPT, receipt)
        return receipt
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        maps._write(validation / RECEIPT, receipt)
        raise
