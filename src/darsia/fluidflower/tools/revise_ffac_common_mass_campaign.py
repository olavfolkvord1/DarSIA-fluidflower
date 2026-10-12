"""Import unchanged calibration-only optical samples into a new policy revision.

This never imports fitting summaries, a freeze, or held-out computations.
Original producer receipts and every array byte remain independently bound.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from darsia.fluidflower import phase_ffac_common_mass as native
from darsia.fluidflower import phase_ffac_common_mass_inputs as inputs
from darsia.fluidflower.tools import run_ffac_common_mass_campaign as driver


def initialize(source: Path, target: Path, input_plan: Path):
    target.mkdir(parents=True, exist_ok=False)
    job = driver._json(source / "job.json")
    job.update(batch=str(target), previous_job=inputs.asset(source / "job.json"),
               input_plan=inputs.asset(input_plan), status="pre_fit_policy_revision",
               created_utc=datetime.now(timezone.utc).isoformat(),
               reason="Replace our extra zero-bottom-positive restriction with physical dose intervals, before any held-out computation")
    driver._new(target / "job.json", job)
    shutil.copyfile(source / "approved_spec.md", target / "approved_spec.md")
    return job


def import_collection(source: Path, target: Path):
    binding = driver._stamp(driver._json(target / "campaign_binding.json"))
    previous_binding = driver._stamp(driver._json(source / "campaign_binding.json"))
    imports = []
    for run in native.CALIBRATION_RUNS:
        original = source / "work/path_statistics" / run
        previous = driver._resume(original, previous_binding)
        if previous is None or previous.get("heldout_pixels_read") is not False:
            raise ValueError(f"Missing immutable calibration-only sample producer: {run}")
        destination = target / "work/path_statistics" / run
        destination.mkdir(parents=True, exist_ok=False)
        assets = deepcopy(previous["assets"])
        for descriptor in assets.values():
            shutil.copyfile(original / descriptor["path"], destination / descriptor["path"])
            if driver.output.sha_file(destination / descriptor["path"]) != descriptor["sha256"]:
                raise ValueError("Imported sample bytes differ from their original producer")
        provenance = dict(original_producer=inputs.asset(original / "receipt.json"),
            original_campaign_binding=inputs.asset(source / "campaign_binding.json"),
            operation="byte-identical import, not new extraction",
            equivalence="Scientific frame loader, masks, sampling stride and path sampling rule unchanged; only subsequent mass-fit excluded-support policy differs",
            imported_utc=datetime.now(timezone.utc).isoformat())
        receipt = dict(previous, campaign_binding_sha256=binding,
                       import_equivalence=provenance, extraction_reexecuted=False)
        driver._new(destination / "receipt.json", receipt)
        driver._resume(destination, binding)
        imports.append(dict(run=run, source=provenance["original_producer"],
                            imported=inputs.asset(destination / "receipt.json"), assets=assets))
    report = dict(contract="ffac-common-native-calibration-sample-import-v1",
        complete=True, run_count=len(imports), imports=imports,
        fitting_summaries_imported=False, freeze_imported=False, heldout_pixels_read=False)
    driver._new(target / "sample_import_receipt.json", report)
    return dict(run_count=len(imports), report=str(target / "sample_import_receipt.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--input-plan", type=Path)
    parser.add_argument("--stage", choices=("initialize", "import"), required=True)
    args = parser.parse_args()
    if args.stage == "initialize":
        if args.input_plan is None:
            parser.error("Initialize requires the exact existing input plan")
        result = initialize(args.source, args.target, args.input_plan)
    else:
        result = import_collection(args.source, args.target)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
