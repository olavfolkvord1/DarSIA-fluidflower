"""Bind a new interval campaign and import verified calibration-only samples."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path

from darsia.fluidflower.tools import run_ffac_common_mass_campaign as driver
from darsia.fluidflower.tools.revise_ffac_common_mass_campaign import import_collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    job = driver._json(args.target / "job.json")
    workflow_binding = args.target / "calibration_plan_binding_final.json"
    if not workflow_binding.exists():
        workflow_binding = args.target / "calibration_plan_binding.json"
    plan = driver._json(workflow_binding)
    options = ["--job", str(args.target), "--input-plan", job["input_plan"]["path"],
        "--spec-sha256", job["spec_sha256"], "--stage", "paths", "--stride", "16",
        "--workflow-plan", plan["calibration_plan"]["path"],
        "--workflow-binding", str(workflow_binding)]
    with (args.target / "sample_import_paths.stdout.log").open("x", encoding="utf-8") as stream:
        with redirect_stdout(stream):
            try:
                driver.main(options)
            except ValueError as error:
                if str(error) != "Pooled collection incomplete: ac17":
                    raise
            result = import_collection(args.source, args.target)
            driver.main(options)
    checks = []
    for path in sorted((args.source / "calibration/native_color_paths").rglob("*")):
        if path.is_file() and path.name != "metadata.json":
            relative = path.relative_to(args.source / "calibration/native_color_paths")
            other = args.target / "calibration/native_color_paths" / relative
            if driver.output.sha_file(path) != driver.output.sha_file(other):
                raise ValueError(f"Imported samples changed an optical component: {relative}")
            checks.append(dict(component=str(relative), sha256=driver.output.sha_file(other)))
    driver._new(args.target / "optical_import_equivalence.json", dict(
        complete=True, numerical_optical_components_bit_identical=True,
        components=checks, metadata_producer_bindings_revised=True,
        source_path_package=driver.inputs.asset(args.source / "calibration/path_package_receipt.json")))
    print(json.dumps(dict(**result, optical_components=len(checks),
        numerical_optical_components_bit_identical=True)))


if __name__ == "__main__":
    main()
