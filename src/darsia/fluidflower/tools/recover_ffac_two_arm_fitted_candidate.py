"""Continue a fitted FF_AC two-arm job at detached Studio preparation.

The source job, its fitted models, and its failed partial candidate stay immutable.
Preflight is read-only. Execution requires a fresh results job and candidate name;
it does not call either optical or regional training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from darsia.fluidflower import phase_runtime
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import train_ffac_two_arm_optical_region as two_arm
from darsia.fluidflower.tools._paths import process_directory
from darsia.fluidflower.tools.phase_training_photometry_revision import verify_context_binding
from darsia.fluidflower.tools.prepare_optical_region_studio import prepare
from darsia.fluidflower.tools.publish_fitted_phase_candidate import verify_fitted
from darsia.fluidflower.tools.train_phase_model_further import native_source


FORMAT = "ffac-two-arm-fitted-studio-recovery-v1"
CANDIDATE_NAME = re.compile(r"phase-model-[A-Za-z0-9-]+-ffac-unified\Z")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hash(path: Path, expected: str | None = None) -> str:
    _require(path.is_file(), f"Missing immutable source file: {path}")
    actual = sha256_file(path)
    if expected is not None:
        _require(actual == expected.lower(), f"Immutable source SHA changed: {path}")
    return actual


def _same(a: str | Path, b: str | Path) -> bool:
    return Path(a).resolve() == Path(b).resolve()


def verify_frozen_code(job_spec: Path, source_job: Path) -> dict:
    """Verify the original scheduler binding and every frozen Python file."""
    spec = _read(job_spec)
    argv = spec["argv"]
    _require(argv[:3] == ["-u", "-m", "darsia.fluidflower.tools.train_ffac_two_arm_optical_region"]
             and "--execute" in argv and "--job" in argv
             and _same(argv[argv.index("--job") + 1], source_job),
             "Scheduler did not launch this exact two-arm source job")
    snapshot = Path(spec["environment"]["PYTHONPATH"]).resolve()
    manifest_path = snapshot / "code_manifest.json"
    hashes = spec["required_sha256"]
    _hash(manifest_path, hashes[str(manifest_path)])
    manifest = _read(manifest_path)
    _require(manifest["format"] == "ffac-detached-code-snapshot-v1"
             and _same(manifest["destination"], snapshot),
             "Scheduled code snapshot identity changed")
    for entry in manifest["files"]:
        file = (snapshot / entry["path"]).resolve()
        _require(file.is_relative_to(snapshot), "Code snapshot path escaped its root")
        _require(file.stat().st_size == entry["bytes"], f"Code snapshot size changed: {file}")
        _hash(file, entry["sha256"])
        _require(hashes.get(str(file)) == entry["sha256"],
                 f"Scheduler and code manifest disagree: {file}")
    for path_text, digest in hashes.items():
        path = Path(path_text).resolve()
        if not path.is_relative_to(snapshot):
            _hash(path, digest)
    for path_text in spec["required_paths"]:
        _require(Path(path_text).exists(), f"Scheduled source path disappeared: {path_text}")
    return dict(spec_path=str(job_spec), spec_sha256=_hash(job_spec),
                snapshot=str(snapshot), code_manifest_sha256=_hash(manifest_path),
                frozen_files=len(manifest["files"]),
                prepared_request=argv[argv.index("--prepared-request") + 1],
                environment={key: spec["environment"][key]
                             for key in ("DARSIA_SERIES_REGISTRY", "DARSIA_FFAC_PROJECT_ROOT",
                                         "PHASE_PHASE_WORKERS") if key in spec["environment"]})


def _current_code() -> dict:
    from darsia.fluidflower.tools import phase_training_photometry_revision as revision
    from darsia.fluidflower.tools import prepare_optical_region_studio as studio
    return {str(path): _hash(path) for path in
            (Path(__file__).resolve(), Path(revision.__file__).resolve(),
             Path(studio.__file__).resolve())}


def preflight(source_job: Path, job_spec: Path, new_job: Path,
              candidate_name: str) -> dict:
    source_job, job_spec, new_job = (path.resolve() for path in
                                      (source_job, job_spec, new_job))
    _require(source_job.is_dir() and new_job.parent == source_job.parent
             and new_job != source_job and not new_job.exists(),
             "Recovery requires a fresh sibling detached results job")
    _require(CANDIDATE_NAME.fullmatch(candidate_name) is not None,
             "Choose a fresh phase-model-...-ffac-unified candidate name")
    candidate = new_job / "regional_candidate" / candidate_name
    frozen = verify_frozen_code(job_spec, source_job)
    status = _read(source_job / "regional_run" / "status.json")
    _require(status.get("stage") == "failed" and any(
        message in status.get("error", "") for message in (
            "Physical context does not match the exact AC47 photometry revision",
            "Physical context does not match the exact AC47 two-arm revision")),
        "Source is not the expected AC47 Studio-preparation failure")
    launch = _read(source_job / "regional_run" / "launch_request.json")
    optical_parent = Path(launch["parent"]).resolve()
    old_candidate = Path(launch["output"]).resolve()
    regional_run = source_job / "regional_run"
    _require(optical_parent.parent == source_job / "optical_parent"
             and old_candidate.parent == source_job / "regional_candidate"
             and _same(launch["job"], regional_run)
             and launch.get("review_candidate_only") is True
             and launch.get("studio_published") is False,
             "Source launch is not an isolated review candidate")
    _require(old_candidate.is_dir() and
             not (old_candidate / "studio_preparation.json").exists() and
             not (old_candidate / "validation").exists() and
             not (source_job / "result.json").exists(),
             "Source candidate has already advanced; do not reuse it")
    optical_prepared = _read(source_job / "optical_prepared.json")
    prepared_request = Path(optical_prepared["request_path"]).resolve()
    _require(_same(prepared_request, frozen["prepared_request"])
             and _hash(prepared_request) == optical_prepared["request_sha256"],
             "Optical prepared request differs from scheduled source")
    request, binding = two_arm.verify_prepared(prepared_request)
    _require(binding == _read(source_job / "two_arm_source_binding.json")
             and binding["arm"] in two_arm.ARMS
             and request["model_id"] == optical_parent.name,
             "Source binding, arm, or fitted optical identity changed")
    expected_error = ("Physical context does not match the exact AC47 two-arm revision"
                      if binding["arm"] == "global" else
                      "Physical context does not match the exact AC47 photometry revision")
    _require(expected_error in status.get("error", ""),
             "Failure does not match this two-arm source")
    fitted = verify_fitted(optical_parent)
    _require(fitted["series"] == "ffac" and fitted["adapters"] == ["generic"]
             and not fitted["nonconverged_heads"]
             and fitted["physics_runtime"] == binding["physics_runtime"],
             "Saved optical model is not the completed converged two-arm fit")
    convergence = _read(source_job / "optical_convergence.json")
    _require(set(convergence) == {"co2_presence", "gas_within_co2"}
             and all(row["all_selected_cv_folds_converged"] is True
                     and row["final_fit_converged"] is True for row in convergence.values()),
             "Saved optical CV or final fit did not converge")
    fitted_path = regional_run / "full_aqueous" / "regional_model.json"
    regional = _read(fitted_path)
    evidence = regional_run / "evidence.json"
    protocol = regional_run / "full_aqueous" / "protocol.json"
    _require(_same(regional["parent_model"], optical_parent)
             and _same(regional["evidence_path"], evidence)
             and regional["fit_report"]["converged"] is True,
             "Saved regional fit does not belong to the optical parent or did not converge")
    _hash(evidence, regional["evidence_sha256"])
    _hash(protocol, regional["protocol_sha256"])
    _require(_hash(old_candidate / "regional_model.json") == _hash(fitted_path),
             "Partial candidate does not contain the completed regional fit")
    extraction = _read(regional_run / "extraction.json")
    sample_path = Path(extraction["samples"]).resolve()
    _hash(sample_path, extraction["samples_sha256"])
    _require(len(extraction["frames"]) > 0 and len(extraction["coverage"]) > 0,
             "Regional extraction is incomplete")
    _hash(Path(binding["parent"]) / "summary.json", binding["parent_summary_sha256"])
    profile = phase_runtime.load_physics_profile(optical_parent)
    ac47_context = profile["runs"]["ffac/ac47"]
    collection = optical_parent / "review_source"
    jobs = _read(collection / "postflight_manifest.json")["jobs"]
    ac47 = [job for job in jobs if job["run"].lower() == "ac47"]
    _require(len(ac47) == 1, "Frozen collection lacks one exact AC47 source")
    _, native_path, _ = native_source(ac47[0])
    verify_context_binding(ac47[0], ac47_context["native_manifest_sha256"],
                           _hash(Path(native_path)), parent_context=ac47_context)
    return dict(format=FORMAT, source_job=str(source_job), new_job=str(new_job),
                candidate=str(candidate), optical_parent=str(optical_parent),
                original_partial_candidate=str(old_candidate),
                fitted_regional_model=str(fitted_path), fitted_regional_sha256=_hash(fitted_path),
                evidence_sha256=_hash(evidence), regional_samples=str(sample_path),
                regional_samples_sha256=extraction["samples_sha256"],
                regional_frames=len(extraction["frames"]),
                regional_regions=len(extraction["coverage"]),
                optical_model_sha256=fitted["model_sha256"],
                optical_selected_cv_folds={name: row["selected_cv_folds"]
                                           for name, row in convergence.items()},
                physical_context_ac47_sha256=ac47_context["sha256"],
                source_binding_sha256=_hash(source_job / "two_arm_source_binding.json"),
                source_status_sha256=_hash(regional_run / "status.json"),
                frozen_code=frozen, recovery_code_sha256=_current_code(),
                review_candidate_only=True, studio_published=False,
                optical_fit_reused=True, regional_fit_reused=True,
                native_studio_maps_built=False)


def _map_command(candidate: Path, *, cpus: int, memory_gb: float,
                 backend: str) -> list[str]:
    return [sys.executable, "-u", "-m",
            "darsia.fluidflower.tools.prepare_optical_region_studio",
            "--root", str(candidate), "--resume", "--fresh-optical",
            "--workers", "1", "--cpus", str(cpus),
            "--memory-gb", str(memory_gb), "--backend", backend]


def _run_maps(report: dict, *, cpus: int, memory_gb: float, backend: str) -> dict:
    candidate = Path(report["candidate"])
    launch = _read(Path(report["new_job"]) / "regional_run" / "launch_request.json")
    _require(launch["review_candidate_only"] is True
             and launch["studio_published"] is False
             and _same(launch["output"], candidate),
             "Recovery candidate lost its detached launch binding")
    _require(_current_code() == report["recovery_code_sha256"],
             "Recovery or Studio code changed after preflight")
    profile = _read(candidate / "phase_optical_regions.json")
    _require("bottom_gas_guard" in profile and "region_edge_guard" in profile,
             "Recovery candidate lacks both frozen regional guards")
    _hash(candidate / "regional_model.json", report["fitted_regional_sha256"])
    env = os.environ.copy()
    env.update(report["frozen_code"]["environment"])
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(_map_command(candidate, cpus=cpus, memory_gb=memory_gb,
                                backend=backend), cwd=process_directory(),
                   env=env, check=True)
    readiness = _read(candidate / "studio_preparation.json")
    if readiness.get("presentation_ready") is not True:
        subprocess.run([sys.executable, "-u", "-m",
                        "darsia.fluidflower.tools.finish_optical_region_studio",
                        "--root", str(candidate), "--cpus", str(cpus),
                        "--memory-gb", str(memory_gb)],
                       cwd=process_directory(), env=env, check=True)
        readiness = _read(candidate / "studio_preparation.json")
    _require(readiness.get("presentation_ready") is True,
             "Native Studio maps are not presentation ready")
    result = dict(report, native_studio_maps_built=True, qa_pending=True,
                  studio_published=False)
    _write_new(Path(report["new_job"]) / "result.json", result)
    return result


def prepare_only(report: dict) -> dict:
    new_job = Path(report["new_job"])
    new_job.mkdir(parents=False, exist_ok=False)
    _write_new(new_job / "recovery_preflight.json", report)
    launch = dict(format=FORMAT, parent=report["optical_parent"],
                  output=report["candidate"], job=str(new_job / "regional_run"),
                  source_job=report["source_job"],
                  fitted_regional_sha256=report["fitted_regional_sha256"],
                  bottom_gas_guard=True, region_edge_guard=True,
                  review_candidate_only=True, studio_published=False)
    _write_new(new_job / "regional_run" / "launch_request.json", launch)
    prepare(Path(report["fitted_regional_model"]).parent,
            Path(report["optical_parent"]) / "review_source",
            Path(report["candidate"]), fresh_optical=True,
            bottom_gas_guard=True, region_edge_guard=True)
    _write_new(new_job / "recovery_preparation.json", dict(
        format=FORMAT, candidate=report["candidate"],
        source_job=report["source_job"],
        fitted_regional_sha256=report["fitted_regional_sha256"],
        recovery_code_sha256=report["recovery_code_sha256"],
        review_candidate_only=True, studio_published=False))
    return dict(report, studio_prepared=True)


def execute(report: dict, *, cpus: int, memory_gb: float, backend: str) -> dict:
    prepare_only(report)
    return _run_maps(report, cpus=cpus, memory_gb=memory_gb, backend=backend)


def resume_maps(new_job: Path, *, cpus: int, memory_gb: float,
                backend: str) -> dict:
    new_job = new_job.resolve()
    report = _read(new_job / "recovery_preflight.json")
    prepared = _read(new_job / "recovery_preparation.json")
    _require(_same(report["new_job"], new_job)
             and prepared["format"] == FORMAT
             and prepared["candidate"] == report["candidate"]
             and prepared["fitted_regional_sha256"] == report["fitted_regional_sha256"]
             and prepared["recovery_code_sha256"] == report["recovery_code_sha256"]
             and prepared["review_candidate_only"] is True
             and prepared["studio_published"] is False
             and not (new_job / "result.json").exists(),
             "Only a prepared, incomplete detached recovery may resume maps")
    return _run_maps(report, cpus=cpus, memory_gb=memory_gb, backend=backend)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--resume-maps", action="store_true")
    parser.add_argument("--source-job", type=Path)
    parser.add_argument("--job-spec", type=Path)
    parser.add_argument("--new-job", type=Path, required=True)
    parser.add_argument("--candidate-name")
    parser.add_argument("--cpus", type=int, default=12)
    parser.add_argument("--memory-gb", type=float, default=64)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    _require(args.cpus > 0 and args.memory_gb > 0, "Positive CPU and memory budgets required")
    if args.resume_maps:
        result = resume_maps(args.new_job, cpus=args.cpus,
                             memory_gb=args.memory_gb, backend=args.backend)
    else:
        if args.source_job is None or args.job_spec is None or args.candidate_name is None:
            parser.error("--preflight/--prepare-only/--execute require source job, job spec, and candidate name")
        result = preflight(args.source_job, args.job_spec, args.new_job,
                           args.candidate_name)
        if args.prepare_only:
            result = prepare_only(result)
        elif args.execute:
            result = execute(result, cpus=args.cpus,
                             memory_gb=args.memory_gb, backend=args.backend)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
