"""Create a refit job (pipeline ``n1-calibration-refit``) from finished phase-one runs.

The job has one task ``refit`` (fit on every fit run, then freeze) and, when
asked, one task ``loro-<run>`` per fit run (the same fit without that run, and
its predictions). Each task gets its own ``darsia-n1-refit-plan-v1`` plan: the
fit runs' per-frame statistics files, the baseline calibration and the rules.
The plans and an execution TOML that declares every task are written to a new
folder under the series root; the job and its tasks are then written as for
any alias job, with the same role catalogue, so the worker's replay verifies
every statistics file before the fit reads it.

Series values (run roles and rules) are arguments, read by the caller from the
series' own configuration. Nothing here computes a fit.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PureWindowsPath

STATISTICS_NAME = "native_statistics.json.gz"
ENVIRONMENT_LOCK = "requirements-worker.lock"
REFIT_FOLDER = ("calibration", "refit")


def _descriptor(path):
    """Full path, SHA-256 and size of an existing file."""
    data = Path(path).read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def completed_attempt(job_root, run):
    """The attempt folder of a run's published completion in a finished job."""
    from .job_queue import name, read_json
    completion = read_json(Path(job_root) / "results" / name(run) / "completed.json")
    if completion.get("complete") is not True or completion.get("run") != run:
        raise ValueError("Run has no complete published result: " + run)
    manifest = completion["manifest"]
    if _descriptor(manifest["path"])["sha256"] != manifest["sha256"]:
        raise ValueError("Completed run manifest changed: " + run)
    return Path(manifest["path"]).parent


def statistics_entries(attempt):
    """Every per-frame statistics file of one attempt, by frame index.

    Phase one keeps them under ``preparation/<case>``, the native-phase-n1
    route under ``frames/<case>``; an attempt uses exactly one of the two.
    """
    attempt = Path(attempt)
    found = [sorted(attempt.glob(folder + "/*/" + STATISTICS_NAME)) for folder in ("preparation", "frames")]
    paths = [paths for paths in found if paths]
    if len(paths) != 1:
        raise ValueError("Attempt needs per-frame statistics in exactly one layout: " + str(attempt))
    entries = []
    for path in paths[0]:
        row = json.loads(gzip.decompress(path.read_bytes()))
        if row.get("selected_target_h") is None:
            raise ValueError("Statistics row lacks its selected target time: " + str(path))
        entries.append(dict(case_id=path.parent.name, frame_index=row["frame_index"],
                            selected_target_h=row["selected_target_h"], statistics=_descriptor(path)))
    entries.sort(key=lambda entry: entry["frame_index"])
    if len({entry["frame_index"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate frame indices in " + str(attempt))
    return entries


def build_plans(*, statistics, roles, rules, baseline_calibration, loro=True, variant_f=None):
    """One refit plan per task: 'refit' and, if asked, 'loro-<run>' for every fit run."""
    from .refit_runner import FOLD_PREFIX, GLOBAL_RUN, PLAN_FORMAT, validate_plan
    baseline = json.loads(Path(baseline_calibration["path"]).read_text(encoding="utf-8-sig"))
    if baseline.get("parameters") != rules["baseline"]:
        raise ValueError("The rules' baseline parameters differ from the baseline calibration")
    common = dict(format=PLAN_FORMAT, runs=deepcopy(roles), rules=deepcopy(rules),
                  statistics={run: deepcopy(statistics[run]) for run in roles["fit"]},
                  baseline=dict(calibration=dict(baseline_calibration)),
                  variant_f=dict(variant_f or {"enabled": False}),
                  colour_paths="as in the phase-one statistics")
    names = [(GLOBAL_RUN, None)] + ([(FOLD_PREFIX + run, run) for run in roles["fit"]] if loro else [])
    plans = {}
    for run, fold in names:
        plan = dict(common, run=run, fold=fold)
        validate_plan(plan)
        plans[run] = plan
    return plans


def write_refit_inputs(folder, plans, *, job, job_id):
    """Write every plan and the execution TOML that declares them; return its path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    lines = ['format = "darsia-series-v1"', "adopted_for_execution = true", f'job_id = "{job_id}"', "",
             "[job]"]
    for key in ("commit", "calibration_sha256", "precision", "created_utc"):
        lines.append(f'{key} = "{job[key]}"')
    lines += ["", "[job.environment_lock]", f'path = "{job["environment_lock"]["path"]}"',
              f'sha256 = "{job["environment_lock"]["sha256"]}"']
    for run, plan in plans.items():
        path = folder / (run + "_plan.json")
        path.write_bytes(json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        bound = _descriptor(path)
        lines += ["", "[[runs]]", f'run = "{run}"', "", "[runs.configuration]",
                  'path = "' + str(path).replace("\\", "\\\\") + '"', f'sha256 = "{bound["sha256"]}"',
                  f'size_bytes = {bound["size_bytes"]}', "", "[runs.resources]", 'backend = "cpu"', "threads = 1"]
    execution = folder / "execution.toml"
    execution.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return execution


def job_identity(repository, *, calibration_sha256, precision="float32"):
    """Commit, environment lock and time for a new job from a clean checkout.

    The worker starts a job only from a clean checkout at the job's commit, so
    the job takes the commit and tracked lock of the checkout that creates it.
    """
    import subprocess
    repository = Path(repository)
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip():
        raise ValueError("A refit job is created from a clean checkout")
    lock = subprocess.check_output(["git", "-C", str(repository), "show", head + ":" + ENVIRONMENT_LOCK])
    return dict(commit=head, calibration_sha256=calibration_sha256, precision=precision,
                created_utc=datetime.now(timezone.utc).isoformat(),
                environment_lock=dict(path=ENVIRONMENT_LOCK, sha256=hashlib.sha256(lock).hexdigest()))


def create_refit_job(series_manifest, *, phase_job, roles, rules, baseline_calibration, job_id, jobs_root,
                     job, loro=True):
    """Find the fit runs' statistics, write the plans and create the queued refit job.

    ``series_manifest`` is the bound series.toml; the plans go to
    ``calibration/refit/<job id>`` under its folder, the job to ``jobs_root``.
    ``job`` comes from :func:`job_identity`. Returns the prepared job record.
    """
    from .job_queue import name
    name(job_id)
    statistics = {run: statistics_entries(completed_attempt(phase_job, run)) for run in roles["fit"]}
    plans = build_plans(statistics=statistics, roles=roles, rules=rules,
                        baseline_calibration=baseline_calibration, loro=loro)
    folder = Path(series_manifest["path"]).parent.joinpath(*REFIT_FOLDER, job_id)
    execution = write_refit_inputs(folder, plans, job=job, job_id=job_id)
    return prepare_refit_job(execution, Path(jobs_root) / job_id, series_manifest=series_manifest)


def prepare_refit_job(execution, job_root, *, series_manifest, storage_profile=None):
    """Write job.json and enqueue one alias task per declared refit run.

    ``series_manifest`` is the bound series.toml descriptor. The tasks carry
    the same catalogue, series and manifest references as any alias task.
    """
    import tomllib
    from .. import storage_profile as storage
    from .alias_job import JOB_FORMAT, TASK_FORMAT, _manifest, shared_output_admission
    from .job_inputs import InputCatalog, validate_catalog
    from .job_queue import JobQueue, descriptor, name, validate_task, write_immutable
    from .refit_runner import PIPELINE
    profile = storage.new_job_profile(storage_profile)
    shared_output_admission(job_root)
    _, manifest, roots = _manifest(series_manifest)
    execution_bound = _descriptor(execution)
    value = tomllib.loads(Path(execution).read_text(encoding="utf-8"))
    job_id = name(value["job_id"])
    admission = InputCatalog(roots)
    series_reference = admission.add(execution_bound, "configuration", "execution_series")
    manifest_reference = admission.add(series_manifest, "configuration", "series_manifest")
    job = dict(value["job"])
    lock = job["environment_lock"]
    if PureWindowsPath(lock["path"]).is_absolute() or ".." in PureWindowsPath(lock["path"]).parts:
        raise ValueError("Environment lock must remain repository-relative")
    version = storage.FORMAT_VERSIONS[profile]
    tasks = []
    for row in value["runs"]:
        plan = json.loads(Path(row["configuration"]["path"]).read_text(encoding="utf-8"))
        if plan.get("run") != row["run"]:
            raise ValueError("Execution TOML and plan runs differ")
        catalog = InputCatalog(roots)
        catalog.add(execution_bound, "configuration", "run.execution_series")
        catalog.add(series_manifest, "configuration", "run.series_manifest")
        inputs = validate_catalog(catalog.collect(row["configuration"], plan))
        inputs["runtime_support"] = "darsia-role-io-runtime-v1"
        cases = [entry["case_id"] for run in sorted(plan["statistics"]) for entry in plan["statistics"][run]]
        task = dict(deepcopy(row["resources"]), pipeline=PIPELINE, run=row["run"],
                    commit=job["commit"], calibration_sha256=job["calibration_sha256"], precision=job["precision"],
                    job_id=job_id, format=TASK_FORMAT, format_version=version, legacy_paths=False,
                    execution_ready=True, execution_blocker="", configuration=inputs["configuration"],
                    series=series_reference, series_manifest=manifest_reference, input_catalog=inputs,
                    ordered_case_ids=cases, storage_profile=profile)
        validate_task(task)
        tasks.append(task)
    job = dict(job, expected_runs=[task["run"] for task in tasks], series=series_reference, format=JOB_FORMAT,
               format_version=version, legacy_paths=False, execution_ready=True, execution_blocker="",
               runtime_support="darsia-role-io-runtime-v1", series_manifest=manifest_reference,
               series_id=manifest["series_id"], input_reference_format="root-alias-relative-v1",
               storage_profile=profile, pipeline=PIPELINE, publish=[])
    root = Path(job_root)
    if (root / "job.json").exists():
        raise FileExistsError("A refit job is created once: " + str(root))
    root.mkdir(parents=True, exist_ok=True)
    write_immutable(root / "job.json", job)
    queue = JobQueue(root)
    prepared = [dict(run=task["run"], task=descriptor(queue.enqueue(task))) for task in tasks]
    return dict(complete=True, job=descriptor(root / "job.json"), tasks=prepared, pipeline=PIPELINE,
                numerical_calls_started=False, created_utc=datetime.now(timezone.utc).isoformat())
