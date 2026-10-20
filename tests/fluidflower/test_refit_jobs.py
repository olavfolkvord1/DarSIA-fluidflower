"""A refit job built from finished runs is admitted and run by the worker's own I/O path."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

import darsia
from darsia.fluidflower import job_filesystem
from darsia.fluidflower.pipeline import alias_job, refit_jobs, refit_runner, store_runtime
from test_n1_refit import TRUE_BETA, TRUE_GAIN, make_plan, make_rows

RUNS = ["r1", "r2"]
REPOSITORY = Path(darsia.__file__).resolve().parents[2]


def bound(path):
    data = Path(path).read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


@pytest.fixture
def series(tmp_path, monkeypatch):
    """A series with two finished runs' statistics, a baseline and a shared job folder."""
    admitted = job_filesystem.SharedFolderAdmission(True, "smb", "unit transport", "no filesystem qualification")
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda path: admitted)
    root = tmp_path / "series"
    (root / "calibration").mkdir(parents=True)
    (root / "series.toml").write_text('schema_version = 1\nseries_id = "TEST"\n\n[roots]\nseries = "."\n',
                                      encoding="utf-8")
    (tmp_path / "out").mkdir()
    phase_job = tmp_path / "jobs" / "phase-one"
    for row in make_rows(RUNS):
        attempt = root / "results" / "phase-one" / row["run"] / ".attempts" / "a1"
        frame = attempt / "frames" / f"{row['run']}_f{row['frame_index']}"
        frame.mkdir(parents=True)
        (frame / "native_statistics.json.gz").write_bytes(gzip.compress(json.dumps(row).encode()))
        manifest = attempt / "manifest.json"
        if not manifest.exists():
            manifest.write_text("{}", encoding="utf-8")
            completion = phase_job / "results" / row["run"] / "completed.json"
            completion.parent.mkdir(parents=True)
            completion.write_text(json.dumps(dict(complete=True, run=row["run"], manifest=bound(manifest))),
                                  encoding="utf-8")
    rules = make_plan(RUNS)["rules"]
    baseline = root / "calibration" / "baseline.json"
    baseline.write_text(json.dumps(dict(parameters=rules["baseline"])), encoding="utf-8")
    return dict(root=root, phase_job=phase_job, rules=rules, baseline=bound(baseline), tmp=tmp_path)


def prepare(series, loro=True):
    statistics = {run: refit_jobs.statistics_entries(refit_jobs.completed_attempt(series["phase_job"], run))
                  for run in RUNS}
    plans = refit_jobs.build_plans(statistics=statistics, roles=dict(fit=RUNS, heldout=["held"], excluded=[]),
                                   rules=series["rules"], baseline_calibration=series["baseline"], loro=loro)
    job = dict(commit="c" * 40, calibration_sha256=series["baseline"]["sha256"], precision="float32",
               created_utc="2026-10-21T00:00:00+00:00", environment_lock=dict(path="uv.lock", sha256="d" * 64))
    execution = refit_jobs.write_refit_inputs(series["root"] / "calibration" / "refit" / "job1", plans,
                                              job=job, job_id="refit-job1")
    job_root = series["tmp"] / "jobs" / "refit-job1"
    prepared = refit_jobs.prepare_refit_job(execution, job_root, series_manifest=bound(series["root"] / "series.toml"))
    return prepared, job_root


def test_statistics_are_found_from_published_completions(series):
    entries = refit_jobs.statistics_entries(refit_jobs.completed_attempt(series["phase_job"], "r1"))
    assert [entry["frame_index"] for entry in entries] == list(range(1, 24))
    assert entries[0]["case_id"] == "r1_f1" and entries[0]["selected_target_h"] == .5


def test_the_job_has_a_refit_task_and_one_fold_per_fit_run(series):
    prepared, job_root = prepare(series)
    assert [task["run"] for task in prepared["tasks"]] == ["refit", "loro-r1", "loro-r2"]
    job = json.loads((job_root / "job.json").read_text(encoding="utf-8"))
    assert job["pipeline"] == refit_runner.PIPELINE and job["expected_runs"] == ["refit", "loro-r1", "loro-r2"]
    task = json.loads(Path(prepared["tasks"][0]["task"]["path"]).read_text(encoding="utf-8"))
    assert task["pipeline"] == refit_runner.PIPELINE and task["calibration_sha256"] == series["baseline"]["sha256"]
    uses = {use for item in task["input_catalog"]["inputs"] for use in item["uses"]}
    assert {"run.execution_series", "run.series_manifest", "refit.baseline_calibration"} <= uses
    assert sum(use.endswith(".statistics") for use in uses) == 2 * 23
    with pytest.raises(FileExistsError):
        refit_jobs.prepare_refit_job(job_root.parent.parent / "series" / "calibration" / "refit" / "job1" /
                                     "execution.toml", job_root,
                                     series_manifest=bound(series["root"] / "series.toml"))


def test_the_worker_replay_admits_a_refit_task_and_the_fit_runs_under_it(series, monkeypatch):
    from darsia.fluidflower.pipeline.job_queue import LocalOutputBinding
    from darsia.fluidflower.pipeline.run_io import RunIOBinding
    from darsia.fluidflower.store_map import parse_store_map
    prepared, _ = prepare(series, loro=False)
    task = json.loads(Path(prepared["tasks"][0]["task"]["path"]).read_text(encoding="utf-8"))
    task["attempt_id"] = "e" * 32  # the queue's claim gives each attempt its identity
    stores = parse_store_map(('[machine]\nid = "Main"\n\n[stores.TEST]\n'
                              f"series = '{series['root']}'\nresults = '{series['tmp'] / 'out'}'\n").encode())
    rows = store_runtime.resolve_catalog(task, stores, "TEST")
    verified = store_runtime.verify_inputs(task, rows)
    output = series["tmp"] / "out" / "refit-job1" / "refit" / ".attempts" / task["attempt_id"]
    admission = job_filesystem.LocalFileAdmission(True, "NTFS", "unit results root", "unit",
                                                  checked_container=str(series["tmp"] / "out"))
    binding = RunIOBinding.replay(task, verified,
                                  LocalOutputBinding("Main", series["tmp"] / "out", output, admission),
                                  repository=REPOSITORY, admitted_commit=task["commit"])
    with binding.activate():
        result = refit_runner.run_refit(task, output, "cpu", "float32")
    assert result["complete"] and result["frozen_calibration"] is not None
    assert result["parameters"]["gain"] == pytest.approx(TRUE_GAIN, rel=1e-4)
    assert result["parameters"]["beta"] == pytest.approx(TRUE_BETA, rel=1e-3)
