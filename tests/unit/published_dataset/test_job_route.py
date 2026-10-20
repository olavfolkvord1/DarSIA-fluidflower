"""The launcher's job route: a moved dataset runs through the job layer's own preparation.

The shared-folder admission and the role catalogue are test doubles, as in the
job layer's own lifecycle tests; the plan, run configurations, selection and
series manifest are read by the job layer's real readers. A fake worker writes
the finished runs that a real worker would publish.
"""

import json
import os
from pathlib import Path
import shutil
import tomllib
import zipfile

import pytest

from darsia.fluidflower.job_filesystem import SharedFolderAdmission
from darsia.fluidflower.pipeline import alias_job, job_inputs
from darsia.fluidflower.published_dataset import builder, job_route, scan
from darsia.fluidflower.published_dataset.comparison import IDENTICAL
from darsia.fluidflower.published_dataset.reproduce import Reproduction, Settings, choose_scope, execute
from tests.unit.published_dataset.synthetic import COMMIT, make_series, write_run_results

INSTALLED = {"commit": COMMIT, "clean": True, "source": "Git checkout"}


class TinyCatalog(job_inputs.InputCatalog):
    """The configuration and the calibration only (the real role graph is tested elsewhere)."""

    def collect(self, bound, config):
        reference = self.add(bound, "configuration", "run.configuration")
        self.add(config["mass"]["calibration"], "calibration", "mass.calibration")
        return dict(format=job_inputs.CATALOG_FORMAT, complete=True, run=config["run"], configuration=reference,
                    inputs=list(self.entries.values()), payload_hashes_verified=False, numerical_calls_started=False)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """A built R record, downloaded elsewhere and unpacked (raw archives stay packed)."""
    monkeypatch.setattr(builder, "machine_names", lambda: (
        scan.ScanName("example-build-host", "build machine host name"),))
    series = make_series(tmp_path / "producer-site")
    report = builder.build_record(builder.BuildRequest(series.series, series.job, "R", "S50",
                                                       tmp_path / "record", deny=(), fingerprint=False))
    assert report.built, report.text()
    folder = tmp_path / "downloads" / "dataset"
    shutil.copytree(tmp_path / "record", folder)
    for name in ("series.zip", "expected.zip"):
        with zipfile.ZipFile(folder / name) as archive:
            archive.extractall(folder)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    monkeypatch.setattr(alias_job, "shared_output_admission",
                        lambda root: SharedFolderAdmission(True, "smb", "test double", "fixture"))
    monkeypatch.setattr(alias_job, "InputCatalog", TinyCatalog)
    series.dataset = folder
    return series


def fake_worker(series, launches):
    """A stand-in for the job layer's workers: finishes every queued run with the expected layer."""
    def launch(job_root, *, backend, workers, machine, stores, exit_when_idle):
        launches.append(dict(backend=backend, workers=workers, machine=machine, stores=stores,
                             exit_when_idle=exit_when_idle))
        store = tomllib.loads(Path(stores).read_text(encoding="utf-8"))
        results = Path(next(iter(store["stores"].values()))["results"])
        for queued in sorted((Path(job_root) / "queue").glob("*.json")):
            task = json.loads(queued.read_text())
            run = task["run"]
            write_run_results(Path(job_root), results, run, series.runs.index(run), task["ordered_case_ids"],
                              series.raws[run], token=task["attempt_id"], machine="reproduction")
            queued.unlink()
    return launch


def snapshot(folder):
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in Path(folder).rglob("*")}


def test_a_moved_dataset_reproduces_identically_and_stays_read_only(dataset, tmp_path):
    launches = []
    runner = job_route.JobLayerRunner(launch=fake_worker(dataset, launches), poll_seconds=0,
                                      sleep=lambda seconds: None, installed=INSTALLED)
    reproduction = Reproduction(dataset.dataset, tmp_path / "results", runner=runner)
    check = reproduction.open(installed=INSTALLED, fingerprint={})
    assert check.problems == [] and check.code["same"] is True
    before = snapshot(dataset.dataset)
    scope = choose_scope(check.manifest, "all")
    settings = Settings(workers=2, threads=2, remove_scratch=True)
    states = []
    result = execute(reproduction, scope, settings, progress=states.append,
                     machine={"cpu": "fixture"})
    assert result["summary"]["dataset"] == IDENTICAL and result["summary"]["reproduced"] is True
    assert snapshot(dataset.dataset) == before
    assert states[-1] == {"r1": "done", "r2": "done"}
    assert launches == [dict(backend="cpu", workers=2, machine="reproduction", stores=runner.setup["stores"],
                             exit_when_idle=True)]
    # The job layer read the published plan through the launcher's aliases.
    plan = tomllib.loads(runner.setup["plan"].read_text(encoding="utf-8"))
    aliases = {row["logical"]["path"]: row["physical"]["path"] for row in plan["data_aliases"]}
    assert Path(aliases["config/runs/r1.json"]) == (dataset.dataset / "config" / "runs" / "r1.json").resolve()
    assert Path(aliases["calibration/frozen.json"]) == (dataset.dataset / "calibration" / "frozen.json").resolve()
    job = json.loads((tmp_path / "results" / "job" / "job.json").read_text())
    assert job["commit"] == COMMIT and job["expected_runs"] == ["r1", "r2"]
    task = json.loads((tmp_path / "results" / "job" / "results" / "r1" / "task.json").read_text())
    inputs = {entry["reference"]["relative_path"]: entry["reference"]["root_alias"]
              for entry in task["input_catalog"]["inputs"]}
    assert inputs["config/runs/r1.json"] == "dataset" and inputs["calibration/frozen.json"] == "dataset"
    series_toml = tomllib.loads((tmp_path / "results" / "setup" / "series" / "series.toml").read_text())
    assert series_toml["roots"]["series"] == "." and Path(series_toml["roots"]["dataset"]) == dataset.dataset.resolve()
    assert runner.setup["stores"].is_relative_to(tmp_path / "local-app-data")
    # Raw images were unpacked into scratch and removed after their run.
    assert not any((tmp_path / "results" / "scratch").rglob("*.ARW"))
    report = json.loads(result["files"]["json"].read_text(encoding="utf-8"))
    assert report["verdict"]["dataset"] == IDENTICAL and report["reproduction"] is True
    assert result["files"]["html"].is_file() and result["files"]["figure r1"].is_file()
    text = result["files"]["json"].read_text(encoding="utf-8") + result["files"]["html"].read_text(encoding="utf-8")
    scanner = scan.NameScanner(scan.machine_names())
    assert not [hit for hit in scanner.text(text, file="report", where="all") if hit.kind == "name"]
    assert str(tmp_path) not in text and str(dataset.dataset) not in text


def test_quick_check_runs_the_first_frames_and_keeps_scratch_on_request(dataset, tmp_path):
    runner = job_route.JobLayerRunner(launch=fake_worker(dataset, []), poll_seconds=0, sleep=lambda s: None,
                                      installed=INSTALLED)
    reproduction = Reproduction(dataset.dataset, tmp_path / "quick", runner=runner)
    check = reproduction.open(installed=INSTALLED, fingerprint={})
    scope = choose_scope(check.manifest, "quick", run="r2", quick_frames=2)
    assert scope.runs == ["r2"] and scope.frames == {"r2": 2}
    result = execute(reproduction, scope, Settings(workers=1, threads=2, remove_scratch=False))
    assert result["summary"]["frames"][IDENTICAL] == 2
    task = json.loads((tmp_path / "quick" / "job" / "results" / "r2" / "task.json").read_text())
    assert task["ordered_case_ids"] == ["r2_f1", "r2_f2"]
    quick = json.loads((tmp_path / "quick" / "setup" / "series" / "quick" / "r2.json").read_text())
    assert [frame["case_id"] for frame in quick["frames"]] == ["r2_f1", "r2_f2"]
    assert sorted(path.name for path in (tmp_path / "quick" / "scratch" / "raw" / "r2").iterdir()) == [
        "BASE_0000.ARW", "IMG_0001.ARW", "IMG_0002.ARW", "IMG_0003.ARW"]


def test_pause_and_resume_publish_the_existing_job_controls(dataset, tmp_path):
    launches = []
    runner = job_route.JobLayerRunner(launch=fake_worker(dataset, launches), poll_seconds=0,
                                      sleep=lambda s: None, installed=INSTALLED)
    reproduction = Reproduction(dataset.dataset, tmp_path / "paused", runner=runner)
    check = reproduction.open(installed=INSTALLED, fingerprint={})
    settings = Settings(workers=1, threads=2)
    execute(reproduction, choose_scope(check.manifest, "run", run="r1"), settings)
    from darsia.fluidflower.job_control import read_control

    runner.pause()
    assert read_control(runner.job_root)["action"] == "pause" and runner.paused
    runner.resume(settings)
    assert read_control(runner.job_root)["action"] == "run" and len(launches) == 2


def test_a_results_folder_off_a_share_is_refused_with_its_reason(tmp_path):
    problems = job_route.JobLayerRunner().preflight(tmp_path / "results")
    assert len(problems) == 1 and "SMB or NFS share" in problems[0]


def test_helpers_bind_roots_aliases_and_progress(tmp_path):
    text = 'series_id = "X"\n[roots]\nseries = "."\nraw = "//share/x"\n[paths]\nresults = "results"\n'
    replaced = job_route.with_roots(text, {"series": ".", "raw": "D:/raw"})
    assert tomllib.loads(replaced)["roots"] == {"series": ".", "raw": "D:/raw"}
    assert tomllib.loads(replaced)["paths"] == {"results": "results"}
    assert tomllib.loads(job_route.with_roots('series_id = "X"\n', {"series": "."}))["roots"] == {"series": "."}
    document = {"a": {"path": "x.json", "sha256": "1" * 64}, "b": [{"path": "y", "sha256": "2" * 64, "n": 1}],
                "c": {"path": "no-sha"}}
    assert [item["path"] for item in job_route.descriptors(document)] == ["x.json", "y"]
    (tmp_path / "x.json").write_text("x")
    rows = job_route.data_aliases([document, document], {"1" * 64: tmp_path / "x.json"})
    assert rows == [{"logical": {"path": "x.json", "sha256": "1" * 64},
                     "physical": {"path": str(tmp_path / "x.json"), "sha256": "1" * 64, "size_bytes": 1}}]
    with pytest.raises(ValueError):
        job_route.toml_text({"a": None})
    (tmp_path / "job" / "done").mkdir(parents=True)
    (tmp_path / "job" / "done" / "r1.json").write_text("{}")
    (tmp_path / "job" / "failed").mkdir()
    (tmp_path / "job" / "failed" / "r2.json").write_text("{}")
    (tmp_path / "job" / "queue").mkdir()
    (tmp_path / "job" / "queue" / "r3.json").write_text("{}")
    assert job_route.job_progress(tmp_path / "job", ["r1", "r2", "r3", "r4"]) == {
        "r1": "done", "r2": "failed", "r3": "queued", "r4": "running"}
    assert job_route.store_map_path("reproduction-1", environment={"LOCALAPPDATA": str(tmp_path)}) == (
        tmp_path / "DarSIA" / "reproductions" / "reproduction-1" / "stores.toml") if os.name == "nt" else True
