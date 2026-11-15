"""Tests of the job engine: jobs, their queue and the step context, in this process."""

import importlib.metadata
import json
import os
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import pytest
from threadpoolctl import threadpool_info

from darsia.jobs import job as job_module
from darsia.jobs import queue
from darsia.jobs.job import create_job, locations_of, read_job, read_tables, revision
from darsia.jobs.step import (
    Memory,
    Order,
    Scope,
    Selection,
    Step,
    StepContext,
    file_identity,
    table,
)
from darsia.presets.workflows.config.utils import settings_from_table
from darsia.utils import results
from darsia.utils.locations import Frame

ROOT = Path(__file__).resolve().parents[2]
KERNEL = "Haswell"


@pytest.fixture(autouse=True)
def no_share_wait(monkeypatch):
    """A local folder has no share's cache: the looks again do not wait."""
    monkeypatch.setattr(results, "SHARE_CACHE", 0.0)


@dataclass(frozen=True)
class ToySettings:
    """Settings of the test steps."""

    scale: float
    """Factor of every stored number."""
    fail: tuple[str, ...] = ()
    """Parts that raise."""
    pause: float = 0.0
    """Seconds a frame takes."""


def _write(folder, **values):
    """Store numbers as one JSON file."""
    (Path(folder) / "values.json").write_text(json.dumps(values), encoding="utf-8")


def read_values(folder):
    """A test result's numbers."""
    return json.loads((Path(folder) / "values.json").read_text(encoding="utf-8"))


def _numbers(ctx: StepContext) -> None:
    """A number per frame, with the process's environment and when it ran."""
    toy = ctx.settings["toy"]
    for frame in ctx.task.frames:
        if ctx.done(frame):
            continue
        began = time.time()
        time.sleep(toy.pause)
        kernels = [
            row["architecture"]
            for row in threadpool_info()
            if row.get("internal_api") == "openblas"
        ]
        value = float(np.float32(toy.scale) * np.float32(len(frame.name)))
        values = dict(
            value=value,
            kernel=os.environ.get("OPENBLAS_CORETYPE"),
            found=kernels,
            threads=os.environ.get("OPENBLAS_NUM_THREADS"),
            began=began,
            ended=time.time(),
        )
        ctx.publish(lambda folder: _write(folder, **values), frame=frame)


def _sums(ctx: StepContext) -> None:
    """The running sum of a run's numbers, in frame order."""
    total, order = 0.0, []
    for frame in ctx.task.frames:
        total += ctx.read("numbers", read_values, frame=frame)["value"]
        order.append(frame.name)
    ctx.publish(lambda folder: _write(folder, total=total, order=order))


def _parts(selection, tables):
    """A part per letter."""
    return ("a", "b", "c")


def _fit(ctx: StepContext) -> None:
    """A part over every run's sum; a part named in the settings raises."""
    toy = ctx.settings["toy"]
    if ctx.task.part in toy.fail:
        raise RuntimeError(f"Part {ctx.task.part} fails")
    total = sum(
        ctx.read("sums", read_values, run=run)["total"] for run in ctx.selection.frames
    )
    ctx.publish(lambda folder: _write(folder, total=total * toy.scale))


def _summary(ctx: StepContext) -> None:
    """Every part of the fit."""
    totals = [ctx.read("fit", read_values, part=p)["total"] for p in _parts(0, 0)]
    ctx.publish(lambda folder: _write(folder, totals=totals))


def _files(tables, project, folders):
    """The toy's protocol and model folder, which its configuration names."""
    return [tables["protocols"]["injection"], tables["model"]["folder"]]


NUMBERS = Step(
    "numbers",
    {},
    Order.FRAMES,
    Memory(),
    _numbers,
    {"toy": ToySettings},
    sections=("protocols",),
    files=_files,
)
SUMS = Step("sums", {"numbers": Scope.OWN}, Order.RUN, Memory(), _sums)
DOMAIN_SUMS = Step(
    "domain-sums", {"numbers": Scope.OWN}, Order.RUN, Memory(), _sums, domains=True
)
FIT = Step(
    "fit",
    {"sums": Scope.ALL},
    Order.PARTS,
    Memory(),
    _fit,
    {"toy": ToySettings},
    _parts,
)
SUMMARY = Step("summary", {"fit": Scope.ALL}, Order.PARTS, Memory(), _summary)
STEPS = (NUMBERS, SUMS, DOMAIN_SUMS, FIT, SUMMARY)
"""The test steps, which a task process finds by their module."""


def _project(
    tmp_path, scale=2.0, fail=(), pause=0.0, kernel=KERNEL, threads=1, backend="cpu"
):
    """A project folder with its configuration, its protocol and model folder, and a
    selection of two runs."""
    project = tmp_path / "project"
    (project / "model").mkdir(parents=True, exist_ok=True)
    (project / "injection.csv").write_text("start,rate\n0,1\n", encoding="utf-8")
    (project / "model" / "head.json").write_text("[1, 2]\n", encoding="utf-8")
    config = project / f"config-{kernel}-{threads}-{scale}-{backend}.toml"
    config.write_text(
        f'[data]\nresults = "results"\n[jobs]\nblas_kernel = "{kernel}"\n'
        f'backend = "{backend}"\nthreads = {threads}\n'
        f"[toy]\nscale = {scale}\nfail = {list(fail)}\n"
        f'pause = {pause}\n[protocols]\ninjection = "injection.csv"\n'
        '[model]\nfolder = "model"\n',
        encoding="utf-8",
    )
    frames = {
        run: tuple(Frame("day1", f"{run}-{k:02d}.npz") for k in range(count))
        for run, count in (("r1", 6), ("r2", 3))
    }
    return config, Selection(frames, {run: {"day1": "photos"} for run in frames})


def context(job: Path, task, token: str) -> StepContext:
    """A task's context in this process, as a task process makes it."""
    record = read_job(job)
    step = {s.name: s for s in STEPS}[task.step]
    config = [job / name for name in record["config"]]
    tables = read_tables(config)
    settings = {
        section: settings_from_table(kind, table(tables, section))
        for section, kind in step.settings.items()
    }
    selection = Selection.from_json(record["selection"])
    locations = locations_of(job, record)
    return StepContext(
        step,
        task,
        selection,
        config,
        tables,
        settings,
        locations,
        token,
        record,
        lambda name: queue.accepted(job, name),
        revision(),
    )


def work(job: Path, machine: str = "here") -> None:
    """Run a job's ready tasks in this process until none is ready."""
    while ready := queue.ready(job):
        for task in ready:
            claim = queue.claim(job, task.id, machine)
            queue.beat(claim, 1, os.getpid(), psutil.Process().create_time())
            ctx = context(job, task, claim.token)
            try:
                ctx.step.run(ctx)
                queue.finish(claim, ctx.adopted)
            except Exception:
                queue.finish(claim, traceback=traceback.format_exc())


def _stored(project, key):
    """A stored result's description and numbers."""
    folder = project / "results" / key
    return results.read_description(folder), read_values(folder)


def test_a_global_fit_waits_for_every_selected_run(tmp_path):
    """The fit of every run's sums is not ready while one run's sum is unfinished;
    its inputs are every selected run's sum."""
    config, selection = _project(tmp_path)
    job = create_job(
        tmp_path / "jobs", (NUMBERS, SUMS, FIT, SUMMARY), selection, config
    )
    fit = queue.tasks(job)["fit.a"]
    assert [i.key for i in fit.inputs] == ["sums/r1", "sums/r2"]
    assert set(fit.after) == {"sums.r1", "sums.r2"}
    for task in queue.ready(job):  # every number
        claim = queue.claim(job, task.id, "here")
        ctx = context(job, task, claim.token)
        ctx.step.run(ctx)
        queue.finish(claim)
    assert {t.id for t in queue.ready(job)} == {"sums.r1", "sums.r2"}
    claim = queue.claim(job, "sums.r1", "here")
    ctx = context(job, queue.tasks(job)["sums.r1"], claim.token)
    ctx.step.run(ctx)
    queue.finish(claim)
    assert {t.id for t in queue.ready(job)} == {"sums.r2"}
    work(job)
    assert queue.waiting(job) == set()
    assert _stored(tmp_path / "project", "fit/a")[1]["total"] == 324.0


def test_an_input_made_outside_the_job_is_bound_by_its_token(tmp_path):
    """A second job reads the first job's sums by their tokens, reuses its stored
    results, and is refused when a stored result does not serve it."""
    config, selection = _project(tmp_path)
    jobs = tmp_path / "jobs"
    first = create_job(jobs, (NUMBERS, SUMS), selection, config)
    work(first)
    second = create_job(jobs, (FIT,), selection, config, known=(SUMS,))
    tokens = read_job(second)["bound"]
    assert set(tokens) == {"sums/r1", "sums/r2"}
    assert tokens["sums/r1"] in queue.accepted(first, "sums.r1")
    again = create_job(jobs, (NUMBERS, SUMS), selection, config)
    assert queue.tasks(again) == {}
    other, _ = _project(tmp_path, scale=3.0)
    with pytest.raises(ValueError, match="numbers/r1/day1/r1-00.npz is stored and"):
        create_job(jobs, (NUMBERS,), selection, other)
    with pytest.raises(ValueError, match="sums/r1, which fit reads, is missing"):
        create_job(jobs, (FIT,), selection, config, known=(SUMS,), project=tmp_path)


def test_a_superseded_owner_commits_nothing(tmp_path, monkeypatch):
    """A beat between the reclaimer's stale read and its rename, and a paused owner
    returning after it was superseded: neither superseded token reaches a done
    marker, and readers see only the accepted token."""
    config, selection = _project(tmp_path)
    job = create_job(tmp_path / "jobs", (NUMBERS,), selection, config)
    task = queue.tasks(job)["numbers.r2.0000"]
    first = queue.claim(job, task.id, "here")
    queue.beat(first, 1, os.getpid(), 0.0)
    seen = {}
    assert queue.reclaim(job, seen) == []  # first seen now, by this clock
    assert queue.reclaim(job, seen) == []  # unchanged, but not for STALE yet
    key = next(iter(seen))
    seen[key] = (seen[key][0], seen[key][1] - results.STALE)  # stale by our clock
    content = queue._content

    def read_then_beat(path):
        found = content(path)
        queue.beat(first, 2, os.getpid(), 0.0)  # the owner's beat comes in between
        return found

    monkeypatch.setattr(queue, "_content", read_then_beat)
    assert queue.reclaim(job, seen) == [task.id]
    monkeypatch.setattr(queue, "_content", content)
    with pytest.raises(queue.Revoked):
        queue.beat(first, 3, os.getpid(), 0.0)
    # The next owner is paused before it publishes; a third claims and finishes.
    paused = queue.claim(job, task.id, "here")
    paused_ctx = context(job, task, paused.token)
    queue.reclaim(job, seen)
    key = next(k for k in seen if paused.token in k)
    seen[key] = (seen[key][0], seen[key][1] - results.STALE)
    assert queue.reclaim(job, seen) == [task.id]
    last = queue.claim(job, task.id, "here")
    ctx = context(job, task, last.token)
    ctx.step.run(ctx)
    queue.finish(last, ctx.adopted)
    paused_ctx.step.run(paused_ctx)  # it returns: every location is the last's
    with pytest.raises(queue.Revoked):
        queue.finish(paused, paused_ctx.adopted)
    with pytest.raises(queue.Revoked):
        queue.finish(first)
    assert queue.accepted(job, task.id) == {last.token}
    done = list((job / queue.DONE).iterdir())
    assert [p.name for p in done] == [f"{task.id}.{last.token}"]
    for frame in task.frames:
        key = f"numbers/r2/day1/{frame.name}"
        assert _stored(tmp_path / "project", key)[0]["token"] == last.token
    # A take-back between the owner's done content and its rename commits nothing.
    other = queue.tasks(job)["numbers.r1.0001"]
    owner = queue.claim(job, other.id, "here")
    write = queue._write

    def written_then_taken_back(claim, value):
        write(claim, value)
        queue._transfer(claim.path, job / queue.QUEUE / f"{claim.task}.back")

    monkeypatch.setattr(queue, "_write", written_then_taken_back)
    with pytest.raises(queue.Revoked):
        queue.finish(owner)
    assert other.id not in queue.markers(job, queue.DONE)
    assert queue.markers(job, queue.QUEUE)[other.id] == "back"


def test_a_retried_task_adopts_its_earlier_attempt(tmp_path, monkeypatch):
    """A task whose process ended after two of its frames resumes: it adopts their
    results, names their token in its done marker, and readers accept both."""
    config, selection = _project(tmp_path)
    job = create_job(tmp_path / "jobs", (NUMBERS, SUMS), selection, config)
    task = queue.tasks(job)["numbers.r1.0000"]
    first = queue.claim(job, task.id, "here")
    ctx = context(job, task, first.token)
    publish = ctx.publish
    calls = []

    def end_after_two(*args, **kwargs):
        if len(calls) == 2:
            raise SystemExit("killed")
        calls.append(kwargs["frame"])
        publish(*args, **kwargs)

    monkeypatch.setattr(ctx, "publish", end_after_two)
    with pytest.raises(SystemExit):
        ctx.step.run(ctx)
    queue.release(first)
    work(job)
    accepted = queue.accepted(job, task.id)
    assert first.token in accepted and len(accepted) == 2
    tokens = {
        _stored(tmp_path / "project", f"numbers/r1/day1/{f.name}")[0]["token"]
        for f in task.frames
    }
    assert tokens == accepted
    assert _stored(tmp_path / "project", "sums/r1")[1]["total"] == 108.0
    # A reader accepts only the tokens the producer's done marker names.
    token = queue.markers(job, queue.DONE)[task.id]
    marker = job / queue.DONE / f"{task.id}.{token}"
    content = queue.read_marker(marker)
    content["accepted"] = [t for t in content["accepted"] if t != first.token]
    marker.write_text(json.dumps(content) + "\n", encoding="utf-8")
    sums = queue.tasks(job)["sums.r1"]
    ctx = context(job, sums, "reader")
    with pytest.raises(PermissionError, match="did not accept"):
        ctx.step.run(ctx)


def test_failed_parts_made_again_with_other_settings(tmp_path):
    """A job over the failed part with other settings publishes only that part; the
    completed parts keep their tokens, and a consumer lists both jobs' tokens."""
    config, selection = _project(tmp_path, fail=("b",))
    jobs = tmp_path / "jobs"
    first = create_job(jobs, (NUMBERS, SUMS, FIT), selection, config)
    work(first)
    assert set(queue.markers(first, queue.FAILED)) == {"fit.b"}
    fixed, _ = _project(tmp_path, scale=3.0)
    parts = Selection(selection.frames, selection.sessions, parts=("b",))
    with pytest.raises(ValueError, match="names the job it completes"):
        create_job(jobs, (FIT,), parts, fixed, known=(SUMS,))
    second = create_job(jobs, (FIT,), parts, fixed, basis=(first.name,), known=(SUMS,))
    assert set(queue.tasks(second)) == {"fit.b"}
    work(second)
    third = create_job(jobs, (SUMMARY,), selection, config, known=(FIT,))
    work(third)
    project = tmp_path / "project"
    stored = {p: _stored(project, f"fit/{p}")[0] for p in "abc"}
    assert stored["a"]["job"] == stored["c"]["job"] == first.name
    assert stored["b"]["job"] == second.name
    assert stored["b"]["settings"]["toy"]["scale"] == 3.0
    summary, values = _stored(project, "summary")
    assert summary["inputs"] == {f"fit/{p}": stored[p]["token"] for p in "abc"}
    assert values["totals"] == [324.0, 486.0, 324.0]


def test_reuse_compares_what_a_step_reads(tmp_path):
    """A stored result is reused only while every configuration section it was
    made from, the run's own tables, the content of its configured files and the
    selected domains are unchanged; else the job is refused, naming what differs."""
    config, selection = _project(tmp_path)
    jobs, project = tmp_path / "jobs", tmp_path / "project"
    domains = Selection(selection.frames, selection.sessions, domains=("a",))
    work(create_job(jobs, (NUMBERS, DOMAIN_SUMS), domains, config))
    assert queue.tasks(create_job(jobs, (NUMBERS, DOMAIN_SUMS), domains, config)) == {}
    stored, _ = _stored(project, "numbers/r1/day1/r1-00.npz")
    assert stored["inputs"]["file:injection.csv"] == file_identity(
        project / "injection.csv"
    )
    protocols = {"protocols": {"injection": "injection.csv"}}
    assert stored["settings"]["runs"] == {"r1": protocols}
    numbers = "numbers/r1/day1/r1-00.npz is stored and does not serve the job"
    for name, edit in (("injection.csv", "1,2\n"), ("model/head.json", "[3]\n")):
        text = (project / name).read_text(encoding="utf-8")
        (project / name).write_text(text + edit, encoding="utf-8")  # edited by hand
        with pytest.raises(ValueError, match=f"{numbers}: Made with other inputs "):
            create_job(jobs, (NUMBERS,), selection, config)
        (project / name).write_text(text, encoding="utf-8")
    own = config.with_name("own.toml")
    own.write_text(config.read_text() + "[runs.r2.protocols]\nsheet = 2\n")
    with pytest.raises(ValueError, match="r2-00.npz is stored and does not") as found:
        create_job(jobs, (NUMBERS,), selection, own)
    assert "Made with other settings" in str(found.value)
    assert "r1-00.npz" not in str(found.value)  # the other run's tables are its own
    more = Selection(selection.frames, selection.sessions, domains=("a", "b"))
    with pytest.raises(ValueError, match="domain-sums/r1 is stored and does not") as e:
        create_job(jobs, (NUMBERS, DOMAIN_SUMS), more, config)
    assert "Made with other scope" in str(e.value) and "numbers/" not in str(e.value)


def test_a_result_just_committed_elsewhere_is_reused(tmp_path, monkeypatch):
    """What a creation does not find is looked at once more after the share's cache
    time: results another job commits meanwhile, and their done markers, are
    reused rather than made again."""
    config, selection = _project(tmp_path)
    jobs = tmp_path / "jobs"
    first = create_job(jobs, (NUMBERS,), selection, config)
    waits = []

    def committed_meanwhile(seconds):
        waits.append(seconds)
        work(first)  # another host commits while this creation waits

    monkeypatch.setattr(job_module, "sleep", committed_meanwhile)
    second = create_job(jobs, (NUMBERS,), selection, config)
    assert waits == [results.SHARE_CACHE] and queue.tasks(second) == {}


def test_a_result_records_the_revision_that_made_it(tmp_path):
    """A result records the revision of the code of the process that made it, not
    the job's."""
    config, selection = _project(tmp_path)
    job = create_job(tmp_path / "jobs", (NUMBERS,), selection, config)
    task = queue.tasks(job)["numbers.r2.0000"]
    claim = queue.claim(job, task.id, "here")
    ctx = context(job, task, claim.token)
    ctx.revision = "another-revision"
    ctx.step.run(ctx)
    stored, _ = _stored(tmp_path / "project", "numbers/r2/day1/r2-00.npz")
    assert stored["revision"] == "another-revision" != read_job(job)["revision"]


def test_an_installed_package_has_a_revision_without_git(tmp_path, monkeypatch):
    """Without git, or outside a checkout, the revision is the package's version
    and the hash of its source files, so that a laptop install creates jobs; a
    changed source file is another revision. A revision not known still refuses a
    job: a job locks the revision of its code."""

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    config, selection = _project(tmp_path)
    monkeypatch.setattr(job_module.subprocess, "run", no_git)
    package = tmp_path / "darsia"
    (package / "jobs").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "jobs" / "job.py").write_text("A = 1\n", encoding="utf-8")
    (package / "notes.txt").write_text("not source\n", encoding="utf-8")
    installed = revision(package)
    version, _, digest = installed.partition("+")
    assert version == importlib.metadata.version("darsia") and len(digest) == 64
    (package / "notes.txt").write_text("changed\n", encoding="utf-8")
    assert revision(package) == installed
    (package / "jobs" / "job.py").write_text("A = 2\n", encoding="utf-8")
    assert revision(package) != installed
    job = create_job(tmp_path / "jobs", (NUMBERS,), selection, config)
    assert read_job(job)["revision"] == revision() != installed
    monkeypatch.setattr(job_module, "installed_revision", no_git)
    assert revision() is None
    with pytest.raises(ValueError, match="revision is not known"):
        create_job(tmp_path / "jobs", (NUMBERS,), selection, config)


def test_a_package_in_an_ignored_folder_of_a_checkout_has_its_own_revision(tmp_path):
    """A package installed into an ignored folder of another checkout, e.g. a
    virtual environment inside a clone, is known by its version and source files,
    not by that checkout's commit: other code there is another revision."""
    git = ["git", "-c", "user.name=test", "-c", "user.email=test@example.org"]
    (tmp_path / ".gitignore").write_text("venv/\n", encoding="utf-8")
    for command in (["init", "-q"], ["add", ".gitignore"], ["commit", "-q", "-m", "a"]):
        subprocess.run(git + command, cwd=tmp_path, check=True, capture_output=True)
    package = tmp_path / "venv" / "Lib" / "site-packages" / "darsia"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "x.py").write_text("A = 1\n", encoding="utf-8")
    installed = revision(package)
    assert installed == job_module.installed_revision(package)
    (package / "x.py").write_text("A = 2\n", encoding="utf-8")
    assert revision(package) != installed
