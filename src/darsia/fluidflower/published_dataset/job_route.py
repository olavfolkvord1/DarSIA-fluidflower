"""Run a dataset's execution plan with the production job layer, inside a results folder.

The dataset stays read-only. The launcher writes, in the results folder:

* ``setup/series/``: a copy of the dataset's ``series.toml`` whose ``[roots]``
  bind this machine's folders (the dataset, the unpacked raw images), the
  execution plan of this reproduction and, for the quick check, the first
  frames' run configurations. The plan's ``data_aliases`` bind every file the
  published plan and run configurations name to its copy in the dataset, by
  path and SHA-256, so the published files are read as they are;
* ``scratch/``: the raw archives of the runs in scope, unpacked; a run's files
  are removed after the run when the user agreed;
* ``job/``: the job, prepared by the job layer's own ``prepare_job``;
* ``results/``: the workers' results.

The machine's store map (local, outside every series folder) lies in the
per-user DarSIA folder. Workers are the job layer's own, started on this
machine and stopped when idle; Pause and Resume publish the existing job
controls.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import time
import tomllib
import uuid

from .archives import extract_members
from .job_reader import FinishedJob, LAYER_FILE
from .layout import inside, raw_archive_name

MACHINE_ID = "reproduction"
SETUP, JOB, RESULTS, SCRATCH = "setup", "job", "results", "scratch"
_SHA = re.compile(r"[0-9a-f]{64}")


def descriptors(value):
    """Every file descriptor ({path, sha256[, size_bytes]}) in a nested document."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and _SHA.fullmatch(str(value.get("sha256", ""))):
            yield value
        for item in value.values():
            yield from descriptors(item)
    elif isinstance(value, list):
        for item in value:
            yield from descriptors(item)


def raw_roots(check, scratch: Path) -> dict[str, Path]:
    """Where each raw root of the dataset lies here: unpacked in the dataset, or in scratch."""
    roots = {}
    for alias, relative in (check.manifest.get("roots") or {}).items():
        if alias == "series":
            continue
        unpacked = inside(check.root, relative)
        roots[alias] = unpacked if unpacked.is_dir() else Path(scratch) / alias
    return roots


def dataset_files(check, roots: dict[str, Path]) -> dict[str, Path]:
    """Every published file of the dataset here, by SHA-256: series inputs and camera files."""
    files = {}
    for row in (check.manifest.get("inputs") or {}).get("series", ()):
        files.setdefault(row["sha256"], inside(check.root, row["path"]))
    for run in check.manifest["runs"]:
        for item in [frame["raw"] for frame in run["frames"]] + list(run.get("references") or ()):
            alias, _separator, relative = item["path"].partition("/")
            if alias in roots:
                files.setdefault(item["sha256"], roots[alias].joinpath(*relative.split("/")))
    return files


def data_aliases(documents, files: dict[str, Path]) -> list[dict]:
    """Bind each named file to its copy here: {logical: as written, physical: here}.

    A file named twice under the same path is bound once; a descriptor whose
    file the dataset does not hold is left as written (it is not read).
    """
    rows, seen = [], set()
    for document in documents:
        for item in descriptors(document):
            target = files.get(item["sha256"])
            key = os.path.normcase(os.path.normpath(item["path"]))
            if target is None or key in seen:
                continue
            seen.add(key)
            logical = {key_: item[key_] for key_ in ("path", "sha256", "size_bytes") if key_ in item}
            rows.append({"logical": logical, "physical": {"path": str(target), "sha256": item["sha256"],
                                                          "size_bytes": target.stat().st_size}})
    return rows


def with_roots(text: str, roots: dict[str, str]) -> str:
    """A series.toml text with its ``[roots]`` table replaced (or added); the rest unchanged."""
    lines = text.splitlines()
    table = ["[roots]"] + [f"{alias} = {json.dumps(str(path))}" for alias, path in roots.items()]
    start = next((index for index, line in enumerate(lines) if line.strip() == "[roots]"), None)
    if start is None:
        result = lines + [""] + table
    else:
        end = next((index for index in range(start + 1, len(lines)) if lines[index].lstrip().startswith("[")),
                   len(lines))
        result = lines[:start] + table + [""] + lines[end:]
    value = "\n".join(result) + "\n"
    if tomllib.loads(value).get("roots") != {alias: str(path) for alias, path in roots.items()}:
        raise ValueError("The series manifest's roots could not be replaced")
    return value


def toml_text(document: dict) -> str:
    """A TOML text of a plan document, checked by reading it back."""
    import toml

    value = toml.dumps(document)
    if tomllib.loads(value) != json.loads(json.dumps(document)):
        raise ValueError("The execution plan does not survive a TOML round trip")
    return value


def store_map_path(job_id: str, *, environment=None) -> Path:
    """The local store map of one reproduction job, beside the user's own store map."""
    from ..store_map import default_store_map_path

    return default_store_map_path(environment=environment).parent / "reproductions" / job_id / "stores.toml"


def unpack_raw(check, scope, roots: dict[str, Path]) -> dict[str, list[Path]]:
    """Unpack the raw archives of the runs in scope into scratch, checking every SHA-256.

    Runs whose raw images are already unpacked in the dataset are read there.
    Returns the files unpacked per run (to remove after the run).
    """
    unpacked = {}
    for row in check.manifest["runs"]:
        if row["run"] not in scope.runs:
            continue
        items = [frame["raw"] for frame in row["frames"]] + list(row.get("references") or ())
        wanted = {}
        for item in items:
            alias, _separator, relative = item["path"].partition("/")
            target = roots[alias].joinpath(*relative.split("/"))
            if not target.is_file() and not inside(check.root, item["path"]).is_file():
                wanted[item["path"]] = item["sha256"]
        if wanted:
            alias = next(iter(wanted)).partition("/")[0]
            scratch_root = roots[alias].parent
            unpacked[row["run"]] = extract_members(check.root / raw_archive_name(row["run"]), scratch_root,
                                                   names=sorted(wanted), expected=wanted)
    return unpacked


def write_setup(check, results: Path, scope, settings, *, installed: dict, environment=None) -> dict:
    """Write this reproduction's series manifest, plan, quick configurations and store map."""
    manifest = check.manifest
    results = Path(results).resolve()
    setup = results / SETUP / "series"
    setup.mkdir(parents=True)
    roots = raw_roots(check, results / SCRATCH)
    series_text = (check.root / "series.toml").read_text(encoding="utf-8-sig")
    series_roots = {"series": ".", "dataset": str(check.root.resolve()),
                    **{alias: str(path) for alias, path in roots.items()}}
    (setup / "series.toml").write_text(with_roots(series_text, series_roots), encoding="utf-8")
    plan = tomllib.loads(inside(check.root, manifest["plan"]["path"]).read_text(encoding="utf-8"))
    files = dataset_files(check, roots)
    configs = {}
    for row in plan["runs"]:
        path = files.get(row["configuration"]["sha256"])
        if path is None:
            raise ValueError(f"The dataset lacks the run configuration of {row['run']}")
        configs[row["run"]] = json.loads(path.read_bytes())
    documents = [plan, *configs.values()]
    for row in (manifest.get("inputs") or {}).get("series", ()):
        if row["path"].endswith((".json", ".toml")) and row["sha256"] in files:
            path = files[row["sha256"]]
            text = path.read_text(encoding="utf-8-sig")
            try:
                documents.append(json.loads(text) if row["path"].endswith(".json") else tomllib.loads(text))
            except ValueError:
                continue
    aliases = data_aliases(documents, files)
    job_id = f"reproduction-{uuid.uuid4().hex[:12]}"
    derived = deepcopy(plan)
    derived["job_id"] = job_id
    derived.pop("data_aliases", None)
    commit = manifest["code"]["published_commit"]
    if settings.override_commit and installed.get("commit"):
        commit = installed["commit"]
    derived["job"]["commit"] = commit
    if commit != manifest["code"]["published_commit"]:
        derived["job"]["environment_lock"] = _installed_lock(derived["job"]["environment_lock"]["path"], commit)
    runs, selected = [], []
    for row in plan["runs"]:
        if row["run"] not in scope.runs:
            continue
        row = deepcopy(row)
        row["resources"] = {"backend": settings.backend, "threads": settings.threads}
        frames = configs[row["run"]]["frames"]
        count = scope.frames.get(row["run"])
        if count:
            frames = frames[:count]
            quick = setup / "quick" / f"{row['run']}.json"
            quick.parent.mkdir(exist_ok=True)
            quick.write_text(json.dumps(dict(configs[row["run"]], frames=frames), indent=1), encoding="utf-8")
            row["configuration"] = _descriptor(quick)
        runs.append(row)
        selected.append({"run": row["run"], "ordered_case_ids": [frame["case_id"] for frame in frames]})
    derived["runs"] = runs
    (setup / "selection.json").write_text(json.dumps({"complete": True, "runs": selected}, indent=1),
                                          encoding="utf-8")
    derived["selection"] = _descriptor(setup / "selection.json")
    derived["data_aliases"] = aliases
    plan_path = setup / "plan.toml"
    plan_path.write_text(toml_text(derived), encoding="utf-8")
    stores = store_map_path(job_id, environment=environment)
    from ..series_paths import series_key

    stores.parent.mkdir(parents=True, exist_ok=True)
    store_roots = {"series": str(setup), "dataset": str(check.root.resolve()),
                   **{alias: str(path) for alias, path in roots.items()}, "results": str(results / RESULTS)}
    stores.write_text(toml_text({"machine": {"id": MACHINE_ID},
                                 "stores": {series_key(manifest["identity"]["series_id"]): store_roots}}),
                      encoding="utf-8")
    (results / RESULTS).mkdir(exist_ok=True)
    return {"job_id": job_id, "plan": plan_path, "series_manifest": setup / "series.toml", "stores": stores,
            "roots": roots, "aliases": len(aliases)}


def _descriptor(path: Path) -> dict:
    """A file descriptor (path, SHA-256, size) of a file this reproduction wrote."""
    from ..pipeline.job_queue import descriptor

    return descriptor(path)


def _installed_lock(path: str, commit: str) -> dict:
    """The environment lock of the installed commit (path and SHA-256 of its tracked bytes)."""
    import hashlib
    import subprocess

    from ..series_paths import darsia_source_checkout

    checkout = darsia_source_checkout()
    raw = subprocess.check_output(["git", "-C", str(checkout), "show", f"{commit}:{path}"])
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}


def write_master(job_root: Path, *, name: str, series_label: str) -> None:
    """The job's UI master (as the Jobs page writes it), so that Pause and Resume can be published."""
    from datetime import datetime, timezone

    from ..pipeline.alias_job import job_metadata_reference
    from ..pipeline.job_queue import write_immutable

    root = Path(job_root)
    write_immutable(root / "ui" / "master.json", {
        "format": "darsia-job-ui-master-v2", "job": job_metadata_reference(root, root / "job.json"),
        "machine": MACHINE_ID, "created_utc": datetime.now(timezone.utc).isoformat(), "name": name,
        "series_label": series_label, "numerical_calls_started": False})


def job_progress(job_root: Path, runs) -> dict:
    """Each run's state: queued, running, done or failed (from the job folder)."""
    root = Path(job_root)
    states = {}
    for run in runs:
        key = run.casefold()
        if (root / "done" / f"{key}.json").is_file():
            states[run] = "done"
        elif (root / "failed" / f"{key}.json").is_file():
            states[run] = "failed"
        elif (root / "queue" / f"{key}.json").is_file():
            states[run] = "queued"
        else:
            states[run] = "running"
    return states


class JobLayerRunner:
    """The launcher's runner: the production job layer on this machine."""

    def __init__(self, *, launch=None, poll_seconds: float = 10.0, sleep=time.sleep, installed=None):
        """``launch`` starts the workers (default: the Jobs page's local worker launch)."""
        self.launch = launch
        self.poll_seconds, self.sleep = poll_seconds, sleep
        self.installed = installed
        self.job_root = None
        self.job = None
        self.setup = None
        self.paused = False

    def preflight(self, results: Path) -> list[str]:
        """Why the job layer cannot hold this reproduction's job in ``results``; empty when it can."""
        from ..pipeline import alias_job

        try:
            alias_job.shared_output_admission(Path(results).resolve() / JOB)
        except (OSError, ValueError) as error:
            return [f"The job layer runs a job only in a folder on an SMB or NFS share ({error}). "
                    "Choose a results folder on such a share."]
        return []

    def run(self, check, results: Path, scope, settings, *, progress=None) -> dict:
        """Prepare the job, start the workers and wait until every run is done or failed."""
        from ..pipeline.job_queue import descriptor
        from ..pipeline.runner import prepare_job
        from .dataset_check import installed_commit

        installed = self.installed if self.installed is not None else installed_commit()
        self.setup = write_setup(check, results, scope, settings, installed=installed)
        unpacked = unpack_raw(check, scope, self.setup["roots"])
        self.job_root = Path(results).resolve() / JOB
        plan = descriptor(self.setup["plan"])
        prepare_job(self.setup["plan"], self.job_root, expected_sha256=plan["sha256"],
                    series_manifest=descriptor(self.setup["series_manifest"]),
                    storage_profile=check.manifest["plan"].get("storage_profile"))
        write_master(self.job_root, name=self.setup["job_id"], series_label=check.manifest["identity"]["title"])
        self.start(settings)
        removed = set()
        while True:
            states = job_progress(self.job_root, scope.runs)
            if progress is not None:
                progress(states)
            for run, state in states.items():
                if state == "done" and settings.remove_scratch and run not in removed:
                    for path in unpacked.get(run, ()):
                        path.unlink(missing_ok=True)
                    removed.add(run)
            if all(state in {"done", "failed"} for state in states.values()):
                break
            self.sleep(self.poll_seconds)
        self.job = FinishedJob(self.job_root) if all(state == "done" for state in states.values()) else None
        if self.job is None:
            failed = [run for run, state in states.items() if state == "failed"]
            raise RuntimeError(f"Runs failed: {', '.join(failed)}; see the job's logs in the results folder")
        return states

    def start(self, settings) -> None:
        """Start the job layer's workers on this machine; they stop when the queue is empty."""
        launch = self.launch
        if launch is None:
            from ..jobs_panel import launch_local_worker as launch
        launch(self.job_root, backend=settings.backend, workers=settings.workers, machine=MACHINE_ID,
               stores=self.setup["stores"], exit_when_idle=True)

    def pause(self) -> None:
        """Publish Pause: workers stop after their current frame."""
        self._control("pause")
        self.paused = True

    def resume(self, settings) -> None:
        """Publish Run again and restart the workers; committed frames are reused."""
        self._control("run")
        self.paused = False
        self.start(settings)

    def _control(self, action: str) -> None:
        """Publish one job control command with the existing control chain."""
        from ..job_control import publish_control, read_control

        current = read_control(self.job_root)
        publish_control(self.job_root, action, machine=MACHINE_ID, expected_previous=current.get("command"))

    def reproduced(self, run: str, case: str) -> tuple[dict, dict]:
        """One reproduced frame: its masses and its stored layer's arrays."""
        from .comparison import load_layer

        result = self.job.frame_result(run, case)
        masses = {"total": result.get("total_mass_g"), "aqueous": result.get("aqueous_mass_g"),
                  "gas": result.get("gas_mass_g")}
        layer = self.job.layer(run, case, LAYER_FILE)
        return masses, ({} if layer is None else load_layer(self.job.product_source(layer)))
