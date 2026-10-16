"""Portable job creation; new metadata requires the explicit admitted J3 runtime.

Previously prepared pending J2 files are never modified or promoted here.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import tomllib

from ..store_map import manifest_roots
from .job_inputs import InputCatalog, validate_catalog
from .job_queue import JobQueue, descriptor, name, read_json, require_unc, write_immutable

JOB_FORMAT = "darsia-alias-job-v1"
TASK_FORMAT = "darsia-alias-task-v1"
RUNTIME_PENDING = "Alias job metadata is prepared; J3 runtime integration is not yet available"


def _manifest(bound):
    if not isinstance(bound, dict) or not {"path","sha256","size_bytes"}.issubset(bound):
        raise ValueError("New jobs require an explicit bound schema-1 series.toml with [roots]")
    path = Path(bound["path"])
    if path.name.casefold() != "series.toml":
        raise ValueError("Choose the registered series.toml, not the execution-series TOML")
    with path.open("rb") as stream:
        raw = stream.read(1024*1024+1)
    if (len(raw)>1024*1024 or len(raw)!=bound["size_bytes"]
            or hashlib.sha256(raw).hexdigest()!=bound["sha256"]):
        raise ValueError("Registered series manifest changed or exceeds 1 MiB")
    value=tomllib.loads(raw.decode("utf-8-sig"))
    return path, value, manifest_roots(path.parent,value)


def shared_output_admission(job_root):
    """Admit the real existing shared ancestor before any new job mkdir."""
    from ..job_filesystem import inspect_shared_folder
    selected=Path(job_root).expanduser()
    parent=selected
    while not parent.exists():
        if parent==parent.parent:
            raise ValueError("No existing shared job ancestor")
        parent=parent.parent
    if not parent.is_dir():
        raise ValueError("Shared job ancestor is not a directory")
    transport=inspect_shared_folder(parent)
    if not transport.allowed:
        raise ValueError("Shared queue transport refused: "+transport.reason+": "+transport.evidence)
    if not selected.resolve().is_relative_to(parent.resolve()):
        raise ValueError("New job escapes its admitted shared ancestor")
    return transport


def read_alias_series(path, *, expected_sha256, series_manifest, catalog_factory=None):
    """Original series admission with explicit rooted metadata inputs only."""
    from . import series
    catalog_factory=catalog_factory or InputCatalog
    _source, _manifest_value, roots=_manifest(series_manifest)
    admission=catalog_factory(roots)
    admission.add({"path":str(path),"sha256":expected_sha256},
                  "configuration","execution_series")
    admission.add(series_manifest,"configuration","series_manifest")
    def reader_factory(aliases):
        catalog=catalog_factory(roots,aliases)
        class RootedReader:
            def __init__(self):
                self.input_index=0
            def json(self,bound):
                use="execution_series.input."+str(self.input_index)
                self.input_index+=1
                return catalog.doc(bound,"configuration",use)
        return RootedReader()
    return series.read_series(path,expected_sha256=expected_sha256,
                              reader_factory=reader_factory)


def prepare_alias_job(series_path, job_root, *, expected_sha256, series_manifest,
                      catalog_factory=None):
    """Resolve all metadata input roles before creating the new shared job.

    No payload arrays, scientific method or executor is invoked. Original
    configuration bytes and source SHA remain external input identities.
    """
    from . import series
    transport = shared_output_admission(job_root)
    catalog_factory = catalog_factory or InputCatalog
    source_manifest, manifest, roots = _manifest(series_manifest)
    value,configs,original_series=read_alias_series(series_path,
        expected_sha256=expected_sha256,series_manifest=series_manifest,
        catalog_factory=catalog_factory)
    admission=catalog_factory(roots)
    series_reference=admission.add(original_series,"configuration","execution_series")
    manifest_reference=admission.add(series_manifest,"configuration","series_manifest")
    # Reuse original task/resource/code/calibration checks before encoding paths.
    job, original_tasks = series.series_tasks(
        value,configs,job_root=job_root,series_asset=original_series,
        shared_admission=transport)
    name(value["job_id"])
    tasks=[]
    for task, config in zip(original_tasks,configs):
        catalog=catalog_factory(roots,[*value.get("data_aliases",()),*config.get("data_aliases",())])
        catalog.add(original_series,"configuration","run.execution_series")
        catalog.add(series_manifest,"configuration","run.series_manifest")
        inputs=validate_catalog(catalog.collect(task["configuration"],config))
        inputs["runtime_support"]="darsia-role-io-runtime-v1"
        portable=dict(deepcopy(task),format=TASK_FORMAT,format_version=1,
            legacy_paths=False,execution_ready=True,execution_blocker="",
            configuration=inputs["configuration"],series=series_reference,
            series_manifest=manifest_reference,input_catalog=inputs,
            ordered_case_ids=[frame["case_id"] for frame in config["frames"]])
        # No producer/consumer may smuggle an unknown resource path into tasks.
        allowed={"backend","threads","device_id","pool_limit_bytes"}
        if set(value["runs"][len(tasks)]["resources"])-allowed:
            raise ValueError("Unsupported alias-task resource keys: "+repr(sorted(set(value["runs"][len(tasks)]["resources"])-allowed)))
        tasks.append(portable)
    allowed_job={"commit","environment_lock","calibration_sha256","precision",
                 "created_utc","publish","expected_runs","series"}
    if set(job)-allowed_job:
        raise ValueError("Unsupported alias-job fields: "+repr(sorted(set(job)-allowed_job)))
    job=dict(job,format=JOB_FORMAT,format_version=1,legacy_paths=False,
        execution_ready=True,execution_blocker="",runtime_support="darsia-role-io-runtime-v1",
        series=series_reference,series_manifest=manifest_reference,
        series_id=manifest["series_id"],input_reference_format="root-alias-relative-v1")
    publish = job.get("publish", ["film", "lossless_export"])
    if (not isinstance(publish,list) or any(
            not isinstance(item,str) or not item or "\\" in item
            or PurePosixPath(item).is_absolute() or PureWindowsPath(item).drive
            or any(part in {"", ".", ".."} for part in item.split("/")) for item in publish)):
        raise ValueError("Publish must name relative products, never machine paths")
    job["publish"] = deepcopy(publish)
    root=Path(job_root)
    # Every role, root, size and metadata SHA has now been admitted.
    target=root/"job.json"
    if target.exists() and read_json(target)!=job:
        raise ValueError("Existing alias job DATA differs from immutable metadata")
    prior_tasks={}
    for task in tasks:
        location=root/"results"/name(task["run"])/"task.json"
        if location.exists():
            prior=read_json(location)
            projected={key:item for key,item in prior.items() if key not in {"attempt_id","output_directory"}}
            if projected!=task:
                raise ValueError("Existing alias run task differs")
            prior_tasks[task["run"]]=location
    root.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        write_immutable(target,job)
    shared=JobQueue(root)
    prepared=[]
    for task in tasks:
        if task["run"] in prior_tasks:
            location=prior_tasks[task["run"]]
            already=True
        else:
            location=shared.enqueue(task)
            already=False
        prepared.append({"run":task["run"],"task":descriptor(location),"already_enqueued":already})
    return {"complete":True,"job":descriptor(root/"job.json"),"series":series_reference,
            "series_manifest":manifest_reference,"tasks":prepared,
            "legacy_paths":False,"execution_ready":True,"execution_blocker":"",
            "numerical_calls_started":False}


def portable_selection_binding(binding, *, series_manifest, catalog_factory=None):
    """Encode only the existing named B4 UI proof fields, before job mkdir."""
    catalog_factory = catalog_factory or InputCatalog
    _path, _manifest_value, roots = _manifest(series_manifest)
    inputs = catalog_factory(roots)
    runs = []
    for row in binding["runs"]:
        cases = []
        for case in row["cases"]:
            cases.append({key: deepcopy(case[key]) for key in
                          ("case_id","elapsed_h","b4_input_index")})
            cases[-1]["original_camera_source"] = inputs.add(
                case["original_camera_source"],"image","UI.B4."+row["run"]+"."+case["case_id"])
        runs.append({"run":row["run"],
            "manifest":inputs.add(row["manifest"],"configuration","UI.series_manifest"),
            "source_metadata":[inputs.add(item,"protocol","UI.B4."+row["run"]+".source_metadata."+str(index)) for index,item in enumerate(row["source_metadata"])],
            "configuration":inputs.add(row["configuration"],"configuration","UI."+row["run"]+".execution_run"),
            "cases":cases})
    return {"format":"darsia-job-ui-image-selection-binding-v2",
            "execution_plan":inputs.add(binding["execution_plan"],"configuration","UI.execution_plan"),
            "series_id":binding["series_id"],"runs":runs,
            "execution_entries_unchanged":True,"raw_hashed_by_this_UI":False}


def job_metadata_reference(job_root, path):
    """Immutable own-job metadata identity, independent of its shared locator."""
    root, path = Path(job_root), Path(path)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        raise ValueError("Own-job metadata is outside its job folder") from None
    if (not relative or PurePosixPath(relative).is_absolute()
            or PureWindowsPath(relative).drive or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("Own-job metadata needs a safe relative path")
    bound = descriptor(path)
    return {"relative_path":relative,"sha256":bound["sha256"],"bytes":bound["size_bytes"]}


def prepared_snapshot(registration, job, ui, *, reader):
    """Show only immutable J2 metadata. No input resolution or runtime inference."""
    from .job_queue import read_json, validate_task
    from ..job_launch_state import read_launch_requests
    root=Path(registration["root"])
    rows=[]
    for run in job["expected_runs"]:
        task=reader(root/"results"/name(run)/"task.json")
        validate_task(task)
        if (task.get("format")!=TASK_FORMAT or task["run"]!=run
                or task["commit"]!=job["commit"]
                or task["calibration_sha256"]!=job["calibration_sha256"]
                or task["precision"]!=job["precision"]):
            raise ValueError("Prepared alias task identity differs from job")
        if ((root/"done"/(name(run)+".json")).exists()
                or (root/"failed"/(name(run)+".json")).exists()
                or list((root/"claimed").glob("*/"+name(run)+".json"))):
            raise ValueError("J2 metadata-only alias job has unexpected execution artifacts")
        rows.append({"run":run,"state":"Prepared (J3 pending)","committed":0,
                     "total":len(task["ordered_case_ids"]),"machine":"Not reported",
                     "worker":"Not reported","error":"","case_ids":[]})
    total=sum(row["total"] for row in rows)
    return {"root":str(root),"name":ui.get("name") or registration.get("name") or root.name,
        "series":ui.get("series_label") or registration.get("series_label") or job.get("series_id"),
        "created_utc":ui.get("created_utc") or job.get("created_utc") or "Not reported",
        "master_machine":ui.get("machine") or "Not recorded","job_facts":dict(job),
        "runs":rows,"workers":[],"machines":[],"logs":[],"registered_machines":[],
        "launch_requests":read_launch_requests(root,reader=reader),
        "run_count":len(rows),"completed_runs":0,"failed_runs":0,
        "committed":0,"total_frames":total,"remaining":total,"frames_per_minute":None,
        "eta_seconds":None,"complete":False,"state":"Prepared (J3 pending)","errors":[],
        "rate_scope":"No actual execution or progress. J3 runtime is not yet available."}
