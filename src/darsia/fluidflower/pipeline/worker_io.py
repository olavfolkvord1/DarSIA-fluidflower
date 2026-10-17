"""Worker-side operational I/O wiring; no scientific executor is replaced."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from .job_queue import LocalOutputBinding, name, read_json
from .store_runtime import (resolve_catalog,verify_inputs,local_attempt_output,
                            RunUnavailable,InputHashMismatch)
from .run_io import RunIOBinding,alias_resume_output


class _PublicationRouter:
    def __init__(self, runtime):
        self.runtime=runtime
    def __call__(self,claim,candidate_descriptor,success):
        return self.runtime.factory(claim)(claim,candidate_descriptor,success)
    def publish_progress(self,claim,event):
        factory=self.runtime.factory(claim)
        callback=getattr(factory,"publish_progress",None)
        if not callable(callback):
            raise RuntimeError("Alias progress requires its actual shared-metadata publisher")
        return callback(claim,event)


class WorkerRunIO:
    def __init__(self,job_root,job,stores,service_id,backend,commit):
        if stores is None:
            raise RunUnavailable("Alias worker has no startup-admitted store map")
        if name(stores.machine_id+"-"+backend)!=service_id:
            raise RunUnavailable("Store-map machine differs from the actual worker service")
        self.root,self.job,self.stores=Path(job_root),deepcopy(job),stores
        self.service_id,self.commit=service_id,commit
        self.bindings,self.factories={},{}
        self.publications=_PublicationRouter(self)

    def output_factory(self,task,token):
        path,evidence=local_attempt_output(self.stores,self.job["series_id"],
                                          task["job_id"],task["run"],token)
        return LocalOutputBinding(self.stores.machine_id,
            Path(self.stores.roots(self.job["series_id"])["results"]),path,evidence)

    def availability(self,task):
        if (task.get("execution_ready") is not True
                or task["input_catalog"].get("runtime_support")!="darsia-role-io-runtime-v1"):
            raise RunUnavailable("Prepared J2 metadata is not runtime-ready")
        rows=resolve_catalog(task,self.stores,self.job["series_id"])
        output=self.output_factory(task,task["attempt_id"])
        # Reuse the exact original resume selector before claim. It reads only
        # named own-job lineage/retirement metadata; full prefix SHA checks still
        # run through original verify_frame_prefix after actual claim.
        from .checkpoint import resume_source_for_claim
        if task.get("pipeline") == "native-phase-statistics":
            from .phase1_protocol import phase1_resume_source
            phase1_resume_source(SimpleNamespace(root=self.root),
                SimpleNamespace(task=task,token=task["attempt_id"],output=output.path,
                                local_output=output))
        else:
            resume_source_for_claim(SimpleNamespace(root=self.root),
                SimpleNamespace(task=task,token=task["attempt_id"],output=output.path,
                                local_output=output))
        return rows

    def prepare(self,claim):
        try:
            verified=verify_inputs(claim.task,claim.input_rows)
            binding=RunIOBinding.replay(claim.task,verified,claim.local_output,
                repository=Path(__file__).resolve().parents[4],
                admitted_commit=self.commit)
        except (InputHashMismatch,RunUnavailable):
            raise
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise InputHashMismatch("Input admission failed before science: "+str(error)) from error
        self.bindings[claim.token]=binding
        return binding

    def factory(self,claim):
        if claim.token not in self.factories:
            if claim.token not in self.bindings:
                raise RuntimeError("No admitted per-run binding for publication")
            from .publication_runtime import make_publication_factory
            self.factories[claim.token]=make_publication_factory(
                self.root,self.job,self.stores.machine_id,
                producer_origin_for_attempt=self.bindings[claim.token].producer_origin_for_attempt)
        return self.factories[claim.token]

    def release(self,claim):
        self.factories.pop(claim.token,None)
        self.bindings.pop(claim.token,None)


def owner_for_attempt(root,task,service_id):
    """Read an exact live or archived owner; never scan another job/attempt."""
    run,token=name(task["run"]),task["attempt_id"]
    root=Path(root)
    sources=[root/"leases"/(run+"."+token)/"owner.json",
             root/"reclaimed"/run/token/"lease"/"owner.json",
             root/"publishing"/run/"owner.json"]
    matches=[]
    for path in sources:
        try: owner=read_json(path)
        except FileNotFoundError: continue
        if owner.get("token")==token:
            if owner.get("machine")!=service_id or owner.get("task")!=task:
                raise ValueError("Actual owner and original task/service differ")
            matches.append(owner)
    if not matches or any(value!=matches[0] for value in matches):
        raise ValueError("No unique actual owner for this original attempt")
    return matches[0]


def retirement_output(root,task,service_id,stores):
    if stores is None:
        raise RunUnavailable("Alias parent has no startup store binding")
    owner=owner_for_attempt(root,task,service_id)
    path,evidence=local_attempt_output(stores,read_json(Path(root)/"job.json")["series_id"],
                                      task["job_id"],task["run"],task["attempt_id"])
    binding=LocalOutputBinding(stores.machine_id,
        Path(stores.roots(read_json(Path(root)/"job.json")["series_id"])["results"]),path,evidence)
    actual=alias_resume_output(owner,binding,task,task["attempt_id"])
    if actual.resolve()!=path.resolve():
        raise RunUnavailable("Actual owned retirement output differs")
    return actual,owner


def claimed_task_after_archive(root,service_id,event):
    """Bind a fast admission reject/publication after its atomic claim moved."""
    root=Path(root);run=name(event["run"]);token=event["attempt_id"]
    tasks=[]
    paths=[(root/"claimed"/service_id/(run+".json"),False),
           (root/"reclaimed"/run/token/"task.json",False),
           (root/"reclaimed"/run/token/"lease"/"owner.json",True),
           (root/"publishing"/run/"owner.json",True)]
    for path,owner in paths:
        try: value=read_json(path)
        except FileNotFoundError: continue
        task=value.get("task") if owner else value
        if isinstance(task,dict) and task.get("attempt_id")==token:
            if owner and value.get("machine")!=service_id:
                raise ValueError("Claim archive belongs to another actual worker service")
            tasks.append(task)
    if not tasks or any(task!=tasks[0] for task in tasks):
        raise ValueError("No exact immutable current/archive task for claimed generation")
    return tasks[0]
