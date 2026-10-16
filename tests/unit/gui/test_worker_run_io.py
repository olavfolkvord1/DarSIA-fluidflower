"""Synthetic worker/parent ownership and callback wiring, no actual process."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import queue

import pytest
from darsia.fluidflower.pipeline import worker,worker_io
from darsia.fluidflower.pipeline.store_runtime import InputHashMismatch


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(json.dumps(value).encode())


def test_fast_input_reject_claim_is_bound_from_its_actual_archive(tmp_path):
    task=dict(format="darsia-alias-task-v1",run="ac52",attempt_id="1"*32)
    owner=dict(machine="main-cpu",token=task["attempt_id"],task=task)
    put(tmp_path/"reclaimed/ac52"/task["attempt_id"]/"lease/owner.json",owner)
    assert worker_io.claimed_task_after_archive(tmp_path,"main-cpu",
        dict(run="ac52",attempt_id=task["attempt_id"]))==task
    with pytest.raises(ValueError,match="worker service"):
        worker_io.claimed_task_after_archive(tmp_path,"olav-cpu",
            dict(run="ac52",attempt_id=task["attempt_id"]))


def test_conflicting_current_archive_task_data_is_never_accepted(tmp_path):
    task=dict(format="darsia-alias-task-v1",run="ac52",attempt_id="1"*32)
    put(tmp_path/"claimed/main-cpu/ac52.json",task)
    put(tmp_path/"reclaimed/ac52"/task["attempt_id"]/"task.json",dict(task,precision="wrong"))
    with pytest.raises(ValueError,match="exact immutable"):
        worker_io.claimed_task_after_archive(tmp_path,"main-cpu",
            dict(run="ac52",attempt_id=task["attempt_id"]))


def test_pre_science_hash_rejection_does_not_publish_fail_or_enter_executor(monkeypatch,tmp_path):
    job=dict(format="darsia-alias-job-v1",precision="float32",calibration_sha256="b"*64)
    task=dict(format="darsia-alias-task-v1",run="ac52",attempt_id="1"*32,
              commit="a"*40,calibration_sha256="b"*64,precision="float32")
    claim=SimpleNamespace(task=task,token=task["attempt_id"],output=tmp_path/"not-created")
    calls=[]
    class Shared:
        def __init__(self,*a,**k):pass
        def recover_publications(self):return []
        def reclaim_stale(self):return []
        def claim(self,*a,**k):
            assert callable(k["availability"]) and callable(k["output_factory"])
            assert callable(k["publication_factory"].publish_progress)
            return claim
        def reject_before_science(self,actual,text,owner):
            assert actual is claim and "literal fixture hash mismatch" in text
            calls.append("rejected");return {"run":"ac52"}
        def fail(self,*a):pytest.fail("Input mismatch must not become a scientific FAIL")
    class Runtime:
        def __init__(self,*a):self.publications=SimpleNamespace(publish_progress=lambda *a:None)
        def availability(self,*a):pass
        def output_factory(self,*a):pass
        def prepare(self,*a):raise InputHashMismatch("literal fixture hash mismatch")
        def release(self,*a):calls.append("released")
    monkeypatch.setattr(worker,"JobQueue",Shared)
    monkeypatch.setattr(worker_io,"WorkerRunIO",Runtime)
    monkeypatch.setattr(worker,"_identity",lambda machine:dict(machine=machine))
    monkeypatch.setattr(worker,"_job_control_boundary",lambda root:None)
    messages=queue.Queue()
    with pytest.raises(SystemExit) as error:
        worker._run_tasks(str(tmp_path),job,"a"*40,"main-cpu","cpu",messages,
            SimpleNamespace(is_set=lambda:False),once=True,exit_when_idle=False,
            stores=SimpleNamespace(machine_id="Main"))
    assert error.value.code==1 and calls.count("rejected")==1
    assert any(event["kind"]=="input_rejected" for event in list(messages.queue))
    assert not claim.output.exists()


def test_publication_router_keeps_original_progress_event(monkeypatch):
    event=dict(state="frame_completed",case_id="ac52_f1",publication={"literal":"source"})
    seen=[]
    class Factory:
        def __call__(self,*args):seen.append(args);return {"candidate":"original"}
        def publish_progress(self,claim,value):
            seen.append((claim,deepcopy(value)));return {"relative_path":"original.json"}
    factory=Factory();runtime=SimpleNamespace(factory=lambda claim:factory)
    router=worker_io._PublicationRouter(runtime);claim=object()
    assert router.publish_progress(claim,event)=={"relative_path":"original.json"}
    assert seen==[(claim,event)] and event["publication"]=={"literal":"source"}
