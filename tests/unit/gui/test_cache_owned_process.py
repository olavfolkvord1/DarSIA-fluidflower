"""Bounded native-control transitions; synthetic data is not a service profile."""
from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from darsia_cache import bootstrap, owned_process as owned
from darsia_cache.policy import CachePolicyError


OWNER = dict(machine="TESTONLY", pid=70, parent_pid=1, process_create_time="123.0",
             creation_filetime_100ns="116444737230000000", argv=["TESTONLY"])
SERVICE = dict(path="C:/TESTONLY/receipts/service.json", sha256="a" * 64, size_bytes=1)


class Lease_TESTONLY:
    def __init__(self):
        self.source = dict(path="C:/TESTONLY/stores.toml", sha256="b" * 64, size_bytes=1)
        self.profile_binding = dict(path="C:/TESTONLY/profile.json", sha256="c" * 64, size_bytes=1)
        self.records = []
    def _record(self, name, document):
        raw = json.dumps(document, sort_keys=True).encode()
        pin = dict(path="C:/TESTONLY/receipts/" + name,
                   sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        self.records.append((name, deepcopy(document), pin))
        return pin


class Native_TESTONLY:
    def __init__(self, lease, *, assignment_failure=False, resume_failure=False,
                 thread_close_failure=False, create_close_failure=False, owner_exited=False,
                 create_interrupt=False):
        self.lease = lease
        self.assignment_failure = assignment_failure
        self.resume_failure = resume_failure
        self.thread_close_failure = thread_close_failure
        self.create_close_failure = create_close_failure
        self.owner_exited = owner_exited
        self.create_interrupt = create_interrupt
        self.calls = []
        self.nested = False
        self.suspended_inherited = False
        self.assigned = False
        self.resumed = False
        self.active = [1, 0]
    def current_in_job(self):
        self.calls.append("nested-check")
        return self.nested
    def new_job(self, name):
        self.calls.append(("job", name)); return 10
    def inherited_job_membership(self, process):
        self.calls.append(("inherited-job-observation", process))
        return self.suspended_inherited
    def member(self, process, job):
        self.calls.append(("membership", process, job))
        return self.suspended_inherited if job is None else self.assigned
    def pipe(self):
        self.calls.append("pipe"); return 11, 12
    def create_suspended(self, command, environment, cwd, channel, *, process_information=None):
        self.calls.append(("create-suspended", list(command), dict(environment), str(cwd), channel))
        result = dict(process=20, thread=21, pid=300)
        if process_information is not None:
            process_information.process, process_information.thread = 20, 21
            process_information.pid, process_information.tid = 300, 301
        if self.create_close_failure:
            raise owned.NativeCloseIncomplete(["TESTONLY duplicate close"], created_process=result)
        if self.create_interrupt:
            error = KeyboardInterrupt("TESTONLY post-CreateProcess interrupt")
            error.created_process = result
            raise error
        return result
    def identity(self, handle):
        self.calls.append(("identity", handle))
        if handle == 22:
            return dict(pid=301, creation_filetime_100ns="116444737270000000")
        return dict(pid=300, creation_filetime_100ns="116444737250000000")
    def assign(self, job, process):
        self.calls.append(("assign", job, process))
        if self.assignment_failure:
            raise OSError("TESTONLY assignment refused")
        self.assigned = True
    def resume(self, thread):
        self.calls.append(("resume", thread))
        self.resumed = True
        if self.resume_failure:
            raise CachePolicyError("TESTONLY unknown ResumeThread result")
    def close(self, handle):
        self.calls.append(("close", handle))
        if handle == 21 and self.thread_close_failure:
            raise OSError("TESTONLY primary thread close unknown")
    def read_message(self, handle):
        self.calls.append(("read", handle))
        observed = next(pin for name, _, pin in self.lease.records if name.endswith("-observed.json"))
        return dict(format="darsia-native-service-bootstrap-v1", service_receipt=SERVICE,
                    native_launch=observed, owner=dict(OWNER, pid=301,
                        creation_filetime_100ns="116444737270000000"))
    def open_owner_while_live(self, owner, job):
        self.calls.append(("bind-live-owner", deepcopy(owner), job))
        if self.owner_exited:
            raise CachePolicyError("TESTONLY actual owner exited before live handle binding")
        return 22
    def terminate_before_resume(self, process):
        self.calls.append(("terminate-before-instruction", process))
    def accounting(self, job):
        if not self.assigned:
            return dict(total_processes=0, active_processes=0, terminated_by_limit_processes=0)
        if not self.resumed:
            return dict(total_processes=1, active_processes=1, terminated_by_limit_processes=0)
        active = self.active.pop(0) if len(self.active) > 1 else self.active[0]
        self.calls.append(("accounting", active))
        return dict(total_processes=3, active_processes=active, terminated_by_limit_processes=0)
    def require_job_flags(self, job):
        self.calls.append("flags-check"); return 0
    def signaled(self, process):
        self.calls.append(("signaled", process)); return True
    def exit_code(self, process):
        return 17


@pytest.fixture
def native_world(monkeypatch):
    monkeypatch.setattr(bootstrap, "_owner", lambda: (deepcopy(OWNER), {"TESTONLY": True}))
    monkeypatch.setattr(owned.time, "sleep", lambda duration: None)
    def original_control(lease, binding):
        return next(deepcopy(document) for _, document, pin in lease.records if pin == binding)
    monkeypatch.setattr(owned, "_bound_control", original_control)
    return Lease_TESTONLY()


def test_original_public_argv_and_child_environment_not_rewritten(native_world):
    api = Native_TESTONLY(native_world)
    service = owned.OwnedWorker(native_world, api=api)
    command = ["TESTONLY_python", "-B", "-m", "darsia.fluidflower.worker", "--job", "TESTONLY_UNC"]
    parent = dict(TEMP="original", OMP_NUM_THREADS="7", UNRELATED="preserved")
    original = dict(parent)
    try:
        assert service.start(command, parent, Path("TESTONLY_CWD")) == SERVICE
        assert service.info["pid"] == 300 and service.native_primary["pid"] == 301
        assert service.owner_handle == 22 and service.primary_binding is not None
        created = next(row for row in api.calls if isinstance(row, tuple) and row[0] == "create-suspended")
        assert created[1] == command and created[2]["OMP_NUM_THREADS"] == "7"
        assert created[2]["TEMP"] == "original" and parent == original
        assert owned.CONTEXT_ENV in created[2] and owned.CHANNEL_ENV in created[2]
        labels = [row[0] if isinstance(row, tuple) else row for row in api.calls]
        assert labels.index("create-suspended") < labels.index("identity") < labels.index("inherited-job-observation") < labels.index("assign") < labels.index("resume")
        terminal = service.await_family()
        assert terminal["accounting"]["active_processes"] == 0
        assert terminal["wrapper_exit_code"] == 17
    finally:
        service.close()
    assert not any(isinstance(row, tuple) and row[0] == "terminate-before-instruction" for row in api.calls)
    assert ("close", 22) in api.calls


@pytest.mark.parametrize("inherited", [False, True])
def test_existing_parent_membership_is_recorded_but_own_assignment_is_single_attempt(native_world, inherited):
    api = Native_TESTONLY(native_world); api.nested = True
    api.suspended_inherited = inherited
    service = owned.OwnedWorker(native_world, api=api)
    assert service.start(["TESTONLY"], {}, Path("TESTONLY")) == SERVICE
    assert api.calls.count(("assign", 10, 20)) == 1 and ("resume", 21) in api.calls
    observed = next(value for name, value, _ in native_world.records if name.endswith("-observed.json"))
    assert observed["inherited_job_membership_observed"] is inherited
    assert observed["own_job_assignment_verified_before_resume"] is True
    assert observed["own_job_accounting_before_resume"] == dict(
        total_processes=1, active_processes=1, terminated_by_limit_processes=0)
    service.close()
    assert "nested-check" not in api.calls
    assert ("inherited-job-observation", 20) in api.calls


@pytest.mark.parametrize("failure", ["assignment_failure", "create_close_failure", "create_interrupt"])
def test_failed_preinstruction_launch_retires_only_exact_returned_handle(native_world, failure):
    api = Native_TESTONLY(native_world, **{failure: True})
    service = owned.OwnedWorker(native_world, api=api)
    with pytest.raises((OSError, owned.NativeCloseIncomplete, KeyboardInterrupt)):
        service.start(["TESTONLY"], {}, Path("TESTONLY"))
    assert ("terminate-before-instruction", 20) in api.calls
    assert ("resume", 21) not in api.calls
    if failure == "create_close_failure":
        with pytest.raises(owned.NativeCloseIncomplete): service.close()
    else:
        service.close()
    assert ("close", 20) in api.calls and ("close", 21) in api.calls


@pytest.mark.parametrize("boundary", ["check", "finally"])
def test_post_create_interrupt_preserves_native_output_before_python_dict(monkeypatch, boundary):
    """Actual ctypes output shape, synthetic functions: no Windows API call."""
    calls, last_error = [], [0]
    monkeypatch.setattr(owned.C, "set_last_error", lambda value: last_error.__setitem__(0, value))
    monkeypatch.setattr(owned.C, "get_last_error", lambda: last_error[0])
    def initialize(attributes, count, flags, size):
        size._obj.value = 64
        if attributes is None:
            last_error[0] = 122
            return False
        return True
    def create(*args):
        info = args[-1]._obj
        info.process, info.thread, info.pid, info.tid = 20, 21, 300, 301
        calls.append("native-returned")
        return True
    def check(okay, operation):
        if boundary == "check" and operation == "Exact worker CreateProcessW failed before resume":
            raise KeyboardInterrupt("TESTONLY immediately after native return")
        assert okay
    def delete(attributes):
        calls.append("attributes-closed-or-interrupted")
        if boundary == "finally":
            raise KeyboardInterrupt("TESTONLY native API finally boundary")
    kernel = SimpleNamespace(GetCurrentProcess=lambda: 99, GetStdHandle=lambda number: None,
        InitializeProcThreadAttributeList=initialize, UpdateProcThreadAttribute=lambda *args: True,
        DeleteProcThreadAttributeList=delete, CreateProcessW=create)
    api = SimpleNamespace(k=kernel, check=check, close=lambda handle: calls.append(("close", handle)))
    caller_output = owned._PROCESSINFO()
    with pytest.raises((KeyboardInterrupt, owned.NativeCloseIncomplete)) as error:
        owned.NativeAPI.create_suspended(api, ["TESTONLY"], {}, Path("TESTONLY"), 12,
                                        process_information=caller_output)
    assert error.value.created_process == dict(process=20, thread=21, pid=300)
    assert caller_output.process == 20 and caller_output.thread == 21 and caller_output.pid == 300
    if boundary == "check":
        assert error.value.resume_not_attempted_by_create is True
        assert error.value.native_create_flags == 0x4 | 0x400 | 0x80000
    else:
        assert error.value.close_failures[0]["error_type"] == "KeyboardInterrupt"
    assert calls == ["native-returned", "attributes-closed-or-interrupted"]


@pytest.mark.parametrize("boundary", ["api_finally_without_exception_attributes", "return_to_self_assignment"])
def test_outer_controller_recovers_caller_owned_handles_even_without_api_error_evidence(native_world, boundary):
    api = Native_TESTONLY(native_world)
    original = api.create_suspended
    def create(*args, **kwargs):
        result = original(*args, **kwargs)
        if boundary == "api_finally_without_exception_attributes":
            raise KeyboardInterrupt("TESTONLY after native API body, no decorated exception")
        return result
    api.create_suspended = create
    class InterruptedAssignment(owned.OwnedWorker):
        interrupt_once = True
        def __setattr__(self, name, value):
            if (boundary == "return_to_self_assignment" and name == "info" and value is not None
                    and self.interrupt_once):
                self.interrupt_once = False
                raise KeyboardInterrupt("TESTONLY return before self.info stores")
            super().__setattr__(name, value)
    service = InterruptedAssignment(native_world, api=api)
    with pytest.raises(KeyboardInterrupt):
        service.start(["TESTONLY"], {}, Path("TESTONLY"))
    assert service.info == dict(process=20, thread=21, pid=300)
    assert service.creation_output.process == 20 and service.creation_output.thread == 21
    assert ("terminate-before-instruction", 20) in api.calls
    assert ("resume", 21) not in api.calls
    service.close()
    assert ("close", 20) in api.calls and ("close", 21) in api.calls


def test_outer_close_recovers_original_output_when_no_return_assignment_exists(native_world):
    api = Native_TESTONLY(native_world)
    service = owned.OwnedWorker(native_world, api=api)
    service.creation_output.process, service.creation_output.thread = 20, 21
    service.creation_output.pid = 300
    assert service.info is None
    service.close()
    assert ("close", 20) in api.calls and ("close", 21) in api.calls


def test_actual_owner_refusal_and_unconfirmed_temporary_close_are_both_retained():
    native = dict(pid=301, creation_filetime_100ns="116444737270000000")
    calls = []
    def close(handle):
        calls.append(handle)
        raise OSError("TESTONLY temporary owner CloseHandle unconfirmed")
    api = SimpleNamespace(k=SimpleNamespace(OpenProcess=lambda *args: 22),
        check=lambda okay, message: None, identity=lambda handle: native,
        signaled=lambda handle: False, member=lambda handle, job: False, close=close)
    with pytest.raises(owned.NativeCloseIncomplete) as error:
        owned.NativeAPI.open_owner_while_live(api, native, 10)
    assert calls == [22]
    evidence = error.value.close_failures[0]
    assert evidence["handle"] == 22 and evidence["close_confirmed"] is False
    assert "outside the original retained Job" in evidence["primary_message"]
    assert "CloseHandle unconfirmed" in evidence["message"]
    assert evidence["no_automatic_close_retry"] is True


@pytest.mark.parametrize("member", [False, True])
def test_inherited_membership_only_observes_actual_child_handle_not_parent(member):
    calls = []
    api = SimpleNamespace(member=lambda process, job: calls.append((process, job)) or member)
    assert owned.NativeAPI.inherited_job_membership(api, 20) is member
    assert calls == [(20, None)]


@pytest.mark.parametrize("defect", ["identity", "membership", "flags", "accounting"])
def test_post_assignment_own_job_proof_refuses_before_resume(native_world, defect):
    api = Native_TESTONLY(native_world)
    original_identity, original_accounting = api.identity, api.accounting
    if defect == "identity":
        api.identity = lambda handle: dict(original_identity(handle), pid=999) if api.assigned else original_identity(handle)
    elif defect == "membership":
        api.member = lambda process, job: False
    elif defect == "flags":
        api.require_job_flags = lambda job: 0x1000 if api.assigned else 0
    else:
        api.accounting = lambda job: dict(total_processes=2, active_processes=2,
            terminated_by_limit_processes=0) if api.assigned else original_accounting(job)
    service = owned.OwnedWorker(native_world, api=api)
    with pytest.raises(CachePolicyError, match="own-Job identity/flags/accounting"):
        service.start(["TESTONLY"], {}, Path("TESTONLY"))
    assert api.calls.count(("assign", 10, 20)) == 1
    assert ("resume", 21) not in api.calls and ("terminate-before-instruction", 20) in api.calls
    service.close()


@pytest.mark.parametrize("failure", ["resume_failure", "thread_close_failure"])
def test_unknown_after_resume_attempt_never_becomes_preinstruction_kill(native_world, failure):
    api = Native_TESTONLY(native_world, **{failure: True})
    service = owned.OwnedWorker(native_world, api=api)
    with pytest.raises((CachePolicyError, owned.NativeCloseIncomplete)):
        service.start(["TESTONLY"], {}, Path("TESTONLY"))
    assert service.resume_attempted is True
    assert ("terminate-before-instruction", 20) not in api.calls
    if failure == "thread_close_failure":
        assert service.close_failures
        with pytest.raises(owned.NativeCloseIncomplete): service.close()
    else:
        service.close()


def test_fast_python_owner_before_live_binding_stays_unqualified(native_world):
    api = Native_TESTONLY(native_world, owner_exited=True)
    service = owned.OwnedWorker(native_world, api=api)
    with pytest.raises(CachePolicyError, match="owner exited"):
        service.start(["TESTONLY"], {}, Path("TESTONLY"))
    assert service.owner_handle is None and service.primary_binding is None
    assert not any(name.endswith("-retirement.json") for name, _, _ in native_world.records)
    assert ("terminate-before-instruction", 20) not in api.calls
    service.close()


@pytest.mark.parametrize("defect", ["exited", "filetime", "outside"])
def test_actual_owner_binding_refuses_unknown_and_closes_temporary_handle(defect):
    closes = []
    native = dict(pid=301, creation_filetime_100ns="116444737270000000")
    owner = dict(native)
    if defect == "filetime":
        owner["creation_filetime_100ns"] = "116444737280000000"
    api = SimpleNamespace(k=SimpleNamespace(OpenProcess=lambda *args: 22),
        check=lambda okay, message: None, identity=lambda handle: native,
        signaled=lambda handle: defect == "exited", member=lambda handle, job: defect != "outside",
        close=lambda handle: closes.append(handle))
    with pytest.raises(CachePolicyError):
        owned.NativeAPI.open_owner_while_live(api, owner, 10)
    assert closes == [22]


def test_live_job_or_unbound_owner_cannot_satisfy_retirement(native_world):
    api = Native_TESTONLY(native_world)
    service = owned.OwnedWorker(native_world, api=api)
    service.start(["TESTONLY"], {}, Path("TESTONLY"))
    terminal = service.await_family()
    binding = service.record_closed_family(terminal)
    member = service.observed_binding
    snapshot = dict(binding=dict(manifest=dict(native_launch_context=member, owner=service.native_primary),
                                 policy=native_world.source, profile=native_world.profile_binding),
        owners=[dict(document=dict(value=dict(owner=deepcopy(service.native_primary), native_job_membership=member)))],
        launches=[], legacy_launches=[])
    assert service.verify_retirement(snapshot, binding) is True
    api.active = [1]
    assert service.verify_retirement(snapshot, binding) is False
    api.active = [0]
    snapshot["owners"][0]["document"]["value"]["native_job_membership"] = None
    assert service.verify_retirement(snapshot, binding) is False
    service.close()


@pytest.mark.parametrize("missing", ["profile", "startup", "bytecode"])
def test_missing_qualified_startup_refuses_without_native_api_or_allocation(monkeypatch, missing):
    executable = dict(path="TESTONLY_python", sha256="d" * 64, size_bytes=1)
    profile = dict(python_executable=executable, module_closure=[])
    block = dict(format="darsia-native-worker-supervision-profile-v1",
        completion=dict(path="TESTONLY_completion", sha256="e" * 64, size_bytes=1),
        startup=dict(path="TESTONLY_startup", sha256="f" * 64, size_bytes=1),
        maximum_family_processes=3, publication_contract="alias-done-small-metadata-v1")
    profile["native_supervision"] = block
    startup = dict(format="darsia-qualified-worker-startup-source-v1",
        python_executable=executable, module_closure=[], include_system_site_packages=False,
        source_files=[dict(path="TESTONLY_source", sha256="a" * 64, size_bytes=1)],
        sys_path=[str(Path("TESTONLY_SOURCE").resolve())], allowed_loaded_modules={})
    def exact(pin, package_file):
        if pin is block["completion"]:
            return dict(complete=True, passed=True)
        return startup
    monkeypatch.setattr(bootstrap, "_exact", exact)
    monkeypatch.setattr(owned, "_exact_source", lambda pin: None)
    monkeypatch.setattr(owned, "sys", SimpleNamespace(path=["TESTONLY_SOURCE"],
        modules={}, flags=SimpleNamespace(dont_write_bytecode=missing != "bytecode")))
    if missing == "profile":
        profile.pop("native_supervision")
    elif missing == "startup":
        startup["source_files"] = []
    with pytest.raises(CachePolicyError):
        owned._checked_contract(SimpleNamespace(profile=profile, package_file="TESTONLY_PACKAGE"))


def test_public_command_remains_same_original_worker_argv(monkeypatch):
    argv = ["TESTONLY_python", "-B", "-m", "darsia.fluidflower.worker", "--workers", "2"]
    monkeypatch.setattr(owned, "sys", SimpleNamespace(orig_argv=argv,
        executable="TESTONLY_python", flags=SimpleNamespace(dont_write_bytecode=True)))
    monkeypatch.setattr(bootstrap, "_python_module_invocation",
        lambda: ("darsia.fluidflower.worker", ["--workers", "2"]))
    monkeypatch.setattr(owned, "_exact_source", lambda pin: None)
    lease = SimpleNamespace(profile=dict(python_executable=dict(path="TESTONLY_python",
        sha256="a" * 64, size_bytes=1)))
    assert owned._public_command(lease) == argv


@pytest.mark.parametrize("defect", [None, "foreign_argv0", "missing_base_binding", "wrong_selected", "changed_base"])
def test_uv_base_argv0_keeps_qualified_selected_venv_and_all_worker_arguments(monkeypatch, defect):
    selected, base = "TESTONLY_venv_python", "TESTONLY_base_python"
    argv = [base, "-X", "utf8", "-B", "-m", "darsia.fluidflower.worker", "--once", "--stores", "TESTONLY_map"]
    selected_pin = dict(path=selected, sha256="a" * 64, size_bytes=1)
    base_pin = dict(path=base, sha256="b" * 64, size_bytes=1)
    startup_pin = dict(path="TESTONLY_startup", sha256="c" * 64, size_bytes=1)
    lease = SimpleNamespace(profile=dict(python_executable=selected_pin,
        native_supervision=dict(startup=startup_pin)), package_file="TESTONLY_PACKAGE")
    startup = dict(source_files=[] if defect == "missing_base_binding" else [base_pin])
    if defect == "foreign_argv0": argv[0] = "FOREIGN_python"
    if defect == "wrong_selected": selected_pin["path"] = "FOREIGN_python"
    checked = []
    def exact_source(pin):
        checked.append(pin)
        if defect == "changed_base" and pin is base_pin:
            raise CachePolicyError("TESTONLY changed qualified base executable")
    monkeypatch.setattr(owned, "_exact_source", exact_source)
    monkeypatch.setattr(bootstrap, "_exact", lambda pin, package: startup)
    monkeypatch.setattr(bootstrap, "_python_module_invocation", lambda: ("darsia.fluidflower.worker", argv[6:]))
    monkeypatch.setattr(owned, "sys", SimpleNamespace(orig_argv=argv,
        executable=selected, _base_executable=base, flags=SimpleNamespace(dont_write_bytecode=True)))
    before = list(argv)
    if defect is None:
        result = owned._public_command(lease)
        assert result == [selected, *argv[1:]] and result[0] != base
        assert checked == [selected_pin, base_pin]
    else:
        with pytest.raises(CachePolicyError): owned._public_command(lease)
    assert argv == before


def test_close_failure_does_not_skip_remaining_original_handles():
    calls = []
    def close(handle):
        calls.append(handle)
        if handle == 1:
            raise OSError("TESTONLY close unknown")
    with pytest.raises(owned.NativeCloseIncomplete) as error:
        owned._close_handles(SimpleNamespace(close=close), [("first", 1), ("second", 2)])
    assert calls == [1, 2]
    assert error.value.close_failures[0]["kind"] == "first"


@pytest.mark.parametrize("result,state,attempted,expected", [
    (None, None, False, False),
    (None, None, True, None),
    ({"service_budget_closed": True}, None, True, True),
    (None, {"service_budget_closed": True}, True, True),
    (None, {"service_budget_closed": None}, True, None),
])
def test_partial_audit_never_relabels_closed_or_unknown_budget(result, state, attempted, expected):
    error = SimpleNamespace(finalization_state=state)
    assert owned._budget_state(result, error, attempted) is expected


@pytest.mark.parametrize("reused", [False, True])
def test_native_filetime_refuses_psutil_pid_reuse_and_closes_all_handles(monkeypatch, reused):
    calls = []
    pin = dict(path="TESTONLY_observed", sha256="f" * 64, size_bytes=1)
    monkeypatch.setattr(owned, "_ACTIVE_CHILD", dict(binding=pin, job_name="TESTONLY_JOB"))
    api = SimpleNamespace(k=SimpleNamespace(OpenProcess=lambda *args: 20,
        OpenJobObjectW=lambda *args: 30),
        check=lambda okay, message: None,
        identity=lambda handle: dict(pid=300, creation_filetime_100ns="116444737250000000"),
        member=lambda handle, job: True, close=lambda handle: calls.append(handle))
    process = SimpleNamespace(pid=300, create_time=lambda: 126.0 if reused else 125.0)
    if reused:
        with pytest.raises(CachePolicyError, match="incarnations differ"):
            owned.native_launch_identity(process, api=api)
    else:
        assert owned.native_launch_identity(process, api=api)["creation_filetime_100ns"] == "116444737250000000"
    assert calls == [30, 20]


@pytest.mark.parametrize("kind", ["same", "newer", "missing", "unknown_native"])
def test_retained_sentinel_never_merges_newer_psutil_epoch_or_argv(kind):
    native = dict(pid=300, creation_filetime_100ns="116444737250000000",
                  native_launch_context=dict(TESTONLY="original"))
    observation = dict(native, process_create_time="125.0", command_line=["original"])
    if kind == "newer":
        observation.update(creation_filetime_100ns="116444737260000000",
                           process_create_time="126.0", command_line=["foreign-newer"])
    elif kind == "missing":
        observation = None
    elif kind == "unknown_native":
        observation.pop("creation_filetime_100ns")
    result = owned.bind_retained_native_observation(native, observation)
    assert result["creation_filetime_100ns"] == native["creation_filetime_100ns"]
    if kind == "same":
        assert result["process_create_time"] == "125.0" and result["command_line"] == ["original"]
    else:
        assert result["process_create_time"] == "125" and result["command_line"] is None
        assert result["psutil_epoch_and_argv_not_bound"] is True
    assert observation is None or observation.get("command_line") != result["command_line"] or kind == "same"


def test_console_interrupt_observer_does_not_kill_the_running_worker(native_world):
    api = Native_TESTONLY(native_world)
    original = api.accounting
    count = [0]
    def interrupted_once(job):
        count[0] += 1
        if count[0] == 1:
            raise KeyboardInterrupt
        return original(job)
    service = owned.OwnedWorker(native_world, api=api)
    service.start(["TESTONLY"], {}, Path("TESTONLY"))
    api.accounting = interrupted_once  # Interrupt the live observer, not empty-Job admission.
    assert service.await_family()["accounting"]["active_processes"] == 0
    assert not any(isinstance(row, tuple) and row[0] == "terminate-before-instruction" for row in api.calls)
    service.close()


@pytest.mark.skipif(os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8,
                    reason="The native worker API is Windows x64 only")
def test_actual_native_job_starts_empty_without_breakaway_or_kill_on_close():
    """Real kernel ABI/accounting only; no child, science, cache policy or quota."""
    api = owned.NativeAPI()
    import uuid
    handle = api.new_job("Local\\DarSIA-cache-" + uuid.uuid4().hex)
    try:
        assert api.require_job_flags(handle) & (0x800 | 0x1000 | 0x2000) == 0
        assert api.accounting(handle)["active_processes"] == 0
        assert api.accounting(handle)["total_processes"] == 0
    finally:
        api.close(handle)


@pytest.mark.skipif(os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8,
                    reason="The native worker API is Windows x64 only")
def test_actual_suspended_stdlib_child_is_in_job_before_resume_and_retires(tmp_path):
    """One tiny stdlib child tests handles/ABI, never the scientific worker."""
    import uuid
    api = owned.NativeAPI()
    job = api.new_job("Local\\DarSIA-cache-" + uuid.uuid4().hex)
    read = write = info = owner_handle = None
    caller_output = owned._PROCESSINFO()
    resume_attempted = False
    ack = tmp_path / "owner-bound.TESTONLY"
    script = '''import ctypes as C, os, msvcrt, json, time
from ctypes import wintypes as W
from pathlib import Path
k=C.WinDLL("kernel32",use_last_error=True)
k.GetCurrentProcess.restype=W.HANDLE
k.GetProcessTimes.argtypes=[W.HANDLE,*([C.POINTER(W.FILETIME)]*4)]
k.GetProcessTimes.restype=W.BOOL
t=[W.FILETIME() for _ in range(4)]
assert k.GetProcessTimes(k.GetCurrentProcess(),*[C.byref(x) for x in t])
owner=dict(pid=os.getpid(),creation_filetime_100ns=str((t[0].dwHighDateTime<<32)|t[0].dwLowDateTime))
fd=msvcrt.open_osfhandle(int(os.environ["TESTONLY_CHANNEL"]),os.O_WRONLY|os.O_BINARY)
data=(json.dumps(owner,separators=(",",":"))+"\\n").encode()
assert os.write(fd,data)==len(data)
os.close(fd)
deadline=time.monotonic()+15
while not Path(os.environ["TESTONLY_ACK"]).exists():
    assert time.monotonic()<deadline
    time.sleep(.01)
'''
    try:
        read, write = api.pipe()
        environment = dict(os.environ, TESTONLY_CHANNEL=str(write), TESTONLY_ACK=str(ack),
                           PYTHONDONTWRITEBYTECODE="1")
        try:
            info = api.create_suspended([sys.executable, "-B", "-c", script], environment, tmp_path, write,
                                        process_information=caller_output)
        except BaseException as error:
            info = getattr(error, "created_process", None)
            if info is None and (caller_output.process or caller_output.thread):
                info = dict(process=caller_output.process, thread=caller_output.thread, pid=int(caller_output.pid))
            raise
        identity = api.identity(info["process"])
        assert identity["pid"] == info["pid"]
        inherited = api.inherited_job_membership(info["process"])
        assert type(inherited) is bool
        api.assign(job, info["process"])
        assert api.member(info["process"], job) is True
        assert api.identity(info["process"]) == identity
        assert api.require_job_flags(job) == 0
        assert api.accounting(job) == dict(total_processes=1, active_processes=1,
                                          terminated_by_limit_processes=0)
        assert api.signaled(info["process"]) is False
        resume_attempted = True
        api.resume(info["thread"])
        api.close(info.pop("thread"))
        api.close(write); write = None
        actual_owner = api.read_message(read)
        owner_handle = api.open_owner_while_live(actual_owner, job)
        assert api.identity(owner_handle) == actual_owner
        assert api.member(owner_handle, job) is True
        ack.write_bytes(b"bound")
        api.close(read); read = None
        deadline = time.monotonic() + 15
        while (not api.signaled(info["process"]) or api.accounting(job)["active_processes"] != 0):
            assert time.monotonic() < deadline, "TESTONLY native child/Job did not retire"
            time.sleep(.02)
        assert api.identity(info["process"]) == identity
        assert api.identity(owner_handle) == actual_owner
        assert api.signaled(owner_handle) is True
        assert api.exit_code(info["process"]) == 0
        assert api.accounting(job)["active_processes"] == 0
        assert api.require_job_flags(job) & (0x800 | 0x1000 | 0x2000) == 0
    finally:
        ack.write_bytes(b"teardown")
        if info is not None:
            if not resume_attempted:
                api.terminate_before_resume(info["process"])
            for name in ("thread", "process"):
                if name in info:
                    api.close(info[name])
        for handle in (owner_handle, read, write, job):
            if handle is not None:
                api.close(handle)
