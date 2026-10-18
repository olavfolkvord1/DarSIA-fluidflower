"""Synthetic protocol cases; never a real cache profile or host admission."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from darsia_cache import first_qualification as first
from darsia_cache.policy import CachePolicyError, PHYSICAL_RESERVE_BYTES
from darsia_cache.worker_adapter import WorkerCacheAdapter


def pin(name, digit="a"):
    return dict(path="C:/TESTONLY/"+name, sha256=digit*64, size_bytes=1)


def provider():
    producers=[pin("producer.py")]
    runtime=[pin("runtime.py", "b")]
    return dict(format=first.PROVIDER, qualification_id="1"*32,
        scope=dict(commit="c"*40, pipeline="native-phase-n1", backend="cpu",
            precision="float32", run="ac52", ordered_case_ids=["ac52_f"+str(i) for i in range(39)]),
        namespace_admission=pin("namespace.json"),
        output_namespace=dict(path="C:/TESTONLY/receipts",identity=dict(TESTONLY="not admitted")),
        slots={name:name+".json" for name in first.SLOTS},
        python_executable=pin("python.exe"), startup=pin("startup.json"),
        producer_source_closure=producers,runtime_source_closure=runtime,
        module_closure=producers+runtime)


def protocol():
    value=provider(); binding=pin("provider.json"); stores=pin("stores.toml")
    request=dict(format=first.REQUEST, execution_authorized=True, authority_ids=["Q97","Q99"],
        root_release=pin("release.json"), provider=binding, stores_source=stores,
        namespace_admission=value["namespace_admission"], scope=value["scope"],
        task=pin("task.json"),
        cache_stat_period_seconds=30,
        public_command=["C:/TESTONLY/python.exe","-B","-m","darsia.fluidflower.worker"])
    release=dict(format="darsia-cache-first-qualification-root-release-v1",
        execution_authorized=True,provider=binding,stores_source=stores,scope=value["scope"],
        request_payload_sha256=first._digest({k:v for k,v in request.items() if k!="root_release"}))
    task=deepcopy(value["scope"]);task.pop("backend")
    task.update(attempt_id="2"*32,job_id="TESTONLY-job")
    objects={binding["path"]:value,request["root_release"]["path"]:release,
        "C:/TESTONLY/task.json":task,
        "C:/TESTONLY/request.json":request}
    read=lambda descriptor:deepcopy(objects[descriptor["path"]])
    state=first.FirstQualification(pin("request.json"),provider_binding=binding,
        stores_source=stores,namespace=value["namespace_admission"],read=read)
    return state,objects


def report(**changes):
    value=dict(hard_filesystem_quota_verified=True,reserve_bytes=PHYSICAL_RESERVE_BYTES,
        quota_headroom_bytes=500,free_physical_bytes=PHYSICAL_RESERVE_BYTES+400,
        active_cache_remaining_bytes=20,required_noncache_bytes=30,
        current_cache_logical_bytes=0)
    value.update(changes)
    return value


def test_predeclared_provider_is_acyclic_not_a_fake_completed_profile():
    state,objects=protocol()
    assert set(state.provider)==set(provider())
    assert not any(key in state.provider for key in ("stores_source","request","profiles","passed"))
    assert state.binding not in state.provider.values()
    assert "maximum_family_processes" not in state.provider
    assert state.request["stores_source"]==pin("stores.toml")
    assert objects["C:/TESTONLY/release.json"]["request_payload_sha256"]==first._digest(
        {k:v for k,v in state.request.items() if k!="root_release"})


@pytest.mark.parametrize("mutation",[
    lambda v:v["scope"].update(pipeline="n1"),
    lambda v:v["scope"].update(precision="float64"),
    lambda v:v["scope"].update(backend="cuda"),
    lambda v:v["scope"].update(run="ac18"),
    lambda v:v["scope"].update(ordered_case_ids=["same"]*39),
    lambda v:v["scope"].update(ordered_case_ids=["a"]),
    lambda v:v["slots"].update(completion="../foreign.json"),
    lambda v:v["slots"].update(completion=v["slots"]["observation"]),
    lambda v:v["module_closure"].pop(),
    lambda v:v.update(profiles={"TESTONLY":"forbidden cycle"}),
])
def test_finite_provider_rejects_wrong_stage_cases_sources_and_output_aliases(mutation):
    value=provider();mutation(value)
    with pytest.raises(CachePolicyError):first.checked_provider(value)


def test_administrative_entitlement_uses_actual_headroom_less_existing_reserves():
    assert first.administrative_entitlement(report())==350
    assert first.administrative_entitlement(report(quota_headroom_bytes=100))==100
    # This number is not presented as an earlier observed or measured peak.


@pytest.mark.parametrize("changes",[
    dict(hard_filesystem_quota_verified=False),dict(reserve_bytes=0),
    dict(quota_headroom_bytes=0),dict(free_physical_bytes=PHYSICAL_RESERVE_BYTES),
])
def test_first_admission_keeps_real_cap_and_original_free_reserve(changes):
    with pytest.raises(CachePolicyError):first.administrative_entitlement(report(**changes))


def test_modified_request_after_root_payload_release_refuses():
    state,objects=protocol()
    objects["C:/TESTONLY/request.json"]["public_command"].append("--changed")
    with pytest.raises(CachePolicyError,match="Root-released"):
        first.FirstQualification(pin("request.json"),provider_binding=state.provider_binding,
            stores_source=state.stores_source,namespace=state.provider["namespace_admission"],
            read=lambda bound:deepcopy(objects[bound["path"]]))


class Ledger_TESTONLY:
    def __init__(self):
        self.documents={};self.control=self;self.maximums=[]
    @contextmanager
    def locked(self,**kwargs):yield self
    def project(self,maximum):
        assert maximum==0
        return report(active_cache_remaining_bytes=0)
    def write_once(self,name,raw):
        if name in self.documents:raise CachePolicyError("TESTONLY immutable marker already exists")
        self.documents[name]=raw
    def names(self):return list(self.documents)
    def read_once(self,name):return self.documents[name],{}
    def require_reservation(self,*args):pass
    def require_unbound_scope(self,*args):pass


def lease_for(state):
    ledger=Ledger_TESTONLY()
    return SimpleNamespace(first_qualification=state,ledger=ledger,
        document={"cache_bootstrap":{"namespace_admission":state.provider["namespace_admission"]}},
        policy=SimpleNamespace(volume_cap=object(),quota_bytes=500*1024**3,
            profiles={"jobs":state.provider_binding}),
        admission={"receipts":state.provider["output_namespace"]},
        owner=dict(pid=12,process_create_time="123.0"),source=state.stores_source)


def test_once_marker_is_shared_host_authority_and_repeated_provider_cannot_retry(monkeypatch):
    state,_=protocol();lease=lease_for(state)
    monkeypatch.setattr(first,"_outputs_absent",lambda value:None)
    state.consume(lease)
    assert state.started is True and len(lease.ledger.documents)==1
    replacement,_=protocol();replacement.provider["qualification_id"]="2"*32
    with pytest.raises(CachePolicyError,match="already exists"):replacement.consume(lease)


def test_existing_output_refuses_before_native_launch(monkeypatch):
    state,_=protocol();lease=lease_for(state)
    def output_exists(value):raise CachePolicyError("existing completed/partial profile")
    monkeypatch.setattr(first,"_outputs_absent",output_exists)
    with pytest.raises(CachePolicyError,match="existing completed/partial"):state.consume(lease)
    assert not lease.ledger.documents


def test_service_entitlement_requires_original_native_context_and_once_marker(monkeypatch):
    state,_=protocol();lease=lease_for(state)
    monkeypatch.setattr(first,"_outputs_absent",lambda value:None)
    with pytest.raises(CachePolicyError,match="controlled native Job"):
        state.service_entitlement(lease)
    state.consume(lease);lease.native_launch_context=pin("actual-native-launch.json")
    assert state.service_entitlement(lease)==370
    assert state.administrative["measured_profile"] is False
    assert state.administrative["admission"]["reserve_bytes"]==PHYSICAL_RESERVE_BYTES
    assert len(lease.ledger.documents)==1  # No service reservation is duplicated here.


def test_different_spawned_owner_cannot_reuse_first_claim():
    state,_=protocol();lease=lease_for(state);task=task_for(state)
    state.require_task(lease,task,remember=True)
    other,_=protocol();lease.first_qualification=other;lease.owner=dict(pid=99,process_create_time="999.0")
    with pytest.raises(CachePolicyError,match="different first claim/owner"):
        other.require_task(lease,task,remember=True)


def test_exact_task_contract_binds_calibration_and_all_original_input_metadata():
    state,_=protocol();lease=lease_for(state);task=task_for(state)
    altered=deepcopy(task);altered["calibration_sha256"]="f"*64
    with pytest.raises(CachePolicyError,match="exact released"):
        state.require_task(lease,altered)
    altered=deepcopy(task);altered["input_catalog"]={"TESTONLY":"different actual RAW binding"}
    with pytest.raises(CachePolicyError,match="exact released"):
        state.require_task(lease,altered)


def task_for(state):
    task=deepcopy(state.provider["scope"]);task.pop("backend")
    task.update(attempt_id="2"*32,job_id="TESTONLY-job")
    return task


def test_one_real_claim_can_bind_but_second_attempt_or_changed_cases_cannot():
    state,_=protocol();lease=lease_for(state);task=task_for(state)
    assert state.require_task(lease,task,remember=True) is True
    changed=deepcopy(task);changed["attempt_id"]="3"*32
    with pytest.raises(CachePolicyError):state.require_task(lease,changed)
    changed=deepcopy(task);changed["ordered_case_ids"].reverse()
    with pytest.raises(CachePolicyError,match="exact released"):state.require_task(lease,changed)


def test_first_claim_uses_existing_service_without_inventing_measured_zero_or_new_scope(monkeypatch):
    state,_=protocol();lease=lease_for(state)
    adapter=WorkerCacheAdapter.__new__(WorkerCacheAdapter)
    adapter.lease=lease;adapter.runtime=SimpleNamespace(backend="cpu",commit="c"*40)
    adapter.profiles={};adapter.claims={}
    task=task_for(state)
    assert adapter.prepare_first_claim(task,None)==state.binding
    proof=adapter.profiles[task["attempt_id"]]
    assert proof["first_administration"] is True and proof["coverage_sha256"] is None
    assert "maximum_observed_cache_bytes" not in proof
    lease.reservation=dict(token="4"*32,pool="5"*64,sha256="6"*64)
    records=[]
    lease._record=lambda name,value:(records.append(deepcopy(value)) or pin(name))
    monkeypatch.setattr("darsia_cache.worker_adapter._owner",lambda:(lease.owner,{}))
    claim=SimpleNamespace(token=task["attempt_id"],task=task)
    adapter.register_claim_allocation(claim,lease.ledger,{"token":claim.token},lease.owner,0,state.binding)
    assert records[0]["format"]=="darsia-worker-cache-first-claim-service-binding-v1"
    assert records[0]["claim_scope_created"] is False
    assert records[0]["service_reservation"]==lease.reservation
    assert records[0]["writer_coverage_sha256"] is None
    assert len(lease.ledger.documents)==1  # Only the first-claim authority record.


def test_sampled_peak_is_not_lifetime_peak_and_kernel_total_is_not_concurrency():
    state,_=protocol();collector=first.FiniteObservations(state)
    lease=lease_for(state);lease.policy.root="C:/TESTONLY/cache"
    rows=iter([dict(logical_bytes=90),dict(logical_bytes=12)])
    lease.measure=lambda root:next(rows)
    owned=SimpleNamespace(lease=lease,service_binding=pin("service.json"),
        observed_binding=pin("native.json"),job=1,
        api=SimpleNamespace(lifetime_memory=lambda job:dict(peak_job_committed_bytes=123)))
    collector.observe(owned,dict(active_processes=2));collector.observe(owned,dict(active_processes=0),force_cache=True)
    result=collector.document(owned,dict(accounting=dict(total_processes=5,active_processes=0),wrapper_exit_code=0),
        pin("retirement.json"),dict(TESTONLY="not production eligibility"))
    assert result["sampled_cache_logical_high_water_bytes"]==90
    assert result["sampled_cache_logical_high_water_is_lifetime_peak"] is False
    assert result["sampled_active_processes_high_water"]==2
    assert result["kernel_total_processes"]==5
    assert result["kernel_total_processes_is_concurrent_peak"] is False
    assert result["conservative_family_process_bound_from_kernel_total"]==5
    assert result["passed"] is False and result["writer_lifetime_cache_bytes_proof"] is None


def test_cache_stat_cadence_is_declared_bounded_and_preserves_every_cheap_kernel_count():
    state,_=protocol();now=[0.0]
    collector=first.FiniteObservations(state,clock=lambda:now[0])
    lease=lease_for(state);lease.policy.root="C:/TESTONLY/cache";scans=[]
    def measure(root):
        scans.append(now[0]);return dict(logical_bytes=len(scans))
    lease.measure=measure;owned=SimpleNamespace(lease=lease)
    for instant,active in [(0,2),(.2,3),(.4,4),(29.9,5),(30,6),(30.2,0)]:
        now[0]=instant;collector.observe(owned,dict(active_processes=active))
    assert scans==[0,30]
    assert collector.native_poll_count==6
    assert collector.sampled_active_processes_high_water==6
    collector.observe(owned,dict(active_processes=0),force_cache=True)
    assert scans==[0,30,30.2]
    assert collector.cache_samples[-1]["forced_terminal"] is True
    assert collector.cache_samples[1]["actual_gap_since_previous_start_seconds"]==30
    assert collector.native_poll_count==6


def test_exiting_module_inventory_preserves_unknown_sources_instead_of_inheriting_old_closure(monkeypatch):
    state,_=protocol()
    calls=[]
    monkeypatch.setattr("darsia_cache.owned_process._exact_source",lambda value:calls.append(value))
    monkeypatch.setattr(first,"sys",SimpleNamespace(modules={
        "known":SimpleNamespace(__file__="C:/TESTONLY/producer.py"),
        "unclassified":SimpleNamespace(__file__="C:/TESTONLY/unclassified.py"),
        "builtin":SimpleNamespace()}))
    result=state.loaded_modules()
    assert calls==[state.provider["producer_source_closure"][0]]
    assert result["all_file_backed_modules_predeclared_and_exact"] is False
    assert result["undeclared"][0]["name"]=="unclassified"
    assert result["actual_task"] is None
    assert result["this_is_not_process_retirement"] is True


@pytest.mark.parametrize("evidence",[None,{},dict(complete=True,passed=True),
    dict(maximum_observed_cache_bytes=1,all_writers=True),
    dict(required_capacity_bound=500*1024**3,physical_cap=True)])
def test_polling_or_fixed_cap_cannot_fabricate_a_passing_lifetime_profile(evidence):
    with pytest.raises(CachePolicyError):
        first.require_publishable(dict(format=first.OBSERVATION,complete=True),evidence)


def test_direct_old_profile_reader_is_unchanged_and_new_provider_never_falls_back(monkeypatch):
    old=dict(format="darsia-cache-import-profile-v1",TESTONLY="old reader returns exact same object")
    assert first.resolve_profile(pin("old.json"),"import",lambda binding:old) is old
    def incomplete(value,role):raise CachePolicyError("no completed provider bundle")
    monkeypatch.setattr(first,"_slot_read",incomplete)
    with pytest.raises(CachePolicyError,match="no completed"):
        first.resolve_profile(pin("new.json"),"import",lambda binding:provider())


def bound_plan_TESTONLY():
    capacity = 500 * 1024**3 - 2 * 1024**2
    return dict(format=first.CAPACITY_BOUND,
        authority_id="Q106-answer-sampled-operational-reserve-hard-bound-is-Q87-volume-no-new-seam",
        volume_cap_observation=dict(format="darsia-readonly-fixed-vhdx-cap-observation-v1",
            verified=True, filesystem="NTFS", is_loaded=True, volume_total_bytes=capacity),
        volume_capacity_bytes=capacity, free_volume_reserve_bytes=PHYSICAL_RESERVE_BYTES,
        required_capacity_bound_bytes=capacity,
        operational_admission_headroom_bytes=capacity-PHYSICAL_RESERVE_BYTES,
        reserve_watchdog_period_seconds=1,
        hard_bound_semantics="actual-fixed-volume-only-not-capacity-minus-reserve",
        reserve_semantics="operational-admission-plus-sampled-watchdog-not-hard-floor",
        exclusive_volume_writer_required=True,
        sampled_observed_highwater_is_complete=False,
        enforcement_state="PRELAUNCH_HOLD_N43_REFERENCE_OUTPUT_BUDGET_AND_PROFILE_PRODUCER")


def test_typed_capacity_bound_uses_actual_usable_capacity_not_sampled_highwater():
    value = bound_plan_TESTONLY()
    actual = first.checked_required_capacity_bound(value)
    assert actual == value and actual is not value
    assert "maximum_observed_cache_bytes" not in actual
    assert actual["enforcement_state"] == "PRELAUNCH_HOLD_N43_REFERENCE_OUTPUT_BUDGET_AND_PROFILE_PRODUCER"
    assert actual["required_capacity_bound_bytes"]==actual["volume_capacity_bytes"]
    assert actual["operational_admission_headroom_bytes"]==actual["volume_capacity_bytes"]-PHYSICAL_RESERVE_BYTES


@pytest.mark.parametrize("change", [
    lambda v: v.update(required_capacity_bound_bytes=v["required_capacity_bound_bytes"]+1),
    lambda v: v.update(free_volume_reserve_bytes=0),
    lambda v: v.update(volume_capacity_bytes=500*1024**3),
    lambda v: v.update(sampled_observed_highwater_is_complete=True),
    lambda v: v.update(enforcement_state="PASSED"),
    lambda v: v["volume_cap_observation"].update(verified=False),
    lambda v: v.update(required_capacity_bound_bytes=v["operational_admission_headroom_bytes"]),
    lambda v: v.update(reserve_watchdog_period_seconds=30),
    lambda v: v.update(exclusive_volume_writer_required=False),
    lambda v: v.update(reserve_semantics="hard-never-consumed-free-floor"),
])
def test_typed_bound_cannot_shrink_reserve_relabel_samples_or_claim_enforcement(change):
    value = bound_plan_TESTONLY(); change(value)
    with pytest.raises(CachePolicyError): first.checked_required_capacity_bound(value)


@pytest.fixture
def output_entitlement_TESTONLY():
    state, _ = protocol(); task = task_for(state)
    task_binding = pin("task.json"); footprint_binding = pin("R4-footprint.json")
    footprint = dict(format="first-qualification-predeclared-output-entitlement-source-data-not-profile-v1",
        authority_id="Q104-answer-first-qualification-produces-result-space-profile-under-predeclared-entitlement",
        run="ac52", case_count=39, ordered_case_ids=task["ordered_case_ids"], factor=2,
        not_current_measurement_or_profile=True, no_first_qualification_or_adoption=True,
        files=[dict(path="C:/TESTONLY/R4/"+case+"/"+role,size_bytes=17,case_id=case,roles=[role],
            actual_stat=dict(size_bytes=17),declared_original_sha256="a"*64)
            for case in task["ordered_case_ids"]
            for role in ("products.npz","result.json","preview.png","native_statistics.json.gz","geometry")])
    total = sum(row["size_bytes"] for row in footprint["files"])
    footprint.update(unique_output_file_count=len(footprint["files"]),
        actual_result_footprint_bytes=total, administrative_output_entitlement_bytes=2*total)
    value = dict(format="darsia-first-qualification-output-entitlement-v1",
        authority_id=footprint["authority_id"],task=task_binding,
        scope={key:task[key] for key in ("commit","pipeline","precision","run","ordered_case_ids")},
        R4_footprint=footprint_binding,administrative_output_entitlement_bytes=2*total,
        measured_profile=False,enforcement_state="PRELAUNCH_HOLD_N43_REFERENCE_OUTPUT_BUDGET_AND_PROFILE_PRODUCER")
    return task,task_binding,value,footprint


def test_output_entitlement_is_exact_Q104_factor_without_measured_profile_claim(output_entitlement_TESTONLY):
    task,binding,value,footprint=output_entitlement_TESTONLY
    actual=first.checked_output_entitlement(task,value,task_binding=binding,read=lambda _:footprint)
    assert actual==value and actual["measured_profile"] is False
    assert actual["administrative_output_entitlement_bytes"]==2*footprint["actual_result_footprint_bytes"]


@pytest.mark.parametrize("change",[
    lambda v,f:v.update(administrative_output_entitlement_bytes=35),
    lambda v,f:v.update(measured_profile=True),
    lambda v,f:v["scope"].update(commit="d"*40),
    lambda v,f:v.update(task=pin("other-task.json")),
    lambda v,f:f.update(factor=3),
    lambda v,f:f["files"].append(deepcopy(f["files"][0])),
    lambda v,f:f["files"][0]["actual_stat"].update(size_bytes=18),
    lambda v,f:f["files"][0].update(roles=["unrelated"]),
])
def test_output_entitlement_refuses_unbound_case_source_size_or_factor(output_entitlement_TESTONLY,change):
    task,binding,value,footprint=output_entitlement_TESTONLY;change(value,footprint)
    with pytest.raises(CachePolicyError):
        first.checked_output_entitlement(task,value,task_binding=binding,read=lambda _:footprint)


def test_new_bound_mode_refuses_before_consumption_or_any_native_start(monkeypatch):
    state,_=protocol();state.bound_mode=True;lease=lease_for(state)
    monkeypatch.setattr(first,"_outputs_absent",lambda _:pytest.fail("No output/namespace access before HOLD"))
    with pytest.raises(CachePolicyError,match="N43 PRELAUNCH HOLD"):
        state.consume(lease)
    assert not lease.ledger.documents and state.started is False


def test_Q106_reserve_samples_are_one_second_separate_from_full_stat_and_latch_STOP(monkeypatch):
    state,_=protocol();state.bound_mode=True;state.required_capacity_bound=bound_plan_TESTONLY()
    now=[0.0];collector=first.FiniteObservations(state,clock=lambda:now[0])
    lease=lease_for(state);lease.policy.root="C:/TESTONLY/cache";scans=[];free_samples=[];stops=[]
    lease.measure=lambda root:(scans.append(now[0]) or dict(logical_bytes=7))
    free_values=iter([PHYSICAL_RESERVE_BYTES+1,PHYSICAL_RESERVE_BYTES,PHYSICAL_RESERVE_BYTES+2])
    def volume(root):
        free_samples.append(now[0]);return dict(free_bytes=next(free_values),volume_id="TESTONLY")
    monkeypatch.setattr("darsia_cache.capacity.volume_info",volume)
    monkeypatch.setattr(first,"request_existing_frame_boundary_stop",
        lambda f,l:(stops.append((f,l)) or dict(requested=False,state="TESTONLY_UNSUPPORTED")))
    owned=SimpleNamespace(lease=lease)
    for instant in [0,.2,.8,1,1.2,2]:
        now[0]=instant;collector.observe(owned,dict(active_processes=1))
    assert scans==[0] and free_samples==[0,1,2]
    assert collector.native_poll_count==6 and len(stops)==1
    assert collector.reserve_crossed is True
    assert collector.reserve_samples[1]["at_or_below_operational_reserve"] is True
    assert collector.reserve_samples[2]["at_or_below_operational_reserve"] is False
    assert collector.reserve_samples[1]["actual_gap_since_previous_start_seconds"]==1
    assert collector.stop_response["state"]=="TESTONLY_UNSUPPORTED"


def test_Q106_existing_stop_uses_actual_master_and_preserves_current_command_CAS(monkeypatch):
    from darsia.fluidflower import job_control
    state,_=protocol();lease=lease_for(state);lease.owner["machine"]="TESTONLY-Main"
    state.request["public_command"] += ["--job","C:/TESTONLY/TESTONLY-job"]
    monkeypatch.setattr(job_control,"read_json",lambda path:dict(commit=state.task["commit"]))
    previous=pin("previous.json")
    monkeypatch.setattr(job_control,"read_control",lambda root:dict(action="run",sequence=1,command=previous))
    calls=[]
    def publish(root,action,**kwargs):
        calls.append((root,action,kwargs));return pin("stop.json")
    monkeypatch.setattr(job_control,"publish_control",publish)
    actual=first.request_existing_frame_boundary_stop(state,lease)
    assert actual["requested"] is True
    assert calls==[("C:/TESTONLY/TESTONLY-job","stop",dict(machine="TESTONLY-Main",expected_previous=previous))]


def test_Q106_foreign_master_or_failed_stop_records_STOP_without_retry_or_fallback(monkeypatch):
    from darsia.fluidflower import job_control
    state,_=protocol();lease=lease_for(state);lease.owner["machine"]="TESTONLY-Olav"
    state.request["public_command"] += ["--job","C:/TESTONLY/TESTONLY-job"]
    monkeypatch.setattr(job_control,"read_json",lambda path:dict(commit=state.task["commit"]))
    monkeypatch.setattr(job_control,"read_control",lambda root:dict(action="run",sequence=0))
    calls=[]
    def refusal(*args,**kwargs):
        calls.append(kwargs);raise ValueError("Only the job's recorded master machine writes commands")
    monkeypatch.setattr(job_control,"publish_control",refusal)
    actual=first.request_existing_frame_boundary_stop(state,lease)
    assert actual["requested"] is False and actual["error_type"]=="ValueError"
    assert actual["state"]=="STOP_COMMAND_NOT_PUBLISHED"
    assert calls==[dict(machine="TESTONLY-Olav",expected_previous=None)]


def test_empty_python_module_is_exact_source_not_empty_metadata():
    from darsia_cache.policy import _descriptor
    from darsia_cache.worker_adapter import _source_pins
    empty = dict(path="C:/TESTONLY/__init__.py", size_bytes=0,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    value = provider()
    value["producer_source_closure"].append(empty)
    value["module_closure"].append(empty)
    assert first.checked_provider(value) == value
    assert _source_pins([empty]) == {'c:\\testonly\\__init__.py'}
    with pytest.raises(CachePolicyError):
        _descriptor(empty)
    for role in ("namespace_admission", "python_executable", "startup"):
        invalid = deepcopy(value)
        invalid[role] = empty
        with pytest.raises(CachePolicyError):
            first.checked_provider(invalid)


@pytest.mark.parametrize("changes", [
    dict(sha256="a"*64), dict(size_bytes=-1), dict(size_bytes=False),
    dict(size_bytes=16*1024**2+1), dict(path="C:/TESTONLY/empty.json"),
    dict(path="C:/TESTONLY/empty.pyd"), dict(path="C:/TESTONLY/python.exe"),
    dict(path=""), dict(size_bytes=0.0), dict(extra="not source identity"),
])
def test_empty_source_exception_keeps_exact_sha_type_suffix_size_and_fields(changes):
    from darsia_cache.worker_adapter import _source_pins
    value = dict(path="C:/TESTONLY/__init__.py", size_bytes=0,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    value.update(changes)
    with pytest.raises(CachePolicyError):
        first._pins([value])
    with pytest.raises(CachePolicyError):
        _source_pins([value])


def test_module_source_descriptor_keeps_positive_executable_and_16MiB_boundary():
    from darsia_cache.policy import _descriptor, _module_source_descriptor
    for size in (1, 16*1024**2):
        value = dict(path="C:/TESTONLY/python.exe", size_bytes=size, sha256="a"*64)
        assert _module_source_descriptor(value) == _descriptor(value)
    for missing in ("path", "sha256", "size_bytes"):
        value = dict(path="C:/TESTONLY/__init__.py", size_bytes=0,
            sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        del value[missing]
        with pytest.raises(CachePolicyError):
            first._pins([value])
