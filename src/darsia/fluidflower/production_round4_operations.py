"""Main operational bridge for the locked round-four whole-run calculation.

Only Root calls coordinate. The native entrypoint claims its real incarnation
before numerical imports and waits for Root's exact release. Scientific work
is delegated to production_round4.run_selected_worker without relabelling any
old B1/127 test, bypassing any native guard, or starting a shared NAS queue.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time


FORMAT = "ffac-round4-main-operational-packet-v1"
SPEC_SHA256 = "bf69e427d39e4b351dbdeb2f4e3067a6724f51f6b4cc84587ba41e959e3d9143"
AMENDMENT_SHA256 = "f1567b3f8724b41c6401e8d72d139c45910cd4a6aeba09dba75436da2a0882d0"
GIB = 1024**3
PRIORITY = ("ac19", "ac28", "ac47", "ac52", "ac56")
OVERNIGHT_END = datetime(2026,10,13,5,tzinfo=timezone.utc)
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def _bound(value):
    if (not isinstance(value, dict) or set(value) != {"path","sha256","size_bytes"}
            or not Path(value.get("path", "")).is_absolute()
            or re.fullmatch("[0-9a-f]{64}", value.get("sha256", "")) is None
            or type(value.get("size_bytes")) is not int or value["size_bytes"] < 0):
        raise ValueError("An exact absolute path/SHA/integer-size descriptor is required")
    return value


def _read_bound(value, *, limit=512*1024**2):
    value = _bound(value)
    if value["sha256"] == "4491b8ea239853d441ece789a371e92f6516874b6ea62308eeafb057981d938c":
        raise ValueError("The opaque phase PASS result must never be JSON-parsed")
    if value["size_bytes"] > limit: raise ValueError("Operational metadata exceeds its bounded read")
    path = Path(value["path"])
    if path.stat().st_size != value["size_bytes"]: raise ValueError("Bound metadata size changed")
    with path.open("rb") as stream: data = stream.read(value["size_bytes"]+1)
    if len(data) != value["size_bytes"] or hashlib.sha256(data).hexdigest() != value["sha256"]:
        raise ValueError("Bound metadata bytes changed")
    def pairs(rows):
        value={}
        for key,item in rows:
            if key in value: raise ValueError("Duplicate operational metadata key")
            value[key]=item
        return value
    def nonfinite(value): raise ValueError("Nonfinite operational metadata")
    return json.loads(data.decode("utf-8-sig"),object_pairs_hook=pairs,parse_constant=nonfinite)


def _load(name, descriptor):
    bound = _bound(descriptor)
    path = Path(bound["path"])
    if bound["size_bytes"] > 8*1024**2 or path.stat().st_size != bound["size_bytes"]:
        raise ValueError("Bound source size changed or exceeds 8MiB: "+name)
    with path.open("rb") as stream: source = stream.read(bound["size_bytes"]+1)
    if len(source) != bound["size_bytes"] or hashlib.sha256(source).hexdigest() != bound["sha256"]:
        raise ValueError("Operational/scientific source changed: "+name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec);sys.modules[name] = module
    exec(compile(source,str(path),"exec"),module.__dict__)
    return module


def _bootstrap(packet):
    # Reuse only the original stdlib loader/status/mutex/thread facts. No old
    # scientific packet validator, B1 scope, guard or calibration is called.
    common = _load("_round4_common", packet["sources"]["common"])
    baseline = common.load_baseline(packet["sources"])
    return common, baseline


def _get(document, keys):
    value = document
    for key in keys.split("."): value = value[key]
    return value


def validate_packet(packet, *, io, normalize):
    if (packet.get("format") != FORMAT or packet.get("host") != "Main"
            or packet.get("stage") not in {"extraction","heldout","loro","production","provisional"}
            or packet.get("reserve_bytes") != 40*GIB or packet.get("fence_refresh_seconds") != 10
            or packet.get("fence_max_age_seconds") != 30):
        raise ValueError("Exact Main round4 scope and fixed40GiB/10s/30s control rules required")
    if (packet["specs"]["v4"]["sha256"] != SPEC_SHA256
            or packet["specs"]["urest_amendment"]["sha256"] != AMENDMENT_SHA256):
        raise ValueError("Both actual round4 specification bindings are required")
    authority=packet.get("overnight_authority",{})
    if (authority.get("delegated") is not True or authority.get("scope") != "round4_v4_amendment1"
            or authority.get("spec_sha256") != SPEC_SHA256 or authority.get("amendment_sha256") != AMENDMENT_SHA256
            or datetime.fromisoformat(authority.get("valid_until_utc","")) != OVERNIGHT_END):
        raise PermissionError("Explicit actual delegated overnight scope/specification/deadline required")
    for value in [packet["code_lock"],packet["python"],*packet["specs"].values(),*packet["sources"].values()]:
        io.verify(_bound(value))
    lock = io.read_json(packet["code_lock"], max_bytes=packet["code_lock"]["size_bytes"])
    if (lock.get("complete") is not True or lock.get("round4_spec_sha256") != SPEC_SHA256
            or lock.get("urest_amendment_sha256") != AMENDMENT_SHA256
            or lock.get("precision") not in {"float32","float64"}
            or re.fullmatch("[0-9a-f]{40}", lock.get("commit", "")) is None):
        raise ValueError("A completed actual PartA lock, with amendment, is required")
    assignment = io.read_json(packet["assignment"], max_bytes=packet["assignment"]["size_bytes"])
    if (assignment.get("format") != "darsia-production-static-whole-run-assignment-v1"
            or assignment.get("assignment_only") is not True
            or assignment.get("production_permission_granted") is not False):
        raise ValueError("Original static whole-AC assignment is required")
    rows = {row["run"]:row for row in assignment["assignments"]}
    if len(rows) != len(assignment["assignments"]): raise ValueError("A whole AC cannot have two owners")
    slots = {}
    for run,task in packet["tasks"].items():
        assigned, settings = rows[run], task["settings"]
        budget = task["worker_budget_bytes"]
        if (task["run"] != run or task["stage"] != packet["stage"] or assigned["host"] != "Main"
                or task["worker"] != assigned["worker"] or settings["host"] != "Main"
                or settings["precision"] != lock["precision"] or settings["backend"] not in {"cpu","cuda"}
                or type(settings["threads"]) is not int or settings["threads"] < 1
                or type(budget) is not int or budget < 20*GIB):
            raise ValueError("Stage/settings/budget differ from the locked static whole-run owner")
        key = "Main/"+settings["backend"]+"/"+settings["precision"]
        if lock.get("qualified_resources", {}).get(key, {}).get("passed") is not True:
            raise ValueError("This actual host/backend/precision did not pass PartA")
        if settings["backend"] == "cuda" and lock["precision"] != "float32":
            raise ValueError("Round4 GPU use requires the passed common float32 choice")
        signature = (settings["backend"],settings["precision"],settings["threads"],budget)
        if task["worker"] in slots and slots[task["worker"]] != signature:
            raise ValueError("A static slot cannot silently change resource or precision")
        slots[task["worker"]] = signature
        for name in ("manifest","configuration"): _bound(task[name])
        manifest = io.read_json(task["manifest"],max_bytes=task["manifest"]["size_bytes"])
        purpose = {"extraction":"calibration","heldout":"validation","loro":"validation",
                   "production":"production","provisional":"validation"}[packet["stage"]]
        if manifest.get("purpose") != purpose or manifest.get("settings") != settings:
            raise ValueError("Manifest purpose/settings must truthfully identify this round4 stage")
        if manifest.get("code",{}).get("commit") != lock["commit"]:
            raise ValueError("Stage manifest execution commit differs from the actual lock")
        if packet["stage"] == "provisional" and run not in PRIORITY:
            raise ValueError("Provisional computation is restricted to the five authorized runs")
        if packet["stage"] == "provisional" and (task.get("failed_check") != packet.get("failed_check")
                or task.get("provisional_delegation") != packet.get("provisional_delegation")):
            raise ValueError("Provisional task must retain the actual failed-check delegation")
    if not slots: raise ValueError("No actual whole-run tasks are assigned")
    if sum(value[2] for value in slots.values()) > packet["logical_cpus"]:
        raise ValueError("Static CPU/GPU host threads exceed actual logical CPUs")
    if packet["stage"] == "production":
        score = io.read_json(_bound(packet["computed_checks"]), max_bytes=packet["computed_checks"]["size_bytes"])
        if (score.get("complete") is not True or score.get("all_checks_passed") is not True
                or score.get("round4_spec_sha256") != SPEC_SHA256
                or score.get("urest_amendment_sha256") != AMENDMENT_SHA256):
            raise PermissionError("Actual computed checks1–3 must pass under the amendment before production")
    if packet["stage"] == "provisional":
        io.verify(_bound(packet["provisional_delegation"]))
        if not packet.get("failed_check"): raise ValueError("The actual failed-check name is required")
    for name in ("operations_root","stream_root","cache_root","coordinator_fence"):
        path = Path(packet[name])
        if not path.is_absolute() or normalize(path).startswith("\\\\"):
            raise ValueError("Owner/control/staging paths must be local and absolute")
    fence = Path(packet["coordinator_fence"]).resolve()
    if not fence.is_relative_to(Path(packet["operations_root"]).resolve()):
        raise ValueError("The coordinator fence must belong to this exclusive local owner")
    return lock, assignment, slots


def _runtime(packet,lock,jobs):
    if jobs.code_identity(packet["worktree"])["commit"] != lock["commit"]:
        raise ValueError("Actual clean worktree differs from the PartA lock")
    if jobs.environment_lock() != lock["environments_by_host"]["Main"]:
        raise ValueError("Actual environment differs from its completed lock")


def _current_guard(packet, *, common, baseline, identity, task=None, coordinator=False):
    import psutil
    jobs, io = baseline["production_job"], baseline["production_io"].NasIO()
    # OVERNIGHT_END bounds NEW Claude decisions. The direct Olav specification
    # authorizes these fixed actions conditionally on the live guards below;
    # an already authorized batch does not expire at the films target time.
    # Root's required current guard must not grant new post-deadline decisions.
    if jobs.worker_identity() != identity: raise RuntimeError("Actual PID/start identity changed")
    status = {name:common.status_data(path) for name,path in packet["status_paths"].items()}
    claude = status["claude"]
    if (claude["label_check"]["state"] != "PASS"
            or claude["label_check"]["manifest_sha256"] != packet["phase_pass_sha256"]
            or any(any(row.get(key) == "CONCERN" for key in ("review","verdict","state"))
                   for row in claude.get("technical_fix_review", []))):
        raise RuntimeError("Original phase PASS/current CONCERN guard denies round4")
    for name, expectations in packet["live_expectations"].items():
        for key,value in expectations.items():
            actual=_get(status[name],key)
            if isinstance(value,list):
                if (not value or any(v in {"STOP","CONCERN","FAIL","FAILED","DENIED"} for v in value)
                        or actual not in value):
                    raise RuntimeError("Actual Root control state denies current stage: "+key)
            elif actual != value: raise RuntimeError("Actual Root live authority changed: "+key)
    if psutil.virtual_memory().available <= 40*GIB: raise RuntimeError("Preserved40GiB reserve unavailable")
    if task is not None:
        process = psutil.Process()
        rss = sum(p.memory_info().rss for p in [process,*process.children(recursive=True)] if p.is_running())
        if max(rss,process.memory_info().peak_wset) > task["worker_budget_bytes"]:
            raise RuntimeError("Actual owned process memory exceeds the frozen worker budget")
    for descriptor in packet["sources"].values(): io.verify(descriptor)
    io.verify(packet["code_lock"])
    if not coordinator:
        coordinator_sha = packet.get("coordinator_packet_sha256",packet["packet_sha256"])
        with common.local_fence_mutex(coordinator_sha): fence = io.read_json(packet["coordinator_fence"])
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(fence["updated_utc"])).total_seconds()
        if (not 0 <= age <= 30 or fence.get("continue") is not True
                or fence.get("packet_sha256") != coordinator_sha
                or jobs.worker_identity(fence["owner"]["pid"]) != fence["owner"]):
            raise RuntimeError("Fresh actual coordinator identity/permission absent")


def qualify_native(value, *, task, process, wrapper, argv, packet_bound, packet, jobs, psutil, normalize):
    identity = value.get("actual_native_worker", {})
    if (value.get("contract") != "ffac-round4-native-ready-v1" or value.get("complete") is not True
            or value.get("packet") != packet_bound or value.get("run") != task["run"]
            or value.get("stage") != task["stage"] or value.get("sources") != packet["sources"]
            or value.get("scientific_arrays_read") is not False or value.get("numerical_imports_started") is not False
            or type(identity.get("pid")) is not int or not isinstance(identity.get("process_create_time"),str)
            or jobs.worker_identity(process.pid) != wrapper or jobs.worker_identity(identity["pid"]) != identity):
        raise ValueError("Exact pre-array native identity/source/stage handshake differs")
    native = psutil.Process(identity["pid"])
    if identity["pid"] == process.pid or not any(parent.pid == process.pid for parent in native.parents()[:3]):
        raise ValueError("Actual native is not this launched Windows wrapper descendant")
    observed = native.cmdline()
    paths = {3,argv.index("--packet")+1}
    if len(observed) != len(argv) or any(
            normalize(observed[i]) != normalize(argv[i]) if i in paths else observed[i] != argv[i]
            for i in range(1,len(argv))):
        raise ValueError("Actual native argv differs from the locked whole-run invocation")
    return identity


def _dependencies(packet, settings):
    modules = {key:_load("_r4_"+key,packet["sources"][key]) for key in
               ("round4","precision_adapter","kernel","sandmap","products","renderer","lifecycle","stream_io")}
    def provider_factory(value):
        if value["backend"] == "cpu": return modules["kernel"].PixelMath(value["precision"])
        if value["backend"] == "cuda":
            provider = _load("_r4_cuda",packet["sources"]["cuda"])
            return provider.CudaHostPixelMath(value["precision"],device_id=value["device"]["id"],
                                              pool_limit_bytes=value["device"]["pool_limit_bytes"])
        raise ValueError("This bridge only executes qualified Main CPU/CUDA resources")
    def prefetch_plan(context,reader,case_id):
        phase = reader.json(context.cases[case_id]["entry"]["receipt"])
        descriptor,plan = phase["full_extended_scientific"],{}
        if "native_centre" in descriptor:
            plan.update(centre=descriptor["native_centre"]["log_linear_rgb"],centre_valid=descriptor["native_centre"]["valid_mask"])
        if Path(phase["labels_map"]["path"]).suffix.lower() == ".npy": plan["labels"] = phase["labels_map"]
        if descriptor.get("ring_windows"): plan["ring"] = descriptor["ring_windows"][0]["log_linear_rgb"]
        return plan
    modules.update(provider_factory=provider_factory,prefetch_plan=prefetch_plan)
    return modules


def make_staged_reader(task, packet, lock, *, baseline, dependencies):
    """Exact staged reader for a released worker or Root's guarded B1 measurement.

    This performs no frame loading, optical calculation, calibration or release.
    Original logical source provenance and actual executed-source pairs stay
    separate; the caller supplies the actual locked manifest and copy evidence.
    """
    io=baseline["production_io"].NasIO()
    cls=dependencies["round4"].make_reader_class(baseline["production_inputs"].ManifestAssetReader,
                                               dependencies["stream_io"],packet["code_lock"],io=io)
    manifest=io.read_json(task["manifest"],max_bytes=task["manifest"]["size_bytes"])
    return cls(manifest,task["staged_inputs"],cache_root=packet["cache_root"],worktree=packet["worktree"],
        source_provenance=lock["provenance_by_host"]["Main"],source_qualification={"round4_lock":packet["code_lock"]},
        logical_aliases=task["logical_aliases"],opaque_descriptors=task["opaque_descriptors"],
        verified_staging_receipt=task["verified_stage"],array_contracts=task["array_contracts"],io=io)


def build_phase(stage, job_root, lock_descriptor, capacity, assignment_descriptor, sourcepins,
                physical_correction, *, selected_inventory, template_packet, specs,
                freeze=None, folds=None, statistics=None, computed_checks=None,
                provisional_delegation=None, failed_check=None, before_copy, _measurement_scope=False):
    """Root explicitly builds essential phase manifests and the existing copier.

    capacity supplies Main worktree/python/logical_cpus/threads/budget/cache_root,
    nas_root, status_paths, live_expectations and phase_pass_sha256. Assignment is
    the already-fixed whole-AC assignment; stage filtering preserves its order
    and owners. physical_correction is the bound B1 decision JSON with an explicit
    correction_matrix. No PartA proof, source review, freeze or GO is invented.
    Return packet descriptor, host_stream and prepare_task for coordinate(...).
    """
    if not callable(before_copy): raise PermissionError("Root's fresh copy/control guard is required")
    if stage not in {"extraction","heldout","loro","production","provisional"}:
        raise ValueError("Unknown round4 phase")
    if stage != "extraction" and freeze is None:
        raise ValueError("Round4 validation/products require the actual newly frozen calibration")
    if stage == "loro" and folds is None: raise ValueError("All actual omitted-run fold descriptors are required")
    common,baseline=_bootstrap(dict(sources=sourcepins))
    io_module=baseline["production_io"];io=io_module.NasIO();jobs=baseline["production_job"]
    normalize=io_module.normalize_locator
    root=Path(job_root).resolve()
    if not root.is_absolute() or normalize(root).startswith("\\\\"):
        raise ValueError("A fresh owned local phase root is required")
    io.verify(lock_descriptor);lock=io.read_json(lock_descriptor,max_bytes=lock_descriptor["size_bytes"])
    _runtime(capacity,lock,jobs)
    inventory=io.read_json(selected_inventory,max_bytes=selected_inventory["size_bytes"])
    old=io.read_json(template_packet,max_bytes=template_packet["size_bytes"])
    full_assignment=io.read_json(assignment_descriptor,max_bytes=assignment_descriptor["size_bytes"])
    if _measurement_scope and stage != "extraction":raise ValueError("Boundary inputs are calibration-only")
    correction=None if _measurement_scope else io.read_json(physical_correction,max_bytes=physical_correction["size_bytes"])
    if (inventory.get("round4_spec_sha256") != SPEC_SHA256 or inventory.get("complete") is not True
            or inventory.get("scientific_arrays_read") is not False or len(inventory["runs"]) != 39
            or not _measurement_scope and "correction_matrix" not in correction):
        raise ValueError("Actual unchanged 39-run metadata and explicit B1 physical decision required")
    aliases=inventory["logical_aliases"]
    alias_targets={normalize(key):_bound(value) for key,value in aliases.items()}
    def canonical(bound):
        target=alias_targets.get(normalize(bound["path"]),bound)
        if (bound["sha256"],bound["size_bytes"]) != (target["sha256"],target["size_bytes"]):
            raise ValueError("Explicit logical alias changes frozen bytes")
        return dict(target)
    # Reuse ONLY original SHA-bound copy rows whose current local identity still
    # equals the coordinator-captured identity. No global payload rehash.
    streams=_load("_round4_stream_builder",sourcepins["stream_io"])
    lifecycle=_load("_round4_lifecycle_builder",sourcepins["lifecycle"])
    science=_load("_round4_science_builder",sourcepins["round4"])
    reusable={}
    for row in old["runs"].values():
        previous=row["float64"]
        receipt=io.read_json(previous["verified_stage"],max_bytes=previous["verified_stage"]["size_bytes"])
        if (receipt.get("complete") is not True or receipt.get("payload_sha_verified_by_coordinator") is not True
                or receipt.get("arrays_decoded") is not False or receipt.get("manifest") != previous["manifest"]):
            raise ValueError("Original common-copy evidence differs")
        for item in receipt["inputs"].values():
            key=normalize(item["logical"]["path"])
            if key in reusable and reusable[key]["logical"] != item["logical"]:
                if (reusable[key]["logical"]["sha256"],reusable[key]["logical"]["size_bytes"]) != (item["logical"]["sha256"],item["logical"]["size_bytes"]):
                    raise ValueError("Conflicting original copied logical asset")
            reusable[key]=item
    purpose={"extraction":"calibration","heldout":"validation","loro":"validation",
             "production":"production","provisional":"validation"}[stage]
    eligible={run for run,row in inventory["runs"].items() if
        (row["calibration"] if stage in {"extraction","loro"} else
         not row["calibration"] and not row["descriptive"] if stage == "heldout" else
         run in PRIORITY if stage == "provisional" else True)}
    if _measurement_scope:
        if inventory.get("boundary_measurement_only") is not True:
            raise ValueError("Explicit two FIT-baseline metadata scope required")
        eligible={"ac28","ac39"}
    assigned=[row for row in full_assignment["assignments"] if row["host"] == "Main" and row["run"] in eligible]
    if stage in {"production","provisional"}:
        assigned=sorted(assigned,key=lambda row:PRIORITY.index(row["run"]) if row["run"] in PRIORITY else len(PRIORITY))
    if not assigned or len({row["run"] for row in assigned}) != len(assigned):
        raise ValueError("Static phase assignment is empty or duplicates an owner")
    phase_assignment=dict(full_assignment,assignments=assigned,round4_stage=stage,
                          original_assignment=assignment_descriptor)
    root.mkdir(parents=True,exist_ok=False)
    shared=root/"shared_inputs";shared.mkdir()
    shared_copies,identities={},{}
    def stage_common(bound):
        bound=canonical(_bound(bound));key=normalize(bound["path"])
        if key in shared_copies: return shared_copies[key]
        before_copy();item=reusable.get(key)
        if (item is not None and (item["logical"]["sha256"],item["logical"]["size_bytes"]) ==
                (bound["sha256"],bound["size_bytes"])):
            local=dict(item["local"])
            if streams.local_identity(local["path"]) != item["local_identity"]:
                raise ValueError("Previously SHA-verified shared copy changed; cannot silently requalify it")
        else:
            locator=hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
            target=shared/bound["sha256"]/locator/Path(bound["path"]).name
            target.parent.mkdir(parents=True,exist_ok=True)
            local=io.copy_file(bound,target,before_attempt=before_copy)
        if not Path(local["path"]).resolve().is_relative_to(Path(capacity["cache_root"]).resolve()):
            raise ValueError("Shared input copy escaped the declared local cache")
        shared_copies[key]=local;identities[normalize(local["path"])]=streams.local_identity(local["path"])
        return local
    common_bounds={normalize(canonical(inventory["assets"][i]["descriptor"])["path"]):
                   canonical(inventory["assets"][i]["descriptor"]) for i in inventory["common_roles"]}
    # All canonical alias targets are explicit old frozen declarations, including
    # the unused old selection target required by the unchanged reader ctor.
    for descriptor in alias_targets.values(): common_bounds[normalize(descriptor["path"])]=descriptor
    for descriptor in [lock_descriptor,*specs.values(),old["boxes"],*sourcepins.values(),
                       *([] if physical_correction is None else [physical_correction])]:
        common_bounds[normalize(descriptor["path"])]=descriptor
    if freeze is not None:
        freeze_value=io.read_json(freeze,max_bytes=freeze["size_bytes"])
        if (freeze_value.get("round4_spec_sha256") != SPEC_SHA256
                or freeze_value.get("amendment_sha256") != AMENDMENT_SHA256 or freeze_value.get("frozen") is not True):
            raise ValueError("Only the actual amended round4 freeze can validate or produce products")
        common_bounds[normalize(freeze["path"])]=freeze
        def declared(value):
            if isinstance(value,dict):
                if "path" in value and "sha256" in value:
                    key=normalize(canonical(value)["path"]) if "size_bytes" in value else normalize(value["path"])
                    if key not in common_bounds:
                        asset=io.verified_asset(value);common_bounds[key]=asset
                else:
                    for item in value.values(): declared(item)
            elif isinstance(value,list):
                for item in value: declared(item)
        declared(freeze_value["calibration_files"])
    for descriptor in common_bounds.values(): stage_common(descriptor)
    io.write_json(root/"assignment.json",phase_assignment);phase_bound=io.asset(root/"assignment.json")
    tasks={};manifests={}
    for assigned_row in assigned:
        run=assigned_row["run"];row=inventory["runs"][run]
        inputs={"asset_%05d"%i:canonical(inventory["assets"][i]["descriptor"]) for i in row["input_roles"]}
        for i,descriptor in enumerate(common_bounds.values()):inputs["common_%05d"%i]=descriptor
        # Fixed Root statistics/fold descriptors enter the flat scientific table,
        # never through an unregistered live NAS read.
        for case,descriptor in (statistics or {}).get(run,{}).items(): inputs["statistics_"+case]=_bound(descriptor)
        if stage == "loro": inputs["omitted_run_fold"]=_bound(folds[run])
        config=io.read_json(row["configuration"],max_bytes=row["configuration"]["size_bytes"])
        config=dict(config,purpose=purpose,backend=capacity["backend"],round4_amendment_sha256=AMENDMENT_SHA256,
                    production_permission_granted=stage == "production",
                    actual_R3_freeze_role="original source guard only; not the round4 selected calibration")
        if freeze is not None:config["round4_selected_calibration"]=freeze
        path=root/run/"configuration.json";io.write_json(path,config);configuration=io.asset(path)
        inputs["round4_configuration"]=configuration;stage_common(configuration)
        unique,seen={},{}
        for role,descriptor in inputs.items():
            key=normalize(descriptor["path"])
            if key in seen:
                if (seen[key]["sha256"],seen[key]["size_bytes"]) != (descriptor["sha256"],descriptor["size_bytes"]):
                    raise ValueError("Canonical flat input locator has contradictory frozen bytes")
                continue
            seen[key]=descriptor;unique[role]=descriptor
        inputs=unique
        settings=dict(host="Main",precision=lock["precision"],backend=capacity["backend"],threads=capacity["threads"],
                      workers=len({value["worker"] for value in assigned}),logical_cpus=capacity["logical_cpus"])
        if settings["backend"] != "cpu":settings["device"]=capacity["device"]
        assurance={role for role,value in inputs.items() if normalize(value["path"]) in
                   {normalize(v["path"]) for v in [lock_descriptor,*sourcepins.values(),*specs.values()]}}
        frames=row["case_ids"]
        if stage == "loro":
            selected=io.read_json(row["selection"],max_bytes=row["selection"]["size_bytes"])["cases"]
            frames=[case for case in frames if float(selected[case]["entry"]["selected_target_h"]) in (56.,64.,72.,96.)]
        manifest=dict(format=jobs.MANIFEST_FORMAT,job_id=root.name+"_"+run,created_utc=jobs.utc_now(),purpose=purpose,
            code=jobs.code_identity(capacity["worktree"]),environment=lock["environments_by_host"]["Main"],
            settings=settings,calibration=freeze or config["actual_R3_freeze"],specs=specs,inputs=inputs,
            scientific_input_roles=[key for key in inputs if key not in assurance],assurance_input_roles=sorted(assurance),
            frames=frames,
            calibration_role="round4 selected calibration" if freeze is not None else "original R3 source guard only; no round4 calibration selected",
            round4_stage=stage,round4_spec_sha256=SPEC_SHA256,urest_amendment_sha256=AMENDMENT_SHA256,
            boundary_measurement_only=_measurement_scope,whole_run_calculation=not _measurement_scope,
            production_permission_granted=stage == "production")
        jobs.validate_manifest(manifest)
        manifest_path=root/run/"manifest.json";io.write_json(manifest_path,manifest);manifest_bound=io.asset(manifest_path)
        # Existing RunStream uses SHA/basename content paths. Declare duplicate
        # role-to-copy links explicitly so the original copier never overwrites
        # an identical destination. Logical provenance/aliases remain unchanged.
        working,copy_roles,copy_names={},{},{}
        common_contents={(value["sha256"],value["size_bytes"]) for value in common_bounds.values()}
        for role,value in inputs.items():
            if normalize(value["path"]) in shared_copies:continue
            if (value["sha256"],value["size_bytes"]) in common_contents:
                stage_common(value);continue
            target=(value["sha256"],Path(value["path"]).name.casefold())
            first=copy_names.get(target)
            if first is None:
                copy_names[target]=role;working[role]=value;first=role
            elif working[first]["size_bytes"] != value["size_bytes"]:
                raise ValueError("Same content destination has conflicting expected sizes")
            copy_roles[role]=first
        common_staged={role:shared_copies[normalize(value["path"])] for role,value in inputs.items()
                       if normalize(value["path"]) in shared_copies}
        tasks[run]=dict(run=run,stage=stage,worker=assigned_row["worker"],settings=settings,
            worker_budget_bytes=capacity["worker_budget_bytes"],manifest=manifest_bound,configuration=configuration,
            working_inputs=working,working_copy_roles=copy_roles,staged_inputs={},verified_stage=None,array_contracts=[],
            local_results=str(root/"streams"/("worker_"+str(assigned_row["worker"]))/run/"results"),
            nas_results=str(Path(capacity["nas_root"])/stage/run),logical_aliases=aliases,
            opaque_descriptors=inventory["opaque_descriptors"],boxes=old["boxes"],physical_correction=physical_correction,
            correction_matrix=None if _measurement_scope else correction["correction_matrix"],statistics_by_case=(statistics or {}).get(run,{}),
            code_sha256=sourcepins["round4"]["sha256"],solubility_limit_g_l=capacity["solubility_limit_g_l"],
            failed_check=failed_check,provisional_delegation=provisional_delegation)
        if freeze is not None:tasks[run]["freeze"]=freeze
        if stage == "loro":tasks[run]["fold"]=folds[run]
        manifests[run]=(manifest,common_staged)
    packet=dict(format=FORMAT,stage=stage,host="Main",worktree=capacity["worktree"],python=capacity["python"],
        code_lock=lock_descriptor,specs=specs,sources=sourcepins,assignment=phase_bound,tasks=tasks,
        reserve_bytes=40*GIB,fence_refresh_seconds=10,fence_max_age_seconds=30,logical_cpus=capacity["logical_cpus"],
        operations_root=str(root/"operations"),stream_root=str(root/"streams"),cache_root=capacity["cache_root"],
        coordinator_fence=str(root/"operations/control_fence.json"),status_paths=capacity["status_paths"],
        live_expectations=capacity["live_expectations"],phase_pass_sha256=capacity["phase_pass_sha256"],
        overnight_authority=capacity["overnight_authority"],
        computed_checks=computed_checks,provisional_delegation=provisional_delegation,failed_check=failed_check,
        selected_inventory=selected_inventory,template_packet=template_packet,physical_correction=physical_correction)
    packet["boundary_measurement_only"]=_measurement_scope
    validate_packet(packet,io=io,normalize=normalize)
    io.write_json(root/"packet.json",packet);packet_bound=io.asset(root/"packet.json")
    host_stream=lifecycle.HostRunStream(root/"streams",host="Main",assignment=phase_assignment,io=io,
                                      common_bindings=list(common_bounds.values()))
    def prepare_task(run,slot,working_copies,owned_results):
        manifest,common_staged=manifests[run]
        staged=dict(common_staged,**{role:working_copies[first] for role,first in tasks[run]["working_copy_roles"].items()})
        for local in working_copies.values():identities[normalize(local["path"])]=streams.local_identity(local["path"])
        receipt=science.verified_copy_stage(manifest,tasks[run]["manifest"],staged,cache_root=capacity["cache_root"],
                                           copy_identities=identities,stream_module=streams)
        stage_path=root/run/"verified_stage.json";io.write_json(stage_path,receipt);stage_bound=io.asset(stage_path)
        verified=streams.VerifiedStage(stage_bound,manifest,staged,cache_root=capacity["cache_root"],io=io)
        contracts=streams.npy_contracts(verified)
        return dict(tasks[run],staged_inputs=staged,verified_stage=stage_bound,array_contracts=contracts,
                    local_results=str(owned_results))
    def cleanup_go(owner,run,task,result,retired):
        return cleanup_run(owner,run,task,result,workers_retired_guard=retired,boundary_guard=before_copy,io=io)
    return dict(packet=packet_bound,host_stream=host_stream,prepare_task=prepare_task,cleanup_go=cleanup_go,io=io)


def cleanup_run(owner,run,task,result,*,workers_retired_guard,boundary_guard,io):
    """Finish existing NAS publication, then delete only exact retired owned files."""
    if result.get("complete") is not True or result.get("worker_references_closed") is not True:
        raise ValueError("Actual completed worker must have closed every input reference")
    workers_retired_guard();boundary_guard()
    folder=owner.runs[run]["results"]
    if Path(task["local_results"]).resolve() != folder.resolve():
        raise ValueError("Cleanup target differs from the freshly owned results")
    # Align the result archive with LocalProductionJob's existing NAS root:
    # manifest+frames+ops+static. Existing immutable files are only verified;
    # missing closed metadata/static files are copied with no overwrite.
    io.copy_file(task["manifest"],folder/"manifest.json",before_attempt=boundary_guard)
    paths=list(folder.rglob("*"))
    if any(path.is_symlink() or ".partial" in path.name for path in paths):
        raise ValueError("Preserve every partial or substituted owned file")
    inventory={path.relative_to(folder).as_posix():io.asset(path) for path in paths if path.is_file()}
    for relative,descriptor in inventory.items():
        boundary_guard();io.copy_file(descriptor,Path(task["nas_results"])/relative,before_attempt=boundary_guard)
    boundary_guard();io.verify_directory(task["nas_results"],inventory)
    owner.bind_results(run,inventory)
    def authorized(which,paths):
        if which != run:raise ValueError("Cleanup authorization cannot move to another run")
        boundary_guard();return True  # Direct human whole-AC verified cleanup instruction.
    owner.cleanup(run,nas_result_directory=task["nas_results"],user_deletion_go=authorized,
                  workers_retired_guard=workers_retired_guard)
    return dict(run=run,owned_inputs_and_results_removed=True,exact_NAS_results_verified=True,
                original_sources_common_and_partials_untouched=True)


def prepare_boundary_contexts(job_root,lock_descriptor,capacity,assignment_descriptor,sourcepins,*,
                              selected_inventory,template_packet,specs,boundary_guard,actual_root_identity):
    """Root-only AC28/AC39 first-acquisition setup; no physical decision fabricated.

    Replays original full39 admission while collecting only the two actual FIT
    baseline cases, include_anchors=False. Returns context/adapter rows accepted
    by production_round4.measure_boundaries. This explicit Root action performs
    original native initialization after genuine identity/lock/control checks.
    """
    if not callable(boundary_guard):raise PermissionError("Actual Root native boundary guard required")
    common,baseline=_bootstrap(dict(sources=sourcepins))
    io=baseline["production_io"].NasIO();jobs=baseline["production_job"]
    if jobs.worker_identity() != actual_root_identity:raise ValueError("Actual Root measurement incarnation differs")
    boundary_guard()
    lock=io.read_json(lock_descriptor,max_bytes=lock_descriptor["size_bytes"]);_runtime(capacity,lock,jobs)
    # This B1 action comes directly from Olav's bound specification. Root's
    # live guard, not the deadline for new Claude decisions, governs its use.
    sys.path.insert(0,str(Path(capacity["worktree"])/"src"))
    from darsia.fluidflower import production_native,production_retained
    selection=_load("_r4_boundary_metadata",sourcepins["selection"])
    science=_load("_r4_boundary_science",sourcepins["round4"])
    original=io.read_json(selected_inventory,max_bytes=selected_inventory["size_bytes"])
    old=io.read_json(template_packet,max_bytes=template_packet["size_bytes"])
    prior=old["runs"]["ac19"]["float64"]
    old_manifest=io.read_json(prior["manifest"],max_bytes=prior["manifest"]["size_bytes"])
    meta=selection.MetadataReader(old_manifest,prior["staged_inputs"],old["logical_aliases"],io=io)
    root=Path(job_root);root.mkdir(parents=True,exist_ok=False)
    assets=list(original["assets"]);runs=dict(original["runs"])
    indexes={baseline["production_io"].normalize_locator(row["descriptor"]["path"]):i for i,row in enumerate(assets)}
    modules=production_native._modules()
    for run in ("ac28","ac39"):
        selected=io.read_json(runs[run]["selection"],max_bytes=runs[run]["selection"]["size_bytes"])["cases"]
        case=run+"_f1"
        if case != runs[run]["case_ids"][0] or selected[case]["entry"]["run"] not in modules["engine"].FIT_RUNS:
            raise ValueError("Only exact first indexed FIT acquisitions can supply boundary measurements")
        cases={case:selected[case]}
        path=root/run/"baseline_selection.json"
        io.write_json(path,dict(round4_spec_sha256=SPEC_SHA256,run=run,cases=cases,boundary_measurement_only=True))
        scope=io.asset(path);meta.register(scope)
        config=io.read_json(runs[run]["configuration"],max_bytes=runs[run]["configuration"]["size_bytes"])
        config=dict(config,profile_selection=scope,purpose="calibration",backend="cpu")
        path=root/run/"baseline_configuration.json";io.write_json(path,config);config_bound=io.asset(path);meta.register(config_bound)
        def exact_config(value,expected=config):
            if value != expected:raise ValueError("Boundary configuration changed")
            return value
        def exact_cases(value,expected=cases,selection_bound=scope):
            if value != meta.json(selection_bound):raise ValueError("Boundary indexed scope changed")
            return expected
        collector=science.clone(production_native.enumerate_native_closure,_configuration=exact_config,_cases=exact_cases,
                                LOCKED_CASES=cases)
        closure=collector(config,meta,modules=modules,size_resolver=meta.size,include_anchors=False)
        if (closure["scientific_arrays_read"] is not False or closure["case_ids"] != [case]
                or closure["includes_anchor_payloads"] is not False):
            raise ValueError("Exact one-baseline/no-anchor metadata closure required")
        required=list(original["common_roles"])
        for member in closure["assets"]+[dict(descriptor=config_bound,roles=["boundary_configuration"])]:
            key=baseline["production_io"].normalize_locator(member["descriptor"]["path"])
            if key not in indexes:
                indexes[key]=len(assets);assets.append(dict(member,working_runs=[run]))
            elif (assets[indexes[key]]["descriptor"]["sha256"],assets[indexes[key]]["descriptor"]["size_bytes"]) != (member["descriptor"]["sha256"],member["descriptor"]["size_bytes"]):
                raise ValueError("Boundary closure contradicts original frozen source bytes")
            required.append(indexes[key])
        runs[run]=dict(runs[run],selection=scope,configuration=config_bound,case_ids=[case],frame_count=1,input_roles=sorted(set(required)))
    document=dict(original,assets=assets,runs=runs,boundary_measurement_only=True)
    path=root/"boundary_input_inventory.json";io.write_json(path,document);inventory_bound=io.asset(path)
    # Two Root-owned photo measurements are not whole-AC production ownership.
    # Keep the real global assignment bound and unchanged even if those AC runs
    # will execute on Olav; never silently omit a required camera group.
    global_assignment=io.read_json(assignment_descriptor,max_bytes=assignment_descriptor["size_bytes"])
    if not {"ac28","ac39"} <= {row["run"] for row in global_assignment["assignments"]}:
        raise ValueError("Both actual FIT runs must exist in the original global assignment")
    measurement_assignment=dict(global_assignment,
        assignments=[dict(run=run,host="Main",worker=0) for run in ("ac28","ac39")],
        boundary_measurement_only=True,whole_AC_ownership_reassigned=False,original_assignment=assignment_descriptor)
    path=root/"boundary_photo_assignment.json";io.write_json(path,measurement_assignment);measurement_bound=io.asset(path)
    built=build_phase("extraction",root/"staged",lock_descriptor,capacity,measurement_bound,sourcepins,None,
        selected_inventory=inventory_bound,template_packet=template_packet,specs=specs,before_copy=boundary_guard,
        _measurement_scope=True)
    packet=io.read_json(built["packet"],max_bytes=built["packet"]["size_bytes"])
    deps=_dependencies(packet,dict(backend="cpu"))
    rows=[];providers=[]
    for run in ("ac28","ac39"):
        task=packet["tasks"][run];slot=task["worker"]
        # Measurements are sequential, including when the fixed assignment
        # puts both camera groups on one slot. Root remains the active consumer;
        # no false retirement/cleanup is claimed here.
        owner=built["host_stream"].workers[slot]
        copies=owner._stage(run,task["working_inputs"],boundary_guard)
        task=built["prepare_task"](run,slot,copies,owner.runs[run]["results"])
        reader=make_staged_reader(task,packet,lock,baseline=baseline,dependencies=deps)
        config=reader.json(task["configuration"]);cases=reader.json(config["profile_selection"])["cases"]
        with reader.frame_inputs(boundary_guard=boundary_guard):
            context=science.prepare_context(config,reader,cases,native_module=production_native,boundary_guard=boundary_guard)
        context.provenance.update(whole_run=False,boundary_measurement_only=True,
                                  whole_AC_ownership_reassigned=False)
        provider=deps["kernel"].PixelMath(lock["precision"]);providers.append(provider)
        adapter=deps["precision_adapter"].PrecisionAdapter(context,lock["precision"],pixel_math=provider,retained_module=production_retained)
        rows.append(dict(run=run,case_id=run+"_f1",context=context,adapter=adapter,task=task))
    def close():
        for value in providers:
            if callable(getattr(value,"close",None)):value.close()
        built["host_stream"].close()
    return dict(sources=rows,close=close,packet=built["packet"],actual_root_identity=actual_root_identity,
                boundary_measurement_only=True,whole_run_calculation=False)


def worker(packet_bound, run):
    packet = _read_bound(packet_bound);packet["packet_sha256"] = packet_bound["sha256"]
    if packet.get("boundary_measurement_only") is True:raise PermissionError("Boundary setup is not a whole-run worker packet")
    common,baseline = _bootstrap(packet)
    io,jobs = baseline["production_io"].NasIO(),baseline["production_job"]
    lock,_,_ = validate_packet(packet,io=io,normalize=baseline["production_io"].normalize_locator)
    task = packet["tasks"][run]
    control = Path(packet["operations_root"])/run
    identity = jobs.worker_identity()
    for key in THREAD_ENV: os.environ[key] = str(task["settings"]["threads"])
    def guard(): _current_guard(packet,common=common,baseline=baseline,identity=identity,task=task)
    guard()
    if "numpy" in sys.modules or "darsia" in sys.modules:
        raise RuntimeError("Native actual identity must precede numerical/package imports")
    io.write_json(control/"native_ready.json",dict(contract="ffac-round4-native-ready-v1",complete=True,
        packet=packet_bound,run=run,stage=task["stage"],actual_native_worker=identity,sources=packet["sources"],
        scientific_arrays_read=False,numerical_imports_started=False))
    deadline = time.monotonic()+120
    release_path = control/"release.json"
    while not release_path.exists():
        guard()
        if time.monotonic() >= deadline: raise TimeoutError("Exact Root native release absent")
        time.sleep(.1)
    release = io.read_json(release_path)
    if (release.get("packet") != packet_bound or release.get("worker") != identity or release.get("run") != run
            or jobs.worker_identity(release["owner"]["pid"]) != release["owner"]):
        raise ValueError("Root release does not bind this exact actual native")
    guard();_runtime(packet,lock,jobs)
    # Numerical imports and physical setup begin only beyond this boundary.
    sys.path.insert(0,str(Path(packet["worktree"])/"src"))
    import cv2,threadpoolctl
    from darsia.fluidflower import production_native
    production_native._modules()  # Original imports may set thread variables.
    for key in THREAD_ENV: os.environ[key] = str(task["settings"]["threads"])
    cv2.setNumThreads(task["settings"]["threads"])
    deps = _dependencies(packet,task["settings"])
    reader=make_staged_reader(task,packet,lock,baseline=baseline,dependencies=deps)
    job = jobs.LocalProductionJob(task["manifest"],task["local_results"],task["nas_results"],io=io)
    job.assert_runtime(packet["worktree"],production_go=task["stage"] == "production")
    started = time.perf_counter()
    with threadpoolctl.threadpool_limits(limits=task["settings"]["threads"]):
        guard();common.thread_facts(task["settings"]["threads"],cv2=cv2,poolctl=threadpoolctl)
        value = deps["round4"].run_selected_worker(task,dependencies=deps,reader=reader,identity=identity,boundary_guard=guard)
    if reader._verified_stage.references or reader._frame_stack is not None:
        raise RuntimeError("Completed run still has frame/prefetch input references")
    value.update(packet=packet_bound,code_lock=packet["code_lock"],round4_spec_sha256=SPEC_SHA256,
                 urest_amendment_sha256=AMENDMENT_SHA256,whole_run_runtime_seconds=time.perf_counter()-started,
                 worker_references_closed=True,arrays_read_only_after_actual_release=True)
    guard();io.write_json(control/"completed.json",value)
    return value


def coordinate(packet_bound, *, before_launch, host_stream, prepare_task, cleanup_go=None):
    """Explicit Root action. Existing lifecycle owns staging/retirement/cleanup.

    prepare_task(run,slot,working_copies,owned_results) is Root's metadata-only
    binder: it uses verified_copy_stage and returns the exact task with staged
    inputs/header contracts. No source manifest, data selection or guard is
    synthesized here. HostRunStream supplies ONE host-wide next-run copier.
    """
    if not callable(before_launch) or not callable(prepare_task):
        raise PermissionError("Actual Root live guard and metadata stage binder are required")
    import psutil
    packet = _read_bound(packet_bound);packet["packet_sha256"] = packet_bound["sha256"]
    if packet.get("boundary_measurement_only") is True:raise PermissionError("No numerical worker launch from a boundary input packet")
    common,baseline = _bootstrap(packet)
    io,jobs = baseline["production_io"].NasIO(),baseline["production_job"]
    lock,assignment,slots = validate_packet(packet,io=io,normalize=baseline["production_io"].normalize_locator)
    _runtime(packet,lock,jobs)
    if psutil.cpu_count() != packet["logical_cpus"]:
        raise ValueError("Actual logical CPU count differs")
    if psutil.virtual_memory().available < 40*GIB+sum(row[3] for row in slots.values()):
        raise RuntimeError("Initial frozen worker reservations would consume the40GiB reserve")
    if host_stream.host != "Main" or host_stream._allowed != {run:task["worker"] for run,task in packet["tasks"].items()}:
        raise ValueError("Existing host lifecycle differs from these fixed stage owners")
    root = Path(packet["operations_root"]);root.mkdir(parents=True,exist_ok=False)
    owner = jobs.worker_identity()
    io.write_json(root/"owner.json",dict(packet=packet_bound,owner=owner,stage=packet["stage"],claim_exclusive=True))
    active,completed,identities,handles = {},{},[],[]
    stop_event = threading.Event()
    def parent_guard():
        before_launch();_current_guard(packet,common=common,baseline=baseline,identity=owner,coordinator=True)
    def publish(continuing=True):
        def current():
            parent_guard()
            return dict(owner=owner,packet_sha256=packet_bound["sha256"],continue_=continuing,
                        updated_utc=datetime.now(timezone.utc).isoformat())
        def provider():
            value=current();value["continue"]=value.pop("continue_");return value
        with common.local_fence_mutex(packet_bound["sha256"]): io.replace_json(packet["coordinator_fence"],provider)
    publisher_errors=[]
    def publisher():
        try:
            while not stop_event.wait(10): publish()
        except BaseException as error:
            publisher_errors.append(error);stop_event.set()
    parent_guard();publish()
    thread=threading.Thread(target=publisher,name="round4-fresh-authority",daemon=False);thread.start()
    orders={slot:[row["run"] for row in assignment["assignments"]
                  if row["host"] == "Main" and row["worker"] == slot and row["run"] in packet["tasks"]] for slot in slots}
    positions={slot:0 for slot in slots}
    def start(slot):
        run=orders[slot][positions[slot]];task=packet["tasks"][run]
        parent_guard()
        copies=host_stream.start(slot,run,task["working_inputs"],boundary_guard=parent_guard)
        owner_stream=host_stream.workers[slot]
        bound_task=prepare_task(run,slot,copies,owner_stream.runs[run]["results"])
        operational = {"staged_inputs","verified_stage","array_contracts","local_results"}
        if (set(bound_task) != set(task) or any(bound_task[key] != task[key] for key in set(task)-operational)
                or Path(bound_task["local_results"]).resolve() != Path(owner_stream.runs[run]["results"]).resolve()):
            raise ValueError("Root stage binder changed the immutable scientific task")
        # A one-run packet is newly immutable after its real copy/stat bindings.
        child_packet=dict(packet,tasks={run:bound_task},coordinator_packet_sha256=packet_bound["sha256"])
        control=root/run;control.mkdir(exist_ok=False)
        path=control/"worker_packet.json";io.write_json(path,child_packet);bound=io.asset(path)
        # Every child uses the GLOBAL coordinator-fence key; its own immutable
        # packet descriptor is additionally bound in Root's ready/release pair.
        argv=[packet["python"]["path"],"-I","-B",packet["sources"]["operations"]["path"],"worker",
              "--packet",bound["path"],"--sha256",bound["sha256"],"--size",str(bound["size_bytes"]),"--run",run]
        log=(control/"worker.log").open("xb");handles.append(log)
        parent_guard()
        if psutil.virtual_memory().available < 40*GIB+bound_task["worker_budget_bytes"]:
            raise RuntimeError("New whole-run worker would consume the preserved40GiB reserve")
        process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0)|getattr(subprocess,"BELOW_NORMAL_PRIORITY_CLASS",0))
        wrapper=jobs.worker_identity(process.pid);deadline=time.monotonic()+120
        while not (control/"native_ready.json").exists():
            parent_guard()
            if process.poll() is not None: raise RuntimeError("Worker exited before actual native handshake")
            if time.monotonic()>=deadline: raise TimeoutError("Actual native-ready handshake absent")
            time.sleep(.1)
        ready=io.read_json(control/"native_ready.json")
        native=qualify_native(ready,task=bound_task,process=process,wrapper=wrapper,argv=argv,
            packet_bound=bound,packet=child_packet,jobs=jobs,psutil=psutil,normalize=baseline["production_io"].normalize_locator)
        jobs.LocalProductionJob(task["manifest"],bound_task["local_results"],task["nas_results"],io=io).register_worker(native,
            frame_id=io.read_json(task["manifest"])["frames"][0])
        identities.extend([wrapper,native])
        io.write_json(control/"release.json",dict(packet=bound,worker=native,run=run,owner=owner))
        active[slot]=dict(run=run,task=bound_task,process=process,native=native,wrapper=wrapper,control=control,
                          identities=[wrapper,native])
        if positions[slot]+1<len(orders[slot]):
            following=orders[slot][positions[slot]+1]
            host_stream.copy_next(slot,following,packet["tasks"][following]["working_inputs"],before_copy=parent_guard)
    try:
        for slot in slots: start(slot)
        while active:
            parent_guard()
            if publisher_errors: raise publisher_errors[0]
            for slot,row in list(active.items()):
                for parent in (row["wrapper"],row["native"]):
                    try:
                        if jobs.worker_identity(parent["pid"]) != parent: continue
                        descendants=psutil.Process(parent["pid"]).children(recursive=True)
                    except psutil.NoSuchProcess:
                        continue
                    for descendant in descendants:
                        try: observed=jobs.worker_identity(descendant.pid)
                        except psutil.NoSuchProcess: continue
                        if observed not in row["identities"]:
                            row["identities"].append(observed);identities.append(observed)
                code=row["process"].poll()
                if code is None: continue
                if code!=0: raise RuntimeError("Actual whole-run worker failed: "+row["run"])
                jobs.require_workers_retired(row["identities"])
                result=io.read_json(row["control"]/"completed.json")
                if (result.get("complete") is not True or result.get("actual_worker") != row["native"]
                        or result.get("worker_references_closed") is not True):
                    raise ValueError("Actual completed whole-run/reference closure differs")
                host_stream.finish(slot,row["run"],workers_retired_guard=lambda r=row:jobs.require_workers_retired(r["identities"]))
                completed[row["run"]]=result
                # Cleanup remains the original strict lifecycle call. Root
                # supplies its full NAS archive/closed inventory and actual GO.
                if cleanup_go is not None:
                    cleanup_go(host_stream.workers[slot],row["run"],row["task"],result,
                               lambda r=row:jobs.require_workers_retired(r["identities"]))
                del active[slot];positions[slot]+=1
                if positions[slot]<len(orders[slot]): start(slot)
            time.sleep(.25)
        jobs.require_workers_retired(identities)
        value=dict(complete=True,packet=packet_bound,owner=owner,workers_retired=identities,
                   results=completed,round4_spec_sha256=SPEC_SHA256,urest_amendment_sha256=AMENDMENT_SHA256)
        io.write_json(root/"completed.json",value)
        return value
    except BaseException as error:
        # No kill, retry or scientific continuation. Stop renewing permission;
        # existing workers stop at their original guarded natural boundary.
        try: publish(False)
        except BaseException: pass
        stop_event.set()
        try: claude_at_stop=common.status_data(packet["status_paths"]["claude"])
        except BaseException as status_error:
            claude_at_stop=dict(unreadable=True,error_type=type(status_error).__name__,error=str(status_error))
        io.write_json(root/"failed.json",dict(complete=False,error_type=type(error).__name__,error=str(error),
            original_stop_utc=datetime.now(timezone.utc).isoformat(),actual_known_identities=identities,
            preserved_all_originals_and_partials=True,claude_at_stop=claude_at_stop))
        raise
    finally:
        stop_event.set();thread.join()
        for handle in handles:handle.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["worker"])
    parser.add_argument("--packet",required=True);parser.add_argument("--sha256",required=True)
    parser.add_argument("--size",type=int,required=True);parser.add_argument("--run",required=True)
    args=parser.parse_args(argv)
    bound=dict(path=str(Path(args.packet).absolute()),sha256=args.sha256,size_bytes=args.size)
    try:worker(bound,args.run)
    except BaseException as error:
        print(json.dumps(dict(error_type=type(error).__name__,error=str(error))),file=sys.stderr,flush=True)
        raise


if __name__ == "__main__": main()
