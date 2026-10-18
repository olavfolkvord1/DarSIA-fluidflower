"""Explicit once-only cold qualification, never an ordinary measured profile.

The immutable provider precedes stores; the released request binds both. Only
the append-once completed bundle names final profile hashes. This is a DAG,
not a placeholder hash or an exception to actual stores/source equality.

Sampling records observations, not lifetime byte peaks. External NumPy/Numba/
Matplotlib/tempfile writers currently have no exact allocation journal here.
Their absence is a publication refusal, not an invented passing profile.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sys
import time
import uuid

from .policy import CachePolicyError, PHYSICAL_RESERVE_BYTES, _descriptor, _module_source_descriptor, nonnegative
from .control import _encoded

REQUEST_ENV = "DARSIA_CACHE_FIRST_QUALIFICATION"
PROVIDER = "darsia-cache-qualified-profile-provider-v1"
REQUEST = "darsia-cache-first-qualification-request-v1"
BOUND_REQUEST = "darsia-cache-first-qualification-bound-request-v1"
REFERENCE_REQUEST = "darsia-n43-reference-build-request-v1"
REFERENCE_ADMINISTRATION = "darsia-n43-reference-build-administration-v1"
REFERENCE_ENTITLEMENT_BYTES = 65728995330
REFERENCE_CASE_IDS = ["ac52_f" + str(i) for i in range(1, 44) if i not in {8, 15, 22, 24}]
CAPACITY_BOUND = "darsia-cache-required-capacity-bound-v1"
OBSERVATION = "darsia-cache-first-qualification-observation-v1"
BUNDLE = "darsia-cache-first-qualification-completed-bundle-v1"
SLOTS = {"observation", "writer_evidence", "measurement", "completion",
         "import_profile", "job_profile", "native_completion"}
WRITERS = ["bytecode", "matplotlib", "numba", "tempfile"]


def _same_path(a, b):
    return str(PureWindowsPath(a)).casefold() == str(PureWindowsPath(b)).casefold()


def _digest(value):
    # Hash-only identity is not a generic control record or an output write.
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":")) + "\n").encode("utf-8")


def _slot_encoded(value):
    if not isinstance(value, dict):
        raise CachePolicyError("First-qualification slot must be a JSON object")
    raw = _canonical_json(value)
    if not 0 < len(raw) <= 16 * 1024**2:
        raise CachePolicyError("First-qualification slot exceeds existing sixteen MiB bound; full samples not recorded")
    return raw


def _exact_keys(value, keys, message):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise CachePolicyError(message)
    return value


def _pins(rows):
    if not isinstance(rows, list) or not rows:
        raise CachePolicyError("Nonempty predeclared exact source/runtime closure required")
    pins = [_module_source_descriptor(row) for row in rows]
    if len({str(PureWindowsPath(pin["path"])).casefold() for pin in pins}) != len(pins):
        raise CachePolicyError("Duplicate first-qualification source identity")
    return pins


def checked_provider(value):
    _exact_keys(value, {"format", "qualification_id", "scope", "namespace_admission",
        "output_namespace", "slots", "python_executable", "module_closure", "startup",
        "producer_source_closure", "runtime_source_closure"}, "Exact immutable provider required")
    if value["format"] != PROVIDER or re.fullmatch(r"[0-9a-f]{32}", value["qualification_id"]) is None:
        raise CachePolicyError("Explicit finite qualification provider required")
    scope = _exact_keys(value["scope"], {"commit", "pipeline", "backend", "precision",
        "run", "ordered_case_ids"}, "Exact case/source/stage scope required")
    if (re.fullmatch(r"[0-9a-f]{40}", scope["commit"]) is None
            or scope["pipeline"] != "native-phase-n1" or scope["backend"] != "cpu"
            or scope["precision"] != "float32" or scope["run"] != "ac52"
            or not isinstance(scope["ordered_case_ids"], list)
            or len(scope["ordered_case_ids"]) != 39
            or len(set(scope["ordered_case_ids"])) != 39
            or any(not isinstance(case, str) or not case for case in scope["ordered_case_ids"])):
        raise CachePolicyError("First qualification is only the predeclared AC52 exact39 CPU float32 route")
    _descriptor(value["namespace_admission"]); _descriptor(value["python_executable"])
    _descriptor(value["startup"])
    target = _exact_keys(value["output_namespace"], {"path", "identity"}, "Owned output namespace required")
    if not isinstance(target["identity"], dict) or not PureWindowsPath(target["path"]).is_absolute():
        raise CachePolicyError("Actual preaccepted output namespace identity required")
    _exact_keys(value["slots"], SLOTS, "All finite append-once output slots required")
    names = list(value["slots"].values())
    if (len(set(names)) != len(names) or any(not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}\.json", name) is None for name in names)):
        raise CachePolicyError("Provider slots must be distinct flat immutable JSON names")
    union = _pins(value["producer_source_closure"]) + _pins(value["runtime_source_closure"])
    actual = _pins(value["module_closure"])
    if len(union) != len(actual) or {_digest(pin) for pin in union} != {_digest(pin) for pin in actual}:
        raise CachePolicyError("Producer/runtime union must exactly equal pre-import module closure")
    return deepcopy(value)


def _slot(provider, role):
    return str(Path(provider["output_namespace"]["path"]) / provider["slots"][role])


def _slot_read(provider, role):
    """Read through the preaccepted directory handle; never resolve a new root."""
    from .windows import WindowsNamespace
    target = provider["output_namespace"]
    namespace = WindowsNamespace(target["path"], target["identity"])
    try:
        raw, _ = namespace.read_once(provider["slots"][role], maximum=16*1024**2)
    finally:
        namespace.close()
    from .capacity import _json_bytes
    return _json_bytes(raw), dict(path=_slot(provider, role),
        sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _slot_write(provider, role, value):
    from .windows import WindowsNamespace
    target = provider["output_namespace"]; raw = _slot_encoded(value)
    namespace = WindowsNamespace(target["path"], target["identity"])
    try:
        namespace.write_once(provider["slots"][role], raw, maximum=16*1024**2)
    finally:
        namespace.close()
    return dict(path=_slot(provider, role), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _outputs_absent(provider):
    from .windows import WindowsNamespace
    bound = provider["output_namespace"]
    namespace = WindowsNamespace(bound["path"], bound["identity"])
    try:
        if set(provider["slots"].values()) & set(namespace.names()):
            raise CachePolicyError("A first-qualification output already exists; no rerun, retry or prior-profile bypass")
    finally:
        namespace.close()


def validate_startup(lease):
    """Same actual pre-import closure checks; no invented prior native profile."""
    from .bootstrap import _exact
    from .owned_process import _exact_source
    provider = lease.first_qualification.provider
    startup = _exact(provider["startup"], lease.package_file)
    if (startup.get("format") != "darsia-qualified-worker-startup-source-v1"
            or startup.get("python_executable") != provider["python_executable"]
            or startup.get("module_closure") != provider["module_closure"]
            or startup.get("include_system_site_packages") is not False
            or not startup.get("source_files") or not sys.flags.dont_write_bytecode
            or [str(Path(value or os.getcwd()).resolve()) for value in sys.path] != startup.get("sys_path")):
        raise CachePolicyError("Exact predeclared first startup/interpreter/-B/sys.path required")
    for pin in startup["source_files"]:
        _exact_source(pin)
    for name, module in tuple(sys.modules.items()):
        if getattr(module, "__file__", None) and not name.startswith("darsia."):
            path = str(Path(module.__file__).resolve())
            if startup.get("allowed_loaded_modules", {}).get(name) != path:
                raise CachePolicyError("Undeclared first pre-import module: " + name)
    if any(name.split(".")[0] in {"numba", "matplotlib", "cupy", "cv2", "numpy", "PIL"}
           for name in sys.modules):
        raise CachePolicyError("Numerical/cache producer imported before finite first admission")
    return dict(startup=provider["startup"], publication_contract="alias-done-small-metadata-v1",
        first_qualification_only=True, maximum_family_processes=None)


def administrative_entitlement(report):
    """Existing quota/free-minus-reserve arithmetic, explicitly NOT a profile."""
    if report.get("hard_filesystem_quota_verified") is not True:
        raise CachePolicyError("Actual enforced fixed-volume cap required before first allocation")
    if report.get("reserve_bytes") != PHYSICAL_RESERVE_BYTES:
        raise CachePolicyError("Original 40 GiB reserve cannot change in first mode")
    headroom = report["quota_headroom_bytes"]
    available = (report["free_physical_bytes"] - report["active_cache_remaining_bytes"]
        - report["required_noncache_bytes"] - PHYSICAL_RESERVE_BYTES)
    if type(headroom) is not int or type(available) is not int or min(headroom, available) <= 0:
        raise CachePolicyError("No actual positive first-mode administrative entitlement")
    return min(headroom, available)


class FirstQualification:
    """Read exact metadata on init; no cache/output writes or environment changes."""
    def __init__(self, binding, *, provider_binding, stores_source, namespace, read,
                 read_authorities=None):
        self.binding = _descriptor(binding)
        self.request = read(self.binding)
        self.reference_build = self.request.get("format") == REFERENCE_REQUEST
        self.bound_mode = self.request.get("format") == BOUND_REQUEST
        fields = {"format", "execution_authorized", "authority_ids", "root_release",
            "provider", "stores_source", "namespace_admission", "scope", "public_command", "task",
            "cache_stat_period_seconds"}
        if self.bound_mode:
            fields |= {"capacity_bound_plan", "output_entitlement_plan"}
        if self.reference_build:
            fields.add("reference_administration")
        _exact_keys(self.request, fields, "Exact Root-released finite first request required")
        expected_format = (REFERENCE_REQUEST if self.reference_build else
            BOUND_REQUEST if self.bound_mode else REQUEST)
        authorities = (["N43", "Q109", "Q106"] if self.reference_build else
            ["Q97", "Q99", "Q102", "Q104", "Q106"] if self.bound_mode else ["Q97", "Q99"])
        if (self.request["format"] != expected_format or self.request["execution_authorized"] is not True
                or self.request["authority_ids"] != authorities
                or self.request["provider"] != provider_binding
                or self.request["stores_source"] != stores_source
                or self.request["namespace_admission"] != namespace):
            raise CachePolicyError("Finite release/stores/provider/authority binding differs")
        period=self.request["cache_stat_period_seconds"]
        if type(period) not in {int,float} or not 0<period<=30:
            raise CachePolicyError("Explicit sampled cache-stat period in (0,30] seconds required")
        release = read(_descriptor(self.request["root_release"]))
        payload={key:value for key,value in self.request.items() if key!="root_release"}
        if (release.get("format") != "darsia-cache-first-qualification-root-release-v1"
                or release.get("execution_authorized") is not True
                or release.get("provider") != provider_binding
                or release.get("stores_source") != stores_source
                or release.get("scope") != self.request["scope"]
                or release.get("request_payload_sha256") != _digest(payload)):
            raise CachePolicyError("Actual Root-released first scope required")
        self.provider_binding = _descriptor(provider_binding)
        self.provider = checked_provider(read(self.provider_binding))
        if (self.request["scope"] != self.provider["scope"]
                or namespace != self.provider["namespace_admission"]
                or not isinstance(self.request["public_command"], list)
                or not self.request["public_command"]
                or any(not isinstance(arg, str) for arg in self.request["public_command"])):
            raise CachePolicyError("First request is not its immutable case/source provider scope")
        self.stores_source = deepcopy(stores_source)
        self.task=read(_descriptor(self.request["task"]))
        if (not isinstance(self.task,dict)
                or {key:self.task.get(key) for key in ("commit","pipeline","precision","run","ordered_case_ids")}
                   != {key:self.provider["scope"][key] for key in ("commit","pipeline","precision","run","ordered_case_ids")}
                or re.fullmatch(r"[0-9a-f]{32}",str(self.task.get("attempt_id"))) is None):
            raise CachePolicyError("Exact original predeclared task/input/calibration binding required")
        self.required_capacity_bound = None
        self.output_entitlement = None
        if self.bound_mode:
            self.required_capacity_bound = checked_required_capacity_bound(
                read(_descriptor(self.request["capacity_bound_plan"])))
            self.output_entitlement = checked_output_entitlement(
                self.task, read(_descriptor(self.request["output_entitlement_plan"])),
                task_binding=self.request["task"], read=read)
        self.reference_administration = None
        self.reference_roots_validated = False
        self.reference_transport_observation = None
        if self.reference_build:
            self.reference_administration = checked_reference_administration(
                read(_descriptor(self.request["reference_administration"])), self, read=read,
                read_authorities=read_authorities)
        self.started = False
        self.claim = None

    def consume(self, lease):
        """Native controller consumes once before CreateProcess; no recovery retry."""
        if self.bound_mode:
            # Q106 permits a sampled operational reserve, not a hard free-floor.
            # N43 reference/GO and the actual all-stage output/profile producer
            # are not yet bound. Refuse before any consume marker/native launch.
            raise CachePolicyError("N43 PRELAUNCH HOLD: OFF reference/GO, all-stage output entitlement and genuine profile producer remain unbound")
        _outputs_absent(self.provider)
        self._namespace(lease)
        key = self._once_key()
        with lease.ledger.locked():
            report=lease.ledger.project(0)
            if (report["current_cache_logical_bytes"]!=0
                    or report["active_cache_remaining_bytes"]!=0
                    or (not self.reference_build
                        and report.get("hard_filesystem_quota_verified") is not True)):
                raise CachePolicyError("First cold qualification requires its actual empty exclusive capped cache")
            if self.reference_build:
                self.reference_admission(lease, report=report, initial=True)
            # This lock/flat immutable marker shares the actual host reservation
            # authority. A new UUID/provider cannot repeat the same qualification.
            lease.ledger.control.write_once("cache-first-" + key + ".json", _encoded(dict(
                format="darsia-cache-first-qualification-consumed-v1", request=self.binding,
                provider=self.provider_binding, stores_source=self.stores_source,
                owner=lease.owner, scope=self.provider["scope"], no_retry=True)))
        self.started = True

    def _once_key(self):
        return _digest(dict(scope=self.provider["scope"], python=self.provider["python_executable"],
            sources=self.provider["module_closure"], namespace=self.provider["namespace_admission"]))

    def _namespace(self, lease):
        if self.reference_build:
            return reference_namespace(self, lease)
        if (self.provider["output_namespace"] != lease.admission["receipts"]
                or self.provider["namespace_admission"] != lease.document["cache_bootstrap"]["namespace_admission"]
                or lease.policy.profiles.get("jobs")!=self.provider_binding
                or lease.policy.volume_cap is None
                or lease.policy.quota_bytes not in {500*1024**3,150*1024**3}):
            raise CachePolicyError("First output/cache namespace is not the actual accepted one-host cap")

    def service_entitlement(self, lease):
        self._namespace(lease)
        # Called inside the original service reservation transaction. A direct
        # worker cannot bypass the consumed native-controller launch marker.
        if getattr(lease,"native_launch_context",None) is None:
            raise CachePolicyError("First service must be inside the original controlled native Job")
        raw,_=lease.ledger.control.read_once("cache-first-"+self._once_key()+".json")
        from .capacity import _json_bytes
        marker=_json_bytes(raw)
        if (marker.get("format")!="darsia-cache-first-qualification-consumed-v1"
                or marker.get("request")!=self.binding or marker.get("provider")!=self.provider_binding
                or marker.get("stores_source")!=self.stores_source
                or marker.get("scope")!=self.provider["scope"] or marker.get("no_retry") is not True):
            raise CachePolicyError("Original one-shot first launch marker differs")
        report = lease.ledger.project(0)
        if self.reference_build:
            admission = self.reference_admission(lease, report=report, initial=True)
            self.administrative = dict(maximum_bytes=REFERENCE_ENTITLEMENT_BYTES,
                measured_profile=False, allocation_model="N43-predeclared-all-stage-aggregate-once",
                request=self.binding, admission=admission)
            return REFERENCE_ENTITLEMENT_BYTES
        maximum = administrative_entitlement(report)
        self.administrative = dict(maximum_bytes=maximum, measured_profile=False,
            semantics="exclusive administrative entitlement from actual quota/free less original reserves",
            admission=report)
        return maximum

    def require_task(self, lease, task, *, remember=False):
        scope = self.provider["scope"]
        if (getattr(lease, "first_qualification", None) is not self
                or {key: task.get(key) for key in ("commit", "pipeline", "precision", "run", "ordered_case_ids")}
                   != {key: scope[key] for key in ("commit", "pipeline", "precision", "run", "ordered_case_ids")}
                or lease.source != self.stores_source or task!=self.task):
            raise CachePolicyError("First claim differs from exact released source/stage/cases/stores")
        if self.claim is not None and self.claim != task:
            raise CachePolicyError("Only one actual exact39 claim may use this first service")
        if self.reference_build and remember:
            # The actual worker slot has the service scope/environment. The
            # external native observer never starts a cache lease of its own.
            reference_writer_coverage(self,lease,writer_environment=True)
        if remember:
            # One original queued attempt/owner across all spawned service
            # actors, not merely one in-memory counter per Python process.
            with lease.ledger.locked():
                name="cache-first-"+self._once_key()+"-claim.json"
                value=dict(format="darsia-cache-first-single-claim-v1",request=self.binding,
                    provider=self.provider_binding,task=task,owner=lease.owner)
                if name in lease.ledger.control.names():
                    raw,_=lease.ledger.control.read_once(name)
                    from .capacity import _json_bytes
                    if _json_bytes(raw)!=value:
                        raise CachePolicyError("A different first claim/owner already consumed this entitlement")
                else:lease.ledger.control.write_once(name,_encoded(value))
            self.claim = deepcopy(task)
        return True

    def reference_admission(self, lease, *, report=None, initial=False, other_remaining=0):
        """One explicit aggregate, independently admitted on each affected volume."""
        if not self.reference_build:
            raise CachePolicyError("Reference administration was not explicitly released")
        self._namespace(lease)
        if report is None:
            report = lease.ledger.project(0)
        if report["reserve_bytes"] != PHYSICAL_RESERVE_BYTES or other_remaining != 0:
            raise CachePolicyError("N43 requires the original reserve and an exclusive result allocation")
        hard=report.get("hard_filesystem_quota_verified") is True
        if hard != (self.reference_administration["host_policy"]=="main-q87-fixed500"):
            raise CachePolicyError("N43 actual hard-cap observation differs from the declared host exception")
        if initial and (report["current_cache_logical_bytes"] != 0
                or report["active_cache_remaining_bytes"] != 0):
            raise CachePolicyError("N43 requires its fresh exclusive cache and no prior service allocation")
        snapshot = reference_scope_snapshot(self, lease)
        if snapshot["observed_scan_anomalies"]:
            raise CachePolicyError("N43 admission requires an actual complete scan without observed identity/rename errors")
        if initial and any(row["logical_bytes"] for row in snapshot["roots"] if row["role"]=="results"):
            raise CachePolicyError("N43 new reference requires its actual empty result namespace; no borrowed/resumed products")
        if snapshot["logical_bytes"] > REFERENCE_ENTITLEMENT_BYTES:
            latch_reference_stop(self, lease, "AGGREGATE_ENTITLEMENT_EXCEEDED", snapshot)
        require_reference_not_stopped(self, lease)
        if any(v["free_bytes"] - PHYSICAL_RESERVE_BYTES < REFERENCE_ENTITLEMENT_BYTES
               for v in snapshot["volumes"]):
            raise CachePolicyError("J6 free minus original40GiB does not cover the whole N43 entitlement")
        return dict(format="darsia-n43-predeclared-admission-v1", request=self.binding,
            entitlement_bytes=REFERENCE_ENTITLEMENT_BYTES, aggregate_counted_once=True,
            measured_profile=False, snapshot=snapshot, sufficient=True,
            reserve_bytes=PHYSICAL_RESERVE_BYTES,
            hard_cap_exists=self.reference_administration["host_policy"]=="main-q87-fixed500",
            no_ordinary_profile_adoption=True)

    def loaded_modules(self):
        """Actual exiting child inventory; unknown files are not relabelled known."""
        from .owned_process import _exact_source
        declared={str(Path(pin["path"]).resolve()).casefold():(index,pin)
            for index,pin in enumerate(self.provider["module_closure"])}
        actual=[];unknown=[];checked=set()
        for name,module in sorted(tuple(sys.modules.items())):
            filename=getattr(module,"__file__",None)
            if filename is None:
                actual.append(dict(name=name,file=None,builtin_or_no_file=True))
                continue
            path=str(Path(filename).resolve());bound=declared.get(path.casefold())
            if bound is None:
                unknown.append(dict(name=name,path=path))
                actual.append(dict(name=name,path=path,source_index=None))
            elif path.casefold() not in checked:
                _exact_source(bound[1]);checked.add(path.casefold())
            if bound is not None:
                actual.append(dict(name=name,source_index=bound[0]))
        return dict(format="darsia-cache-first-actual-loaded-modules-v1",request=self.binding,
            provider=self.provider_binding,actual=actual,undeclared=unknown,
            all_file_backed_modules_predeclared_and_exact=not unknown,
            predeclared_module_closure_sha256=_digest(self.provider["module_closure"]),
            module_inventory_scope="actual Python sys.modules file bindings; not an OS DLL enumeration",
            actual_task=deepcopy(self.claim),this_is_not_process_retirement=True)


def request_existing_frame_boundary_stop(first, lease):
    """Existing master-only append command; never kill, spoof or rewrite a job."""
    argv = first.request["public_command"]
    positions = [i for i, argument in enumerate(argv) if argument == "--job"]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        return dict(requested=False, state="UNSUPPORTED_PUBLIC_JOB_ARGUMENT")
    root = argv[positions[0] + 1]
    if not PureWindowsPath(root).is_absolute():
        return dict(requested=False, state="UNSUPPORTED_NONABSOLUTE_JOB_ROOT")
    try:
        from darsia.fluidflower import job_control
        job = job_control.read_json(Path(root) / "job.json")
        # Alias job.json has no job_id field: the original queue binds it to
        # the job directory name. Never invent a new JSON schema here.
        if (PureWindowsPath(root).name != first.task.get("job_id")
                or job.get("commit") != first.provider["scope"]["commit"]):
            raise CachePolicyError("Actual public job differs from the released first task")
        current = job_control.read_control(root)
        if current["action"] in {"pause", "stop"}:
            return dict(requested=False, state="EXISTING_BOUNDARY_EXIT_COMMAND",
                actual_control=current)
        command = job_control.publish_control(root, "stop", machine=lease.owner["machine"],
            expected_previous=current.get("command"))
        return dict(requested=True, state="EXISTING_MASTER_STOP_COMMAND_PUBLISHED",
            command=command, semantics="cooperative exit at original frame boundary; not immediate termination")
    except Exception as error:
        return dict(requested=False, state="STOP_COMMAND_NOT_PUBLISHED", error_type=type(error).__name__,
            message=str(error), semantics="reserve crossing is STOP even when the command is unavailable")


class FiniteObservations:
    """Truthful samples plus independently kernel-maintained lifetime counters."""
    def __init__(self, first, *, clock=None):
        self.first = first
        self.clock=time.monotonic if clock is None else clock
        self.cache_stat_period_seconds=first.request["cache_stat_period_seconds"]
        self.samples = 0
        self.native_poll_count=0
        self.cache_samples=[]
        self.last_cache_start=None
        self.sampled_cache_logical_high_water_bytes = 0
        self.sampled_active_processes_high_water = 0
        self.last = None
        self.reserve_samples = []
        self.last_reserve_start = None
        self.reserve_crossed = False
        self.stop_response = None
        self.aggregate_samples = []
        self.last_aggregate = None

    def observe_reserve(self, owned, started, *, force=False):
        """Q106 sampled operational reserve; a sample never proves a free-floor."""
        if not (self.first.bound_mode or self.first.reference_build):
            return
        period = (1 if self.first.reference_build else
            self.first.required_capacity_bound["reserve_watchdog_period_seconds"])
        if (not force and self.last_reserve_start is not None
                and started - self.last_reserve_start < period):
            return
        from .bootstrap import _utc
        from .capacity import volume_info
        aggregate = None
        if self.first.reference_build:
            try:
                aggregate = reference_scope_snapshot(self.first,owned.lease)
            except BaseException as error:
                latch_reference_stop(self.first,owned.lease,"AGGREGATE_OBSERVATION_UNAVAILABLE",
                    dict(error_type=type(error).__name__,message=str(error)))
                raise
        row = volume_info(owned.lease.policy.root)
        crossed = (any(v["free_bytes"] <= PHYSICAL_RESERVE_BYTES for v in aggregate["volumes"])
            if aggregate is not None else row["free_bytes"] <= PHYSICAL_RESERVE_BYTES)
        if aggregate is not None:
            aggregate["started_monotonic_seconds"] = started
            aggregate["actual_gap_since_previous_start_seconds"] = (None if self.last_reserve_start is None
                else started-self.last_reserve_start)
            self.last_aggregate = aggregate
            self.aggregate_samples.append(dict(utc=aggregate["utc"],
                started_monotonic_seconds=started,
                actual_gap_since_previous_start_seconds=aggregate["actual_gap_since_previous_start_seconds"],
                logical_bytes=aggregate["logical_bytes"],entries=aggregate["entries"],
                roots_logical_bytes={row["role"]:row["logical_bytes"] for row in aggregate["roots"]},
                volumes_free_bytes={row["volume_id"]:row["free_bytes"] for row in aggregate["volumes"]},
                forced_terminal=force,sampled=True,complete_lifetime_peak=False,
                observed_scan_anomalies=aggregate["observed_scan_anomalies"]))
            if aggregate["logical_bytes"] > REFERENCE_ENTITLEMENT_BYTES:
                latch_reference_stop(self.first, owned.lease, "AGGREGATE_ENTITLEMENT_EXCEEDED", aggregate)
        self.reserve_samples.append(dict(utc=_utc(), monotonic_seconds=started,
            actual_gap_since_previous_start_seconds=(None if self.last_reserve_start is None
                else started - self.last_reserve_start), forced_terminal=force,
            operational_reserve_bytes=PHYSICAL_RESERVE_BYTES,
            free_volume_observation=row, at_or_below_operational_reserve=crossed))
        self.last_reserve_start = started
        if crossed and not self.reserve_crossed:
            self.reserve_crossed = True  # Permanent STOP, even if free bytes later recover.
            if self.first.reference_build:
                latch_reference_stop(self.first, owned.lease, "OPERATIONAL_RESERVE_CROSSED", aggregate)
            self.stop_response = request_existing_frame_boundary_stop(self.first, owned.lease)

    def observe(self, owned, accounting, *, force_cache=False):
        if not force_cache:self.native_poll_count+=1
        self.sampled_active_processes_high_water = max(self.sampled_active_processes_high_water,
            accounting["active_processes"])
        started=self.clock()
        self.observe_reserve(owned, started, force=force_cache)
        if (not force_cache and self.last_cache_start is not None
                and started-self.last_cache_start<self.cache_stat_period_seconds):
            return  # Native retirement/accounting polling remains cheap.
        from .bootstrap import _utc
        utc_started=_utc()
        lease = owned.lease
        if self.first.reference_build:
            # Diagnostic view of the already observed aggregate, NOT a fresh
            # cache-ledger/per-token measurement. No second handle-based walk.
            aggregate=self.last_aggregate
            if aggregate is None:
                raise CachePolicyError("N43 diagnostic cache view has no actual aggregate observation")
            cache=[value for value in aggregate["roots"] if value["role"]=="cache"]
            if len(cache)!=1 or not _same_path(cache[0]["path"],lease.policy.root):
                raise CachePolicyError("N43 diagnostic cache row differs from the actual cache root")
            row=dict(format="darsia-n43-aggregate-cache-diagnostic-view-v1",root=cache[0]["path"],
                logical_bytes=cache[0]["logical_bytes"],source_aggregate_utc=aggregate["utc"],
                source_aggregate_monotonic_seconds=aggregate["started_monotonic_seconds"],
                actual_sample_age_seconds=started-aggregate["started_monotonic_seconds"],
                observed_scan_anomalies=aggregate["observed_scan_anomalies"],
                non_atomic_sample=True,per_token_accounting_provided=False,
                ordinary_cache_ledger_measurement=False,complete_lifetime_peak=False)
        else:
            with lease.ledger.locked(create_binding=False):
                row = lease.measure(lease.policy.root)
        ended=self.clock()
        self.cache_samples.append(dict(started_utc=utc_started,finished_utc=_utc(),
            started_monotonic_seconds=started,elapsed_seconds=ended-started,
            actual_gap_since_previous_start_seconds=(None if self.last_cache_start is None
                else started-self.last_cache_start),forced_terminal=force_cache,observation=row))
        self.last_cache_start=started
        self.samples += 1
        self.sampled_cache_logical_high_water_bytes = max(self.sampled_cache_logical_high_water_bytes,
            row["logical_bytes"])
        self.last = row

    def document(self, owned, terminal, retirement, eligibility):
        peaks = owned.api.lifetime_memory(owned.job)
        total = nonnegative(terminal["accounting"]["total_processes"], "Actual kernel total process count")
        if not total or terminal["accounting"]["active_processes"] != 0:
            raise CachePolicyError("Kernel whole-family retirement required for observations")
        result = dict(format=OBSERVATION, request=self.first.binding,
            provider=self.first.provider_binding, stores_source=self.first.stores_source,
            scope=self.first.provider["scope"], service_receipt=owned.service_binding,
            native_launch=owned.observed_binding, retirement=retirement,
            original_publication_and_retirement_eligibility=eligibility,
            actual_public_worker_exit_code=terminal["wrapper_exit_code"],
            original_kernel_terminal=deepcopy(terminal),
            sample_count=self.samples, final_sample=self.last,
            native_accounting_poll_count=self.native_poll_count,
            cache_stat_period_seconds=self.cache_stat_period_seconds,
            cache_samples=self.cache_samples,
            cache_sampling_semantics="initial, declared bounded periodic and forced terminal lstat samples; actual times/gaps retained; no peak interpolation",
            sampled_cache_logical_high_water_bytes=self.sampled_cache_logical_high_water_bytes,
            sampled_cache_logical_high_water_is_lifetime_peak=False,
            sampled_active_processes_high_water=self.sampled_active_processes_high_water,
            kernel_total_processes=total, kernel_total_processes_is_concurrent_peak=False,
            conservative_family_process_bound_from_kernel_total=total,
            kernel_lifetime_memory=peaks,
            reserve_watchdog_period_seconds=(1 if (self.first.bound_mode or self.first.reference_build) else None),
            reserve_samples=self.reserve_samples, operational_reserve_crossed=self.reserve_crossed,
            operational_stop_response=self.stop_response,
            reserve_semantics="operational admission plus sampled watchdog; not a hard free-volume floor",
            hard_bound_semantics=("NO_HARD_CAP: explicit one-run OlavD N43 human exception"
                if self.first.reference_build and self.first.reference_administration["host_policy"]=="olav-n43-one-run-uncapped-d"
                else "actual fixed-volume capacity only; never capacity minus40GiB"),
            reserve_samples_prove_no_crossing_between_samples=False,
            reserve_sampling_scope="existing native await_family observer plus forced terminal sample; initial startup/blocked scan gaps are not continuous coverage",
            writer_lifetime_cache_bytes_proof=None,
            profile_publication_state="REFUSED_UNSUPPORTED_LIFETIME_WRITER_COVERAGE",
            complete=True, passed=False, no_ordinary_profile_created=True)
        if self.first.reference_build:
            result.update(reference_aggregate_samples=self.aggregate_samples,
                final_reference_aggregate=self.last_aggregate,aggregate_samples_are_lifetime_peak=False)
        return result


def supervise_first_worker(lease, *, api=None):
    """Same exact native launch and natural retirement; never a numerical path."""
    from .owned_process import OwnedWorker, _public_command
    first = lease.first_qualification
    validate_startup(lease)
    command = _public_command(lease)
    if command != first.request["public_command"]:
        raise CachePolicyError("Actual normal public command differs from the finite released argv")
    first.consume(lease)
    owned = OwnedWorker(lease, api=api)
    observations = FiniteObservations(first)
    native_completion=None
    terminal=retirement=observation_pin=None
    try:
        owned.start(command, dict(os.environ), Path.cwd())
        terminal = owned.await_family(observer=observations.observe)
        observations.observe(owned,terminal["accounting"],force_cache=True)
        retirement = owned.record_closed_family(terminal)
        from .service_context import resolve_original_service_context
        from .service_lifecycle import ServiceLifecycle
        from .adapters import verify_worker_publication
        from .bootstrap import _utc
        context=resolve_original_service_context(owned.service_binding,package_file=lease.package_file,
            first_request=first.binding)
        eligibility=ServiceLifecycle(context).retention_eligibility(now_utc=_utc(),
            retirement=retirement,verify_retirement=owned.verify_retirement,
            verify_publication=verify_worker_publication)
        from .owned_process import _exact_source
        from .bootstrap import _exact
        for pin in first.provider["module_closure"]:
            _exact_source(pin)
        startup=_exact(first.provider["startup"],lease.package_file)
        for pin in startup["source_files"]:
            _exact_source(pin)
        observation = observations.document(owned, terminal, retirement, eligibility)
        from .owned_process import _control
        lifecycle=ServiceLifecycle(context)
        with context.ledger.locked(create_binding=False):
            records=lifecycle._records()
        inventories=[];tasks=[];modules_exact=True
        for (kind,key),registered in sorted(records.items()):
            if kind!="owner":continue
            owner=registered["document"]["value"]["owner"]
            exit_name=(context.reservation["token"]+".exit-intent-"+str(owner["pid"])
                +"-"+owner["creation_filetime_100ns"]+".json")
            exited,exit_pin=_control(lease,exit_name)
            modules=exited.get("first_qualification_modules",{})
            if (exited.get("owner")!=owner or modules.get("request")!=first.binding
                    or modules.get("provider")!=first.provider_binding
                    or modules.get("predeclared_module_closure_sha256")!=_digest(first.provider["module_closure"])):
                raise CachePolicyError("Actual registered owner's exit inventory differs from first request")
            modules_exact=modules_exact and modules.get("all_file_backed_modules_predeclared_and_exact") is True
            if modules.get("actual_task") is not None:tasks.append(modules["actual_task"])
            inventories.append(dict(owner=owner,registration=registered["binding"],inventory=exit_pin,
                all_file_backed_modules_predeclared_and_exact=modules.get("all_file_backed_modules_predeclared_and_exact"),
                undeclared=modules.get("undeclared")))
        if len(tasks)!=1:
            raise CachePolicyError("Exactly one actual worker-slot first claim is required")
        first.require_task(lease,tasks[0])
        observation["actual_registered_owner_exit_inventories"]=inventories
        observation["actual_claim"]=tasks[0]
        observation["source_runtime_before_after_exact"]=True
        observation_pin=_slot_write(first.provider, "observation", observation)
        seal=_exact(eligibility["seal"],lease.package_file)
        native_completion=dict(format="darsia-native-worker-first-qualification-completion-v1",
            request=first.binding,provider=first.provider_binding,stores_source=lease.source,
            scope=first.provider["scope"],startup=first.provider["startup"],
            module_closure=first.provider["module_closure"],observation=observation_pin,
            retirement=retirement,eligibility=eligibility,
            maximum_family_processes=observation["conservative_family_process_bound_from_kernel_total"],
            family_bound_semantics="actual kernel lifetime TotalProcesses; conservative concurrency bound, not sampled peak",
            sampled_concurrent_high_water=observation["sampled_active_processes_high_water"],
            complete=True,passed=(terminal["wrapper_exit_code"]==0
                and eligibility.get("original_retirement_verifier_accepted") is True
                and seal.get("value",{}).get("outcome")=="success"
                 and modules_exact and bool(inventories) and not observations.reserve_crossed
                 and not (first.reference_build and reference_stopped(first, lease))),
            cache_profile_qualified=False)
        # Deliberately no guessed measured profile, no automatic scope adoption,
        # reservation close or cleanup. Original actual cache remains charged.
    except BaseException as error:
        if first.reference_build and observation_pin is None:
            # Preserve genuine samples even if cooperative STOP prevents the
            # ordinary whole-run publication/eligibility or a final scan fails.
            try:
                observation_pin=_slot_write(first.provider,"observation",dict(format=OBSERVATION,
                    request=first.binding,provider=first.provider_binding,scope=first.provider["scope"],
                    state="STOP_PARTIAL_DIAGNOSTIC",complete=False,passed=False,
                    error_type=type(error).__name__,message=str(error),actual_terminal=terminal,
                    actual_retirement=retirement,reference_aggregate_samples=observations.aggregate_samples,
                    final_reference_aggregate=observations.last_aggregate,
                    reserve_samples=observations.reserve_samples,operational_reserve_crossed=observations.reserve_crossed,
                    sampled_observed_highwater_is_complete=False,partial_files_retained=True,
                    no_ordinary_profile_created=True,no_reference_acceptance=True))
            except BaseException as diagnostic_error:
                error.add_note("Partial observation was not recorded; full samples are not durable: "
                    + type(diagnostic_error).__name__ + ": " + str(diagnostic_error))
        held=dict(request=first.binding,provider=first.provider_binding,error_type=type(error).__name__,
            message=str(error),no_retry=True,reservation_closed=False)
        if first.reference_build:
            held["actual_partial_observation"]=observation_pin
        try:
            lease._record("first-held-"+uuid.uuid4().hex+".json",held)
        except BaseException as diagnostic_error:
            error.add_note("First-held diagnostic was not recorded: "
                + type(diagnostic_error).__name__ + ": " + str(diagnostic_error))
        if owned.resume_attempted and owned.info is not None:
            owned.await_family()
        raise
    finally:
        owned.close()
    native_completion["observer_native_handles_closed"]=True
    if first.reference_build:
        native_completion["producer_complete_unqualified"] = native_completion["passed"]
        native_completion["passed"] = False
        native_completion["state"] = "UNQUALIFIED_REFERENCE_PRODUCER_COMPLETE"
        native_completion["root_reference_acceptance_required"] = True
    _slot_write(first.provider,"native_completion",native_completion)
    if first.reference_build:
        # Durable observations/retirement are genuine; this is deliberately not
        # a passing profile, self-comparison, accepted reference or adoption.
        if not native_completion["producer_complete_unqualified"]:
            raise CachePolicyError("N43 reference producer STOP; partial products retained, no reference promotion")
        return 0  # Original external controller/SystemExit contract; not a profile PASS.
    # A successful scientific job is not an accepted cache profile. Make the
    # missing qualification explicit after durable observations and close.
    raise CachePolicyError("First observations complete; cache profile publication refused: no qualified lifetime writer coverage")


def require_publishable(observation, writer_evidence):
    """No polling/redirect/source-audit assertion can mint a lifetime maximum."""
    if (observation.get("format") != OBSERVATION or observation.get("complete") is not True
            or writer_evidence is None):
        raise CachePolicyError("Actual completed observation and all-writer lifetime evidence required")
    # The existing four external runtime writers do not expose an exact
    # lifetime-allocation proof seam. No caller-supplied boolean/number is a
    # qualified provider. Implement that audited source seam before accepting
    # any new evidence format or publishing first measured profiles.
    raise CachePolicyError("No qualified all-writer lifetime allocation evidence provider is implemented")


def resolve_profile(binding, kind, read, *, stores_source=None, read_authorities=None):
    """Old direct profiles unchanged; incomplete new providers always refuse.

    The append-once DAG reader is explicit. Final publication remains closed
    until a genuine lifetime-writer proof format/provider exists; it cannot be
    activated with fabricated completed JSON or test protocol fixtures.
    """
    value = read(binding)
    if value.get("format") != PROVIDER:
        return value
    provider = checked_provider(value)
    if kind not in {"import","jobs"}:
        raise CachePolicyError("Unknown typed provider profile role")
    bundle, bundle_pin = _slot_read(provider, "completion")
    _exact_keys(bundle, {"format", "request", "provider", "stores_source", "scope",
        "observation", "writer_evidence", "profiles", "retirement"}, "Exact completed provider bundle required")
    if (bundle["format"] != BUNDLE or bundle["provider"] != binding
            or bundle["scope"] != provider["scope"]
            or (stores_source is not None and bundle["stores_source"] != stores_source)):
        raise CachePolicyError("Completed provider differs from actual stores/source/cases")
    first = FirstQualification(bundle["request"], provider_binding=binding,
        stores_source=bundle["stores_source"], namespace=provider["namespace_admission"], read=read,
        read_authorities=read_authorities)
    observation, pin = _slot_read(provider, "observation")
    evidence, evidence_pin = _slot_read(provider, "writer_evidence")
    if (bundle["observation"] != pin or bundle["writer_evidence"] != evidence_pin
            or observation.get("request") != first.binding
            or observation.get("provider") != binding
            or observation.get("stores_source") != bundle["stores_source"]
            or observation.get("scope") != provider["scope"]
            or observation.get("retirement") != bundle["retirement"]
            or observation.get("source_runtime_before_after_exact") is not True):
        raise CachePolicyError("Provider actual producer/observation/retirement binding differs")
    _exact_keys(bundle["profiles"],{"import","jobs"},"Typed import/jobs profile roles required")
    selected,selected_pin=_slot_read(provider,"import_profile" if kind=="import" else "job_profile")
    if selected_pin!=bundle["profiles"][kind]:
        raise CachePolicyError("Completed provider profile differs from its fixed immutable slot")
    native,native_pin=_slot_read(provider,"native_completion")
    if (native.get("request")!=first.binding or native.get("provider")!=binding
            or native.get("stores_source")!=bundle["stores_source"]
            or native.get("scope")!=provider["scope"] or native.get("startup")!=provider["startup"]
            or native.get("module_closure")!=provider["module_closure"]
            or native.get("observation")!=pin or native.get("retirement")!=bundle["retirement"]
            or native.get("complete") is not True or native.get("passed") is not True
            or native.get("observer_native_handles_closed") is not True):
        raise CachePolicyError("Actual completed native/module/retirement evidence differs")
    if kind=="import":
        block=selected.get("native_supervision",{})
        if (selected.get("format")!="darsia-cache-import-profile-v1"
                or selected.get("python_executable")!=provider["python_executable"]
                or selected.get("module_closure")!=provider["module_closure"]
                or block.get("startup")!=provider["startup"] or block.get("completion")!=native_pin
                or block.get("maximum_family_processes")!=native.get("maximum_family_processes")):
            raise CachePolicyError("Import/native profile differs from actual producer evidence")
    elif (selected.get("format")!="darsia-cache-measured-profile-v1"
            or selected.get("commit")!=provider["scope"]["commit"]
            or selected.get("pipeline")!=provider["scope"]["pipeline"]
            or selected.get("precision")!=provider["scope"]["precision"]
            or selected.get("case_count")!=len(provider["scope"]["ordered_case_ids"])):
        raise CachePolicyError("Job profile differs from original case/stage/source scope")
    require_publishable(observation, evidence)
    return selected  # Today unreachable: unproved writer evidence always refuses.


def checked_required_capacity_bound(value):
    """Q106 hard volume bound and separate sampled operational reserve; no GO."""
    _exact_keys(value, {"format", "authority_id", "volume_cap_observation",
        "volume_capacity_bytes", "free_volume_reserve_bytes", "required_capacity_bound_bytes",
        "operational_admission_headroom_bytes", "reserve_watchdog_period_seconds",
        "hard_bound_semantics", "reserve_semantics", "exclusive_volume_writer_required",
        "sampled_observed_highwater_is_complete", "enforcement_state"},
        "Exact typed Q106 volume-bound and operational-reserve plan required")
    observed = value["volume_cap_observation"]
    if (value["format"] != CAPACITY_BOUND
            or value["authority_id"] != "Q106-answer-sampled-operational-reserve-hard-bound-is-Q87-volume-no-new-seam"
            or not isinstance(observed, dict) or observed.get("verified") is not True
            or observed.get("format") != "darsia-readonly-fixed-vhdx-cap-observation-v1"
            or observed.get("filesystem") != "NTFS"
            or observed.get("is_loaded") is not True
            or value["sampled_observed_highwater_is_complete"] is not False
            or value["hard_bound_semantics"] != "actual-fixed-volume-only-not-capacity-minus-reserve"
            or value["reserve_semantics"] != "operational-admission-plus-sampled-watchdog-not-hard-floor"
            or value["exclusive_volume_writer_required"] is not True
            or type(value["reserve_watchdog_period_seconds"]) not in {int,float}
            or value["reserve_watchdog_period_seconds"] != 1
            or value["enforcement_state"] != "PRELAUNCH_HOLD_N43_REFERENCE_OUTPUT_BUDGET_AND_PROFILE_PRODUCER"):
        raise CachePolicyError("Actual volume-only bound, sampled reserve and truthful N43 HOLD required")
    capacity = nonnegative(value["volume_capacity_bytes"], "Actual usable volume capacity")
    if (capacity != observed.get("volume_total_bytes")
            or capacity <= PHYSICAL_RESERVE_BYTES
            or value["free_volume_reserve_bytes"] != PHYSICAL_RESERVE_BYTES
            or type(value["required_capacity_bound_bytes"]) is not int
            or value["required_capacity_bound_bytes"] != capacity
            or type(value["operational_admission_headroom_bytes"]) is not int
            or value["operational_admission_headroom_bytes"] != capacity - PHYSICAL_RESERVE_BYTES):
        raise CachePolicyError("Hard bound is actual volume capacity; capacity minus40GiB is only operational headroom")
    return deepcopy(value)


def checked_output_entitlement(task, value, *, task_binding, read):
    """Q104 source-bound administration, never an expected-space profile."""
    _exact_keys(value, {"format", "authority_id", "task", "scope", "R4_footprint",
        "administrative_output_entitlement_bytes", "measured_profile", "enforcement_state"},
        "Exact predeclared Q104 output entitlement required")
    scope = {key: task.get(key) for key in ("commit", "pipeline", "precision", "run", "ordered_case_ids")}
    if (value["format"] != "darsia-first-qualification-output-entitlement-v1"
            or value["authority_id"] != "Q104-answer-first-qualification-produces-result-space-profile-under-predeclared-entitlement"
            or value["task"] != _descriptor(task_binding)
            or value["scope"] != scope or scope["pipeline"] != "native-phase-n1"
            or scope["run"] != "ac52" or scope["precision"] != "float32"
            or not isinstance(scope["ordered_case_ids"], list)
            or len(scope["ordered_case_ids"]) != 39
            or len(set(scope["ordered_case_ids"])) != 39
            or value["measured_profile"] is not False
            or value["enforcement_state"] != "PRELAUNCH_HOLD_N43_REFERENCE_OUTPUT_BUDGET_AND_PROFILE_PRODUCER"):
        raise CachePolicyError("Q104 requires actual current exact39 task and truthful N43 administrative HOLD")
    footprint = read(_descriptor(value["R4_footprint"]))
    if (footprint.get("format") != "first-qualification-predeclared-output-entitlement-source-data-not-profile-v1"
            or footprint.get("authority_id") != value["authority_id"]
            or footprint.get("run") != scope["run"]
            or footprint.get("case_count") != 39
            or footprint.get("ordered_case_ids") != scope["ordered_case_ids"]
            or footprint.get("factor") != 2
            or footprint.get("not_current_measurement_or_profile") is not True
            or footprint.get("no_first_qualification_or_adoption") is not True):
        raise CachePolicyError("Actual pinned original R4 exact39 footprint required")
    rows = footprint.get("files")
    if not isinstance(rows, list) or not rows:
        raise CachePolicyError("Nonempty actual original output file-size ledger required")
    total = 0
    names = set()
    coverage = {case: set() for case in scope["ordered_case_ids"]}
    for row in rows:
        name = str(PureWindowsPath(row["path"])).casefold()
        size = nonnegative(row["size_bytes"], "Original actual result file length")
        if (name in names or not PureWindowsPath(row["path"]).is_absolute()
                or row.get("actual_stat", {}).get("size_bytes") != size
                or re.fullmatch(r"[0-9a-f]{64}", str(row.get("declared_original_sha256"))) is None):
            raise CachePolicyError("Unique source-bound actual result file lengths required")
        roles = row.get("roles")
        case = row.get("case_id")
        if (not isinstance(roles, list) or not roles or any(not isinstance(role, str) for role in roles)
                or (case is not None and case not in coverage)):
            raise CachePolicyError("Exact original per-case output roles required")
        if case is not None:
            coverage[case].update(roles)
        names.add(name)
        total += size
    required = {"products.npz", "result.json", "preview.png", "native_statistics.json.gz", "geometry"}
    if (footprint.get("unique_output_file_count") != len(rows)
            or any(not required <= roles for roles in coverage.values())):
        raise CachePolicyError("R4 footprint must include every original exact39 output/geometry role")
    if (not total or footprint.get("actual_result_footprint_bytes") != total
            or footprint.get("administrative_output_entitlement_bytes") != 2 * total
            or type(value["administrative_output_entitlement_bytes"]) is not int
            or value["administrative_output_entitlement_bytes"] != 2 * total):
        raise CachePolicyError("Q104 entitlement is exactly twice the bound actual R4 footprint")
    return deepcopy(value)


def checked_reference_administration(value, first, *, read, read_authorities=None):
    """N43/Q109 is an explicit producer entitlement, never a measured profile."""
    _exact_keys(value, {"format", "authority_ids", "answers_source", "accepted_plan",
        "scope", "task", "host", "host_policy", "owned_roots", "entry_limit",
        "aggregate_entitlement_bytes", "counting_semantics", "watchdog_period_seconds",
        "operational_reserve_bytes", "minimum_family_memory_limit_bytes",
        "memory_reserve_bytes", "all_external_writers_redirected", "exclusive_producer",
        "output_configuration", "requested_native_layout", "output_writers", "producer_state"}, "Exact N43 reference administration required")
    if (value["format"] != REFERENCE_ADMINISTRATION
            or value["authority_ids"] != ["N43", "Q109", "Q106"]
            or value["scope"] != first.provider["scope"]
            or value["scope"]["ordered_case_ids"] != REFERENCE_CASE_IDS
            or value["task"] != first.request["task"]
            or value["aggregate_entitlement_bytes"] != REFERENCE_ENTITLEMENT_BYTES
            or value["counting_semantics"] != "all-stages-products-intermediates-receipts-previews-caches-once"
            or value["watchdog_period_seconds"] != 1
            or value["operational_reserve_bytes"] != PHYSICAL_RESERVE_BYTES
            or value["minimum_family_memory_limit_bytes"] != 20*1024**3
            or value["memory_reserve_bytes"] != PHYSICAL_RESERVE_BYTES
            or value["all_external_writers_redirected"] is not True
            or value["exclusive_producer"] is not True
            or value["producer_state"] != "PREDECLARED_UNQUALIFIED_REFERENCE_BUILD"):
        raise CachePolicyError("N43 exact human scope, aggregate entitlement, reserves and one producer required")
    nonnegative(value["entry_limit"], "N43 finite accounting entry limit")
    if not value["entry_limit"]:
        raise CachePolicyError("Positive N43 finite accounting entry limit required")
    authorities=(read if read_authorities is None else read_authorities)(
        _descriptor(value["answers_source"]))
    ids={row.get("id"):row for row in authorities} if isinstance(authorities,list) else {}
    required={"N43-OLAV-GO-execute-plan-9dc0979d-tonight-Main-reference-and-Olav-qualification":"olav",
        "Q109-answer-option-A-whole-output-entitlement-65728995330B-all-stages-count":"claude",
        "Q106-answer-sampled-operational-reserve-hard-bound-is-Q87-volume-no-new-seam":"claude"}
    if any(ids.get(key,{}).get("decided_by")!=by for key,by in required.items()):
        raise CachePolicyError("Actual N43 humanGO/Q109/Q106 source records required")
    argv=first.request["public_command"]
    def option(name, expected):
        positions=[i for i,arg in enumerate(argv) if arg==name]
        return len(positions)==1 and positions[0]+1<len(argv) and argv[positions[0]+1]==expected
    if (not option("--workers","1") or not option("--backend","cpu")
            or argv.count("--once")!=1 or "--watchdog" in argv):
        raise CachePolicyError("N43 preserves one original whole-run CPU worker/once and forbids automatic retry")
    _descriptor(value["accepted_plan"])
    if value["accepted_plan"]["sha256"]!="9dc0979d5cd48ce18df6391253ed3f178f181c0c321626cef179c25a2e59e988":
        raise CachePolicyError("N43 humanGO names the exact retained plan9dc0979d")
    read(value["accepted_plan"])
    if not isinstance(value["requested_native_layout"],dict) or not value["output_writers"]:
        raise CachePolicyError("Predeclared actual native layout and closed output-writer pins required")
    if value["output_configuration"]!=first.task.get("configuration"):
        raise CachePolicyError("Exact original queued run configuration required")
    host=value["host"].casefold() if isinstance(value["host"],str) else ""
    if (host,value["host_policy"]) not in {("moderskipet","main-q87-fixed500"),
            ("olav","olav-n43-one-run-uncapped-d")}:
        raise CachePolicyError("Only explicit MainQ87 or human one-run OlavD policy is supported")
    roots=value["owned_roots"]
    roles=[row.get("role") for row in roots] if isinstance(roots,list) else []
    if (not {"cache","results","job"}<=set(roles) or len(set(roles))!=len(roles)
            or not set(roles)<={"cache","results","job","receipts","capacity_state","host_ledger"}):
        raise CachePolicyError("Explicit disjoint data and optional receipt/state/host-ledger roots required")
    for row in roots:
        _exact_keys(row,{"role","path","identity","volume_id","stat_identity"},"Actual preaccepted N43 root identity required")
        _exact_keys(row["stat_identity"],{"st_dev","st_ino","creation_ns"},"Actual no-follow root stat identity required")
        if any(type(v) is not int or v<=0 for v in row["stat_identity"].values()):
            raise CachePolicyError("Positive actual root device/inode/creation identity required")
        path=PureWindowsPath(row["path"])
        if not path.is_absolute() or not isinstance(row["identity"],dict) or not row["identity"] or not row["volume_id"]:
            raise CachePolicyError("Actual absolute owned namespace and volume binding required")
        if row["role"]=="job":
            if (not path.drive.startswith("\\") or row["identity"].get("format")!="original-j1-shared-folder-admission-v1"
                    or row["identity"].get("protocol") not in {"smb","nfs"}):
                raise CachePolicyError("N43 job retains the original positively admitted UNC RunIO queue route")
        if row["role"] in {"cache","results"} and host=="olav" and path.drive.casefold()!="d:":
            raise CachePolicyError("Olav one-run exception is confined to its explicitly assigned D namespace")
    by_role={row["role"]:row for row in roots}
    for option_name, expected in [("--job",by_role["job"]["path"]),("--stores",first.stores_source["path"])]:
        positions=[i for i,arg in enumerate(argv) if arg==option_name]
        if (len(positions)!=1 or positions[0]+1>=len(argv)
                or not _same_path(argv[positions[0]+1],expected)):
            raise CachePolicyError("N43 original public job/stores argv differs from the released namespaces")
    if not _same_path(argv[0],first.provider["python_executable"]["path"]):
        raise CachePolicyError("N43 exact selected public interpreter required")
    for i,row in enumerate(roots):
        for other in roots[i+1:]:
            left,right=PureWindowsPath(row["path"]),PureWindowsPath(other["path"])
            if left==right or left in right.parents or right in left.parents:
                raise CachePolicyError("N43 roots overlap; all-stage files must count exactly once")
    return deepcopy(value)


def reference_namespace(first, lease):
    value=first.reference_administration
    roots={row["role"]:row for row in value["owned_roots"]}
    host=str(lease.context["host"]).casefold()
    if (value["host"].casefold()!=host
            or first.provider["output_namespace"]!=lease.admission["receipts"]
            or first.provider["namespace_admission"]!=lease.document["cache_bootstrap"]["namespace_admission"]
            or lease.policy.profiles.get("jobs")!=first.provider_binding
            or not _same_path(roots["cache"]["path"],lease.policy.root)
            or roots["cache"]["identity"]!=lease.admission["cache_identity"]):
        raise CachePolicyError("N43 actual host/cache/receipts/provider namespace differs")
    if value["host_policy"]=="main-q87-fixed500":
        if (lease.policy.volume_cap is None or lease.policy.quota_bytes!=500*1024**3
                or lease.policy.volume_cap.maximum_size_bytes!=500*1024**3
                or "49079372-0000-0000-0000-100000000000" not in lease.policy.volume_cap.volume_guid.casefold()
                or PureWindowsPath(lease.policy.root).drive.casefold()!="q:"):
            raise CachePolicyError("Main N43 requires the actual accepted fixed500GiB Q87 volume")
    elif lease.policy.volume_cap is not None:
        raise CachePolicyError("One-run Olav uncapped exception must report the actual absence of a hard cap")
    reference_writer_coverage(first,lease)
    validate_reference_roots(first)
    return True


def reference_root_for_path(first, path):
    target=PureWindowsPath(str(path))
    if not target.is_absolute() or ".." in target.parts:
        raise CachePolicyError("N43 writer path must be explicit absolute and normalized")
    matches=[row for row in first.reference_administration["owned_roots"]
        if target==PureWindowsPath(row["path"]) or PureWindowsPath(row["path"]) in target.parents]
    if len(matches)!=1:
        raise CachePolicyError("N43 actual writer is omitted or counted by more than one aggregate root: "+str(path))
    return matches[0]


def reference_writer_coverage(first, lease, *, writer_environment=False):
    """Actual receipt/control/state paths, not only a predeclared boolean."""
    paths={"receipts":lease.admission["receipts"]["path"],
        "host_ledger":lease.admission["host_ledger"]["path"],
        "capacity_state":lease.document["capacity"]["state_directory"]}
    explicit={row["role"]:row for row in first.reference_administration["owned_roots"]}
    coverage=[]
    for role,path in paths.items():
        row=reference_root_for_path(first,path)
        if role in explicit:
            if not _same_path(explicit[role]["path"],path):
                raise CachePolicyError("N43 extra control root differs from the actual "+role)
            if role in {"receipts","host_ledger"} and explicit[role]["identity"]!=lease.admission[role]["identity"]:
                raise CachePolicyError("N43 extra control root kernel identity differs")
        coverage.append(dict(writer_role=role,path=str(path),aggregate_root_role=row["role"]))
    if writer_environment:
        from .environment import FIRST_WRITER_ENVIRONMENT_KEYS
        for key in FIRST_WRITER_ENVIRONMENT_KEYS:
            path=os.environ.get(key)
            if key=="NUMBA_CACHE_LOCATOR_CLASSES":
                if path!="UserProvidedCacheLocator":
                    raise CachePolicyError("N43 requires the original explicit Numba user-provided locator")
                coverage.append(dict(writer_role=key,value=path,semantics="original locator selector, not a filesystem path"))
                continue
            if path is None:
                raise CachePolicyError("N43 actual writer environment not redirected: "+key)
            actual=PureWindowsPath(path);service=PureWindowsPath(str(lease.root))
            if actual!=service and service not in actual.parents:
                raise CachePolicyError("N43 actual writer environment escapes its existing service scope: "+key)
            row=reference_root_for_path(first,path)
            coverage.append(dict(writer_role=key,path=path,aggregate_root_role=row["role"]))
    return coverage


def _reference_directory_stat(path):
    """Existing no-follow lstat route; no directory handle held over live I/O."""
    import stat
    path=Path(path)
    for part in (path,*path.parents):
        info=part.lstat()
        if (not stat.S_ISDIR(info.st_mode) or getattr(info,"st_reparse_tag",0)
                or getattr(info,"st_file_attributes",0)&0x400):
            raise CachePolicyError("N43 observed non-directory/reparse ancestor refused")
    return path.lstat()


def _reference_stat_identity(info):
    return dict(st_dev=info.st_dev,st_ino=info.st_ino,
        creation_ns=getattr(info,"st_birthtime_ns",info.st_ctime_ns))


def validate_reference_roots(first):
    """Original immutable local admission, or original J1/RunIO SMB contract."""
    if first.reference_roots_validated:
        return
    from .windows import WindowsNamespace
    from dataclasses import asdict
    for row in first.reference_administration["owned_roots"]:
        if _reference_stat_identity(_reference_directory_stat(row["path"]))!=row["stat_identity"]:
            raise CachePolicyError("N43 actual root stat identity differs from its Root-bound namespace")
        if row["role"]=="job":
            from darsia.fluidflower.job_filesystem import inspect_shared_folder
            actual=inspect_shared_folder(row["path"])
            if not actual.allowed or actual.protocol!=row["identity"]["protocol"]:
                raise CachePolicyError("N43 original RunIO shared-folder admission differs")
            first.reference_transport_observation=asdict(actual)
        else:
            # Admit immutable root identity, then CLOSE before scanning live
            # descendants. Never hold shareREAD child handles over a rename.
            namespace=WindowsNamespace(row["path"],row["identity"])
            namespace.close()
    first.reference_roots_validated=True


def reference_scope_snapshot(first, lease):
    """No-follow length samples, identity before/after; not atomic/lifetime proof."""
    import stat
    from .capacity import volume_info
    from .bootstrap import _utc
    value=first.reference_administration
    total=entries=0;roots=[];volumes={};identities=set();terminal_metadata=[];anomalies=[]
    for row in value["owned_roots"]:
        root=Path(row["path"]);subtotal=0
        before=_reference_stat_identity(_reference_directory_stat(root))
        if before!=row["stat_identity"]:
            raise CachePolicyError("N43 actual aggregate root identity changed")
        def walk(directory):
            nonlocal entries,total,subtotal
            try:
                start=_reference_stat_identity(_reference_directory_stat(directory))
                with os.scandir(directory) as stream:
                    for entry in stream:
                        entries+=1
                        if entries>value["entry_limit"]:
                            raise CachePolicyError("N43 finite aggregate accounting entry limit exceeded")
                        try:
                            info=Path(entry.path).lstat()
                        except FileNotFoundError:
                            anomalies.append(dict(path=entry.path,state="VANISHED_DURING_SAMPLE"));continue
                        if getattr(info,"st_reparse_tag",0) or getattr(info,"st_file_attributes",0)&0x400:
                            raise CachePolicyError("N43 aggregate reparse entry refused")
                        if stat.S_ISDIR(info.st_mode):
                            walk(Path(entry.path))
                        else:
                            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:
                                raise CachePolicyError("N43 aggregate file type/hardlink refused")
                            identity=(info.st_dev,info.st_ino)
                            if identity in identities:
                                # A live directory rename can expose the same
                                # one-link file twice. Count once and mark the
                                # non-atomic sample, never call this a peak.
                                anomalies.append(dict(path=entry.path,state="IDENTITY_SEEN_TWICE_DURING_SAMPLE"));continue
                            identities.add(identity);total+=info.st_size;subtotal+=info.st_size
                            relative=Path(entry.path).relative_to(root)
                            if entry.name in {"candidate_completed.json","completed.json"} or (
                                    row["role"]=="job" and relative==Path("done")/"ac52.json"):
                                terminal_metadata.append(dict(path=entry.path,size_bytes=info.st_size,
                                    role="original terminal/candidate metadata; not Root reference acceptance"))
                if _reference_stat_identity(_reference_directory_stat(directory))!=start:
                    anomalies.append(dict(path=str(directory),state="DIRECTORY_IDENTITY_CHANGED_DURING_SAMPLE"))
            except FileNotFoundError:
                anomalies.append(dict(path=str(directory),state="DIRECTORY_MOVED_DURING_SAMPLE"))
        walk(root)
        after=_reference_stat_identity(_reference_directory_stat(root))
        if before!=after:
            raise CachePolicyError("N43 aggregate root changed during length sample")
        volume=volume_info(row["path"])
        if volume["volume_id"]!=row["volume_id"]:
            raise CachePolicyError("N43 actual owned-root volume identity changed")
        volumes[volume["volume_id"]]=volume
        roots.append(dict(role=row["role"],path=row["path"],identity=row["identity"],
            stat_identity_before=before,stat_identity_after=after,logical_bytes=subtotal,
            accounting_scope=("entire accepted shared host-ledger directory including retained legacy control bytes; conservative, not exclusive producer-only"
                if row["role"]=="host_ledger" else "entire explicitly inventoried root")))
    return dict(utc=_utc(),logical_bytes=total,entries=entries,roots=roots,volumes=list(volumes.values()),
        entitlement_bytes=REFERENCE_ENTITLEMENT_BYTES,all_stage_files_counted_once=True,
        sampled=True,non_atomic=True,complete_lifetime_peak=False,
        observed_scan_anomalies=anomalies,observed_bytes_are_lower_bound_if_scan_anomalies=bool(anomalies),
        hard_cap_exists=value["host_policy"]=="main-q87-fixed500",
        shared_job_transport=first.reference_transport_observation,
        actual_writer_path_coverage=reference_writer_coverage(first,lease),
        actual_terminal_metadata_files=terminal_metadata)


def _reference_stop_name(first):
    return "n43-stop-"+first._once_key()+".json"


@contextmanager
def _reference_lock(lease):
    if lease.ledger.lock_active:
        yield
    else:
        with lease.ledger.locked(create_binding=False):
            yield


def reference_stopped(first, lease):
    with _reference_lock(lease):
        return _reference_stop_name(first) in lease.ledger.control.names()


def require_reference_not_stopped(first, lease):
    if reference_stopped(first,lease):
        raise CachePolicyError("N43 persistent STOP: no completed/reference promotion; partial files retained")


def latch_reference_stop(first, lease, reason, snapshot):
    name=_reference_stop_name(first)
    with _reference_lock(lease):
        if name not in lease.ledger.control.names():
            lease.ledger.control.write_once(name,_encoded(dict(format="darsia-n43-reference-producer-stop-v1",
                request=first.binding,provider=first.provider_binding,scope=first.provider["scope"],
                reason=reason,observation=snapshot,no_retry=True,no_reference_promotion=True,
                partial_products_retained=True)))
    return request_existing_frame_boundary_stop(first,lease)


def fence_reference_publication(first, lease):
    """Read the same persistent STOP and fresh aggregate at existing I/O seams."""
    with lease.ledger.locked(create_binding=False):
        try:
            snapshot=reference_scope_snapshot(first,lease)
        except BaseException as error:
            latch_reference_stop(first,lease,"AGGREGATE_OBSERVATION_UNAVAILABLE",
                dict(error_type=type(error).__name__,message=str(error)))
            raise
        if snapshot["observed_scan_anomalies"]:
            latch_reference_stop(first,lease,"PUBLICATION_SCAN_INCOMPLETE",snapshot)
        if snapshot["logical_bytes"]>REFERENCE_ENTITLEMENT_BYTES:
            latch_reference_stop(first,lease,"AGGREGATE_ENTITLEMENT_EXCEEDED",snapshot)
        if any(v["free_bytes"]<=PHYSICAL_RESERVE_BYTES for v in snapshot["volumes"]):
            latch_reference_stop(first,lease,"OPERATIONAL_RESERVE_CROSSED",snapshot)
        require_reference_not_stopped(first,lease)
    return snapshot


class ReferencePublicationFence:
    """Delegate the actual publisher; refuse observed STOP before promotion."""
    def __init__(self, original, first, lease):
        self.original,self.first,self.lease=original,first,lease

    def __call__(self, *args, **kwargs):
        success=kwargs.get("success",args[2] if len(args)>2 else None)
        if type(success) is not bool:
            raise CachePolicyError("Original publication success boolean required")
        if success:
            fence_reference_publication(self.first,self.lease)
        result=self.original(*args,**kwargs)
        if success:
            # Original shared prepare files stay diagnostic on STOP.
            fence_reference_publication(self.first,self.lease)
        # A plain new descriptor dict: no mutation or delayed dict side effect.
        # Failed JSON retains the original traceback; it is diagnostic only.
        return dict(result,n43_reference_guard=reference_publication_binding(self.first,self.lease))

    def publish_progress(self, *args, **kwargs):
        fence_reference_publication(self.first,self.lease)
        return self.original.publish_progress(*args,**kwargs)


def reference_publication_binding(first, lease):
    return dict(format="darsia-n43-reference-publication-guard-v1",request=first.binding,
        provider=first.provider_binding,stores_source=first.stores_source,
        task=first.request["task"],attempt_id=first.task["attempt_id"],commit=first.task["commit"],
        job_root=next(row["path"] for row in first.reference_administration["owned_roots"] if row["role"]=="job"),
        namespace_admission=lease.document["cache_bootstrap"]["namespace_admission"],
        aggregate_namespace_sha256=_digest(first.reference_administration["owned_roots"]))


def check_reference_publication_guard(guard, job_root, task, *, success):
    """Initial and recovery publication need the same actual managed context."""
    if type(success) is not bool:
        raise CachePolicyError("Original publication success boolean required")
    # Ordinary records/default paths retain their old route, without opening a
    # stores/provider or silently bootstrapping any managed service.
    if guard is None and REQUEST_ENV not in os.environ:
        return None
    from .bootstrap import current_import_lease
    lease=current_import_lease()  # Missing managed context is a refusal, never fallback.
    first=getattr(lease,"first_qualification",None)
    if guard is None:
        if first is not None and first.reference_build:
            actual=next(row["path"] for row in first.reference_administration["owned_roots"] if row["role"]=="job")
            if _same_path(actual,job_root):
                raise CachePolicyError("N43 publication/recovery record is missing its typed guard")
        return None
    _exact_keys(guard,{"format","request","provider","stores_source","task","attempt_id",
        "commit","job_root","namespace_admission","aggregate_namespace_sha256"},
        "Exact N43 publication guard required")
    if (first is None or not first.reference_build or lease.entry_kind!="worker" or lease.purpose!="jobs"
            or guard!=reference_publication_binding(first,lease)
            or lease.source!=first.stores_source or task!=first.task
            or guard["attempt_id"]!=task.get("attempt_id") or guard["commit"]!=task.get("commit")
            or not _same_path(guard["job_root"],job_root)):
        raise CachePolicyError("N43 publication/recovery actual source/request/task/namespace differs")
    first._namespace(lease)
    first.require_task(lease,task)
    if success:
        return fence_reference_publication(first,lease)
    # Preserve the primary traceback in original failed.json even after STOP.
    # This is not scientific product, completed/done or reference publication.
    return dict(state="DIAGNOSTIC_FAILURE_ONLY",no_reference_or_profile_promotion=True)
