"""Explicit Setup claims within the original admitted worker and volume locks.

The existing service keeps its native ownership and cache reservation. Setup
adds whole output ceilings, not a measured writer profile or a new cache scope.
Claim controls spend the same metadata allowance as subsequent operation IO.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
import os
from pathlib import Path
import re

from .job_queue import name
from .series_setup_admission import prepare_setup_admission
from .series_setup_protocol import validate_setup_claim
from .series_setup_sink import SetupMetadataSink, _claim_sink
from .store_runtime import VerifiedInputs


def current_claim_sink():
    """Return only the active, finite original claim-control writer."""
    return _claim_sink.get()


def retirement_allowance(limits):
    """Reserve one finite supervisor slice inside the original metadata cap."""
    return min(64 * 1024, limits["maximum_file_bytes"],
               (limits["maximum_output_bytes"] + 1) // 2)


class _RetirementControlSink(SetupMetadataSink):
    """One exact supervisor file and its retained immutable-publisher partials."""

    def __init__(self, task, config, output, owner):
        """Keep the same cumulative supervisor allowance across retries."""
        self.task, self.owner = deepcopy(task), deepcopy(owner)
        self.output = Path(output)
        self.final = self.output / "retirement.json"
        observed = self.output.stat()
        self.directory_identity = observed.st_dev, observed.st_ino
        self.maximum = retirement_allowance(config["metadata_output"])
        self.maximum_file = self.maximum
        self.admission = self
        self.temporaries = {}
        self.written_bytes, self.files, self.open_writers = 0, {}, 0
        self.closed = False

    def require_write(self, path):
        """Permit only the original owned attempt and its named retirement file."""
        from ..rig_raw_write import _canonical
        selected = Path(path)
        observed = self.output.stat()
        if (_canonical(self.output) != self.output
                or (observed.st_dev, observed.st_ino) != self.directory_identity
                or _canonical(selected) != selected):
            raise ValueError("The supervisor retirement output changed its original location")
        if selected not in {self.output, self.final} and selected not in self.temporaries:
            raise ValueError("The supervisor writer permits only its exact retirement control")
        return selected

    def register_temporary(self, final_path, temporary_path):
        """Register only an original immutable-publisher retirement partial."""
        final, temporary = Path(final_path), Path(temporary_path)
        self.require_write(final)
        if (final != self.final or temporary.parent != self.output
                or re.fullmatch(r"\.retirement\.json\.[0-9a-f]{32}\.partial", temporary.name) is None):
            raise ValueError("The temporary is outside the exact supervisor retirement control")
        self.temporaries[temporary] = final
        self.require_write(temporary)

    @contextmanager
    def open_exclusive(self, path):
        """Open only a registered partial or the exact final retirement file."""
        selected = self.require_write(path)
        if selected != self.final and selected not in self.temporaries:
            raise ValueError("The supervisor writer cannot open this control path")
        with super().open_exclusive(selected) as stream:
            yield stream

    def mkdir(self, path, *, parents=False, exist_ok=False):
        """Retain the original publisher's existing-parent directory operation."""
        self._require_open()
        selected = self.require_write(path)
        if selected != self.output:
            raise ValueError("The supervisor writer cannot create another directory")
        return selected.mkdir(parents=parents, exist_ok=exist_ok)

    def rename(self, source, destination, *, replace=False, rename_operation=None):
        """Publish only a registered partial with the original atomic move."""
        source, destination = self.require_write(source), self.require_write(destination)
        if replace or self.temporaries.get(source) != destination or destination != self.final:
            raise ValueError("The move is outside the exact supervisor retirement control")
        return super().rename(source, destination, rename_operation=rename_operation)


def retirement_control_writer(root, task, stores, output, owner):
    """Bind the finite parent writer using only original control identities."""
    from .capacity import configuration_for_task, read_bound
    from .series_setup_admission import _reservation_pin
    from .worker_io import retirement_output
    from ..rig_raw_write import _canonical
    root = Path(root).resolve()
    job_path = root / "job.json"
    size, digest = _reservation_pin(job_path)
    job, _raw = read_bound(dict(path=str(job_path), size_bytes=size, sha256=digest))
    config = configuration_for_task(stores, job["series_id"], task)
    validate_setup_claim(task, job, config=config)
    actual, original_owner = retirement_output(root, task,
        name(task["target_machine"] + "-" + task["backend"]), stores)
    if actual != Path(output) or original_owner != owner:
        raise ValueError("The supervisor writer differs from the original claim output or owner")
    if not any(actual.is_relative_to(_canonical(path)) for path in config["output_scope"]["metadata_roots"]):
        raise ValueError("The original retirement output is outside its declared metadata allowance")
    return _RetirementControlSink(task, config, actual, original_owner)


class _ClaimControlSink(SetupMetadataSink):
    """Spend one metadata cap before the actual claim can bind reservations."""

    def __init__(self, helper):
        self.helper, self.admission = helper, self
        limits = helper.admission.metadata_limits
        self.maximum, self.maximum_file = limits["maximum_output_bytes"], limits["maximum_file_bytes"]
        self.maximum -= helper.retirement_maximum
        self.written_bytes, self.files, self.open_writers = 0, {}, 0
        self.closed = False
        root, task = Path(helper.runtime.root).resolve(), helper.task
        run, token = name(task["run"]), task["attempt_id"]
        self.pending = root / "queue" / (run + ".json")
        self.claimed = root / "claimed" / helper.runtime.service_id / (run + ".json")
        self.lease = root / "leases" / (run + "." + token)
        output = helper.runtime.output_factory(task, token)
        self.output = Path(output.path).resolve()
        self.finals = {self.lease / "claimed.json", self.lease / "owner.json"}
        self.temporaries = {}
        self.directories = {self.claimed.parent, self.lease, self.output}
        for leaf in tuple(self.directories):
            current = leaf.parent
            while any(current.is_relative_to(Path(item))
                      for item in helper.config["output_scope"]["metadata_roots"]):
                self.directories.add(current)
                if current == current.parent:
                    break
                current = current.parent

    def require_write(self, path):
        """Check exact controls and their captured positive output domains."""
        if current_claim_sink() is not self or self.helper._stack is None:
            raise RuntimeError("Setup claim controls need their active locked scope")
        from ..rig_raw_write import _no_links
        selected = Path(path).expanduser()
        _no_links(selected)
        selected = selected.resolve()
        if (selected not in self.finals and selected not in self.temporaries
                and selected not in self.directories
                and selected not in {self.pending, self.claimed}):
            raise ValueError("The path is outside the finite Setup claim controls")
        if not any(selected.is_relative_to(Path(root))
                   for root in self.helper.config["output_scope"]["metadata_roots"]):
            raise ValueError("A Setup claim control is outside its exact metadata roots")
        self.helper.admission._revalidate_domains()
        return selected

    def register_temporary(self, final_path, temporary_path):
        """Register one exact original publisher partial before its creation."""
        final, temporary = Path(final_path).resolve(), Path(temporary_path).resolve()
        self.require_write(final)
        if (final not in self.finals or temporary.parent != final.parent
                or re.fullmatch(r"\." + re.escape(final.name) + r"\.[0-9a-f]{32}\.partial",
                                temporary.name) is None):
            raise ValueError("The temporary is not an original named claim-control partial")
        self.temporaries[temporary] = final
        return temporary

    @contextmanager
    def open_exclusive(self, path):
        """Open only the two named controls or their registered retry partials."""
        selected = self.require_write(path)
        if selected not in self.finals and selected not in self.temporaries:
            raise ValueError("Setup claim controls do not authorize this file")
        with super().open_exclusive(selected) as stream:
            yield stream

    def mkdir(self, path, *, parents=False, exist_ok=False):
        """Create only original claim, lease and attempt directory paths."""
        selected = self.require_write(path)
        if selected not in self.directories:
            raise ValueError("Setup claim controls do not authorize this directory")
        return selected.mkdir(parents=parents, exist_ok=exist_ok)

    def rename(self, source, destination, *, replace=False, rename_operation=None):
        """Keep exact original queue moves and named partial publication."""
        source, destination = self.require_write(source), self.require_write(destination)
        queue_move = (source, destination) in {
            (self.pending, self.claimed), (self.claimed, self.pending)}
        if replace or not (queue_move or self.temporaries.get(source) == destination):
            raise ValueError("The move is outside the original Setup claim controls")
        (os.rename if rename_operation is None else rename_operation)(source, destination)
        if str(source) in self.files:
            self.files[str(destination)] = self.files.pop(str(source))


class SetupWorkerAdmission:
    """Hold all output locks and the genuine service budget through claim bind."""

    def __init__(self, runtime, task, config, owner):
        from .capacity import CapacityLedger
        from darsia_cache.control import CacheBudgetLedger
        from darsia_cache.policy import PHYSICAL_RESERVE_BYTES
        from darsia_cache.worker_adapter import WorkerCacheAdapter
        validate_setup_claim(task, runtime.job, config=config)
        if (task["target_machine"] != runtime.stores.machine_id
                or config["target_machine"] != runtime.stores.machine_id
                or runtime.service_id != name(runtime.stores.machine_id + "-" + runtime.backend)
                or owner.get("machine") != runtime.service_id
                or runtime.backend != task["backend"] or runtime.commit != task["commit"]):
            raise ValueError("Setup needs its exact configured worker source, target and owner")
        verified = runtime.setup_verified_inputs.get(task["attempt_id"])
        if (not isinstance(verified, VerifiedInputs) or verified.run != task["run"]
                or verified.metadata("run.configuration") != config
                or [row.reference for row in verified.rows] !=
                   [row["reference"] for row in task["input_catalog"]["inputs"]]
                or [row.uses for row in verified.rows] !=
                   [tuple(row["uses"]) for row in task["input_catalog"]["inputs"]]):
            raise ValueError("Setup admission needs its exact verified control catalogue")
        self.runtime, self.task, self.config, self.owner = runtime, deepcopy(task), deepcopy(config), deepcopy(owner)
        self.adapter = runtime.cache_adapter
        if (not isinstance(self.adapter, WorkerCacheAdapter)
                or self.adapter.verify_preimport(runtime.stores, self.adapter.lease.policy) is not True
                or not self.adapter._is_current_claim_owner(owner)):
            raise ValueError("Setup requires its original pre-import worker service and actual owner")
        self.lease = self.adapter.lease
        scope = config["output_scope"]
        self.admission = prepare_setup_admission(runtime.stores, runtime.job["series_id"], task, config,
            metadata_roots=scope["metadata_roots"], scratch_roots=scope["scratch_roots"])
        result_capacity = CapacityLedger(runtime.stores, runtime.job["series_id"],
                                        runtime.stores.roots(runtime.job["series_id"])["results"])
        if any(ledger.reserve < PHYSICAL_RESERVE_BYTES for ledger, _estimate in self.admission._scopes):
            raise ValueError("Keep the original physical reserve on every Setup output volume")
        self.cache_ledger = self.adapter.budget_ledger(result_capacity)
        if (not isinstance(self.cache_ledger, CacheBudgetLedger)
                or self.cache_ledger is not self.lease.ledger
                or self.cache_ledger.policy != self.lease.policy
                or self.cache_ledger.admission != self.adapter.namespace_admission):
            raise ValueError("Setup must reuse the original admitted service cache ledger")
        self._stack, self.bound = None, False
        self.sink = None
        self.retirement_maximum = retirement_allowance(self.admission.metadata_limits)
        self.preclaim_sink = _ClaimControlSink(self)
        self.reports, self.cache_report = (), None

    def acquire(self):
        """Use stable output-volume lock order followed by the service lock."""
        if self._stack is not None or self.bound:
            raise RuntimeError("Setup admission cannot acquire twice")
        stack = ExitStack()
        try:
            stack.enter_context(self.admission.locked())
            stack.enter_context(self.cache_ledger.locked())
        except BaseException:
            stack.close()
            raise
        self._stack = stack

    def check(self):
        """Count retained output caps and actual unwritten service allocations."""
        if self._stack is None or self.bound:
            raise RuntimeError("Setup needs its held preclaim admission locks")
        if not self.adapter._is_current_claim_owner(self.owner):
            raise ValueError("Setup admission lost its actual worker owner")
        self.reports = self.admission.check()
        self.cache_ledger.require_reservation(self.lease.reservation,
                                             self.lease.manifest["owner"], self.lease.maximum)
        from darsia_cache.capacity import volume_info
        cache_volume = volume_info(self.lease.policy.root)
        same = {ledger.key for ledger, estimate in self.admission._scopes
                if ledger.remote_scope is None and estimate.volume_id == cache_volume["volume_id"]}
        noncache = sum(row["maximum_output_bytes"] + row["other_open_bytes"]
                       for row in self.reports if row["capacity_key"] in same)
        # This is an existing-budget projection, not a zero scratch profile or
        # a new cache reservation. Explicit operation scratch spends sink caps.
        self.cache_report = self.cache_ledger.project(0, noncache_bytes=noncache)
        if self.cache_report["sufficient"] is not True:
            raise ValueError("Setup output caps exceed the existing service cache-volume headroom")
        return self.reports

    @contextmanager
    def preclaim_scope(self, task, machine):
        """Bound only original atomic claim controls after all locked checks."""
        if task != self.task or machine != self.runtime.service_id or current_claim_sink() is not None:
            raise ValueError("Setup claim-control scope differs from its admitted candidate")
        self.check()
        token = _claim_sink.set(self.preclaim_sink)
        try:
            yield self.preclaim_sink
        finally:
            _claim_sink.reset(token)

    def bind(self, claim):
        """Bind the exact real claim and retain every failed volume prefix."""
        if claim.task != self.task or claim.token != self.task["attempt_id"]:
            raise ValueError("Setup bind differs from the exact admitted claim")
        if self.preclaim_sink.open_writers:
            raise RuntimeError("Setup claim-control streams remain open")
        self.check()
        self.admission.bind(self.runtime.root, claim, self.owner)
        self.bound = True
        self.sink = SetupMetadataSink(self.admission)
        self.sink.maximum -= self.retirement_maximum
        self.sink.written_bytes = self.preclaim_sink.written_bytes
        self.sink.files = dict(self.preclaim_sink.files)
        return self.admission.bound_reservations

    def unlock(self):
        """Close admission locks without releasing any output reservation."""
        if self._stack is not None:
            stack, self._stack = self._stack, None
            stack.close()


def setup_claim_admission(runtime, task, config, owner):
    """Prepare one exact verified Setup candidate without importing science."""
    return SetupWorkerAdmission(runtime, task, config, owner)


def release_retired_setup_capacity(root, task, stores, retirement):
    """Close exact retained volume reservations using original retirement proof.

    Only the named operation and, for Build, its pinned request metadata are
    reopened. Scientific input files and the retained RAW destination need not
    be available or fresh. Missing controls leave all reservations untouched.
    """
    from .capacity import CapacityLedger, _reservation_id, configuration_for_task, read_bound
    from .series_setup_admission import _reservation_pin, validate_retained_reservation
    from ..rig_raw_write import _canonical

    root = Path(root).resolve()
    job_path = root / "job.json"
    size, digest = _reservation_pin(job_path)
    job, _raw = read_bound(dict(path=str(job_path), size_bytes=size, sha256=digest))
    config = configuration_for_task(stores, job["series_id"], task)
    validate_setup_claim(task, job, config=config)
    scope = config["output_scope"]
    roots = [("metadata", path) for path in scope["metadata_roots"]]
    roots.extend(("scratch", path) for path in scope["scratch_roots"])
    if task["operation"] == "build":
        request, _raw = read_bound(config["raw_build_request"])
        if (request.get("format") != "darsia-raw-rig-build-request-v1"
                or request.get("run_id") != task["run"]
                or not isinstance(request.get("output"), dict)):
            raise ValueError("Retired Setup request differs from its exact Build operation")
        roots.append(("raw", request["output"]["intended_root"]))
    pools = {}
    for role, locator in roots:
        path = Path(locator)
        if _canonical(locator) != path:
            raise ValueError("Retired Setup output root changed its physical location")
        ledger = CapacityLedger(stores, job["series_id"], path)
        pool = pools.setdefault(ledger.key, dict(ledger=ledger, roots=[]))
        if pool["ledger"].volume["volume_id"] != ledger.volume["volume_id"]:
            raise ValueError("Retired Setup capacity scope names several actual volumes")
        declared = dict(role=role, path=str(path))
        if declared not in pool["roots"]:
            pool["roots"].append(declared)
    retained = []
    for key in sorted(pools):
        pool, path = pools[key], pools[key]["ledger"].root / ("r-" + _reservation_id(root, task))
        reserved = path / "reserved.json"
        if not reserved.exists():
            continue  # An unbound suffix of a failed claim has no reservation.
        size, digest = _reservation_pin(reserved)
        row, _raw = read_bound(dict(path=str(reserved), size_bytes=size, sha256=digest))
        ledger = pool["ledger"]
        validate_retained_reservation(row, capacity_key=ledger.key,
                                      volume_id=ledger.volume["volume_id"])
        estimate = row["estimate"]
        if (row["run"] != task["run"] or row["attempt_id"] != task["attempt_id"]
                or Path(row["job_root"]).resolve() != root
                or row["ordered_case_ids"] != task["ordered_case_ids"]
                or estimate["source_commit"] != task["commit"]
                or estimate["setup_plan"] != config["setup_plan"]
                or estimate["raw_build_request"] != config.get("raw_build_request")
                or estimate["roots"] != sorted(pool["roots"], key=lambda item: (item["role"], item["path"]))):
            raise ValueError("Retained Setup reservation differs from its exact operation controls")
        retained.append((ledger, path))
    with ExitStack() as locks:
        for ledger, _path in retained:
            locks.enter_context(ledger.locked(create=False))
        for ledger, path in retained:
            ledger.release(path, retirement["supervisor"], retirement=retirement)
    return tuple(str(path) for _ledger, path in retained)
