"""Synthetic read-only original-service context contracts.

Only parsed metadata, immutable bytes and fake namespace/mutex handles are used.
The canonical modules have inert imports; every native/host/path boundary is
replaced before resolution. No accepted production policy, namespace, reader
retirement, cache writer, deletion or service cleanup is qualified here.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

import darsia_cache.bootstrap as bootstrap
import darsia_cache.capacity as capacity
import darsia_cache.control as control
import darsia_cache.service_context as resolver
from darsia_cache.control import _encoded
from darsia_cache.policy import CachePolicyError, cache_settings
from darsia_cache.service_lifecycle import ServiceLifecycle


TOKEN = "a" * 32
PRIMARY = dict(machine="testonly-host", pid=101, process_create_time="TESTONLY_PRIMARY_EPOCH")
OBSERVER = dict(machine="testonly-host", pid=202, process_create_time="TESTONLY_OBSERVER_EPOCH")
PACKAGE = "C:/TESTONLY_NOT_CREATED/source/darsia/__init__.py"


def path_key(path):
    return str(PureWindowsPath(path)).casefold()


def descriptor(label):
    # A synthetic parsed-metadata boundary, never a claimed real file digest.
    return dict(path="C:/TESTONLY_NOT_CREATED/metadata/" + label,
                sha256=hashlib.sha256(label.encode()).hexdigest(), size_bytes=1)


class World:
    """All constructors below allocate Python objects only."""

    def __init__(self):
        self.events, self.namespaces, self.ledgers = [], [], []
        self.lock_depth = 0
        self.writes = []
        self.close_failure_path = None
        self.read_failure_name = None
        self.original_owner = deepcopy(PRIMARY)
        self.observer = deepcopy(OBSERVER)
        self.actual = dict(host="testonly-host", service_account_sid="TESTONLY_SID",
                           common_root="C:/TESTONLY_NOT_CREATED/authority")
        self.profile_pin = descriptor("profile.json")
        self.completion_pin = descriptor("completion.json")
        self.namespace_pin = descriptor("namespace.json")
        self.policy_pin = descriptor("stores.toml")
        self.token = TOKEN
        self.receipts_path = "C:/TESTONLY_NOT_CREATED/receipts"
        self.receipt_pin = descriptor("unused")
        self.receipt_pin["path"] = str(Path(self.receipts_path) / (TOKEN + ".import-service.json"))
        self.document = dict(
            cache=dict(root="C:/TESTONLY_NOT_CREATED/cache", quota_bytes=4096,
                       failed_retention_hours=48,
                       profiles={"jobs": deepcopy(self.profile_pin),
                                 "tests": deepcopy(self.profile_pin),
                                 "display": deepcopy(self.profile_pin)}),
            cache_bootstrap=dict(
                entry_roles=dict(runtime="interactive", worker="worker", tests="tests"),
                roles=dict(
                    interactive=dict(purpose="display", profile=deepcopy(self.profile_pin)),
                    worker=dict(purpose="jobs", profile=deepcopy(self.profile_pin)),
                    tests=dict(purpose="tests", profile=deepcopy(self.profile_pin))),
                namespace_admission=deepcopy(self.namespace_pin), accounting_entry_limit=32))
        self.policy = cache_settings(self.document, required=True)
        self.root = self.policy.purpose("jobs") / TOKEN
        self.scope_identity = {"TESTONLY_ID": "original-scope"}
        self.receipt_identity = {"TESTONLY_ID": "original-receipts"}
        self.authority_identity = {"TESTONLY_ID": "original-authority"}
        self.cache_identity = {"TESTONLY_ID": "original-cache"}
        self.admission = dict(
            host=self.actual["host"], authorized_sids=["TESTONLY_SID"],
            cache_identity=deepcopy(self.cache_identity),
            host_ledger=dict(path=self.actual["common_root"], identity=deepcopy(self.authority_identity)),
            receipts=dict(path=self.receipts_path, identity=deepcopy(self.receipt_identity)),
            purposes={"jobs": dict(path=str(self.policy.purpose("jobs")),
                                   identity={"TESTONLY_ID": "jobs-parent"})})
        self.profile = dict(
            format="darsia-cache-import-profile-v1", purpose="jobs",
            role_identifier="worker", entry_kind="worker",
            writer_contract="cooperative-start-admission-v1", maximum_cache_bytes=128,
            completion=deepcopy(self.completion_pin))
        self.completion = dict(complete=True, passed=True)
        self.pool = hashlib.sha256((self.actual["host"] + "\0cache-host-pool-v2").encode()).hexdigest()
        self.reservation_row = dict(
            format="darsia-cache-budget-reservation-v2", pool=self.pool, utc="2026-01-01T00:00:00Z",
            owner=deepcopy(PRIMARY), token=TOKEN, maximum_bytes=128)
        self.reservation_raw = _encoded(self.reservation_row)
        self.reservation = dict(token=TOKEN, pool=self.pool,
                                sha256=hashlib.sha256(self.reservation_raw).hexdigest())
        self.manifest = dict(
            format="darsia-cache-external-import-scope-v1", token=TOKEN,
            owner=deepcopy(PRIMARY), purpose="jobs", root=str(self.root),
            root_identity=deepcopy(self.scope_identity), maximum_bytes=128,
            entry_kind="worker", role_identifier="worker", reservation=deepcopy(self.reservation))
        self.receipt = dict(
            format="darsia-cache-import-service-receipt-v1",
            policy=deepcopy(self.policy_pin), profile=deepcopy(self.profile_pin),
            namespace=deepcopy(self.namespace_pin), entry_kind="worker", role_identifier="worker",
            manifest=self.manifest, directories={"": deepcopy(self.scope_identity),
                                               "tmp": {"TESTONLY_ID": "original-tmp"}})
        self.policy_source = deepcopy(self.policy_pin)
        self.metadata = {
            path_key(self.receipt_pin["path"]): (self.receipt_pin, self.receipt),
            path_key(self.profile_pin["path"]): (self.profile_pin, self.profile),
            path_key(self.completion_pin["path"]): (self.completion_pin, self.completion),
            path_key(self.namespace_pin["path"]): (self.namespace_pin, self.admission)}
        self.host_record = dict(
            format="darsia-one-host-cache-binding-v2", host=self.actual["host"],
            authorized_sids=["TESTONLY_SID"], cache_root=self.policy.root,
            cache_identity=deepcopy(self.cache_identity), quota_bytes=4096)
        self.records = {
            "cache-" + self.pool + ".json": _encoded(self.host_record),
            "cache-r-" + TOKEN + ".json": self.reservation_raw,
            "cache-s-" + TOKEN + ".json": _encoded(self.manifest)}
        self.directory_ids = {
            path_key(self.receipts_path): deepcopy(self.receipt_identity),
            path_key(self.root): deepcopy(self.scope_identity),
            path_key(self.actual["common_root"]): deepcopy(self.authority_identity)}
        self.reservation_checks = []

    def outside(self, path, package_file):
        assert package_file == PACKAGE
        self.events.append(("outside", path_key(path)))
        return Path(path)

    def read_bound(self, binding):
        self.events.append(("metadata-read", path_key(binding["path"])))
        expected, value = self.metadata[path_key(binding["path"])]
        if binding != expected:
            raise CachePolicyError("TESTONLY immutable metadata descriptor differs")
        return deepcopy(value), b"TESTONLY_PARSED_BOUNDARY"

    def explicit_document(self, path, package_file):
        assert path_key(path) == path_key(self.policy_pin["path"]) and package_file == PACKAGE
        self.events.append(("policy-read", path_key(path)))
        return deepcopy(self.document), deepcopy(self.policy_source)

    def observe(self):
        self.events.append(("actual-observer",))
        return deepcopy(self.observer), deepcopy(self.actual)

    def sync_manifest_record(self):
        self.records["cache-s-" + TOKEN + ".json"] = _encoded(self.manifest)

    def replace_reservation_owner(self, owner):
        row = {**self.reservation_row, "owner": deepcopy(owner)}
        raw = _encoded(row)
        self.records["cache-r-" + TOKEN + ".json"] = raw
        self.manifest["reservation"] = dict(self.reservation, sha256=hashlib.sha256(raw).hexdigest())
        self.sync_manifest_record()

    def assert_no_mutation(self):
        assert self.writes == []
        assert not any(event[0] in {"producer", "ImportLease"} for event in self.events)

    def assert_closed(self):
        assert self.lock_depth == 0
        assert all(namespace.closed for namespace in self.namespaces)
        assert all(ledger.control is None and not ledger.lock_active for ledger in self.ledgers)


@pytest.fixture
def world(monkeypatch):
    w = World()

    class Namespace:
        def __init__(self, path, expected):
            self.path, self.closed, self.close_attempted = path_key(path), False, False
            w.events.append(("open-existing", self.path, deepcopy(expected)))
            if w.directory_ids.get(self.path) != expected:
                raise CachePolicyError("TESTONLY existing directory identity differs")
            w.namespaces.append(self)

        def names(self):
            assert not self.closed and w.lock_depth == 1
            w.events.append(("names", self.path))
            return list(w.records)

        def read_once(self, name, maximum=1024**2):
            assert not self.closed and w.lock_depth == 1
            w.events.append(("control-read", name, maximum))
            if name == w.read_failure_name:
                raise CachePolicyError("TESTONLY control read failure")
            raw = w.records[name]
            assert len(raw) <= maximum
            return raw, {"TESTONLY_NOT_A_REAL_FILE_ID": True}

        def write_once(self, name, raw, maximum=1024**2):
            assert not self.closed and w.lock_depth == 1
            assert isinstance(raw, bytes) and len(raw) <= maximum
            if name in w.records:
                raise FileExistsError("TESTONLY immutable record already exists")
            w.writes.append((name, raw, maximum))
            w.records[name] = raw

        def scope(self, *args, **kwargs):
            w.events.append(("producer", "scope/create"))
            raise AssertionError("Resolver must never create/open a producer scope")

        def close(self):
            self.close_attempted = True
            w.events.append(("close", self.path))
            if self.path == w.close_failure_path:
                raise CachePolicyError("TESTONLY reported close failure; actual closure unknown")
            self.closed = True

    @contextmanager
    def mutex(path, key):
        assert path_key(path) == path_key(w.actual["common_root"])
        assert key == w.pool and w.lock_depth == 0
        w.events.append(("mutex-enter",))
        w.lock_depth += 1
        try:
            yield
        finally:
            w.lock_depth -= 1
            w.events.append(("mutex-exit",))

    def forbidden_lease(*args, **kwargs):
        w.events.append(("ImportLease",))
        raise AssertionError("Resolver must not start/attach/register a producer")

    original_init = control.CacheBudgetLedger.__init__
    original_require = control.CacheBudgetLedger.require_reservation

    def ledger_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        w.ledgers.append(self)

    def require(self, reservation, owner, needed):
        w.reservation_checks.append((deepcopy(reservation), deepcopy(owner), needed))
        return original_require(self, reservation, owner, needed)

    monkeypatch.setattr(resolver, "_outside", w.outside)
    monkeypatch.setattr(resolver, "read_bound", w.read_bound)
    monkeypatch.setattr(resolver, "_explicit_document", w.explicit_document)
    monkeypatch.setattr(resolver, "_owner", w.observe)
    monkeypatch.setattr(resolver, "WindowsNamespace", Namespace)
    monkeypatch.setattr(control, "WindowsNamespace", Namespace)
    monkeypatch.setattr(control, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(control.CacheBudgetLedger, "__init__", ledger_init)
    monkeypatch.setattr(control.CacheBudgetLedger, "require_reservation", require)
    monkeypatch.setattr(capacity, "_capacity_host_context", lambda: deepcopy(w.actual))
    monkeypatch.setattr(capacity, "_LocalVolumeLock", mutex)
    monkeypatch.setattr(bootstrap, "ImportLease", forbidden_lease)
    return w


def resolve(world):
    return resolver.resolve_original_service_context(deepcopy(world.receipt_pin), package_file=PACKAGE)


def test_context_keeps_original_owner_and_uses_distinct_actual_observer_once(world):
    context = resolve(world)
    assert context.owner == OBSERVER and context.manifest["owner"] == PRIMARY
    assert context.reservation == world.reservation
    assert world.reservation_checks == [(world.reservation, PRIMARY, 128)]
    assert context.root == world.root and context.maximum == 128 and context.entry_limit == 32
    assert context.resolver_is_not_retirement_proof is True
    assert len(context.original_metadata) == 5
    reads = [event[1] for event in world.events if event[0] in {"metadata-read", "policy-read"}]
    assert len(reads) == len(set(reads)) == 5
    # Receipt/scope ID checks open existing handles but never enumerate payloads.
    assert [event[1] for event in world.events if event[0] == "names"] == [
        path_key(world.actual["common_root"]), path_key(world.actual["common_root"])]
    world.assert_no_mutation()
    world.assert_closed()
    context.owner["pid"] = 999
    context.manifest["owner"]["pid"] = 888
    assert world.observer == OBSERVER and world.manifest["owner"] == PRIMARY


@pytest.mark.parametrize("operation", ["read", "remember"])
def test_once_metadata_cache_refuses_conflicting_same_path_descriptor(world, operation):
    once = resolver._MetadataOnce(PACKAGE)
    first = once.read(world.profile_pin)
    assert once.read(deepcopy(world.profile_pin)) == first
    first["purpose"] = "changed returned copy"
    assert once.read(world.profile_pin)["purpose"] == "jobs"
    conflict = {**world.profile_pin, "sha256": "f" * 64}
    with pytest.raises(CachePolicyError):
        if operation == "read":
            once.read(conflict)
        else:
            once.remember(conflict, {})
    assert sum(event[0] == "metadata-read" for event in world.events) == 1
    world.assert_no_mutation()


@pytest.mark.parametrize("case", [
    "policy-source", "policy-quota", "profile-binding", "namespace-binding",
    "profile-purpose", "profile-completion", "original-host", "live-primary", "actual-account",
])
def test_wrong_original_source_policy_profile_namespace_or_actor_refuses(world, case):
    if case == "policy-source":
        world.policy_source["sha256"] = "f" * 64
    elif case == "policy-quota":
        world.document["cache"]["quota_bytes"] = 0
    elif case == "profile-binding":
        world.receipt["profile"] = descriptor("wrong-profile.json")
    elif case == "namespace-binding":
        world.receipt["namespace"] = descriptor("wrong-namespace.json")
    elif case == "profile-purpose":
        world.profile["purpose"] = "tests"
    elif case == "profile-completion":
        world.completion["passed"] = False
    elif case == "original-host":
        world.manifest["owner"]["machine"] = "other-host"
    elif case == "live-primary":
        world.observer = deepcopy(PRIMARY)
    elif case == "actual-account":
        world.actual["service_account_sid"] = "OTHER_TESTONLY_SID"
    with pytest.raises(CachePolicyError):
        resolve(world)
    world.assert_no_mutation()
    world.assert_closed()


@pytest.mark.parametrize("case", ["missing-host-binding", "wrong-host-binding", "reservation-owner", "manifest"])
def test_only_existing_exact_host_reservation_and_manifest_are_accepted(world, case):
    if case == "missing-host-binding":
        del world.records["cache-" + world.pool + ".json"]
    elif case == "wrong-host-binding":
        world.records["cache-" + world.pool + ".json"] = _encoded({**world.host_record, "quota_bytes": 1})
    elif case == "reservation-owner":
        world.replace_reservation_owner(OBSERVER)
    elif case == "manifest":
        world.records["cache-s-" + TOKEN + ".json"] = _encoded({**world.manifest, "maximum_bytes": 127})
    with pytest.raises(CachePolicyError):
        resolve(world)
    world.assert_no_mutation()
    world.assert_closed()
    assert any(event[0] == "mutex-exit" for event in world.events)


@pytest.mark.parametrize("case", ["scope-id", "receipt-path", "control-read"])
def test_resolution_failure_closes_all_fresh_handles_without_allocating(world, case):
    if case == "scope-id":
        world.directory_ids[path_key(world.root)] = {"TESTONLY_ID": "replaced-scope"}
    elif case == "receipt-path":
        world.receipt_pin["path"] = "C:/TESTONLY_NOT_CREATED/elsewhere/service.json"
        world.metadata[path_key(world.receipt_pin["path"])] = (world.receipt_pin, world.receipt)
    elif case == "control-read":
        world.read_failure_name = "cache-s-" + TOKEN + ".json"
    with pytest.raises(CachePolicyError):
        resolve(world)
    world.assert_no_mutation()
    world.assert_closed()


def test_reported_close_failure_propagates_without_claiming_handle_closure(world):
    world.close_failure_path = path_key(world.receipts_path)
    with pytest.raises(CachePolicyError, match="close failure"):
        resolve(world)
    assert len(world.namespaces) == 1 and world.namespaces[0].close_attempted
    assert not world.namespaces[0].closed  # Failure means closure unknown, never synthetic PASS.
    world.assert_no_mutation()
    assert world.lock_depth == 0


@pytest.mark.parametrize("operation", ["reserve", "admit", "bind_scope", "close", "measure"])
def test_observer_has_no_producer_allocation_or_budget_closure_capability(world, operation):
    context = resolve(world)
    with pytest.raises(CachePolicyError):
        getattr(context.ledger, operation)()
    assert not hasattr(context.ledger, "release")
    assert not hasattr(context, "start") and not hasattr(context, "attach") and not hasattr(context, "attach_owner")
    world.assert_no_mutation()
    world.assert_closed()


def test_observer_lock_refuses_initialization_before_open_and_restores_view_on_failure(world):
    context = resolve(world)
    before = len(world.namespaces)
    with pytest.raises(CachePolicyError):
        with context.ledger.locked(create_binding=True):
            pytest.fail("Read-only observer initialized a binding")
    assert len(world.namespaces) == before
    with pytest.raises(RuntimeError, match="TESTONLY caller failure"):
        with context.ledger.locked():
            assert isinstance(context.ledger.control, resolver._EligibilityMetadataOnly)
            raise RuntimeError("TESTONLY caller failure")
    world.assert_no_mutation()
    world.assert_closed()


def test_only_explicit_v11_eligibility_evidence_may_use_actual_observer_actor(world):
    context = resolve(world)
    lifecycle = ServiceLifecycle(context)
    original_records = deepcopy(world.records)
    value = dict(observer=deepcopy(OBSERVER), cleanup_authorized=False, service_budget_closed=False,
                 TESTONLY_NOT_RETIREMENT_OR_CLEANUP_PROOF=True)
    # Exercise the V11 writer API only; no retention provider or seal protocol is repeated.
    key = hashlib.sha256(_encoded(value)).hexdigest()
    with context.ledger.locked():
        record = lifecycle._put({}, "eligibility", key, value, actor=deepcopy(OBSERVER))
    assert len(world.writes) == 1
    name, raw, maximum = world.writes[0]
    assert name == "cache-l-" + TOKEN + "-eligibility-" + key + ".json"
    assert record["sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["size_bytes"] == len(raw) and maximum == 1024**2
    assert context.manifest["owner"] == PRIMARY
    assert all(world.records[name] == raw for name, raw in original_records.items())
    world.assert_closed()


@pytest.mark.parametrize("case", ["original-actor", "wrong-value-observer", "producer-record", "wrong-key"])
def test_observer_view_refuses_wrong_actor_and_non_eligibility_writes(world, case):
    context = resolve(world)
    lifecycle = ServiceLifecycle(context)
    value = dict(observer=deepcopy(OBSERVER), cleanup_authorized=False, service_budget_closed=False)
    actor, kind = deepcopy(OBSERVER), "eligibility"
    if case == "original-actor":
        actor = deepcopy(PRIMARY)
    elif case == "wrong-value-observer":
        value["observer"] = deepcopy(PRIMARY)
    elif case == "producer-record":
        kind = "owner"
    key = hashlib.sha256(_encoded(value)).hexdigest()
    if case == "wrong-key":
        key = "f" * 64
    with pytest.raises(CachePolicyError):
        with context.ledger.locked():
            lifecycle._put({}, kind, key, value, actor=actor)
    world.assert_no_mutation()
    world.assert_closed()




def test_typed_authorities_cache_uses_actual_reader_once_and_returns_copies(tmp_path,monkeypatch):
    path=tmp_path/"answers.json";raw=b'[{"id":"N43","decided_by":"olav"}]\n'
    path.write_bytes(raw)
    pin=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
    once=resolver._MetadataOnce(str(tmp_path/"package"/"darsia"/"__init__.py"))
    original=resolver._exact_authority_list;calls=[]
    def actual(binding,package_file):
        calls.append(deepcopy(binding));return original(binding,package_file)
    monkeypatch.setattr(resolver,"_exact_authority_list",actual)
    value=once.read_authorities(pin);value[0]["decided_by"]="changed copy"
    assert once.read_authorities(pin)==[{"id":"N43","decided_by":"olav"}]
    assert calls==[pin]
    assert once.seen[path_key(path)]==(pin,[{"id":"N43","decided_by":"olav"}])
    with pytest.raises(CachePolicyError,match="descriptor changed"):
        once.read_authorities({**pin,"sha256":"f"*64})
    assert calls==[pin]


@pytest.mark.parametrize("document",[{},[None],[1]])
def test_typed_authorities_cache_cannot_relabel_an_object_or_invalid_rows(document):
    once=resolver._MetadataOnce(PACKAGE);pin=descriptor("answers.json")
    once.remember(pin,document)
    with pytest.raises(CachePolicyError,match="list of record objects"):
        once.read_authorities(pin)


@pytest.mark.parametrize("new_format", [
    "darsia-cache-volume-bound-import-profile-v1",
    "darsia-cache-volume-bound-job-profile-v1",
])
def test_direct_new_typed_profile_refuses_before_observer_or_ledger(world, monkeypatch, new_format):
    world.profile["format"] = new_format
    def forbidden_owner():
        raise AssertionError("Unresolved direct typed profile reached actual observer construction")
    monkeypatch.setattr(resolver, "_owner", forbidden_owner)
    with pytest.raises(CachePolicyError, match="original checked provider DAG"):
        resolver.resolve_original_service_context(world.receipt_pin, package_file=PACKAGE)
    assert world.ledgers == []
    assert not any(event[0] == "actual-observer" for event in world.events)
    world.assert_no_mutation()
    world.assert_closed()
