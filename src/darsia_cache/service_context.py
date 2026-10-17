"""Resolve an original service from exact metadata, never from dead handles.

Resolution is read-only. It neither starts/attaches a producer nor establishes
retirement or ownership of externally generated files. A later explicitly
invoked lifecycle review may write only its exact observer eligibility record.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

from .bootstrap import _explicit_document, _outside, _owner, select_entry_role
from .capacity import _json_bytes, read_bound
from .control import CacheBudgetLedger, _encoded, _json, _owner as checked_owner, _token
from .policy import CachePolicyError, _descriptor, cache_settings, nonnegative
from .windows import WindowsNamespace, _parts


def _same_path(left, right):
    return str(PureWindowsPath(left)).casefold() == str(PureWindowsPath(right)).casefold()


def _host_binding(admitted, actual):
    return (admitted.get("host") == actual["host"]
        and actual["service_account_sid"] in admitted.get("authorized_sids", [])
        and _same_path(admitted["host_ledger"]["path"], actual["common_root"]))


def _verify_original_directory(path, identity):
    namespace = WindowsNamespace(path, identity)  # Existing IDs, no create/enumeration.
    namespace.close()


class _MetadataOnce:
    def __init__(self, package_file):
        self.package_file, self.seen = package_file, {}

    def remember(self, pin, document):
        key = str(PureWindowsPath(pin["path"])).casefold()
        if key in self.seen and self.seen[key][0] != pin:
            raise CachePolicyError("One original metadata path has conflicting descriptors")
        self.seen[key] = (deepcopy(pin), deepcopy(document))

    def read(self, binding):
        pin = _descriptor(binding)
        key = str(PureWindowsPath(pin["path"])).casefold()
        if key in self.seen:
            original, document = self.seen[key]
            if original != pin:
                raise CachePolicyError("Original metadata descriptor changed during resolution")
            return deepcopy(document)
        _outside(pin["path"], self.package_file)
        document, _ = read_bound(pin)
        self.remember(pin, document)
        return document


class _EligibilityMetadataOnly:
    """No producer records or host-binding initialization through this view."""
    def __init__(self, namespace, binding, observer):
        self.namespace, self.binding, self.observer = namespace, binding, observer

    def names(self):
        return self.namespace.names()

    def read_once(self, name, maximum=1024**2):
        return self.namespace.read_once(name, maximum=maximum)

    def write_once(self, name, raw, maximum=1024**2):
        row = _json_bytes(raw)
        value = row.get("value", {})
        if (row.get("format") != "darsia-cache-service-lifecycle-v1"
                or row.get("kind") != "eligibility" or row.get("binding") != self.binding
                or row.get("actor") != self.observer or value.get("observer") != self.observer
                or value.get("cleanup_authorized") is not False
                or value.get("service_budget_closed") is not False):
            raise CachePolicyError("Observer context permits only original-token eligibility evidence")
        key = hashlib.sha256(_encoded(value)).hexdigest()
        expected = "cache-l-" + self.binding["reservation"]["token"] + "-eligibility-" + key + ".json"
        if name != expected or row.get("key") != key:
            raise CachePolicyError("Eligibility evidence differs from original service/actual observer")
        return self.namespace.write_once(name, raw, maximum=maximum)


class _ExistingObserverLedger(CacheBudgetLedger):
    """Fresh existing-binding transactions; never initialize or allocate."""
    def __init__(self, *args, service_binding, observer, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_binding, self.observer = deepcopy(service_binding), deepcopy(observer)

    @contextmanager
    def locked(self, *, create_binding=False):
        if create_binding is not False:
            raise CachePolicyError("Observer context cannot initialize a host binding")
        with super().locked(create_binding=False):
            namespace = self.control
            self.control = _EligibilityMetadataOnly(namespace, self.service_binding, self.observer)
            try:
                yield self
            finally:
                self.control = namespace  # Original finally closes this fresh handle.

    def _producer_operation_refused(self, *args, **kwargs):
        raise CachePolicyError("Read-only observer is not a producer/allocation/cleanup context")

    reserve = admit = bind_scope = close = _producer_operation_refused


@dataclass(frozen=True)
class OriginalServiceObserverContext:
    owner: dict
    context: dict
    document: dict
    source: dict
    policy: object
    profile: dict
    profile_binding: dict
    admission: dict
    manifest: dict
    reservation: dict
    root: Path
    directories: dict
    maximum: int
    entry_limit: int
    entry_kind: str
    role_identifier: str
    purpose: str
    receipt: dict
    ledger: object
    original_metadata: tuple
    resolver_is_not_retirement_proof: bool = True


def resolve_original_service_context(receipt_binding, *, package_file):
    """Reopen exact metadata once; no ImportLease constructor or live handles.

    Requires CacheBudgetLedger.locked(create_binding=False). The original
    policy/profile/namespace are metadata bindings, not fresh producer/runtime
    qualification. The actual native observer is distinct from the original
    primary owner. Full registered-owner/launch/reader retirement is still a
    separate mandatory provider; this function does not inspect/kill processes.
    """
    reads = _MetadataOnce(package_file)
    receipt_binding = _descriptor(receipt_binding)
    receipt = reads.read(receipt_binding)
    if receipt.get("format") != "darsia-cache-import-service-receipt-v1":
        raise CachePolicyError("Exact original import/service receipt required")
    policy_binding = _descriptor(receipt["policy"])
    policy_key = str(PureWindowsPath(policy_binding["path"])).casefold()
    if policy_key in reads.seen:
        raise CachePolicyError("Policy and service receipt cannot alias one metadata file")
    document, actual_source = _explicit_document(policy_binding["path"], package_file)
    if actual_source != policy_binding:
        raise CachePolicyError("Original service policy SHA/size/path differs")
    reads.remember(policy_binding, document)
    policy = cache_settings(document, required=True)
    role = select_entry_role(document, entry_kind=receipt["entry_kind"],
                             role_identifier=receipt["role_identifier"])
    if (receipt["profile"] != role["profile"]
            or receipt["namespace"] != document["cache_bootstrap"]["namespace_admission"]):
        raise CachePolicyError("Original service role/profile/namespace metadata differs")
    profile = reads.read(role["profile"])
    if (profile.get("format") != "darsia-cache-import-profile-v1"
            or profile.get("purpose") != role["purpose"]
            or profile.get("role_identifier") != role["role_identifier"]
            or profile.get("entry_kind") != role["entry_kind"]
            or profile.get("writer_contract") != "cooperative-start-admission-v1"):
        raise CachePolicyError("Original import/service profile is not applicable")
    maximum = nonnegative(profile["maximum_cache_bytes"], "Original service maximum")
    completed = reads.read(profile["completion"])
    if not maximum or completed.get("complete") is not True or completed.get("passed") is not True:
        raise CachePolicyError("Original completed import/service profile is required")
    admission = reads.read(receipt["namespace"])
    observer, actual = _owner()
    if not _host_binding(admission, actual):
        raise CachePolicyError("Actual observer host/account/common authority is not admitted")
    manifest = deepcopy(receipt["manifest"])
    original_owner = checked_owner(manifest["owner"])
    if original_owner.get("machine") != actual["host"]:
        raise CachePolicyError("Original service owner belongs to a different physical host")
    if ((observer["pid"], observer["process_create_time"])
            == (original_owner["pid"], original_owner["process_create_time"])):
        raise CachePolicyError("Original live primary is not an independent observer")
    reservation = deepcopy(manifest["reservation"])
    token = _token(reservation["token"])
    purpose = role["purpose"]
    container = policy.purpose(purpose) / ("current" if purpose == "tests" else "")
    root = container / token
    directories = deepcopy(receipt["directories"])
    if (manifest.get("format") != "darsia-cache-external-import-scope-v1"
            or manifest.get("token") != token or not _same_path(manifest["root"], root)
            or manifest["purpose"] != purpose or manifest["maximum_bytes"] != maximum
            or manifest["entry_kind"] != role["entry_kind"]
            or manifest["role_identifier"] != role["role_identifier"]
            or directories.get("") != manifest["root_identity"]
            or not _same_path(admission["purposes"][purpose]["path"], container)):
        raise CachePolicyError("Original service scope/purpose/root identity differs")
    for relative in directories:
        if relative:
            _parts(relative)
    expected_receipt = Path(admission["receipts"]["path"]) / (token + ".import-service.json")
    if not _same_path(receipt_binding["path"], expected_receipt):
        raise CachePolicyError("Service receipt is not in its original admitted receipt namespace")
    _verify_original_directory(admission["receipts"]["path"], admission["receipts"]["identity"])
    _verify_original_directory(root, manifest["root_identity"])
    entry_limit = nonnegative(document["cache_bootstrap"]["accounting_entry_limit"], "Original entry limit")
    if not entry_limit:
        raise CachePolicyError("Original positive accounting limit is required")
    binding = dict(reservation=reservation, policy=policy_binding,
        profile=role["profile"], namespace=receipt["namespace"], manifest=manifest)
    def no_cache_measurement(*args, **kwargs):
        raise CachePolicyError("Observer resolver does not measure or admit producer cache")
    ledger = _ExistingObserverLedger(policy, SimpleNamespace(**actual),
        measurement_provider=no_cache_measurement, namespace_admission=admission,
        verify_host_admission=_host_binding, service_binding=binding, observer=observer)
    with ledger.locked(create_binding=False):
        ledger.require_reservation(reservation, original_owner, maximum)
        original_scope, _ = _json(ledger.control, "cache-s-" + token + ".json")
        if original_scope != manifest:
            raise CachePolicyError("Original bound service manifest differs from current ledger")
    if ledger.control is not None or ledger.lock_active:
        raise CachePolicyError("Resolver did not close its freshly opened control handles")
    return OriginalServiceObserverContext(owner=deepcopy(observer), context=deepcopy(actual),
        document=deepcopy(document), source=deepcopy(policy_binding), policy=policy,
        profile=deepcopy(profile), profile_binding=deepcopy(role["profile"]), admission=deepcopy(admission),
        manifest=manifest, reservation=reservation, root=root, directories=directories,
        maximum=maximum, entry_limit=entry_limit, entry_kind=role["entry_kind"],
        role_identifier=role["role_identifier"], purpose=purpose, receipt=deepcopy(receipt_binding),
        ledger=ledger, original_metadata=tuple(deepcopy(pin) for pin, _ in reads.seen.values()))
