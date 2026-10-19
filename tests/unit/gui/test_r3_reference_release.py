"""TESTONLY: R3, the RAW39 reference, is admitted only by its exact release entry.

Like N43, R3 is pinned in REFERENCE_RELEASES: its answer records with their
deciders and its accepted plan's SHA-256 (N134, N135, N136). Under N139 the
entry also pins the retained Q123 font file R3 may find in the cache, and
admission counts exactly those verified bytes instead of an empty cache. Any
other authority set, record, plan or inventory is refused. Synthetic data only.
"""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from darsia_cache import first_qualification as first
from darsia_cache.policy import CachePolicyError, PHYSICAL_RESERVE_BYTES
from .test_n43_reference_administration import Ledger_TESTONLY, administration_TESTONLY, pin_TESTONLY

R3_PLAN = "3ea1caa8ff432f5c17d2904f7f43080e23aeac039a447d8667a1586749e0426e"
R3_GO = ("N134-OLAV-r3-inputs-into-a-new-ff-ac-folder-reference-build-once-fresh-cache-released-"
         "after-the-code-change")
R3_PLAN_RECORD = "N136-CLAUDE-r3-plan-3ea1caa8-entitlement-65728995330B-pinned-for-the-exact-code-entry"
R3_RETAINED = ("N139-OLAV-r3-starts-with-the-verified-117590-byte-retained-font-file-counted-file-and-old-"
               "approval-untouched")
R3_INVENTORY = "be5c8bf204e6a5021f79bd10125542bcfd87adaf80c2401e66cb7ce9463852a2"
R3_AUTHORITIES = ("N134", "N136", "N139")


def r3_TESTONLY(answers=None, authority_ids=R3_AUTHORITIES, plan=R3_PLAN, inventory=None,
                inventory_sha256=R3_INVENTORY):
    state, value, _read = administration_TESTONLY()
    cache = next(row["path"] for row in value["owned_roots"] if row["role"] == "cache")
    inventory = inventory if inventory is not None else dict(format=first.RETAINED_INVENTORY, root=cache,
        no_cleanup=True, files=[{"TESTONLY": "one retained font file"}], logical_bytes=117590)
    value.update(authority_ids=list(authority_ids),
                 accepted_plan=dict(path="C:/TESTONLY/R3_N135_accepted_plan.json", sha256=plan, size_bytes=1),
                 retained_cache_inventory=dict(path="C:/TESTONLY/retained_inventory.json",
                                               sha256=inventory_sha256, size_bytes=413))
    answers = answers if answers is not None else [
        dict(id=R3_GO, decided_by="olav"), dict(id=R3_PLAN_RECORD, decided_by="claude"),
        dict(id=R3_RETAINED, decided_by="olav")]

    def read(binding):
        if binding == value["answers_source"]:
            return deepcopy(answers)
        if binding == value["retained_cache_inventory"]:
            return deepcopy(inventory)
        return {"TESTONLY": "read exact plan"}

    return state, value, read


def test_r3_is_admitted_by_its_exact_release_entry():
    state, value, read = r3_TESTONLY()
    actual = first.checked_reference_administration(value, state, read=read)
    assert actual["authority_ids"] == ["N134", "N136", "N139"]
    assert actual["aggregate_entitlement_bytes"] == first.REFERENCE_ENTITLEMENT_BYTES
    assert actual["retained_cache_inventory"]["sha256"] == R3_INVENTORY


@pytest.mark.parametrize("kwargs", [
    dict(plan="3ea1caa8" + "0" * 56),  # Only the short prefix matches.
    dict(authority_ids=("N134", "N136")),  # The empty-cache R3 entry is gone.
    dict(authority_ids=("N134", "N136", "N140")),
    dict(authority_ids=("N136", "N134", "N139")),  # Order is part of the entry.
    dict(answers=[dict(id=R3_GO, decided_by="olav"), dict(id=R3_PLAN_RECORD, decided_by="claude")]),
    dict(answers=[dict(id=R3_GO, decided_by="olav"), dict(id=R3_PLAN_RECORD, decided_by="claude"),
                  dict(id=R3_RETAINED, decided_by="claude")]),
    dict(answers=[dict(id=R3_GO, decided_by="claude"), dict(id=R3_PLAN_RECORD, decided_by="claude"),
                  dict(id=R3_RETAINED, decided_by="olav")]),
    dict(answers=[]),
])
def test_r3_refuses_any_other_plan_authority_set_or_record(kwargs):
    state, value, read = r3_TESTONLY(**kwargs)
    with pytest.raises(CachePolicyError):
        first.checked_reference_administration(value, state, read=read)


@pytest.mark.parametrize("change", ["missing", "pin", "bytes", "float_bytes", "format", "cleanup", "root"])
def test_r3_requires_exactly_the_pinned_retained_inventory_of_its_cache_root(change):
    state, value, read = r3_TESTONLY(inventory_sha256="0" * 64 if change == "pin" else R3_INVENTORY)
    inventory = read(value["retained_cache_inventory"])
    edits = dict(bytes=dict(logical_bytes=117591), float_bytes=dict(logical_bytes=117590.0),
                 format=dict(format="darsia-cache-retained-file-inventory-v0"), cleanup=dict(no_cleanup=False),
                 root=dict(root="Q:/TESTONLY/another-cache"))
    if change in edits:
        inventory.update(edits[change])
        state, value, read = r3_TESTONLY(inventory=inventory)
    if change == "missing":
        value.pop("retained_cache_inventory")
    with pytest.raises(CachePolicyError):
        first.checked_reference_administration(value, state, read=read)


def test_n43_names_no_retained_inventory():
    state, value, read = administration_TESTONLY()
    value["retained_cache_inventory"] = pin_TESTONLY("retained_inventory.json")
    with pytest.raises(CachePolicyError, match="Exact N43 reference administration"):
        first.checked_reference_administration(value, state, read=read)


def test_r3_keeps_the_n43_entitlement():
    state, value, read = r3_TESTONLY()
    value["aggregate_entitlement_bytes"] = first.REFERENCE_ENTITLEMENT_BYTES + 1
    with pytest.raises(CachePolicyError):
        first.checked_reference_administration(value, state, read=read)


def test_released_reference_names_only_exact_entries():
    assert first.released_reference(["N43", "Q109", "Q106"]) is first.REFERENCE_RELEASES[("N43", "Q109", "Q106")]
    r3 = first.released_reference(["N134", "N136", "N139"])
    assert r3["plan_sha256"] == R3_PLAN
    assert r3["retained_cache"] == dict(inventory_sha256=R3_INVENTORY, logical_bytes=117590)
    assert "retained_cache" not in first.REFERENCE_RELEASES[("N43", "Q109", "Q106")]
    assert first.released_reference(["N134", "N136"]) is None
    assert first.released_reference(("N134", "N136", "N139")) is None  # A list, as stored in JSON.
    assert first.released_reference([["N134"], "N136", "N139"]) is None
    assert first.released_reference(None) is None


def retained_state_TESTONLY(tmp_path, *, retained=True):
    """A real retained file and its exact inventory; no actual cache is read."""
    root = tmp_path / "cache"
    relative = "jobs/" + "e" * 32 + "/matplotlib/fontlist-v390.json"
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True)
    path.write_bytes(b"f" * 117590)
    info = path.lstat()
    inventory = dict(format=first.RETAINED_INVENTORY, root=str(root), no_cleanup=True,
                     files=[dict(relative_path=relative, dev=info.st_dev, ino=info.st_ino,
                                 size_bytes=info.st_size, mtime_ns=info.st_mtime_ns)],
                     logical_bytes=117590)
    state = first.FirstQualification.__new__(first.FirstQualification)
    state.reference_build = True
    state.bound_mode = False
    state.binding = pin_TESTONLY("request.json")
    state.reference_administration = dict(host_policy="main-q87-fixed500")
    state.reference_retained_inventory = None
    if retained:
        state.reference_retained_inventory = inventory
        state._reference_retained_seal = first._digest(inventory)
    token = "a" * 32
    ledger = Ledger_TESTONLY()
    ledger.documents.update({"cache-r-" + token + ".json": b"{}", "cache-c-" + token + ".json": b"{}"})
    lease = SimpleNamespace(policy=SimpleNamespace(root=str(root)), ledger=ledger)
    return state, lease, path


def test_r3_counts_exactly_its_verified_retained_file(tmp_path):
    state, lease, _ = retained_state_TESTONLY(tmp_path)
    assert state.initial_cache_bytes(lease) == 117590


def test_without_a_retained_release_a_cold_start_stays_empty(tmp_path):
    state, lease, _ = retained_state_TESTONLY(tmp_path, retained=False)
    assert state.initial_cache_bytes(lease) == 0
    assert first.FirstQualification.__new__(first.FirstQualification).initial_cache_bytes(lease) == 0


@pytest.mark.parametrize("change", ["grown", "missing", "open_reservation", "unknown_reservation", "memory"])
def test_r3_retained_count_refuses_any_changed_file_open_reservation_or_inventory(tmp_path, change):
    state, lease, path = retained_state_TESTONLY(tmp_path)
    if change == "grown":
        path.write_bytes(b"f" * 117591)
    elif change == "missing":
        path.unlink()
    elif change == "open_reservation":
        lease.ledger.documents["cache-r-" + "b" * 32 + ".json"] = b"TESTONLY fully written live reservation"
    elif change == "unknown_reservation":
        lease.ledger.documents["cache-r-TESTONLY.json"] = b"{}"
    else:
        state.reference_retained_inventory["logical_bytes"] = 0
    with pytest.raises((CachePolicyError, OSError)):
        state.initial_cache_bytes(lease)


def admission_TESTONLY(monkeypatch, state):
    """Reference admission with only its namespace and length scan replaced."""
    monkeypatch.setattr(state, "_namespace", lambda lease: True, raising=False)
    snapshot = dict(observed_scan_anomalies=[], roots=[dict(role="results", logical_bytes=0)],
                    logical_bytes=117590, volumes=[])
    monkeypatch.setattr(first, "reference_scope_snapshot", lambda *args: snapshot)
    monkeypatch.setattr(first, "require_reference_not_stopped", lambda *args: None)


def report_TESTONLY(**changes):
    value = dict(reserve_bytes=PHYSICAL_RESERVE_BYTES, hard_filesystem_quota_verified=True,
                 current_cache_logical_bytes=117590, active_cache_remaining_bytes=0)
    value.update(changes)
    return value


@pytest.mark.parametrize("retained,changes,admitted", [
    (True, {}, True),
    (True, dict(current_cache_logical_bytes=0), False),
    (True, dict(current_cache_logical_bytes=117591), False),
    (True, dict(active_cache_remaining_bytes=1), False),
    (False, dict(current_cache_logical_bytes=0), True),  # N43 is unchanged.
    (False, {}, False),
])
def test_reference_admission_counts_exactly_the_released_retained_bytes(tmp_path, monkeypatch, retained,
                                                                         changes, admitted):
    state, lease, _ = retained_state_TESTONLY(tmp_path, retained=retained)
    admission_TESTONLY(monkeypatch, state)
    report = report_TESTONLY(**changes)
    with lease.ledger.locked():
        if admitted:
            assert state.reference_admission(lease, report=report, initial=True)["sufficient"] is True
        else:
            with pytest.raises(CachePolicyError, match="fresh exclusive cache"):
                state.reference_admission(lease, report=report, initial=True)


class ProjectingLedger_TESTONLY(Ledger_TESTONLY):
    def __init__(self, report):
        super().__init__()
        self.report = report

    def project(self, needed):
        assert self.lock_active and needed == 0
        return deepcopy(self.report)


@pytest.mark.parametrize("retained,current,admitted", [
    (True, 117590, True), (True, 0, False), (False, 0, True), (False, 117590, False)])
def test_consume_admits_r3_with_exactly_its_retained_bytes(tmp_path, monkeypatch, retained, current, admitted):
    state, lease, _ = retained_state_TESTONLY(tmp_path, retained=retained)
    lease.ledger = ProjectingLedger_TESTONLY(report_TESTONLY(current_cache_logical_bytes=current))
    calls = []
    monkeypatch.setattr(first, "_outputs_absent", lambda provider: None)
    monkeypatch.setattr(state, "_namespace", lambda lease: True, raising=False)
    monkeypatch.setattr(state, "_once_key", lambda: "c" * 64, raising=False)
    monkeypatch.setattr(state, "reference_admission", lambda lease, **kwargs: calls.append(kwargs), raising=False)
    state.provider = {"scope": {"TESTONLY": "scope"}}
    state.provider_binding, state.stores_source = pin_TESTONLY("provider.json"), pin_TESTONLY("stores.toml")
    lease.owner = {"TESTONLY": "owner"}
    if admitted:
        state.consume(lease)
        assert state.started is True and [call["initial"] for call in calls] == [True]
    else:
        with pytest.raises(CachePolicyError, match="empty exclusive capped cache"):
            state.consume(lease)
        assert calls == [] and lease.ledger.documents == {}
