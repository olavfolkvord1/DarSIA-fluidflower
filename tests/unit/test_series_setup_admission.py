"""Original private ledgers plus explicit Setup output ceilings.

Only filesystem/volume observations are synthetic. The tests use actual
CapacityLedger locks, immutable reservations, accounting and release records,
and the original full RAW-request reader over tiny private byte inputs. They
do not qualify a physical volume, RAW decoder, scientific grid or cache model.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import job_filesystem as filesystem
from darsia.fluidflower.pipeline import capacity
from darsia.fluidflower.pipeline import series_setup_admission as admission
from .test_series_setup_capacity import prepared_raw_request, _json
from .test_series_setup_protocol import _task, _operation


def _admission(tmp_path, monkeypatch, *, metadata_cap, raw_cap=0, shared=True,
               scratch=False, bind=True, metadata_file_cap=None, metadata_roots=None):
    """Return a real bound SetupAdmission for other bounded-writer tests.

    Every written byte remains under ``tmp_path``. The tiny RAW request has
    deliberately synthetic leaf files; its original reader still verifies
    their exact descriptors. No admission, reservation or release is mocked.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    operation = "build" if raw_cap else "apply"
    task = _task(operation, run="sample")
    task.update(job_id="setup-job", attempt_id="1" * 32)
    config = _operation(operation, run="sample")
    config["metadata_output"] = dict(maximum_output_bytes=metadata_cap,
                                     maximum_file_bytes=metadata_cap if metadata_file_cap is None else metadata_file_cap)
    raw_root = None
    if raw_cap:
        _old_task, _old_config, request_path, request = prepared_raw_request.__wrapped__(tmp_path)
        request["output"].update(maximum_output_bytes=raw_cap, maximum_file_bytes=raw_cap)
        config["raw_build_request"] = _json(request_path, request)
        raw_root = Path(request["output"]["intended_root"])
    if metadata_roots is None:
        metadata_root = tmp_path / "publication"
        metadata_root.mkdir()
        metadata_roots = (metadata_root,)
    else:
        metadata_roots = tuple(Path(root).resolve() for root in metadata_roots)
        assert metadata_roots and all(root.is_dir() for root in metadata_roots)
        metadata_root = metadata_roots[0]
    scratch_root = tmp_path / "scratch"
    if scratch:
        scratch_root.mkdir()
    config["output_scope"] = dict(metadata_roots=list(map(str, metadata_roots)),
                                  scratch_roots=[str(scratch_root)] if scratch else [])
    roots = dict(series=str(tmp_path / "series"), results=str(metadata_root))
    profile = tmp_path / "unused-schema-profile.json"
    profile.write_bytes(b'{"unused":true}\n')
    profile_pin = dict(path=str(profile), sha256=hashlib.sha256(profile.read_bytes()).hexdigest(), size_bytes=profile.stat().st_size)
    stores = SimpleNamespace(machine_id="fixture", roots=lambda _series: roots,
        document=dict(capacity=dict(state_directory=str(tmp_path / "capacity-state"),
                                   profiles={"sample-series": profile_pin}, reserve_bytes=7)))

    def local(path, **_kwargs):
        container = Path(path).resolve().parent
        while not container.exists():
            container = container.parent
        return filesystem.LocalFileAdmission(True, "fixture-fixed", "synthetic OS observation",
                                             "private test only", str(container))

    monkeypatch.setattr(filesystem, "inspect_local_file", local)
    states = {}

    def volume(path):
        path = Path(path).resolve()
        is_raw = raw_root is not None and path.is_relative_to(raw_root)
        is_scratch = scratch and path.is_relative_to(scratch_root)
        identity = "shared-volume" if shared else "raw-volume" if is_raw or is_scratch else "publication-volume"
        state = states.setdefault(identity, dict(volume_id=identity, volume_anchor=str(tmp_path),
                                                 free_bytes=10**9, checked_container=str(tmp_path)))
        return dict(state)

    context = dict(host="fixture-host", service_account_sid="S-1-5-21-100",
                   common_root=str(tmp_path / "machine-common"))

    def ledger_factory(selected, series_id, root):
        return capacity.CapacityLedger(selected, series_id, root, volume_provider=volume,
                                       context_provider=lambda: dict(context))

    actual = admission.prepare_setup_admission(stores, "sample-series", task, config,
        metadata_roots=metadata_roots, scratch_roots=(scratch_root,) if scratch else (),
        ledger_factory=ledger_factory)
    owner = dict(machine="fixture-cpu", pid=123, process_create_time="12345.125")
    claim = SimpleNamespace(task=deepcopy(task), token=task["attempt_id"])
    actual.test_context = SimpleNamespace(task=task, config=config, owner=owner, claim=claim,
        states=states, stores=stores, volume=volume, context=context, raw_root=raw_root,
        metadata_root=metadata_root, scratch_root=scratch_root if scratch else None,
        job_root=tmp_path / "job")
    if bind:
        with actual.locked():
            actual.check()
            actual.bind(actual.test_context.job_root, claim, owner)
    return actual


def test_two_actual_volumes_keep_raw_and_metadata_caps_separate(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, shared=False)
    ceilings = {row.volume_id: row.maximum_output_bytes for row in actual.volume_reservations}
    assert ceilings == {"raw-volume": 100, "publication-volume": 20}
    assert len(actual.bound_reservations) == 2
    for row in actual.bound_reservations:
        record = json.loads((Path(row["path"]) / "reserved.json").read_bytes())
        assert record["format"] == "darsia-local-capacity-reservation-v1"
        assert record["not_a_measured_profile"] is True
        assert record["maximum_output_bytes"] in {20, 100}
    assert actual.metadata_limits == dict(maximum_output_bytes=20, maximum_file_bytes=20)


def test_same_volume_counts_one_global_metadata_allowance_even_with_scratch(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, shared=True, scratch=True)
    assert [row.maximum_output_bytes for row in actual.volume_reservations] == [120]
    assert len(actual.bound_reservations) == 1
    row = actual.volume_reservations[0]
    assert row.raw_maximum_output_bytes == 100 and row.metadata_maximum_output_bytes == 20
    assert {root["role"] for root in row.roots} == {"raw", "metadata", "scratch"}
    assert actual.require_write(actual.test_context.metadata_root / "next.json")
    assert actual.require_write(actual.test_context.scratch_root / "partial.tmp")


def test_distinct_metadata_and_scratch_volumes_each_keep_full_shared_allowance(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, shared=False, scratch=True)
    ceilings = {row.volume_id: row.maximum_output_bytes for row in actual.volume_reservations}
    assert ceilings == {"raw-volume": 120, "publication-volume": 20}
    # The bounded sink still has only one 20-byte global budget, not 20 per disk.
    assert actual.metadata_limits["maximum_output_bytes"] == 20


def test_existing_open_claims_are_included_before_any_new_bind(tmp_path, monkeypatch):
    first = _admission(tmp_path, monkeypatch, metadata_cap=100)
    data = first.test_context
    original = capacity.CapacityLedger(data.stores, "sample-series", data.metadata_root,
        volume_provider=data.volume, context_provider=lambda: dict(data.context))
    next_task = deepcopy(data.task)
    next_task["attempt_id"] = "2" * 32
    second = admission.prepare_setup_admission(data.stores, "sample-series", next_task, data.config,
        metadata_roots=(data.metadata_root,), ledger_factory=lambda *_args: original)
    data.states["shared-volume"]["free_bytes"] = 206  # 100 + open100 + reserve7 needs207.
    with second.locked():
        with pytest.raises(ValueError, match="existing open claims"):
            second.check()
    assert not second.bound_reservations
    with original.locked():
        assert original.remaining() == 100


def test_insufficient_second_volume_does_not_bind_first_volume(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, shared=False, bind=False)
    actual.test_context.states["publication-volume"]["free_bytes"] = 26  # 20+7.
    with actual.locked():
        with pytest.raises(ValueError, match="exceed free space"):
            actual.check()
    assert not actual.bound_reservations
    assert not list((tmp_path / "capacity-state").glob("*/r-*/reserved.json"))


def test_locks_use_stable_original_scope_order_and_leave_reservations_live(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, shared=False, bind=False)
    events = []
    original = capacity.CapacityLedger.locked

    @contextmanager
    def observed(ledger, *args, **kwargs):
        with original(ledger, *args, **kwargs):
            events.append(("enter", ledger.key))
            try:
                yield ledger
            finally:
                events.append(("leave", ledger.key))

    monkeypatch.setattr(capacity.CapacityLedger, "locked", observed)
    with actual.locked():
        actual.check()
        actual.bind(actual.test_context.job_root, actual.test_context.claim, actual.test_context.owner)
    expected = sorted(row.capacity_key for row in actual.volume_reservations)
    assert [key for action, key in events if action == "enter"] == expected
    assert [key for action, key in events if action == "leave"] == list(reversed(expected))
    assert all(not (Path(row["path"]) / "released.json").exists() for row in actual.bound_reservations)


def test_whole_cap_does_not_shrink_from_frame_progress_and_history_survives_release(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=100)
    data = actual.test_context
    ledger = capacity.CapacityLedger(data.stores, "sample-series", data.metadata_root,
        volume_provider=data.volume, context_provider=lambda: dict(data.context))
    reserved = Path(actual.bound_reservations[0]["path"])
    before = (reserved / "reserved.json").read_bytes()
    with ledger.locked():
        assert ledger.remaining() == 100
        with pytest.raises(ValueError, match="whole|Setup|setup"):
            ledger.progress(reserved, data.owner,
                dict(state="frame_completed", complete=True, run="sample", case_id="apply"))
        assert ledger.remaining() == 100 and not list(reserved.glob("p-*.json"))
    actual.release(data.owner)
    with ledger.locked():
        assert ledger.remaining() == 0
    assert (reserved / "reserved.json").read_bytes() == before
    assert (reserved / "released.json").is_file()
    with pytest.raises(RuntimeError, match="active Setup reservation"):
        actual.require_write(data.metadata_root / "late.json")


def test_partial_bind_failure_preserves_truthful_prefix_without_automatic_release(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, shared=False, bind=False)
    original = capacity.CapacityLedger.bind
    calls = []

    def fail_second(ledger, *args):
        calls.append(ledger.key)
        if len(calls) == 2:
            raise OSError("synthetic second-volume IO failure")
        return original(ledger, *args)

    monkeypatch.setattr(capacity.CapacityLedger, "bind", fail_second)
    with actual.locked():
        actual.check()
        with pytest.raises(admission.SetupBindError, match="retained reservations") as failed:
            actual.bind(actual.test_context.job_root, actual.test_context.claim, actual.test_context.owner)
    assert failed.value.bound_reservations == actual.bound_reservations
    assert len(actual.bound_reservations) == 1
    path = Path(actual.bound_reservations[0]["path"])
    assert (path / "reserved.json").is_file() and not (path / "released.json").exists()
    with pytest.raises(RuntimeError, match="active Setup reservation"):
        actual.require_write(actual.test_context.metadata_root / "never.json")


def test_root_or_request_drift_between_plan_and_bind_is_refused(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, bind=False)
    request = Path(actual.configuration["raw_build_request"]["path"])
    old = request.read_bytes()
    request.write_bytes(b" " + old[1:])
    with actual.locked():
        with pytest.raises(ValueError, match="changed"):
            actual.check()
    assert not actual.bound_reservations


def test_unlisted_path_or_changed_active_reservation_cannot_authorize_a_write(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100)
    with pytest.raises(ValueError, match="outside the explicit"):
        actual.require_write(tmp_path / "foreign.json")
    with pytest.raises(ValueError, match="outside the explicit"):
        actual.require_write(actual.test_context.raw_root / "bypass.json")
    path = Path(actual.bound_reservations[0]["path"]) / "reserved.json"
    before = path.read_bytes()
    path.write_bytes(before + b" ")
    with pytest.raises(ValueError, match="frozen bytes"):
        actual.require_write(actual.test_context.metadata_root / "never.json")


def test_active_domain_volume_change_is_refused_before_write(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20)
    actual.test_context.states["shared-volume"]["volume_id"] = "changed-volume"
    with pytest.raises((ValueError, RuntimeError), match="volume"):
        actual.require_write(actual.test_context.metadata_root / "never.json")


def test_replaced_output_directory_is_not_the_captured_domain(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20)
    root = actual.test_context.metadata_root
    root.rename(root.with_name("retained-original-directory"))
    root.mkdir()
    with pytest.raises(ValueError, match="directory identity"):
        actual.require_write(root / "never.json")


def test_new_work_at_raw_destination_cannot_be_relaunched_after_planning(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, bind=False)
    Path(actual.raw_authority["destination"]).mkdir()
    with actual.locked():
        with pytest.raises(ValueError, match="cannot be relaunched"):
            actual.check()
    assert not actual.bound_reservations


def test_retained_literal_caps_do_not_reopen_old_request_or_plan_sources(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100)
    reserved = Path(actual.bound_reservations[0]["path"]) / "reserved.json"
    record = json.loads(reserved.read_bytes())
    request = Path(actual.configuration["raw_build_request"]["path"])
    old = request.read_bytes()
    original = Path.read_bytes

    def no_old_source_read(path):
        if path == request:
            raise AssertionError("Historical accounting must not reopen the original request")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", no_old_source_read)
    row = admission.validate_retained_reservation(record,
        capacity_key=actual.volume_reservations[0].capacity_key,
        volume_id=actual.volume_reservations[0].volume_id)
    assert row["maximum_output_bytes"] == 120
    assert original(request) == old


@pytest.mark.parametrize("field,value", [("maximum_output_bytes", 119), ("case_count", 2),
    ("frame_progress_release_allowed", True), ("source_commit", "wrong"),
    ("volume_id", "foreign"), ("capacity_key", "foreign")])
def test_retained_records_cannot_relax_whole_cap_or_scope(tmp_path, monkeypatch, field, value):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100)
    path = Path(actual.bound_reservations[0]["path"]) / "reserved.json"
    record = json.loads(path.read_bytes())
    record["estimate"][field] = value
    original = actual.volume_reservations[0]
    with pytest.raises(ValueError):
        admission.validate_retained_reservation(record, capacity_key=original.capacity_key,
                                               volume_id=original.volume_id)


def test_plan_and_declared_reservation_are_defensive_copies(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100)
    changed = actual.configuration
    changed["metadata_output"]["maximum_output_bytes"] = 10**8
    changed = actual.raw_authority
    changed["maximum_output_bytes"] = 10**8
    assert actual.metadata_limits["maximum_output_bytes"] == 20
    assert actual.raw_authority["maximum_output_bytes"] == 100
    assert asdict(actual.volume_reservations[0])["maximum_output_bytes"] == 120
    changed = actual.volume_reservations[0]
    changed.setup_plan["sha256"] = "f" * 64
    changed.roots[0]["path"] = str(tmp_path / "foreign")
    assert actual.volume_reservations[0].setup_plan["sha256"] == "d" * 64
    assert actual.volume_reservations[0].roots[0]["path"] != str(tmp_path / "foreign")


def test_original_reader_still_counts_the_positive_setup_record(tmp_path, monkeypatch):
    """An existing worker sees a conservative whole cap, not an unknown format."""
    import ast
    import os
    import subprocess
    actual = _admission(tmp_path, monkeypatch, metadata_cap=100)
    data = actual.test_context
    ledger = capacity.CapacityLedger(data.stores, "sample-series", data.metadata_root,
        volume_provider=data.volume, context_provider=lambda: dict(data.context))
    # Immutable pre-Setup source; no production module is replaced or imported.
    repository = Path(__file__).resolve().parents[2]
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith("GIT_")}
    source = subprocess.check_output(["git", "--no-optional-locks", "-C", str(repository),
        "show", "098465c1b82f92ae6ce129571e9bf8e2e6267ddf:src/darsia/fluidflower/pipeline/capacity.py"],
        env=environment)
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "CapacityLedger")
    original = next(node for node in cls.body
                    if isinstance(node, ast.FunctionDef) and node.name == "remaining")
    namespace = dict(vars(capacity))
    exec(compile(ast.Module(body=[original], type_ignores=[]), "<immutable-original-capacity>", "exec"), namespace)
    with ledger.locked():
        assert namespace["remaining"](ledger) == 100
        assert ledger.remaining() == 100
