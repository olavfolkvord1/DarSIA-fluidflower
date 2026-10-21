"""Synthetic ownership/refusal contracts; no RAW decoder or production writes."""

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from darsia.fluidflower import rig_raw_write as write


def reserve(tmp_path, monkeypatch, name="fresh"):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    root = tmp_path / "products"
    root.mkdir(exist_ok=True)
    from darsia.fluidflower import job_filesystem
    monkeypatch.setattr(job_filesystem, "inspect_local_file", lambda path:
        SimpleNamespace(allowed=True, filesystem="synthetic-local", evidence="unit test"))
    target = root / name
    authority = dict(destination=str(target), intended_root=str(root), forbidden_raw_roots=[str(raw)],
                     maximum_output_bytes=100_000, maximum_file_bytes=20_000)
    return write.OwnedRawOutput(target, authority, operation_id="a" * 32,
        request=dict(synthetic=True)), authority


def test_atomic_reservation_refuses_a_peer_even_when_only_owner_marker_exists(tmp_path, monkeypatch):
    owner, authority = reserve(tmp_path, monkeypatch)
    before = (owner.destination / "raw_build_owner.json").read_bytes()
    with pytest.raises(FileExistsError):
        write.OwnedRawOutput(owner.destination, authority, operation_id="b" * 32,
                             request=dict(synthetic=True))
    assert (owner.destination / "raw_build_owner.json").read_bytes() == before


def test_same_operation_cannot_overwrite_or_save_another_rig(tmp_path, monkeypatch):
    owner, _ = reserve(tmp_path, monkeypatch)
    rig = object()
    owner.begin_setup(rig, owner.destination / "figures")
    with pytest.raises(ValueError, match="original rig"):
        owner.begin_save(object(), owner.destination)
    owner.begin_save(rig, owner.destination)
    image = SimpleNamespace(img=np.ones((2, 3)), metadata=lambda: dict(synthetic=True))
    target = owner.destination / "baseline.npz"
    owner.image(image, target)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        owner.image(image, target)
    assert target.read_bytes() == before


def test_owner_marker_tamper_and_root_escape_refuse(tmp_path, monkeypatch):
    owner, _ = reserve(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="leaves"):
        with owner.open(tmp_path / "outside.bin"):
            pytest.fail("writer must not open")
    (owner.destination / "raw_build_owner.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="ownership"):
        owner.check()


def test_nonlocal_root_and_protected_acquisition_refuse_before_reservation(tmp_path, monkeypatch):
    from darsia.fluidflower import job_filesystem
    root = tmp_path / "raw"
    root.mkdir()
    target = root / "forbidden"
    authority = dict(destination=str(target), intended_root=str(root), forbidden_raw_roots=[str(root)],
                     maximum_output_bytes=100_000, maximum_file_bytes=20_000)
    with pytest.raises(ValueError, match="protected acquisition"):
        write.OwnedRawOutput(target, authority, operation_id="a" * 32, request={})
    assert not target.exists()
    authority["forbidden_raw_roots"] = [str(tmp_path / "acquisitions")]
    monkeypatch.setattr(job_filesystem, "inspect_local_file", lambda path:
        SimpleNamespace(allowed=False))
    with pytest.raises(ValueError, match="local filesystem"):
        write.OwnedRawOutput(target, authority, operation_id="a" * 32, request={})
    assert not target.exists()


def test_existing_reparse_ancestor_is_checked_before_resolution(tmp_path, monkeypatch):
    original = Path.lstat
    ancestor = tmp_path / "alias"
    def lstat(path, *args, **kwargs):
        if path == ancestor:
            return SimpleNamespace(st_mode=0o040755, st_file_attributes=0x400)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="reparse"):
        write._canonical(ancestor / "not-created" / "output")


def test_budget_refuses_before_an_over_limit_byte_and_keeps_partial(tmp_path, monkeypatch):
    owner, _ = reserve(tmp_path, monkeypatch)
    target = owner.destination / "partial.bin"
    with pytest.raises(ValueError, match="byte ceiling"):
        with owner.open(target) as stream:
            stream.write(b"kept")
            stream.write(b"x" * owner.authority["maximum_file_bytes"])
    assert target.read_bytes() == b"kept"
    assert owner.failed is True


def test_raw_npz_stream_retains_original_correction_writer_contract(tmp_path, monkeypatch):
    import darsia
    owner, _ = reserve(tmp_path, monkeypatch)
    correction = darsia.CurvatureCorrection(config={})
    correction.cache = dict(grid=np.arange(12, dtype=np.float32).reshape(2, 6), shape=(2, 3))
    target = owner.destination / "shape_correction_0_curvaturecorrection.npz"
    owner.correction(correction, target)
    restored = darsia.CurvatureCorrection()
    restored.load(target)
    np.testing.assert_array_equal(restored.cache["grid"], correction.cache["grid"])


def test_owned_type_correction_keeps_original_archive_members_and_exclusive_output(tmp_path, monkeypatch):
    """The original type serializer writes identical payloads through the bounded stream."""
    import darsia
    owner, _ = reserve(tmp_path, monkeypatch)
    correction = darsia.TypeCorrection(np.float32)
    legacy = tmp_path / "legacy-type.npz"
    correction.save(legacy)
    target = owner.destination / "shape_correction_0_typecorrection.npz"
    owner.correction(correction, target)
    with ZipFile(legacy) as original, ZipFile(target) as actual:
        assert actual.namelist() == original.namelist() == ["class_name.npy", "data_type.npy"]
        assert all(actual.read(name) == original.read(name) for name in original.namelist())
    restored = darsia.TypeCorrection()
    restored.load(target)
    assert restored.data_type is np.float32
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        owner.correction(correction, target)
    assert target.read_bytes() == before and owner.failed is False


def test_owned_type_correction_retains_partial_without_exceeding_operation_cap(tmp_path, monkeypatch):
    """The original NPZ serializer cannot bypass the operation ceiling through its stream."""
    import darsia
    correction = darsia.TypeCorrection(np.float32)
    legacy = tmp_path / "legacy-type.npz"
    correction.save(legacy)
    owner, _ = reserve(tmp_path, monkeypatch)
    ceiling = owner.authority["maximum_output_bytes"]
    initial = sum(path.stat().st_size for path in owner.destination.rglob("*") if path.is_file())
    remaining = ceiling - initial - (legacy.stat().st_size - 1)
    for index in range((remaining + owner.authority["maximum_file_bytes"] - 1)
                       // owner.authority["maximum_file_bytes"]):
        count = min(remaining, owner.authority["maximum_file_bytes"])
        with owner.open(owner.destination / ("retained-" + str(index) + ".bin")) as stream:
            stream.write(b"x" * count)
        remaining -= count
    target = owner.destination / "shape_correction_0_typecorrection.npz"
    with pytest.raises(ValueError, match="byte ceiling"):
        owner.correction(correction, target)
    assert owner.failed is True and target.is_file()
    assert sum(path.stat().st_size for path in owner.destination.rglob("*") if path.is_file()) <= ceiling
    assert target.stat().st_size < legacy.stat().st_size
