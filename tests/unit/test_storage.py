"""The one writer of a result's arrays, plain or compressed losslessly."""

import time
import zipfile

import numpy as np
import pytest

import darsia


@pytest.fixture(autouse=True)
def plain_after():
    """Each test leaves the process's storage settings as it found them."""
    found = darsia.storage_settings()
    yield
    darsia.use_storage(found)


def _arrays() -> dict:
    """Arrays of each kind a result stores, with the bits a careless writer loses:
    a NaN with a payload, a negative zero, a Fortran order and pickled objects."""
    rng = np.random.default_rng(3)
    single = rng.normal(size=(40, 60)).astype(np.float32)
    single[0, :3] = [np.nan, -0.0, np.inf]
    single.view(np.uint32)[1, 0] = 0x7FC00ABC  # a NaN with its own payload
    return dict(
        single=single,
        double=np.asfortranarray(rng.normal(size=(30, 20))),
        labels=rng.integers(0, 4, size=(40, 60)).astype(np.uint8),
        mask=rng.random((40, 60)) > 0.5,
        empty=np.zeros((0, 3)),
        metadata=np.array({"name": "frame", "dimensions": [0.3, 0.45]}, object),
    )


def _members(path) -> set[int]:
    """The compression methods of an archive's members."""
    with zipfile.ZipFile(path) as archive:
        return {member.compress_type for member in archive.infolist()}


@pytest.mark.parametrize("compress", [False, True])
def test_arrays_read_back_bit_for_bit_plain_and_compressed(tmp_path, compress):
    """What is written is read back by np.load with its dtype, shape, order and
    every byte; compressed, every member is deflated and the archive smaller."""
    arrays = _arrays()
    darsia.use_storage(darsia.StorageSettings(compress=compress))
    darsia.save_arrays(tmp_path / "arrays", **arrays)  # .npz appended
    with np.load(tmp_path / "arrays.npz", allow_pickle=True) as stored:
        assert stored.files == list(arrays)
        for name, values in arrays.items():
            found = stored[name]
            assert found.dtype == values.dtype and found.shape == values.shape
            if values.dtype == object:
                assert found.item() == values.item()
            else:
                assert found.tobytes(order="A") == values.tobytes(order="A")
                assert found.flags.f_contiguous == values.flags.f_contiguous
    assert _members(tmp_path / "arrays.npz") == {
        zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    }
    labels = np.zeros((300, 400), np.uint8)  # a map with one region compresses
    labels[100:200, 150:250] = 2
    darsia.save_arrays(tmp_path / "labels.npz", labels=labels)
    size = (tmp_path / "labels.npz").stat().st_size
    assert (size < labels.nbytes / 100) == compress


def test_plain_storage_writes_numpys_savez_byte_for_byte(tmp_path, monkeypatch):
    """Without settings, and with compression off, the archive is the one
    np.savez writes at the same moment, byte for byte."""
    monkeypatch.setattr(time, "time", lambda: 1.8e9)  # the members' time stamps
    arrays = _arrays()
    np.savez(tmp_path / "numpy.npz", **arrays)
    expected = (tmp_path / "numpy.npz").read_bytes()
    darsia.save_arrays(tmp_path / "default.npz", **arrays)
    assert darsia.storage_settings() == darsia.StorageSettings()
    darsia.use_storage(darsia.StorageSettings(compress=False, level=9))
    darsia.save_arrays(tmp_path / "plain.npz", **arrays)
    assert (tmp_path / "default.npz").read_bytes() == expected
    assert (tmp_path / "plain.npz").read_bytes() == expected


def test_an_image_is_saved_as_before_and_compressed_by_the_settings(
    tmp_path, monkeypatch
):
    """Image.save writes the file np.savez wrote before, byte for byte, without
    settings; with compression, darsia.imread reads the same image back."""
    monkeypatch.setattr(time, "time", lambda: 1.8e9)
    values = np.zeros((90, 120), np.float32)
    values[30:60, 40:80] = np.linspace(0, 1, 40, dtype=np.float32)
    values[0, 0] = np.nan
    image = darsia.ScalarImage(values, dimensions=[0.3, 0.45], name="frame")
    np.savez(tmp_path / "before.npz", array=image.img, metadata=image.metadata())
    image.save(tmp_path / "plain.npz")
    assert (tmp_path / "plain.npz").read_bytes() == (
        tmp_path / "before.npz"
    ).read_bytes()
    darsia.use_storage(darsia.StorageSettings(compress=True))
    image.save(tmp_path / "compressed.npz")
    assert _members(tmp_path / "compressed.npz") == {zipfile.ZIP_DEFLATED}
    with zipfile.ZipFile(tmp_path / "plain.npz") as plain:
        with zipfile.ZipFile(tmp_path / "compressed.npz") as packed:
            assert packed.namelist() == plain.namelist()
            for name in plain.namelist():  # the array and the pickled metadata
                assert packed.read(name) == plain.read(name)
    found = darsia.imread(tmp_path / "compressed.npz")
    assert found.img.dtype == values.dtype
    assert found.img.tobytes() == values.tobytes()
