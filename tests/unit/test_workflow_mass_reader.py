"""Legacy mass result discovery, without acquisition or job execution."""

from types import SimpleNamespace

import numpy as np

import darsia
from darsia.presets.workflows.utils.mass import load_data


def test_load_data_skips_same_suffix_directories(tmp_path, monkeypatch):
    """Only mass files are offered to protocol time matching and image loading."""
    (tmp_path / "a.npz").mkdir()
    path = tmp_path / "b.npz"
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    darsia.ScalarImage(values, dimensions=[2, 3]).save(path)
    config = SimpleNamespace(
        analysis=SimpleNamespace(mass=SimpleNamespace(folder=tmp_path))
    )

    def find_images_for_times(*, times, tol, data):
        assert (times, tol) == (3, 3600)
        assert data == [path]
        return data[0]

    monkeypatch.setattr(
        darsia.ProtocolledExperiment,
        "init_from_config",
        lambda config: SimpleNamespace(find_images_for_times=find_images_for_times),
    )
    result = load_data(config, "mass", 3, tol=1)
    np.testing.assert_array_equal(result.img, values)
