"""The stored colour paths of the colour-path steps."""

import numpy as np

from darsia.presets.workflows.setup.color_path_steps import (
    read_color_paths,
    write_color_paths,
)


def test_stored_key_colours_keep_their_precision(tmp_path):
    """Key colours stored and read back are bit-equal in their precision, so the
    shared scale of the runs is computed as from the fit itself."""
    knots = np.array([[0, 0, 0], [1e-5, 3.3e-5, -7e-6], [0.02, 0.031, 1e-3]])
    paths = {13: knots.astype(np.float32), 1: knots[:2].astype(np.float32)}
    write_color_paths(tmp_path, paths, {1: 20, 13: 30, 5: 0}, np.float32)
    found = read_color_paths(tmp_path)
    assert set(found) == {1, 13}
    for sand, values in paths.items():
        assert found[sand].dtype == np.float32
        np.testing.assert_array_equal(found[sand], values)
