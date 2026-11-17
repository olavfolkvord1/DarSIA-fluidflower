import json
from pathlib import Path

import pytest

from darsia.presets.workflows.calibration.calibration_color_to_mass_analysis import (
    _load_baseline_color_spectrum_for_color_to_mass,
    _stored_ignore_mask,
)


def _write_spectrum_file(folder: Path, label: int):
    folder.mkdir(parents=True, exist_ok=True)
    content = {
        "base_color": [0.0, 0.0, 0.0],
        "spectrum": [[[False]]],
        "histogram": [[[0.0]]],
        "color_range": {
            "min_color": [0.0, 0.0, 0.0],
            "max_color": [1.0, 1.0, 1.0],
            "color_mode": "RELATIVE",
        },
    }
    (folder / f"color_spectrum_{label}.json").write_text(json.dumps(content))


def test_load_baseline_color_spectrum_no_files_returns_none(tmp_path):
    result = _load_baseline_color_spectrum_for_color_to_mass(
        baseline_color_spectrum_folder=tmp_path / "missing",
        required_labels={0},
    )
    assert result is None


def test_load_baseline_color_spectrum_requires_all_labels(tmp_path):
    folder = tmp_path / "spectra"
    _write_spectrum_file(folder, 0)

    with pytest.raises(FileNotFoundError, match="Missing labels: \\[1\\]"):
        _load_baseline_color_spectrum_for_color_to_mass(
            baseline_color_spectrum_folder=folder,
            required_labels={0, 1},
        )


def test_load_baseline_color_spectrum_loads_valid_folder(tmp_path):
    folder = tmp_path / "spectra"
    _write_spectrum_file(folder, 0)
    _write_spectrum_file(folder, 1)

    result = _load_baseline_color_spectrum_for_color_to_mass(
        baseline_color_spectrum_folder=folder,
        required_labels={0, 1},
    )

    assert set(result.keys()) == {0, 1}


def test_stored_ignore_mask_skips_same_suffix_directory(tmp_path):
    """A directory sorted before a stored calibration cannot hide its mode."""
    folder = tmp_path / "color_path_interpretation"
    folder.mkdir()
    (folder / "a.json").mkdir()
    assert _stored_ignore_mask(tmp_path) is None
    (folder / "b.json").write_text('{"ignore_mode": "spectrum"}')
    assert _stored_ignore_mask(tmp_path) == "spectrum"


def test_baseline_spectrum_presence_requires_files(tmp_path):
    """A matching directory alone does not mean a calibration was stored."""
    (tmp_path / "color_spectrum_1.json").mkdir()
    assert (
        _load_baseline_color_spectrum_for_color_to_mass(
            baseline_color_spectrum_folder=tmp_path, required_labels={0}
        )
        is None
    )
    _write_spectrum_file(tmp_path, 0)
    result = _load_baseline_color_spectrum_for_color_to_mass(
        baseline_color_spectrum_folder=tmp_path, required_labels={0}
    )
    assert set(result) == {0}
