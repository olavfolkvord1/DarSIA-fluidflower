from __future__ import annotations

from pathlib import Path

import numpy as np

import darsia


def _signal() -> darsia.Image:
    return darsia.Image(
        np.linspace(0.0, 1.0, 35, dtype=np.float64).reshape(5, 7),
        width=1.4,
        height=1.0,
        space_dim=2,
        scalar=True,
    )


def test_titration_flash_initial_ph_and_blend_round_trip(tmp_path: Path) -> None:
    flash = darsia.TitrationFlash(
        0.0,
        1.0,
        0.0,
        1.0,
        initial_pH=8.1,
        aqueous_blend=0.2,
        n_lut=96,
    )
    concentration, saturation = flash(_signal())

    assert np.all(np.isfinite(concentration.img))
    assert np.all(np.diff(concentration.img.ravel()) >= 0.0)
    assert float(concentration.img.min()) == 0.0
    assert np.isclose(float(concentration.img.max()), 1.0)
    assert np.allclose(saturation.img, _signal().img)

    path = tmp_path / "titration-flash"
    flash.save(path)
    restored = darsia.SimpleFlash.load(path)
    restored_concentration, restored_saturation = restored(_signal())

    assert isinstance(restored, darsia.TitrationFlash)
    assert restored.initial_pH == 8.1
    assert restored.aqueous_blend == 0.2
    assert np.allclose(restored_concentration.img, concentration.img)
    assert np.allclose(restored_saturation.img, saturation.img)

