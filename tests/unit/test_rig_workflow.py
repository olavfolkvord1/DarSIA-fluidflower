"""Unit tests for Rig correction workflow setup behavior."""

import errno
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import darsia
from darsia.presets.workflows import rig as rig_module
from darsia.presets.workflows.config.corrections import (
    ColorCorrectionConfig,
    CorrectionsConfig,
    IlluminationCorrectionConfig,
    RelativeColorCorrectionConfig,
)
from darsia.presets.workflows.rig import Rig
from darsia.utils import results


@dataclass
class DummyImage:
    applied: list[str] | None = None

    def __post_init__(self) -> None:
        if self.applied is None:
            self.applied = []

    def copy(self):
        return DummyImage(self.applied.copy())


class DummyCorrection:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, image: DummyImage) -> DummyImage:
        image.applied.append(self.name)
        return image


def test_rig_corrections_property_concatenates_shape_then_color():
    rig = Rig()
    rig.shape_corrections = [DummyCorrection("shape_a"), DummyCorrection("shape_b")]
    rig.color_corrections = [DummyCorrection("color_a")]

    all_corrections = rig.corrections

    assert [c.name for c in all_corrections] == ["shape_a", "shape_b", "color_a"]


def test_setup_color_corrections_requires_shape_corrected_baseline():
    rig = Rig()
    config = CorrectionsConfig()

    with pytest.raises(RuntimeError, match="Shape-corrected baseline missing"):
        rig.setup_color_corrections(config)


def test_setup_color_corrections_illumination_return_is_assigned_and_appended(
    monkeypatch,
):
    rig = Rig()
    rig.shape_corrected_baseline = DummyImage()

    illumination = DummyCorrection("illumination")
    color = DummyCorrection("color")
    monkeypatch.setattr(
        rig,
        "setup_illumination_correction",
        lambda *_args, **_kwargs: illumination,
    )

    from darsia.presets.workflows import rig as rig_module

    monkeypatch.setattr(
        rig_module.darsia,
        "find_colorchecker",
        lambda *_args, **_kwargs: (None, "dummy-roi"),
    )
    monkeypatch.setattr(
        rig_module.darsia,
        "ColorCorrection",
        lambda *_args, **_kwargs: color,
    )

    config = CorrectionsConfig(
        illumination=IlluminationCorrectionConfig(),
        color=ColorCorrectionConfig(colorchecker="upper_left"),
    )
    rig.setup_color_corrections(config)

    assert rig.illumination_correction is illumination
    assert rig.color_corrections == [illumination, color]
    assert rig.baseline.applied == ["illumination", "color"]


def test_setup_color_corrections_loads_relative_color_from_path_and_preserves_order(
    monkeypatch,
):
    rig = Rig()
    rig.shape_corrected_baseline = DummyImage()
    rig.shape_corrections = [DummyCorrection("shape")]

    illumination = DummyCorrection("illumination")
    color = DummyCorrection("color")
    monkeypatch.setattr(
        rig,
        "setup_illumination_correction",
        lambda *_args, **_kwargs: illumination,
    )

    from darsia.presets.workflows import rig as rig_module

    monkeypatch.setattr(
        rig_module.darsia,
        "find_colorchecker",
        lambda *_args, **_kwargs: (None, "dummy-roi"),
    )
    monkeypatch.setattr(
        rig_module.darsia,
        "ColorCorrection",
        lambda *_args, **_kwargs: color,
    )

    class DummyRelativeColorCorrection(DummyCorrection):
        def __init__(self, *_args, **_kwargs):
            super().__init__("relative")
            self.loaded = None

        def load(self, path):
            self.loaded = path

    monkeypatch.setattr(
        rig_module.darsia,
        "RelativeColorCorrection",
        DummyRelativeColorCorrection,
    )

    config = CorrectionsConfig(
        illumination=IlluminationCorrectionConfig(),
        relative_color=RelativeColorCorrectionConfig(path=Path("relative.npz")),
        color=ColorCorrectionConfig(colorchecker="upper_left"),
    )

    rig.setup_color_corrections(config)

    assert [c.name for c in rig.color_corrections] == [
        "illumination",
        "relative",
        "color",
    ]
    assert rig.baseline.applied == ["illumination", "relative", "color"]
    assert rig.relative_color_correction.loaded.name == "relative.npz"


def test_setup_color_corrections_raises_for_noninteractive_relative_color_images():
    rig = Rig()
    rig.shape_corrected_baseline = DummyImage()
    rig.shape_corrections = []

    config = CorrectionsConfig(
        relative_color=RelativeColorCorrectionConfig(
            images=[Path("calibration_a.jpg")], interactive=False
        )
    )

    with pytest.raises(ValueError, match="Interactive calibration is required"):
        rig.setup_color_corrections(config)


def test_setup_illumination_correction_creates_uninitialized_correction_when_config_is_none(
    monkeypatch,
):
    rig = Rig()
    rig.shape_corrected_baseline = DummyImage()

    from darsia.presets.workflows import rig as rig_module

    class DummyIlluminationCorrection:
        def __init__(self):
            self.setup_called = False

        def setup(self, **_kwargs):
            self.setup_called = True

        def select_random_samples(self, **_kwargs):
            return []

    monkeypatch.setattr(
        rig_module.darsia,
        "IlluminationCorrection",
        DummyIlluminationCorrection,
    )

    correction = rig.setup_illumination_correction(None)

    assert isinstance(correction, DummyIlluminationCorrection)
    assert correction.setup_called is False


def test_import_from_csv_supports_matrix_layout(tmp_path):
    rig = Rig()
    rig.baseline = darsia.ScalarImage(
        np.zeros((2, 3), dtype=float),
        dimensions=[2.0, 3.0],
    )

    values = np.array(
        [
            [0.5, 0.5, 3.0],
            [1.5, 0.5, 3.0],
            [2.5, 0.5, 3.0],
            [0.5, 1.5, 6.0],
            [1.5, 1.5, 6.0],
            [2.5, 1.5, 6.0],
        ]
    )
    csv_path = tmp_path / "result.csv"
    np.savetxt(csv_path, values, delimiter=",")

    image = rig.import_from_csv(csv_path)

    assert isinstance(image, darsia.ScalarImage)
    assert np.allclose(image.img, np.array([[6.0, 6.0, 6.0], [3.0, 3.0, 3.0]]))


def test_import_from_csv_supports_coordinate_layout(tmp_path):
    rig = Rig()
    baseline = darsia.ScalarImage(
        np.zeros((2, 2), dtype=float),
        dimensions=[2.0, 2.0],
    )
    rig.baseline = baseline

    reference = darsia.ScalarImage(
        np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float),
        **baseline.metadata(),
    )
    csv_path = tmp_path / "coords.csv"
    reference.to_csv(csv_path, delimiter=",")

    image = rig.import_from_csv(csv_path)

    assert isinstance(image, darsia.ScalarImage)
    assert np.allclose(image.img, reference.img)


def test_rig_maps_baseline_and_reloads_the_mapping(tmp_path):
    """A configured mapping corrects the baseline once and survives save/load."""
    i, j = np.mgrid[0:4, 0:5].astype(np.float32)
    mapping = darsia.ImageMappingCorrection(
        1.5 + 1.25 * j, 0.75 + 1.5 * i, source_shape=(8, 10), dimensions=[0.4, 0.5]
    )
    mapping.save(tmp_path / "mapping.npz")
    rows, cols = np.mgrid[0:8, 0:10].astype(np.float32)
    source = np.stack([rows, cols, rows + cols], axis=-1)
    darsia.OpticalImage(source, color_space="RGB").save(tmp_path / "baseline.npz")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'[corrections.mapping]\npath = "{(tmp_path / "mapping.npz").as_posix()}"\n'
    )

    rig = Rig()
    rig.setup_reading(
        tmp_path / "baseline.npz",
        experiment=None,
        corrections_config=CorrectionsConfig().load(config_path),
    )
    rig.baseline_path = tmp_path / "baseline.npz"
    rig.color_corrections = []
    with pytest.warns(UserWarning):
        rig.save(tmp_path / "rig")
    loaded = Rig()
    loaded.load_corrections(tmp_path / "rig")
    corrections = loaded.corrections
    reread = darsia.imread(tmp_path / "baseline.npz", transformations=corrections)

    assert rig.baseline.img.shape == (4, 5, 3)
    assert rig.baseline.dimensions == [0.4, 0.5]
    assert any(isinstance(c, darsia.ImageMappingCorrection) for c in corrections)
    np.testing.assert_array_equal(reread.img, rig.baseline.img)
    # The corrected baseline is not corrected again.
    with pytest.raises(ValueError, match="not on the source grid"):
        darsia.imread(tmp_path / "rig" / "baseline.npz", transformations=corrections)


@pytest.mark.parametrize(
    "grid, factor, dtype",
    [
        ("", 1, np.float64),
        (
            '[corrections.grid]\nprecision = "float32"\npixel_size_mm = 300.0\n'
            "area_factor = 1.25\n",
            3,
            np.float32,
        ),
    ],
)
def test_run_grid_sets_precision_and_scales_once(tmp_path, grid, factor, dtype):
    """Without a grid nothing changes. With one, the images are converted first
    and averaged once over the grid's cells, the facies take each cell's
    majority, each cell's area sums its pixels' measured areas, and the saved
    corrections give the same baseline."""
    i, j = np.mgrid[0:6, 0:9].astype(np.float32)
    mapping = darsia.ImageMappingCorrection(
        j, i, source_shape=(7, 10), dimensions=[0.6, 0.9]
    )
    mapping.save(tmp_path / "mapping.npz")
    source = np.random.default_rng(0).random((7, 10, 3))
    darsia.OpticalImage(source, color_space="RGB").save(tmp_path / "baseline.npz")
    facies = np.ones((6, 9), int)
    facies[0::3, 0::3] = facies[1::3, 1::3] = 2
    darsia.ScalarImage(facies, dimensions=[0.6, 0.9]).save(tmp_path / "facies.npz")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'[corrections.mapping]\npath = "{(tmp_path / "mapping.npz").as_posix()}"\n'
        + grid
    )

    rig = Rig()
    rig.setup_reading(
        tmp_path / "baseline.npz",
        experiment=None,
        corrections_config=CorrectionsConfig().load(config_path),
    )
    rig.setup_facies(tmp_path / "facies.npz", apply_corrections=True)
    rig.baseline_path = tmp_path / "baseline.npz"
    rig.color_corrections = []
    with pytest.warns(UserWarning):
        rig.save(tmp_path / "rig")
    loaded = Rig()
    loaded.load_corrections(tmp_path / "rig")
    reread = darsia.imread(
        tmp_path / "baseline.npz", transformations=loaded.corrections
    )

    rows, cols = 6 // factor, 9 // factor
    blocks = source[:6, :9].astype(dtype).reshape(rows, factor, cols, factor, 3)
    assert rig.baseline.img.dtype == dtype
    assert rig.baseline.dimensions == [0.6, 0.9]
    np.testing.assert_allclose(rig.baseline.img, blocks.mean(axis=(1, 3)), rtol=1e-6)
    np.testing.assert_array_equal(
        rig.facies.img, facies if factor == 1 else np.ones((2, 3))
    )
    np.testing.assert_array_equal(reread.img, rig.baseline.img)
    if grid:
        # Nine pixels of 0.1 m x 0.1 m per cell, at the optical area factor.
        assert rig.pixel_area.img.dtype == dtype
        np.testing.assert_allclose(rig.pixel_area.img, np.full((2, 3), 0.1125))
        saved = darsia.imread(tmp_path / "rig" / "pixel_area.npz")
        np.testing.assert_array_equal(saved.img, rig.pixel_area.img)
    else:
        assert rig.pixel_area is None


FIT = (
    "[corrections.material_illumination.fit]\n"
    "spatial_ridge = 0.002\noffset_ridge = 0.02\nresidual_constant_ridge = 0.02\n"
    "residual_spatial_ridge = 0.01\niterations = 8\nhuber_threshold = 1.5\n"
    "clip_level = 0.98\ndark_level = 8.0\nminimum_block_cells = 32\n"
    "chroma_cutoff = 0.08\nchroma_sigmas = 3.5\n"
)
"""The fit's settings our production ran with."""


def test_rig_canvas_records_settings_not_locations(tmp_path):
    """The canvas holds the format, the corrections and the grid in rig metres; a
    moved project keeps it, a changed setting changes it, and the rig saves it.
    Whole metric cells keep the canvas' extent from before they extend it, and the
    run's optical area factor is not part of the canvas."""
    i, j = np.mgrid[0:4, 0:5].astype(np.float32)
    mapping = darsia.ImageMappingCorrection(
        1.5 + 1.25 * j, 0.75 + 1.5 * i, source_shape=(8, 10), dimensions=[0.4, 0.5]
    )
    rows, cols = np.mgrid[0:8, 0:10].astype(np.float32)
    canvases = []
    float32 = '[corrections.type]\ntarget_type = "float32"\n'
    cells = (
        '[corrections.grid]\nprecision = "float32"\npixel_size_mm = 150.0\n'
        "area_factor = 1.0337\n"
    )
    for project, extra in (("a", ""), ("b", ""), ("b", float32), ("c", cells)):
        folder = tmp_path / project
        folder.mkdir(exist_ok=True)
        mapping.save(folder / "mapping.npz")
        darsia.OpticalImage(
            np.stack([rows, cols, rows + cols], -1), color_space="RGB"
        ).save(folder / "baseline.npz")
        mapping_path = (folder / "mapping.npz").as_posix()
        (folder / "config.toml").write_text(
            f'{extra}[corrections.mapping]\npath = "{mapping_path}"\n'
        )
        rig = Rig()
        rig.setup_reading(
            folder / "baseline.npz",
            experiment=None,
            corrections_config=CorrectionsConfig().load(folder / "config.toml"),
        )
        canvases.append(rig.canvas)
    rig.baseline_path = tmp_path / "b" / "baseline.npz"
    rig.color_corrections = []
    with pytest.warns(UserWarning):
        rig.save(tmp_path / "rig")
    saved = json.loads((tmp_path / "rig" / "meta_data.json").read_text())

    assert (
        canvases[0]
        == canvases[1]
        == {
            "source": {"format": "NPZ"},
            "corrections": {"mapping": {"path": "mapping.npz"}},
            "grid": {
                "shape": [4, 5],
                "dimensions": [0.4, 0.5],
                "origin": [0.0, 0.4],
                "extent": {"dimensions": [0.4, 0.5], "origin": [0.0, 0.4]},
            },
        }
    )
    assert canvases[2]["corrections"]["type"] == {"target_type": "float32"}
    # 150 mm cells extend the 0.4 x 0.5 m canvas to 3 x 4 cells upwards and right.
    assert canvases[3]["corrections"]["grid"] == {
        "precision": "float32",
        "pixel_size_mm": 150.0,
    }
    assert canvases[3]["grid"] == {
        "shape": [3, 4],
        "dimensions": [0.45, 0.6],
        "origin": [0.0, 0.45],
        "extent": {"dimensions": [0.4, 0.5], "origin": [0.0, 0.4]},
    }
    assert saved["canvas"] == canvases[3]


def _material_rig(tmp_path) -> tuple[Rig, CorrectionsConfig, np.ndarray]:
    """Rig with a shape-corrected baseline under a smooth illumination, and the
    configuration of its material illumination correction."""
    rows, cols = np.mgrid[0:200, 0:300]
    x = (cols + 0.5) * 2 / 300 - 1
    gain = np.exp(np.stack([0.2 + 0.3 * x, 0.1 - 0.2 * x, 0.1 * x], axis=-1))
    facies = np.ones((200, 300), dtype=int)
    facies[70:140] = 2
    facies[140:] = 3
    colours = {1: [0.3, 0.2, 0.1], 2: [0.2, 0.25, 0.3], 3: [0.4, 0.35, 0.2]}
    image = np.array([colours[m] for m in facies.ravel()]).reshape(200, 300, 3)
    rig = Rig()
    rig.shape_corrected_baseline = darsia.OpticalImage(
        image / gain, color_space="RGB", dimensions=[0.2, 0.3]
    )
    rig.facies = darsia.ScalarImage(facies, dimensions=[0.2, 0.3])
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[corrections.material_illumination]\n"
        "fit_shape = [100, 150]\nblock_size = 10\nedge_margin = 0.0\n"
        "dn_scale = 1000.0\n[corrections.material_illumination.targets]\n"
        + "".join(
            f'"{m}" = {np.log(np.array(colour) * 1000).tolist()}\n'
            for m, colour in colours.items()
        )
        + FIT
    )
    return rig, CorrectionsConfig().load(config_path), image


def test_material_illumination_is_fitted_and_reloaded(tmp_path):
    """The configured material correction is fitted on the baseline and saved."""
    rig, config, image = _material_rig(tmp_path)

    rig.setup_color_corrections(config)
    rig.baseline_path = tmp_path / "baseline.npz"
    rig.shape_corrections = []
    with pytest.warns(UserWarning):
        rig.save(tmp_path / "rig")
    loaded = Rig()
    loaded.load_corrections(tmp_path / "rig")

    assert isinstance(
        rig.illumination_correction, darsia.MaterialIlluminationCorrection
    )
    np.testing.assert_allclose(rig.baseline.img, image, rtol=2e-3)
    assert [type(c) for c in loaded.corrections] == [
        darsia.MaterialIlluminationCorrection
    ]
    np.testing.assert_array_equal(
        loaded.corrections[0](rig.shape_corrected_baseline).img, rig.baseline.img
    )


def test_material_illumination_reads_its_reference_run(tmp_path):
    """The configured reference run is read and fits on its grid; the run as its
    own reference gives the same correction, which is saved with that grid."""
    rig, config, _ = _material_rig(tmp_path)
    rig.setup_color_corrections(config)
    rig.shape_corrected_baseline.save(tmp_path / "reference.npz")
    rig.facies.save(tmp_path / "facies.npz")
    referenced, _, _ = _material_rig(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text(
        path.read_text()
        + "[corrections.material_illumination.reference]\n"
        + f'baseline = "{(tmp_path / "reference.npz").as_posix()}"\n'
        + f'facies = "{(tmp_path / "facies.npz").as_posix()}"\n'
        + "homography = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n"
    )

    referenced.setup_color_corrections(CorrectionsConfig().load(path))
    correction = referenced.illumination_correction
    correction.save(tmp_path / "correction.npz")
    loaded = darsia.MaterialIlluminationCorrection()
    loaded.load(tmp_path / "correction.npz")

    assert loaded.grid_to_field is not None
    np.testing.assert_array_equal(
        correction.common_coefficients, rig.illumination_correction.common_coefficients
    )
    np.testing.assert_array_equal(referenced.baseline.img, rig.baseline.img)
    np.testing.assert_array_equal(
        loaded.correct_array(referenced.shape_corrected_baseline.img),
        referenced.baseline.img,
    )


def test_official_and_material_illumination_give_one_form(monkeypatch, tmp_path):
    """Swapping the illumination method changes the correction, not the result form."""
    material_rig, config, _ = _material_rig(tmp_path)
    material_rig.setup_color_corrections(config)
    gain = material_rig.illumination_correction.gain(np.float64)

    # The official illumination correction with the same gain field.
    official = darsia.IlluminationCorrection()
    official.colorspace = "rgb"
    official.local_scaling = [darsia.ScalarImage(gain[..., i]) for i in range(3)]
    official_rig, _, _ = _material_rig(tmp_path)
    monkeypatch.setattr(
        official_rig, "setup_illumination_correction", lambda *_a, **_k: official
    )
    official_rig.setup_color_corrections(
        CorrectionsConfig(illumination=IlluminationCorrectionConfig())
    )

    for rig in (material_rig, official_rig):
        assert isinstance(rig.illumination_correction, darsia.BaseCorrection)
        assert isinstance(rig.baseline, darsia.OpticalImage)
    assert official_rig.baseline.dimensions == material_rig.baseline.dimensions
    np.testing.assert_array_equal(
        official_rig.baseline.origin, material_rig.baseline.origin
    )
    np.testing.assert_allclose(
        official_rig.baseline.img, material_rig.baseline.img, rtol=1e-12
    )


def test_a_rig_saved_again_without_a_grid_has_no_pixel_areas(tmp_path):
    """Saving without pixel areas leaves none of an earlier save in the folder."""
    i, j = np.mgrid[0:6, 0:9].astype(np.float32)
    darsia.ImageMappingCorrection(
        j, i, source_shape=(7, 10), dimensions=[0.6, 0.9]
    ).save(tmp_path / "mapping.npz")
    darsia.OpticalImage(np.ones((7, 10, 3)), color_space="RGB").save(
        tmp_path / "baseline.npz"
    )
    mapping = (
        f'[corrections.mapping]\npath = "{(tmp_path / "mapping.npz").as_posix()}"\n'
    )
    grid = '[corrections.grid]\nprecision = "float64"\narea_factor = 1.25\n'
    for text in (mapping + grid, mapping):
        (tmp_path / "config.toml").write_text(text)
        rig = Rig()
        rig.setup_reading(
            tmp_path / "baseline.npz",
            experiment=None,
            corrections_config=CorrectionsConfig().load(tmp_path / "config.toml"),
        )
        rig.baseline_path = tmp_path / "baseline.npz"
        rig.color_corrections = []
        with pytest.warns(UserWarning):
            rig.save(tmp_path / "rig")

    loaded = Rig()
    loaded.load_corrections(tmp_path / "rig")
    assert loaded.pixel_area is None


def test_a_rig_records_its_shape_corrected_baseline_and_pixel_areas(
    tmp_path, monkeypatch
):
    """A saved rig records whether it has a shape-corrected baseline and pixel
    areas, and the files of its corrections: a file it records that the share does
    not find raises a file not found, never a fallback to the baseline, to no areas
    or to no corrections; a rig saved before the record has them where the files
    are found, and a listing of a share that is out raises."""
    i, j = np.mgrid[0:6, 0:9].astype(np.float32)
    darsia.ImageMappingCorrection(
        j, i, source_shape=(7, 10), dimensions=[0.6, 0.9]
    ).save(tmp_path / "mapping.npz")
    darsia.OpticalImage(np.ones((7, 10, 3)), color_space="RGB").save(
        tmp_path / "baseline.npz"
    )
    (tmp_path / "config.toml").write_text(
        f'[corrections.mapping]\npath = "{(tmp_path / "mapping.npz").as_posix()}"\n'
        '[corrections.grid]\nprecision = "float64"\narea_factor = 1.25\n'
    )
    rig = Rig()
    rig.setup_reading(
        tmp_path / "baseline.npz",
        experiment=None,
        corrections_config=CorrectionsConfig().load(tmp_path / "config.toml"),
    )
    rig.baseline_path = tmp_path / "baseline.npz"
    rig.color_corrections = []
    folder = tmp_path / "rig"
    with pytest.warns(UserWarning):
        rig.save(folder)
    meta = json.loads((folder / "meta_data.json").read_text())
    assert meta["shape_corrected_baseline"] is True and meta["pixel_area"] is True

    def missing(name):
        """The file a load of the rig does not find first, with ``name`` hidden."""
        (folder / name).rename(tmp_path / f"hidden-{name}")
        try:
            with pytest.raises(FileNotFoundError) as error:
                Rig.load(folder)
        finally:
            (tmp_path / f"hidden-{name}").rename(folder / name)
        return Path(error.value.filename).name

    # The rig has no depth map: a load that reads every recorded file stops there.
    assert missing("shape_corrected_baseline.npz") == "shape_corrected_baseline.npz"
    assert missing("pixel_area.npz") == "pixel_area.npz"
    shape = meta["shape_corrections"]
    assert meta["color_corrections"] == [] and len(shape) == len(rig.shape_corrections)
    assert sorted(shape) == sorted(p.name for p in folder.glob("shape_correction_*"))
    assert missing(shape[-1]) == shape[-1]
    del meta["shape_corrected_baseline"], meta["pixel_area"]
    del meta["shape_corrections"], meta["color_corrections"]
    (folder / "meta_data.json").write_text(json.dumps(meta))
    assert missing("shape_corrected_baseline.npz") == "depth.npz"
    assert missing("pixel_area.npz") == "depth.npz"
    assert missing(shape[-1]) == "depth.npz"

    # Before the record its optional files are looked for: a share that is out
    # raises once its interruption outlasts the grace, never reading none.
    for name in ("pixel_area.npz", "shape_corrected_baseline.npz"):
        target = folder / name
        out = OSError(errno.ENOENT, "The network path was not found", str(target), 53)
        if getattr(out, "winerror", None) != 53:
            continue  # the share's own error has a network code on Windows only
        stat, clock = os.stat, []

        def outage(path, *args, target=target, out=out, **kwargs):
            if Path(path) == target:
                raise out
            return stat(path, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(results, "sleep", clock.append)
            patched.setattr(results, "monotonic", lambda: sum(clock))
            patched.setattr(os, "stat", outage)
            with pytest.raises(results.Unavailable):
                Rig.load(folder)

    # Before the record the corrections are listed: a share that is out raises.
    out = OSError(errno.ENETDOWN, "The network is down", str(folder))

    def outage(path):
        raise out

    monkeypatch.setattr(rig_module.os, "scandir", outage)
    with pytest.raises(OSError) as raised:
        Rig.load(folder)
    assert raised.value is out


def test_a_rig_saved_again_drops_obsolete_split_corrections(tmp_path):
    """Re-saving keeps only the current pipelines and leaves unrelated files alone."""
    rig = Rig()
    rig.baseline_path = tmp_path / "baseline.npz"
    rig.baseline = darsia.OpticalImage(np.ones((4, 5, 3)), color_space="RGB")
    i, j = np.mgrid[0:4, 0:5].astype(np.float32)
    mapping = darsia.ImageMappingCorrection(
        j, i, source_shape=(4, 5), dimensions=[0.4, 0.5]
    )
    rig.shape_corrections = [darsia.TypeCorrection(np.float32), mapping]
    rig.color_corrections = [
        darsia.MaterialIlluminationCorrection(
            np.zeros((6, 3)), {}, np.zeros((4, 5), dtype=int), (4, 5)
        )
    ]
    folder = tmp_path / "rig"
    with pytest.warns(UserWarning):
        rig.save(folder)
    before = Rig()
    before.load_corrections(folder)
    assert len(before.shape_corrections) == 2
    assert len(before.color_corrections) == 1
    unrelated = folder / "unrelated.npz"
    unrelated.write_bytes(b"unrelated saved data")

    # Removing the first correction also changes the mapping's saved index.
    rig.shape_corrections = [mapping]
    rig.color_corrections = []
    with pytest.warns(UserWarning):
        rig.save(folder)

    loaded = Rig()
    loaded.load_corrections(folder)
    assert len(loaded.shape_corrections) == 1
    assert isinstance(loaded.shape_corrections[0], darsia.ImageMappingCorrection)
    assert loaded.color_corrections == []
    assert unrelated.read_bytes() == b"unrelated saved data"
    np.testing.assert_array_equal(
        loaded.shape_corrections[0](rig.baseline).img, mapping(rig.baseline).img
    )


def test_rig_listed_skips_same_suffix_directories(tmp_path):
    """A legacy correction list contains sorted files, never child folders."""
    for name in ("correction_2.npz", "correction_1.npz", "other.npz"):
        (tmp_path / name).touch()
    (tmp_path / "correction_0.npz").mkdir()
    assert rig_module._listed(tmp_path, "correction_") == [
        tmp_path / "correction_1.npz",
        tmp_path / "correction_2.npz",
    ]
