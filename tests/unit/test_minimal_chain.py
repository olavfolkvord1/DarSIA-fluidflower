"""The minimal chain on a coarse run grid: image mapping, material light
correction, phase model, phase rules and phase maps as expert knowledge."""

import datetime
import json

import numpy as np

import darsia
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
from darsia.presets.workflows.config.corrections import CorrectionsConfig
from darsia.presets.workflows.rig import Rig
from tests.unit.test_phase_rules import GAS_THRESHOLD, SETTINGS, START

W, D, G, U = darsia.PhaseLabel

COLOURS = np.array([[0.0, 0.0, 0.0], [0.3, 0.2, 0.1], [0.2, 0.25, 0.3]])
"""Colours of sand 1 (upper half) and sand 2 (lower half)."""


def _head(weight: float, intercept: float) -> dict:
    """A delivered head on the change of red against the baseline."""
    names = darsia.PHASE_FEATURE_NAMES
    return {
        "feature_names": list(names),
        "feature_mean": [0.0] * len(names),
        "feature_scale": [1.0] * len(names),
        "coefficients": [weight * (name == "delta_r") for name in names],
        "intercept": intercept,
        "texture_sigma": 2.0,
        "feature_contract": {"input_color_space": "camera-white-relative-linear-rgb"},
    }


def test_minimal_chain_runs_on_a_coarse_grid(tmp_path):
    """On a 5 mm grid in single precision the light is removed (on a fitting grid
    capped at the run grid), the plume is CO2 in the phase map, and the analysis
    keeps the field only there."""
    # Sensor images under a sideways light; the frame has a redder plume.
    facies = np.ones((120, 180), int)
    facies[60:] = 2
    colour = COLOURS[np.pad(facies, ((0, 1), (0, 1)), mode="edge")]
    light = np.exp(0.2 * np.linspace(-1, 1, 181))[None, :, None]
    plume = np.zeros((121, 181, 3))
    plume[40:80, 70:110, 0] = 0.3
    for name, image in (("baseline", colour), ("frame", colour + plume)):
        image = darsia.OpticalImage(image * light, color_space="RGB")
        image.save(tmp_path / f"{name}.npz")
    # A mapping onto a 2.5 mm grid of 0.3 x 0.45 m, with the sand map on it.
    i, j = np.mgrid[0:120, 0:180].astype(np.float32)
    darsia.ImageMappingCorrection(
        j, i, source_shape=(121, 181), dimensions=[0.3, 0.45]
    ).save(tmp_path / "mapping.npz")
    darsia.ScalarImage(facies, dimensions=[0.3, 0.45]).save(tmp_path / "facies.npz")
    targets = "".join(f'"{m}" = {np.log(COLOURS[m] * 1e3).tolist()}\n' for m in (1, 2))
    (tmp_path / "config.toml").write_text(
        '[corrections.grid]\nprecision = "float32"\npixel_size_mm = 5.0\n'
        "area_factor = 1.0337\n"
        f'[corrections.mapping]\npath = "{(tmp_path / "mapping.npz").as_posix()}"\n'
        "[corrections.material_illumination]\nfit_shape = [150, 225]\n"
        "block_size = 6\nedge_margin = 0.0\ndn_scale = 1000.0\n"
        f"[corrections.material_illumination.targets]\n{targets}"
        "[corrections.material_illumination.fit]\nspatial_ridge = 0.002\n"
        "offset_ridge = 0.02\nresidual_constant_ridge = 0.02\n"
        "residual_spatial_ridge = 0.01\niterations = 8\nhuber_threshold = 1.5\n"
        "clip_level = 0.98\ndark_level = 8.0\nminimum_block_cells = 32\n"
        "chroma_cutoff = 0.08\nchroma_sigmas = 3.5\n"
    )
    config = CorrectionsConfig().load(tmp_path / "config.toml")
    hour = datetime.timedelta(hours=1)
    (tmp_path / "protocol.csv").write_text(
        "id,location_x,location_y,start,end,rate_g/s\n"
        f"1,0.225,0.1,{START - hour},{START + hour},0.01\n"
    )
    heads = tmp_path / "model" / "models" / "generic"
    heads.mkdir(parents=True)
    for role, head in (("co2_presence", (20, -3)), ("gas_within_co2", (0, -5))):
        (heads / f"{role}_signal.json").write_text(json.dumps(_head(*head)))
    policy = {"gas_within_co2": {"presence_threshold": GAS_THRESHOLD}}
    (tmp_path / "model" / "phase_decision_policy.json").write_text(
        json.dumps({"models": {"generic": policy}})
    )

    rig = Rig()
    rig.setup_reading(
        tmp_path / "baseline.npz", experiment=None, corrections_config=config
    )
    rig.setup_facies(tmp_path / "facies.npz", apply_corrections=True)
    rig.setup_color_corrections(config)
    baseline, frame = (
        darsia.imread(tmp_path / f"{name}.npz", transformations=rig.corrections)
        for name in ("baseline", "frame")
    )
    pixel_mm = [size * 1000 for size in frame.voxel_size]
    # A folder in the delivered layout records no canvas: read by explicit choice.
    model = darsia.read_phase_model(
        tmp_path / "model", pixel_size_mm=0.5, canvas=rig.canvas, accept_canvas=True
    )
    co2, gas = darsia.phase_probabilities(
        model,
        frame.img,
        baseline.img,
        pixel_size_mm=pixel_mm[0],
    )
    ((labels, _),) = darsia.phase_maps(
        [(co2, gas)],
        [START],
        rig.facies,
        np.ones(rig.facies.img.shape, bool),
        rig.pixel_area.img * 1e6,
        {1: 1.0, 2: 1.0},
        darsia.InjectionProtocol(tmp_path / "protocol.csv"),
        SETTINGS,
        gas_threshold=model.gas_threshold,
    )
    grid = {"dimensions": frame.dimensions, "origin": frame.origin}
    (tmp_path / "maps").mkdir()
    darsia.ScalarImage(labels, **grid).save(tmp_path / "maps" / "frame.npz")
    field = darsia.ScalarImage(np.full((60, 90), 0.8), name="frame.npz", **grid)
    kept = ExpertKnowledgeAdapter(phase_maps=tmp_path / "maps").apply(
        field, "concentration_aq"
    )

    inside = np.zeros((60, 90), bool)
    inside[20:40, 35:55] = True
    assert pixel_mm == [5.0, 5.0]
    assert rig.illumination_correction.field_shape == (60, 90)
    assert frame.img.dtype == co2.dtype == np.float32
    np.testing.assert_allclose(baseline.img, COLOURS[rig.facies.img], rtol=2e-3)
    np.testing.assert_array_equal(labels, np.where(inside, D, W))
    np.testing.assert_array_equal(kept.img, np.where(inside, 0.8, 0.0))
