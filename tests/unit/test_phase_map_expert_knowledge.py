"""Unit tests for phase maps as dynamic expert knowledge in the analysis."""

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
from darsia.presets.workflows.analysis.scalar_products import analysis_scalar_products

W, D, G, U = darsia.PhaseLabel


def _field(value, name):
    """A constant scalar field of the frame ``name``."""
    return darsia.ScalarImage(np.full((2, 4), value), dimensions=[0.2, 0.4], name=name)


def test_each_frame_keeps_co2_inside_its_own_phase_map(tmp_path):
    """The analysis reads each frame's own map; unknown stays unknown."""
    maps = {"a": [[W, D, G, U], [D, D, D, D]], "b": [[G, G, W, W], [U, W, W, W]]}
    for stem, labels in maps.items():
        phase = darsia.ScalarImage(np.array(labels, np.uint8), dimensions=[0.2, 0.4])
        phase.save(tmp_path / f"{stem}.npz")
    adapter = ExpertKnowledgeAdapter(phase_maps=tmp_path)
    for stem, labels in maps.items():
        name = f"{stem}.jpg"
        result = darsia.SimpleMassAnalysisResults(
            name=name,
            date=None,
            time=0.0,
            mass=_field(1.0, name),
            mass_g=_field(1.0, name),
            mass_aq=_field(1.0, name),
            saturation_g=_field(0.5, name),
            color_signal=_field(1.0, name),
            concentration_aq=_field(0.8, name),
        )
        products, _ = analysis_scalar_products(
            mass_analysis_result=result, expert_knowledge_adapter=adapter
        )
        labels = np.array(labels)
        unknown = np.where(labels == U, np.nan, 0)
        np.testing.assert_array_equal(
            products["concentration_aq"].img,
            np.where((labels == D) | (labels == G), 0.8, unknown),
        )
        np.testing.assert_array_equal(
            products["saturation_g"].img, np.where(labels == G, 0.5, unknown)
        )


def test_a_cached_phase_map_on_another_grid_is_refused(tmp_path):
    """The grid check also runs when the image's map is already read."""
    phase = darsia.ScalarImage(np.zeros((2, 4), np.uint8), dimensions=[0.2, 0.4])
    phase.save(tmp_path / "a.npz")
    adapter = ExpertKnowledgeAdapter(phase_maps=tmp_path)
    adapter.apply(_field(1.0, "a.jpg"), "saturation_g")
    moved = darsia.ScalarImage(
        np.ones((2, 4)), dimensions=[0.2, 0.4], origin=[0.1, 0.2], name="a.jpg"
    )
    with pytest.raises(ValueError, match="another grid"):
        adapter.apply(moved, "saturation_g")
