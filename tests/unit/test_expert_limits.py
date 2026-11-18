"""Expert limits at the probability, configuration and workflow layers."""

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.config.expert_knowledge import ExpertKnowledgeConfig
from darsia.presets.workflows.config.roi_registry import RoiRegistry

ROI = """
[[roi]]
name = "top"
shape = "label"
label_index = 3

[[roi]]
name = "corner"
shape = "box"
corner_1 = [0.0, 0.0]
corner_2 = [0.1, 0.1]
"""


def _probabilities(dtype=np.float32):
    co2 = np.array([[0.9, 0.95], [np.nan, 0.2]], dtype)
    gas = np.array([[0.8, 0.1], [np.nan, 0.7]], dtype)
    return co2, gas


def test_without_limits_the_probabilities_are_returned_as_given():
    co2, gas = _probabilities()
    assert darsia.limit_probabilities(co2, gas) == (co2, gas)
    limited = darsia.limit_probabilities(co2, gas, None, None)
    assert limited[0] is co2 and limited[1] is gas


def test_where_no_co2_may_be_both_probabilities_are_zero_and_unknown_stays():
    co2, gas = _probabilities()
    before = co2.copy(), gas.copy()
    keep = np.array([[False, True], [False, False]])
    limited_co2, limited_gas = darsia.limit_probabilities(co2, gas, keep_co2=keep)
    np.testing.assert_array_equal(
        limited_co2, np.array([[0, 0.95], [np.nan, 0]], np.float32)
    )
    np.testing.assert_array_equal(
        limited_gas, np.array([[0, 0.1], [np.nan, 0]], np.float32)
    )
    assert limited_co2.dtype == limited_gas.dtype == np.float32
    np.testing.assert_array_equal(co2, before[0])  # the inputs are not changed
    np.testing.assert_array_equal(gas, before[1])


def test_where_no_gas_may_be_co2_stays_and_its_gas_is_zero():
    co2, gas = _probabilities(np.float64)
    keep_gas = np.array([[False, True], [True, True]])
    limited_co2, limited_gas = darsia.limit_probabilities(co2, gas, keep_gas=keep_gas)
    np.testing.assert_array_equal(limited_co2, co2)
    np.testing.assert_array_equal(
        limited_gas, np.array([[0, 0.1], [np.nan, 0.7]], np.float64)
    )


def test_a_limit_of_another_shape_is_refused():
    co2, gas = _probabilities()
    with pytest.raises(ValueError, match="does not fit"):
        darsia.limit_probabilities(co2, gas, keep_co2=np.ones((3, 2), bool))


def test_a_registry_from_entries_equals_one_loaded_from_its_file(tmp_path):
    import tomllib

    path = tmp_path / "config.toml"
    path.write_text(ROI)
    loaded = RoiRegistry().load(path)
    made = RoiRegistry.from_entries(tomllib.loads(ROI)["roi"])
    assert made.keys() == loaded.keys() == ["top", "corner"]
    assert made.rois["top"].label_index == loaded.rois["top"].label_index == 3
    assert made.rois["corner"].shape == loaded.rois["corner"].shape == "box"


def test_entries_with_a_duplicate_name_or_not_an_array_are_refused():
    entry = {"name": "top", "shape": "label", "label_index": 3}
    with pytest.raises(ValueError, match="duplicated"):
        RoiRegistry.from_entries([entry, dict(entry)])
    with pytest.raises(ValueError, match="array-of-tables"):
        RoiRegistry.from_entries({"top": entry})


def test_a_table_is_read_as_its_file_is_loaded(tmp_path):
    import tomllib

    text = ROI + '\n[expert_knowledge]\nno_concentration_aq = ["top"]\n'
    path = tmp_path / "config.toml"
    path.write_text(text)
    registry = RoiRegistry().load(path)
    loaded = ExpertKnowledgeConfig().load(path, registry)
    made = ExpertKnowledgeConfig.from_table(
        tomllib.loads(text)["expert_knowledge"], registry
    )
    assert made == loaded
    assert made.no_concentration_aq == ["top"] and made.no_saturation_g == []
    with pytest.raises(KeyError, match="not found"):
        ExpertKnowledgeConfig.from_table({"no_saturation_g": ["missing"]}, registry)
    with pytest.raises(ValueError, match="list"):
        ExpertKnowledgeConfig.from_table({"no_saturation_g": "top"}, registry)
