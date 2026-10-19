"""The method names chosen in N141 (names A), and grouping that never reads them.

The sidebar headers, the method radio buttons, the badge and the Calibration
header line use the new names; the internal keys and series.toml stay. Local
manifests only; no calculation.
"""

from darsia.fluidflower.series_method import (
    METHOD_DESCRIPTIONS,
    METHOD_HEADERS,
    METHOD_NAMES,
    METHODS,
    SeriesMethodPanel,
)
from darsia.gui.ui.sidebar import GroupHeaderLabel

from .test_native_series_method import _phase_choice, _roots, method_window  # noqa: F401


def _headers(window, category):
    layout = window.sidebar._sections[category].items_layout
    widgets = [layout.itemAt(index).widget() for index in range(layout.count())]
    return [widget for widget in widgets if isinstance(widget, GroupHeaderLabel)]


def test_names_cover_exactly_the_two_internal_method_keys():
    for table in (METHOD_NAMES, METHOD_HEADERS, METHOD_DESCRIPTIONS):
        assert set(table) == set(METHODS)
    assert METHOD_NAMES == {"colour_path": "Direct colour method", "phase_model": "Phase-map method"}
    assert METHOD_HEADERS["colour_path"] == "Direct colour method (DarSIA original)"
    assert METHOD_HEADERS["phase_model"] == "Phase-map method"


def test_sidebar_headers_use_the_new_names_and_explain_each_method(method_window):
    window = method_window
    calibration = {widget.text(): widget for widget in _headers(window, "calibration")}
    analysis = {widget.text(): widget for widget in _headers(window, "analysis")}
    assert set(calibration) == {METHOD_HEADERS["colour_path"], "Danger zone", METHOD_HEADERS["phase_model"]}
    assert set(analysis) == {METHOD_HEADERS["colour_path"], METHOD_HEADERS["phase_model"], "Jobs"}
    for headers in (calibration, analysis):
        assert headers[METHOD_HEADERS["colour_path"]].toolTip() == METHOD_DESCRIPTIONS["colour_path"]
        assert headers[METHOD_HEADERS["phase_model"]].toolTip() == METHOD_DESCRIPTIONS["phase_model"]
    assert calibration["Danger zone"].toolTip() == ""
    assert analysis["Jobs"].toolTip() == ""
    texts = [header.text() for category in ("calibration", "analysis") for header in _headers(window, category)]
    assert not any("Colour path —" in text or "classification first" in text or text == "Original method"
                   for text in texts)


def test_grouping_reads_group_keys_not_header_text(method_window):
    window = method_window
    before = {category: {method: list(widgets) for method, widgets in groups.items()}
              for category, groups in window._method_group_widgets.items()}
    for category in ("calibration", "analysis"):
        for header in _headers(window, category):
            header.setText("TESTONLY renamed header")
    window._capture_method_groups()
    after = {category: {method: list(widgets) for method, widgets in groups.items()}
             for category, groups in window._method_group_widgets.items()}
    assert after == before
    calibration = before["calibration"]
    assert [key for key, row in window.sidebar._sections["calibration"]._rows.items()
            if row in calibration["phase_model"]] == ["phase_model:training", "phase_model:mass_from_phase"]
    assert window.sidebar._sections["calibration"]._rows["reset"] in calibration["colour_path"]
    analysis = before["analysis"]
    assert [key for key, row in window.sidebar._sections["analysis"]._rows.items()
            if row in analysis["common"]] == ["jobs:overview"]


def test_badge_and_calibration_line_name_the_chosen_method(method_window):
    window = method_window
    _phase_choice(_roots(window)["series-a"])
    window._refresh_method_state()
    assert window.method_badge.text().startswith("Method: phase-map method · model ")
    window._select_row(("calibration", "color"))
    assert window.calibration_header.line.text().endswith(" · Phase-map method")


def test_method_buttons_show_the_new_names_with_their_explanation(qtbot):
    panel = SeriesMethodPanel()
    qtbot.addWidget(panel)
    assert panel.colour_path_button.text() == "Direct colour method"
    assert panel.phase_model_button.text() == "Phase-map method"
    assert panel.colour_path_button.toolTip() == METHOD_DESCRIPTIONS["colour_path"]
    assert panel.phase_model_button.toolTip() == METHOD_DESCRIPTIONS["phase_model"]
