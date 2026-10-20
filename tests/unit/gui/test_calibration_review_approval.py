"""Approve run in Calibration review (N141 spec 2.5, Part 2), on the real page.

A temporary local series; the only write is the explicit approval record.
"""

from darsia.fluidflower import calibration_approvals as approvals
from darsia.fluidflower.series_method import MethodSelection, save_series_method


def _series(tmp_path, *, calibration=True):
    root = tmp_path / "series-a"
    root.mkdir()
    (root / "series.toml").write_text(
        'schema_version = 1\nseries_id = "series-a"\nname = "series-a"\n[runs]\nids = ["ac28", "ac47"]\n',
        encoding="utf-8",
    )
    if calibration:
        (root / "fits").mkdir()
        (root / "fits/frozen.json").write_text('{"frozen": true}', encoding="utf-8")
        save_series_method(root, MethodSelection(active="phase_model", model_version="m",
                                                 mass_calibration="fits/frozen.json"))
    return root


def _page(qtbot, monkeypatch, root):
    from darsia.fluidflower.tools import phase_studio_mass_analysis_workspace as module

    monkeypatch.setattr(module, "configured_series", lambda: (("series-a", "series-a", root),))
    page = module.MassAnalysisWorkspace(series="series-a", project_root=root)
    qtbot.addWidget(page)
    experiments = page.saved_results.view_experiment
    experiments.addItem("AC28", "ac28")
    experiments.addItem("AC47", "ac47")
    return page, experiments


def test_approve_run_appends_one_record_for_the_viewed_run(qtbot, tmp_path, monkeypatch):
    root = _series(tmp_path)
    page, experiments = _page(qtbot, monkeypatch, root)
    try:
        page.show_view("results")
        assert page.approval.isHidden()
        page.show_view("review")
        experiments.setCurrentIndex(0)
        assert not page.approval.isHidden()
        assert page.approve_button.text() == "Approve AC28" and page.approve_button.isEnabled()
        assert "AC28: not approved for calibration fits/frozen.json" in page.approval_label.text()
        before = (root / "series.toml").read_bytes()
        page.approve_button.click()
        records = approvals.read_run_approvals(root)
        assert [record["run"] for record in records] == ["ac28"]
        assert records[0]["calibration"] == approvals.calibration_identity(root, "fits/frozen.json")
        assert "AC28: approved by" in page.approval_label.text()
        assert "1 run(s) approved" in page.approval_label.text()
        assert not page.approve_button.isEnabled()
        experiments.setCurrentIndex(1)
        assert page.approve_button.text() == "Approve AC47" and page.approve_button.isEnabled()
        assert (root / "series.toml").read_bytes() == before
    finally:
        page._stop_lookups()


def test_approval_is_unavailable_without_a_referenced_calibration(qtbot, tmp_path, monkeypatch):
    root = _series(tmp_path, calibration=False)
    page, experiments = _page(qtbot, monkeypatch, root)
    try:
        page.show_view("review")
        experiments.setCurrentIndex(0)
        assert page.approval_label.text().startswith("Approval unavailable: No mass calibration is referenced")
        assert not page.approve_button.isEnabled()
        page.approve_selected_run()
        assert not (root / approvals.APPROVALS_PATH).exists()
    finally:
        page._stop_lookups()
