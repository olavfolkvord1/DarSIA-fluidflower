"""The launcher window and the command line give the same report (offscreen Qt)."""

import json

from darsia.fluidflower.published_dataset import reproduce_cli
from darsia.fluidflower.published_dataset.comparison import IDENTICAL
from tests.unit.published_dataset.test_reproduce import INSTALLED, FakeRunner, dataset  # noqa: F401


def test_command_line_and_window_give_the_same_report(dataset, tmp_path, qtbot):
    from darsia.fluidflower.published_dataset.reproduce_window import ReproduceWindow

    arguments = reproduce_cli.parser().parse_args(["--dataset", str(dataset), "--results", str(tmp_path / "cli"),
                                                   "--workers", "1"])
    assert reproduce_cli.run(arguments, runner=FakeRunner(), machine={"cpu": "fixture"}, installed=INSTALLED) == 0
    window = ReproduceWindow(runner_factory=FakeRunner, installed=INSTALLED, fingerprint={},
                             machine={"cpu": "fixture"},
                             inventory={"cpu": {"physical_cores": 4, "logical_cores": 8},
                                        "memory": {"available_bytes": 16 * 1024**3}})
    qtbot.addWidget(window)
    window.dataset_edit.setText(str(dataset))
    window.results_edit.setText(str(tmp_path / "window"))
    window.open_dataset()
    qtbot.waitUntil(lambda: window.reproduction is not None and not window.busy(), timeout=20000)
    assert window.workers_spin.value() == 2 and "measured on the reference machine" in window.advice_label.text()
    assert window.scope_buttons["quick"].isEnabled() and window.precision_label.text().startswith("float32")
    window.workers_spin.setValue(1)
    window.start()
    qtbot.waitUntil(lambda: window.result is not None or "Stopped" in window.status_label.text(), timeout=20000)
    assert window.status_label.text() == f"Verdict: {IDENTICAL}." and window.report_button.isEnabled()
    assert window.progress_table.rowCount() == 2

    def comparable(folder):
        value = json.loads((folder / "report" / "report.json").read_text(encoding="utf-8"))
        value.pop("created_utc")
        value.pop("timing")
        return value
    assert comparable(tmp_path / "cli") == comparable(tmp_path / "window")
