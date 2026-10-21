"""Typed Setup operation labels beside unchanged image-job presentation."""
from copy import deepcopy

from darsia.fluidflower.jobs_panel import JobsPanel


def _image(root):
    """Supply ordinary image-job display facts without starting a reader."""
    return dict(root=str(root), name="Image job", series="campaign", state="Waiting",
        job_facts=dict(precision="float32"), run_count=1, completed_runs=0,
        committed=1, total_frames=3, remaining=2, frames_per_minute=None, eta_seconds=None,
        workers=[], machines=[], logs=[], errors=[], runs=[], rate_scope="Published image receipts")


def _setup(root):
    """Use the operation fields supplied by the typed shared-result projection."""
    return dict(root=str(root), name="Setup job", series="campaign", state="Waiting",
        job_facts=dict(format="darsia-alias-job-v1", pipeline="series-setup", precision="float32",
            execution_ready=True, runtime_support="darsia-role-io-runtime-v1"),
        run_count=3, completed_runs=1, completed_operations=1, total_operations=3,
        remaining_operations=2, work_unit="operations", committed=0, total_frames=None,
        remaining=None, frames_per_minute=None, eta_seconds=None, workers=[], machines=[],
        logs=[], errors=[], rate_scope="Only verified Setup operation completions",
        runs=[dict(run="setup-apply-fixture", operation="apply", state="Complete"),
              dict(run="run-a", operation="build", state="Waiting for Apply"),
              dict(run="run-b", operation="build", state="Queued")])


def _ready(panel, snapshots):
    """Deliver a finite reader's actual projection envelope to the real widget."""
    panel._read_ready(dict(generation=panel._generation, snapshots=deepcopy(snapshots),
                          known_launch_requests=[], known_launch_errors=[]))


def test_jobs_show_operation_counts_and_each_apply_build_stage(qtbot, tmp_path):
    """Setup counts operations, while unfinished child stages remain explicit."""
    panel = JobsPanel(local_machine="fixture")
    qtbot.addWidget(panel)
    snapshot = _setup(tmp_path / "setup-job")
    _ready(panel, [snapshot])
    assert panel.job_table.horizontalHeaderItem(2).text() == "Operations"
    assert panel.job_table.horizontalHeaderItem(3).text() == "Progress"
    assert panel.job_table.item(0, 2).text() == "3"
    assert panel.job_table.item(0, 3).text() == "1 / 3 operations"
    assert "Completed operations: 1 / 3" in panel.progress_label.text()
    assert "Remaining operations: 2" in panel.progress_label.text()
    assert "Committed images" not in panel.progress_label.text()
    assert not panel.operation_label.isHidden()
    assert panel.operation_label.text().splitlines() == [
        "Apply geometry · setup-apply-fixture: Complete",
        "Build run model · run-a: Waiting for Apply",
        "Build run model · run-b: Queued",
    ]
    assert "limits recorded in each operation" in panel.profile_label.text()
    assert panel.can_close()


def test_mixed_and_image_only_job_views_preserve_image_progress(qtbot, tmp_path):
    """Switching job kinds never relabels image counts as completed operations."""
    panel = JobsPanel(local_machine="fixture")
    qtbot.addWidget(panel)
    image, setup = _image(tmp_path / "image-job"), _setup(tmp_path / "setup-job")
    _ready(panel, [image])
    image_progress, image_profile = panel.progress_label.text(), panel.profile_label.text()
    assert panel.job_table.horizontalHeaderItem(2).text() == "Runs"
    assert panel.job_table.horizontalHeaderItem(3).text() == "Frames"
    _ready(panel, [image, setup])
    assert panel.job_table.horizontalHeaderItem(2).text() == "Runs / operations"
    assert panel.job_table.horizontalHeaderItem(3).text() == "Images / operations"
    assert panel.job_table.item(0, 3).text() == "1 / 3"
    assert panel.job_table.item(1, 3).text() == "1 / 3 operations"
    panel.job_table.selectRow(1)
    assert not panel.operation_label.isHidden()
    panel.job_table.selectRow(0)
    assert panel.operation_label.isHidden() and not panel.operation_label.text()
    assert panel.progress_label.text() == image_progress
    assert panel.profile_label.text() == image_profile
    _ready(panel, [image])
    assert panel.job_table.horizontalHeaderItem(3).text() == "Frames"
    assert panel.can_close()
