from __future__ import annotations

from pathlib import Path

from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.reporting import (
    REPORT_SECTIONS,
    SIGNOFF_ROLES,
    default_reporting_plan,
    reporting_task_contract,
)


def test_reporting_plan_has_no_watermark_or_signature_workflow(
    tmp_path: Path,
) -> None:
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Compact export plan",
        series_id="compact-export",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Olav Folkvord",
    )

    plan = default_reporting_plan(project)
    contract = reporting_task_contract(plan)

    assert "approvals" not in REPORT_SECTIONS
    assert SIGNOFF_ROLES == ()
    assert "draft_watermark" not in plan["exports"]
    assert "signoffs" not in plan
    assert "signoffs" not in contract
