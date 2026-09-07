from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from darsia.fluidflower.analysis_domains import (
    AnalysisDomainStore,
    default_analysis_domain_config,
    domain_from_normalized_rectangle,
    import_analysis_domains,
    load_analysis_domain_reference,
    validate_analysis_domain_config,
)
from darsia.fluidflower.geometry_facies import (
    GeometryRevisionStore,
    import_darsia_rig,
    validate_geometry_config,
)
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.wasserstein import default_wasserstein_plan


def _save_field(path: Path, array: np.ndarray) -> None:
    np.savez_compressed(
        path,
        array=array,
        metadata={
            "space_dim": 2,
            "indexing": "ij",
            "dimensions": [0.49, 0.9],
            "origin": "[0.0, 0.49]",
        },
    )


def _project_with_geometry(tmp_path: Path) -> FluidFlowerProject:
    rig = tmp_path / "rig"
    rig.mkdir()
    labels = np.ones((6, 8), dtype=np.int16)
    for name, array in (
        ("baseline.npz", np.full((6, 8, 3), (30, 80, 145), dtype=np.uint8)),
        ("labels.npz", labels),
        ("facies.npz", labels),
        ("depth.npz", np.full((6, 8), 0.01, dtype=np.float32)),
        ("porosity.npz", np.full((6, 8), 0.40, dtype=np.float32)),
        ("permeability.npz", np.full((6, 8), 44.0, dtype=np.float32)),
    ):
        _save_field(rig / name, array)
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Analysis-domain test",
        series_id="analysis-domain-test",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )
    config = import_darsia_rig(rig, rig_type="medium")
    GeometryRevisionStore(project).save_revision(
        config,
        actor="Reviewer",
        reason="Canonical corrected grid",
        validation=validate_geometry_config(config),
    )
    return project


def test_domain_coordinates_cover_pixel_normalized_and_physical_spaces() -> None:
    domain = domain_from_normalized_rectangle(
        name="Upper half",
        normalized_xyxy_top_left=[0.0, 0.0, 0.5, 0.5],
        source_shape_rc=[100, 200],
        width_m=0.9,
        height_m=0.49,
    )

    assert domain["pixel_bounds_rc"] == [0, 0, 50, 100]
    assert domain["physical_bounds_m"] == {
        "x_min": 0.0,
        "x_max": 0.45,
        "y_min": 0.245,
        "y_max": 0.49,
    }


def test_import_accepts_zero_valued_physical_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "domains.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("name", "x_min_m", "x_max_m", "y_min_m", "y_max_m"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "Lower left",
                "x_min_m": 0,
                "x_max_m": 0.45,
                "y_min_m": 0,
                "y_max_m": 0.245,
            }
        )

    domains = import_analysis_domains(
        path,
        source_shape_rc=[100, 200],
        width_m=0.9,
        height_m=0.49,
    )

    assert domains[0]["normalized_xyxy_top_left"] == [0.0, 0.5, 0.5, 1.0]


def test_saved_domain_revision_flows_into_wasserstein_plan(tmp_path: Path) -> None:
    project = _project_with_geometry(tmp_path)
    reference = load_analysis_domain_reference(project)
    provenance = reference["provenance"]
    domain = domain_from_normalized_rectangle(
        name="Analysis box 1",
        normalized_xyxy_top_left=[0.1, 0.2, 0.9, 0.8],
        source_shape_rc=provenance["source_shape_rc"],
        width_m=provenance["width_m"],
        height_m=provenance["height_m"],
    )
    config = default_analysis_domain_config(project)
    config.update(
        {
            "reference_image_path": provenance["source_path"],
            "reference_provenance": provenance,
            "width_m": provenance["width_m"],
            "height_m": provenance["height_m"],
            "source_shape_rc": provenance["source_shape_rc"],
            "domains": [domain],
        }
    )

    assert validate_analysis_domain_config(project, config)["valid"]
    revision = AnalysisDomainStore(project).save_revision(
        config,
        actor="Reviewer",
        reason="Wasserstein subdomain",
    )
    plan = default_wasserstein_plan(project)

    assert plan["analysis_domain_revision_id"] == revision["revision_id"]
    assert plan["comparison"]["domains"] == [
        "whole_active_domain",
        domain["domain_id"],
    ]
