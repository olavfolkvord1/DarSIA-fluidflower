from __future__ import annotations

from darsia.fluidflower.quality_budget import (
    default_quality_profile,
    default_time_selection_budget,
    numerical_dtype_for_backend,
    resolve_analysis_shape,
    resolve_time_selection,
    summarize_quality_contract,
    validate_quality_profile,
)


def test_time_budget_preserves_endpoints_and_landmark_deterministically() -> None:
    budget = default_time_selection_budget(maximum_time_points=4)
    budget["required_landmark_times_hours"] = [5.1]

    first = resolve_time_selection(list(range(10)), budget)
    second = resolve_time_selection(list(range(10)), budget)

    assert first == second
    assert first["valid"]
    assert first["selected_time_count"] == 4
    assert first["selected_time_indices"][0] == 0
    assert first["selected_time_indices"][-1] == 9
    assert 5 in first["selected_time_indices"]


def test_smallest_task_or_publication_limit_controls_selection() -> None:
    budget = default_time_selection_budget(maximum_time_points=20)
    budget["maximum_solver_tasks"] = 25
    budget["maximum_publication_bytes"] = 350

    result = resolve_time_selection(
        list(range(10)),
        budget,
        tasks_per_time=5,
        fixed_publication_bytes=50,
        publication_bytes_per_time=100,
    )

    assert result["valid"]
    assert result["selected_time_count"] == 3
    assert result["selected_solver_task_count"] == 15
    assert result["estimated_publication_bytes"] == 350
    assert result["active_limits"] == ["maximum_publication_bytes"]


def test_budget_rejects_capacity_smaller_than_required_points() -> None:
    budget = default_time_selection_budget(maximum_time_points=2)
    budget["required_landmark_times_hours"] = [2.0, 5.0]
    result = resolve_time_selection(list(range(8)), budget)

    assert not result["valid"]
    assert "budget_smaller_than_required_landmarks" in {
        row["code"] for row in result["issues"]
    }


def test_quality_profile_separates_preview_and_scientific_resolution() -> None:
    profile = default_quality_profile("production")
    report = validate_quality_profile(profile, backend="cuda-float32")

    assert report["valid"]
    assert profile["numerical_resolution"]["policy"] == "native_registered_grid"
    assert not profile["numerical_resolution"][
        "preview_resolution_is_scientifically_authoritative"
    ]
    assert profile["precision"]["gpu_dtype"] == "float32"
    assert profile["precision"]["cpu_reference_dtype"] == "float64"


def test_quality_profile_allows_independent_downscaling_and_float64() -> None:
    profile = default_quality_profile("exploratory")
    profile["precision"]["compute_dtype"] = "float64"
    report = validate_quality_profile(profile, backend="cuda-float64")

    assert report["valid"]
    assert {row["code"] for row in report["issues"]} == {
        "reduced_scientific_resolution"
    }
    assert resolve_analysis_shape((2000, 3000), profile) == (1000, 1500)
    assert numerical_dtype_for_backend("cuda-float64") == "float64"


def test_cpu_float32_has_explicit_quality_summary_and_float64_reference() -> None:
    profile = default_quality_profile("exploratory")
    profile["preview"].update(
        {"maximum_dimension_px": 1024, "jpeg_quality": 80}
    )

    report = validate_quality_profile(profile, backend="cpu-float32")
    summary = summarize_quality_contract(profile, backend="cpu-float32")

    assert report["valid"]
    assert summary == {
        "backend": "cpu-float32",
        "compute_dtype": "float32",
        "scientific_grid": "50% of registered grid",
        "preview": "max 1024 px, JPEG 80",
        "cpu_reference_dtype": "float64",
        "label": (
            "cpu-float32 | float32 | scientific grid: 50% of registered grid | "
            "preview only: max 1024 px, JPEG 80"
        ),
    }


def test_maximum_dimension_resolution_never_upsamples() -> None:
    profile = default_quality_profile("production")
    profile["numerical_resolution"].update(
        {"policy": "maximum_dimension", "maximum_dimension_px": 1200}
    )

    assert resolve_analysis_shape((2000, 3000), profile) == (800, 1200)
    assert resolve_analysis_shape((400, 600), profile) == (400, 600)
