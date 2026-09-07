from __future__ import annotations

import pytest

from darsia.fluidflower.modules import MODULE_DEFINITIONS
from darsia.fluidflower.page_registry import (
    FluidFlowerPagePlugin,
    FluidFlowerPageRegistry,
    FluidFlowerSubpagePlugin,
    FluidFlowerSubpageRegistry,
    default_page_registry,
    default_subpage_registry,
)


def test_default_page_registry_exposes_every_workflow_module() -> None:
    registry = default_page_registry(discover_external=False)
    expected = {"overview", "execution", *(row.key for row in MODULE_DEFINITIONS)}

    assert set(registry.keys()) == expected
    assert not registry.discovery_errors
    assert registry.plugin("phase_interpretation").service_modules == (
        "darsia.fluidflower.phase_features",
        "darsia.fluidflower.phase_interpretation",
        "darsia.fluidflower.phase_learning",
        "darsia.fluidflower.phase_workflow",
    )
    assert registry.plugin("overflow").service_modules == (
        "darsia.fluidflower.overflow",
        "darsia.fluidflower.overflow_workflow",
    )
    assert all(
        isinstance(registry.plugin(key).factory, str)
        and ".fluidflower.pages." in registry.plugin(key).factory
        and ".gui:" not in registry.plugin(key).factory
        for key in registry.keys()
    )


def test_page_registry_requires_explicit_replacement() -> None:
    original = FluidFlowerPagePlugin("example", lambda project: ("old", project))
    replacement = FluidFlowerPagePlugin("example", lambda project: ("new", project))
    registry = FluidFlowerPageRegistry((original,))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(replacement)

    registry.register(replacement, replace_existing=True)
    assert registry.create("example", "project") == ("new", "project")


def test_builtin_subpages_are_ordered_and_independently_replaceable() -> None:
    registry = default_subpage_registry(discover_external=False)
    geometry = registry.plugins_for("geometry_facies")
    phase = registry.plugins_for("phase_interpretation")
    reporting = registry.plugins_for("reporting")
    setup = registry.plugins_for("setup_wizard")

    assert [plugin.key for plugin in geometry] == [
        "inputs",
        "layers",
        "injection_points",
        "run_geometry",
        "boundary_edits",
        "material_examples",
        "training_data",
        "qc_history",
    ]
    assert [plugin.key for plugin in phase] == [
        "active_learning",
        "model_validation",
    ]
    assert [plugin.key for plugin in reporting] == [
        "report_package",
        "scientific_review",
    ]
    assert [plugin.key for plugin in setup] == [
        "identity",
        "image_sources",
        "summary",
    ]
    assert all(
        isinstance(plugin.factory, str)
        and (".subpages." in plugin.factory or ".setup.steps." in plugin.factory)
        for plugin in (*geometry, *phase, *reporting, *setup)
    )

    replacement = FluidFlowerSubpagePlugin(
        "geometry_facies",
        "injection_points",
        "Custom injection points",
        lambda parent: ("custom", parent),
        order=25,
    )
    with pytest.raises(ValueError, match="already registered"):
        registry.register(replacement)
    registry.register(replacement, replace_existing=True)

    selected = registry.plugins_for("geometry_facies")
    assert selected[2].title == "Custom injection points"


def test_subpage_registry_populates_tabs_and_named_attributes() -> None:
    class Tabs:
        def __init__(self) -> None:
            self.rows: list[tuple[object, str]] = []

        def addTab(self, widget: object, title: str) -> None:
            self.rows.append((widget, title))

    tabs = Tabs()
    widget = object()
    registry = FluidFlowerSubpageRegistry(
        (
            FluidFlowerSubpagePlugin(
                "parent",
                "child",
                "Child",
                lambda _parent: widget,
                attribute_name="child_tab",
            ),
        )
    )

    parent = type("Parent", (), {})()
    registry.populate_tabs("parent", parent, tabs)

    assert tabs.rows == [(widget, "Child")]
    assert parent.child_tab is widget


def test_subpage_registry_populates_ordered_page_containers() -> None:
    class Pages:
        def __init__(self) -> None:
            self.rows: list[object] = []

        def addPage(self, widget: object) -> None:
            self.rows.append(widget)

    first, second = object(), object()
    registry = FluidFlowerSubpageRegistry(
        (
            FluidFlowerSubpagePlugin(
                "wizard", "second", "Second", lambda _parent: second, order=20
            ),
            FluidFlowerSubpagePlugin(
                "wizard",
                "first",
                "First",
                lambda _parent: first,
                order=10,
                attribute_name="first_page",
            ),
        )
    )
    pages = Pages()
    parent = type("Parent", (), {})()

    registry.populate_pages("wizard", parent, pages)

    assert pages.rows == [first, second]
    assert parent.first_page is first
