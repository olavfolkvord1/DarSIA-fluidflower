"""Versioned templates for starting FluidFlower experimental series."""

from __future__ import annotations

from dataclasses import dataclass


RIG_TYPES: dict[str, str] = {
    "a4": "A4 FluidFlower",
    "medium": "Medium FluidFlower (0.90 x 0.49 m)",
    "large": "Large FluidFlower (2.80 x 1.50 m)",
    "custom": "Custom FluidFlower apparatus",
}


@dataclass(frozen=True)
class ExperimentalSeriesTemplate:
    """Editable defaults, never a substitute for measured series metadata."""

    template_id: str
    name: str
    description: str
    default_rig_type: str
    protocol_template: str
    overflow_mode: str
    indicator_system: str


EXPERIMENTAL_SERIES_TEMPLATES: tuple[ExperimentalSeriesTemplate, ...] = (
    ExperimentalSeriesTemplate(
        "ff_um",
        "FF_UM",
        "Published large-FluidFlower series using the universal event-table schema.",
        "large",
        "fluidflower_timeline_v2",
        "none",
        "BTB",
    ),
    ExperimentalSeriesTemplate(
        "ff_ac",
        "FF_AC",
        "Albus/medium-FluidFlower series with optional scale-derived overflow.",
        "medium",
        "fluidflower_timeline_v2",
        "scale_images",
        "BTB",
    ),
    ExperimentalSeriesTemplate(
        "ff_bm",
        "FF_BM",
        "Published FluidFlower benchmark series.",
        "large",
        "fluidflower_timeline_v2",
        "none",
        "BTB",
    ),
    ExperimentalSeriesTemplate(
        "custom",
        "New custom series",
        "Create a new series using the universal FluidFlower event-table schema.",
        "custom",
        "fluidflower_timeline_v2",
        "none",
        "unspecified",
    ),
)

TEMPLATE_BY_ID = {
    template.template_id: template
    for template in EXPERIMENTAL_SERIES_TEMPLATES
}


def template_by_id(template_id: str) -> ExperimentalSeriesTemplate:
    try:
        return TEMPLATE_BY_ID[str(template_id).casefold()]
    except KeyError as exc:
        raise ValueError(f"Unknown FluidFlower series template: {template_id!r}") from exc
