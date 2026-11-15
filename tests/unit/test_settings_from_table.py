"""Tests of the loader of method settings from configuration tables."""

import tomllib
from dataclasses import dataclass

import pytest

import darsia
from darsia.presets.workflows.config.utils import settings_from_table


@dataclass(frozen=True)
class _Named:
    """Settings that map names to settings."""

    parts: dict[str, darsia.CirculationSettings]


def test_settings_from_a_table():
    """Unknown and missing keys are refused; nested settings are built, also in a
    mapping of names to settings."""
    table = tomllib.loads(
        'statuses = ["Verified"]\nrecipe_tolerance = {}\nvariant = "circulation"\n'
        "circulation = {minimum_readings = 30}\n"
    )
    settings = settings_from_table(darsia.OverflowSettings, table)
    assert settings.circulation == darsia.CirculationSettings(minimum_readings=30)
    with pytest.raises(ValueError, match=r"unknown keys \['extra'\]"):
        settings_from_table(darsia.OverflowSettings, {**table, "extra": 1})
    with pytest.raises(ValueError, match=r"lacks \['variant'\]"):
        settings_from_table(
            darsia.OverflowSettings, {k: v for k, v in table.items() if k != "variant"}
        )
    with pytest.raises(ValueError, match=r"CirculationSettings has unknown keys"):
        settings_from_table(
            darsia.OverflowSettings, {**table, "circulation": {"minimum": 3}}
        )
    named = settings_from_table(_Named, {"parts": {"a": {"minimum_readings": 3}}})
    assert named.parts == {"a": darsia.CirculationSettings(minimum_readings=3)}


@dataclass(frozen=True)
class _Listed:
    """Settings with tuples, of settings and of numbers."""

    parts: tuple[darsia.CirculationSettings, ...]
    size: tuple[int, int]
    limits: tuple[tuple[float, float], ...] | None = None


def test_a_list_of_a_tuple_field_becomes_its_tuple():
    """A list given for a tuple field becomes a tuple, its tables settings."""
    table = tomllib.loads(
        "parts = [{minimum_readings = 3}, {minimum_readings = 4}]\nsize = [2, 4]\n"
        "limits = [[0.0, 1.0]]\n"
    )
    settings = settings_from_table(_Listed, table)
    assert settings == _Listed(
        (darsia.CirculationSettings(3), darsia.CirculationSettings(4)),
        (2, 4),
        ((0.0, 1.0),),
    )
    assert settings_from_table(_Listed, {**table, "limits": None}).limits is None
