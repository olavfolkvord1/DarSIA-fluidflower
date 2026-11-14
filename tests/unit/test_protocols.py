"""Unit tests for reading the protocols from workbooks."""

from pathlib import Path

import pandas as pd

import darsia


def test_workbook_path_without_sheet_reads_the_first_sheet(tmp_path):
    """A one-sheet workbook read by its path gives the protocol of that sheet."""
    tables = {
        "imaging": {
            "path": ["run/image_0001.jpg", "run/image_0002.jpg"],
            "datetime": ["2000-01-01 10:00:00", "2000-01-01 10:05:00"],
        },
        "blacklist": {"path": ["run/image_0002.jpg"]},
        "injection": {
            "id": [1],
            "location_x": [0.1],
            "location_y": [0.2],
            "start": ["2000-01-01 10:00:00"],
            "end": ["2000-01-01 11:00:00"],
            "rate_g/s": [0.01],
        },
        "state": {
            "datetime": ["2000-01-01 10:00:00", "2000-01-01 11:00:00"],
            "pressure_bar": [1.0, 1.01],
            "temperature_celsius": [21.0, 21.5],
            "pressure_gradient_bar": [0.0, 0.0],
            "temperature_gradient_celsius": [0.0, 0.0],
        },
    }
    paths = {name: tmp_path / f"{name}.xlsx" for name in tables}
    for name, table in tables.items():
        pd.DataFrame(table).to_excel(paths[name], sheet_name="data", index=False)

    for name, protocol in (
        ("imaging", darsia.ImagingProtocol),
        ("injection", darsia.InjectionProtocol),
        ("state", darsia.PressureTemperatureProtocol),
    ):
        pd.testing.assert_frame_equal(
            protocol(paths[name]).df, protocol((paths[name], "data")).df
        )
    imaging = darsia.ImagingProtocol(paths["imaging"], blacklist=paths["blacklist"])
    assert imaging.is_blacklisted(Path("run/image_0002.jpg"))
    assert not imaging.is_blacklisted(Path("run/image_0001.jpg"))
