"""Candidate helper/panel tests; import becomes package-qualified on adoption."""

import copy
import tomllib

import pytest

from darsia.fluidflower import series_method as methods


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "series.toml"
    path.write_text(
        'schema_version = 1\nseries_id = "demo"\n'
        '[paths]\nraw = "../raw photographs"\n'
        '[runs]\nids = ["ac52", "ac19"]\n'
        "[unrelated]\nflag = true\nlimit = 0.125\n"
        "created = 2026-10-15T12:30:00+02:00\n"
        'nested = { label = "保持", values = [1, 2, 3] }\n',
        encoding="utf-8",
    )
    return path


def test_unset_and_unknown_choice_are_not_chosen_without_writes(manifest):
    before = manifest.read_bytes()
    assert methods.read_series_method(manifest.parent).active == ""
    assert manifest.read_bytes() == before
    manifest.write_bytes(before + b'\n[method]\nactive = "unexpected"\n')
    before = manifest.read_bytes()
    assert methods.read_series_method(manifest.parent).active == ""
    assert manifest.read_bytes() == before


def test_save_preserves_unrelated_and_unknown_values_and_reload(manifest):
    manifest.write_bytes(
        manifest.read_bytes()
        + (
            '\n[method]\nactive = "colour_path"\nnote = "retain"\n'
            '[method.colour_path]\ncalibration = "old/colour"\nfuture = [4, 5]\n'
            '[method.phase_model]\nmodel_version = "old-model"\n'
            'mass_calibration = "old-fit"\nreviewed = true\n'
        ).encode()
    )
    original = tomllib.loads(manifest.read_text(encoding="utf-8"))
    choice = methods.MethodSelection(
        active="phase_model",
        model_version="模型/版本 α",
        mass_calibration="../calibration/β.json",
        colour_calibration="do not replace",
    )
    saved = methods.save_series_method(manifest.parent, choice)
    after = tomllib.loads(manifest.read_text(encoding="utf-8"))
    assert {key: value for key, value in after.items() if key != "method"} == {
        key: value for key, value in original.items() if key != "method"
    }
    assert after["method"]["note"] == "retain"
    assert after["method"]["phase_model"]["reviewed"] is True
    assert after["method"]["colour_path"] == original["method"]["colour_path"]
    assert saved == methods.read_series_method(manifest.parent)
    assert saved.model_version == choice.model_version
    assert saved.mass_calibration == choice.mass_calibration
    assert saved.colour_calibration == "old/colour"
    methods.save_series_method(
        manifest.parent,
        methods.MethodSelection(
            active="colour_path", colour_calibration="relative/new colour"
        ),
    )
    switched = tomllib.loads(manifest.read_text(encoding="utf-8"))
    assert switched["method"]["phase_model"] == after["method"]["phase_model"]
    assert switched["method"]["colour_path"]["future"] == [4, 5]


def test_invalid_manifest_and_invalid_choice_do_not_change_bytes(manifest):
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="Choose"):
        methods.save_series_method(manifest.parent, methods.MethodSelection())
    assert manifest.read_bytes() == before


def test_invalid_reference_table_stays_not_chosen_without_losing_data(manifest):
    manifest.write_bytes(
        manifest.read_bytes()
        + b'\n[method]\nactive = "phase_model"\nphase_model = 17\n'
    )
    before = manifest.read_bytes()
    loaded = methods.read_series_method(manifest.parent)
    assert loaded.active == "" and loaded.error
    with pytest.raises(ValueError, match="must be a TOML table"):
        methods.save_series_method(
            manifest.parent,
            methods.MethodSelection(
                active="phase_model", model_version="model", mass_calibration="fit"
            ),
        )
    assert manifest.read_bytes() == before
    manifest.write_bytes(b"invalid = [\n")
    before = manifest.read_bytes()
    loaded = methods.read_series_method(manifest.parent)
    assert loaded.active == "" and loaded.error
    with pytest.raises(ValueError):
        methods.save_series_method(
            manifest.parent,
            methods.MethodSelection(
                active="phase_model", model_version="model", mass_calibration="fit"
            ),
        )
    assert manifest.read_bytes() == before


def test_old_load_and_edit_during_save_are_rejected(manifest, monkeypatch):
    original = manifest.read_bytes()
    manifest.write_bytes(original + b'\nnew_field = "newer"\n')
    newer = manifest.read_bytes()
    choice = methods.MethodSelection(active="colour_path", colour_calibration="fit")
    with pytest.raises(methods.ManifestChangedError, match="loaded"):
        methods.save_series_method(manifest.parent, choice, expected_bytes=original)
    assert manifest.read_bytes() == newer
    dumps = methods.toml.dumps
    concurrent = newer + b'concurrent = "preserve this edit"\n'

    def editing_dumps(value):
        manifest.write_bytes(concurrent)
        return dumps(value)

    monkeypatch.setattr(methods.toml, "dumps", editing_dumps)
    with pytest.raises(methods.ManifestChangedError, match="while saving"):
        methods.save_series_method(manifest.parent, choice)
    assert manifest.read_bytes() == concurrent
    assert not list(manifest.parent.glob(".series.toml.*.tmp"))


def test_replace_failure_and_semantic_roundtrip_failure_preserve_original(
    manifest, monkeypatch
):
    original = manifest.read_bytes()
    choice = methods.MethodSelection(active="colour_path", colour_calibration="fit")

    def denied(_source, _destination):
        raise PermissionError("simulated local manifest replace denial")

    monkeypatch.setattr(methods.os, "replace", denied)
    with pytest.raises(PermissionError, match="denial"):
        methods.save_series_method(manifest.parent, choice)
    assert manifest.read_bytes() == original
    assert not list(manifest.parent.glob(".series.toml.*.tmp"))
    monkeypatch.undo()
    dumps = methods.toml.dumps

    def broken_dumps(value):
        altered = copy.deepcopy(value)
        altered["unrelated"]["limit"] = 999.0
        return dumps(altered)

    monkeypatch.setattr(methods.toml, "dumps", broken_dumps)
    with pytest.raises(ValueError, match="other TOML values"):
        methods.save_series_method(manifest.parent, choice)
    assert manifest.read_bytes() == original


def test_bom_crlf_and_nonfinite_unknown_values_survive(manifest):
    original = manifest.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    manifest.write_bytes(
        b"\xef\xbb\xbf"
        + original
        + b"\r\n[limits]\r\nmissing = nan\r\nmaximum = inf\r\n"
    )
    methods.save_series_method(
        manifest.parent,
        methods.MethodSelection(
            active="phase_model", model_version="model", mass_calibration="fit"
        ),
    )
    after = manifest.read_bytes()
    assert after.startswith(b"\xef\xbb\xbf")
    assert b"\n" not in after.replace(b"\r\n", b"")
    assert methods.read_series_method(manifest.parent).active == "phase_model"


def test_panel_saved_refs_win_and_only_explicit_use_emits(qtbot, manifest):
    panel = methods.SeriesMethodPanel()
    qtbot.addWidget(panel)
    emitted = []
    panel.use_requested.connect(emitted.append)
    original = manifest.read_bytes()
    saved = methods.MethodSelection(
        active="phase_model",
        model_version="saved-model",
        mass_calibration="saved-fit",
        colour_calibration="saved-colour",
    )
    panel.set_selection(
        saved,
        phase_model_reference="selected-other",
        mass_calibration_reference="other-fit",
    )
    assert panel.selection() == saved
    assert not emitted
    panel.model_version.setText("edited-model")
    panel.colour_path_button.click()
    assert not emitted and manifest.read_bytes() == original
    assert panel.colour_calibration.isEnabled()
    assert not panel.model_version.isEnabled()
    panel.use_button.click()
    assert len(emitted) == 1
    assert emitted[0].active == "colour_path"
    assert emitted[0].colour_calibration == "saved-colour"
    assert emitted[0].model_version == "edited-model"
    assert manifest.read_bytes() == original  # Panel never owns persistence.


def test_panel_never_fabricates_missing_refs_or_chooses_implicitly(qtbot):
    panel = methods.SeriesMethodPanel()
    qtbot.addWidget(panel)
    panel.set_selection(methods.MethodSelection())
    assert not panel.use_button.isEnabled()
    assert panel.selection() == methods.MethodSelection()
    panel.set_selection(
        methods.MethodSelection(),
        phase_model_reference="actual/model",
        mass_calibration_reference="../actual-fit",
    )
    assert panel.selection().model_version == "actual/model"
    assert panel.selection().mass_calibration == "../actual-fit"
    assert panel.selection().colour_calibration == ""
    assert panel.selection().active == ""
    panel.phase_model_button.click()
    assert panel.use_button.isEnabled()
    panel.set_selection(methods.MethodSelection(error="broken TOML"))
    assert not panel.use_button.isEnabled()
    assert panel.status.text() == "broken TOML"
    panel.set_selection(methods.MethodSelection(active="phase_model"), available=False)
    assert not panel.use_button.isEnabled()


@pytest.mark.parametrize(
    "choice",
    [
        methods.MethodSelection(active="phase_model", model_version="model"),
        methods.MethodSelection(active="phase_model", mass_calibration="fit"),
        methods.MethodSelection(active="colour_path", colour_calibration="  \t "),
    ],
)
def test_missing_chosen_reference_rejects_save_without_any_write(manifest, choice):
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match="before saving this method"):
        methods.save_series_method(manifest.parent, choice)
    assert manifest.read_bytes() == original
    assert not list(manifest.parent.glob(".series.toml.*.tmp"))


def test_saved_active_missing_refs_remains_visible_with_warning_and_no_fallback(
    qtbot, manifest
):
    manifest.write_bytes(
        manifest.read_bytes()
        + b'\n[method]\nactive = "phase_model"\n[method.colour_path]\ncalibration = "available-other-method"\n'
    )
    original = manifest.read_bytes()
    loaded = methods.read_series_method(manifest.parent)
    assert loaded.active == "phase_model"
    assert methods.missing_method_references(loaded) == (
        "model version",
        "mass calibration",
    )
    panel = methods.SeriesMethodPanel()
    qtbot.addWidget(panel)
    panel.set_selection(loaded)
    assert panel.phase_model_button.isChecked()
    assert not panel.colour_path_button.isChecked()
    assert "missing" in panel.status.text()
    assert not panel.use_button.isEnabled()
    panel.model_version.setText("  opaque/模型 ")
    assert not panel.use_button.isEnabled()
    panel.mass_calibration.setText("../opaque/fit.json")
    assert panel.use_button.isEnabled()
    assert panel.selection().model_version == "  opaque/模型 "
    assert manifest.read_bytes() == original
