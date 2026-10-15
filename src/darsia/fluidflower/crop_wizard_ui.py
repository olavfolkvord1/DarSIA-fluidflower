"""Native-instance-only crop policy for the existing preparation wizard."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping

from PySide6.QtWidgets import QLabel, QPushButton


CROP_NOTE = (
    "Rig corners are set under Rig corners and injection points. "
    "Apply geometry to Setup writes their crop settings. "
    "Other curvature corrections remain editable below. Entire curvature "
    "presets can be loaded in the full configuration editor."
)
_KEY = "corrections.curvature"
_MARKER = "native_crop_wizard_protected"


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _install_sync_policy(window) -> None:
    """Keep hidden crop and unrendered activations out of widget-side changes.

    The original sync rebuilds active lists from rendered groups only. A wizard
    may jump straight to Curvature without building Type/Resize first. Preserve
    those unrendered memberships and the hidden crop, while allowing edits to
    the visible stage checkboxes. The policy is inert for every unmarked form,
    including the ordinary wizard and the main full configuration editor.
    """
    factory = window.settings_factory
    if getattr(factory, "_native_crop_sync_installed", False):
        return
    factory._native_crop_sync_installed = True
    original = factory._sync_settings_inputs_to_config_dict
    from darsia.gui.ui.schema.dataclass_introspection import (
        get_section_fields,
    )

    fields = get_section_fields("corrections") or []
    outer_names = {
        item["key"].rsplit(".", 1)[-1]
        for item in fields
        if item.get("active_list_key") == "active"
    }
    curvature = next((item for item in fields if item["key"] == _KEY), {})
    inner_names = {
        item["key"].rsplit(".", 1)[-1]
        for item in curvature.get("fields", [])
        if item.get("active_list_key") == "active"
    }

    def sync(*args, **kwargs):
        group = _mapping(window.settings_inputs.get(_KEY))
        widget = group.get("widget")
        if widget is None or not widget.property(_MARKER):
            return original(*args, **kwargs)
        before_outer = deepcopy(_mapping(window.config_dict.get("corrections")))
        before_inner = deepcopy(_mapping(before_outer.get("curvature")))
        recorded = {}
        for value in window.settings_inputs.values():
            if isinstance(value, dict) and value.get("active_list_key") in {
                "corrections.active",
                "corrections.curvature.active",
            }:
                recorded.setdefault(value["active_list_key"], {})[value["name"]] = (
                    value["checkbox"].isChecked()
                )
        result = original(*args, **kwargs)
        after_outer = window.config_dict.setdefault("corrections", {})
        after_inner = after_outer.setdefault("curvature", {})
        if "crop" in before_inner:
            after_inner["crop"] = deepcopy(before_inner["crop"])
        else:
            after_inner.pop("crop", None)
        for before, after, key, known in (
            (before_outer, after_outer, "corrections.active", outer_names),
            (before_inner, after_inner, "corrections.curvature.active", inner_names),
        ):
            previous = before.get("active")
            if previous is not None and not isinstance(previous, list):
                continue  # Original configuration validation owns invalid input.
            enabled = (
                list(previous)
                if previous is not None
                else [
                    name
                    for name in before
                    if name in known and before[name] is not None
                ]
            )
            controls = dict(recorded.get(key, {}))
            if key.endswith("curvature.active") and "crop" in controls:
                controls["crop"] = "crop" in enabled
            active = [
                name for name in enabled if name not in controls or controls[name]
            ]
            active.extend(
                name
                for name, checked in controls.items()
                if checked and name not in active
            )
            present = {
                name for name in after if name in known and after[name] is not None
            }
            if previous is None and set(active) == present:
                after.pop("active", None)
            else:
                after["active"] = active
        return result

    sync.__wrapped__ = original
    factory._sync_settings_inputs_to_config_dict = sync


def prepare_crop_wizard(dialog) -> None:
    """Adapt one native-owned dialog; no config write or calculation occurs."""
    if getattr(dialog, "_native_crop_wizard_policy", False):
        return
    if not hasattr(dialog, "_fill_curvature_page"):
        return
    dialog._native_crop_wizard_policy = True
    window = dialog.main_window
    factory = window.settings_factory
    _install_sync_policy(window)
    original_fill = dialog._fill_curvature_page

    def blocked_picker():
        window.print_log(
            "Rig corners are set under Rig corners and injection points; "
            "use Apply geometry to Setup."
        )

    def fill(layout):
        note = QLabel(CROP_NOTE)
        note.setWordWrap(True)
        layout.addWidget(note)
        first = layout.count()
        create_load = factory._create_load_button

        def protected_load(setting, on_apply):
            button = create_load(setting, on_apply)
            if setting.get("key") == _KEY and button is not None:
                button.setEnabled(False)
                button.hide()
                button.setToolTip(CROP_NOTE)
            return button

        factory._create_load_button = protected_load
        try:
            original_fill(layout)
        finally:
            factory._create_load_button = create_load
        for index in range(first, layout.count()):
            widget = layout.itemAt(index).widget()
            if isinstance(widget, QPushButton):
                widget.setEnabled(False)
                widget.hide()
        group = _mapping(window.settings_inputs.get(_KEY))
        if group.get("widget") is not None:
            group["widget"].setProperty(_MARKER, True)
        crop = _mapping(window.settings_inputs.get(_KEY + ".crop"))
        if crop.get("widget") is not None:
            crop["widget"].setEnabled(False)
            crop["widget"].hide()

    fill.__wrapped__ = original_fill
    dialog._fill_curvature_page = fill
    dialog._launch_crop_correction = blocked_picker
