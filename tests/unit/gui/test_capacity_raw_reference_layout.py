"""TESTONLY: RAW-only frames get a layout only on the released reference route.

The ordinary measured-profile route stays rig-only. The reference route reads
the native grid from the RAW source binding and names that binding in the
layout, so a RAW layout never equals a rig layout. Synthetic JSON only.
"""
import hashlib
import json
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import capacity as cap
from .test_capacity import native_metadata

TASK = dict(run="ac52", pipeline="native-phase-n1", precision="float32")


def raw_metadata_TESTONLY(tmp_path, *, binding=None, frame=None, split_binding=False):
    """The rig fixture with every frame switched to RAW frames on one binding."""
    config, rows, rig_layout = native_metadata(tmp_path)
    rows = [row for row in rows if not row.uses[0].endswith(".native_geometry")]

    def put(stem, value, uses):
        path = tmp_path / (stem + ".json")
        raw = json.dumps(value, sort_keys=True).encode()
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        rows.append(SimpleNamespace(path=path, reference=dict(sha256=digest, bytes=len(raw)), uses=tuple(uses)))
        return dict(path=str(path), sha256=digest, size_bytes=len(raw))

    cases = [row["case_id"] for row in config["frames"]]
    value = dict(contract="darsia-raw-rig-source-v1", run="ac52", native_shape_hw=[2, 4], profile_id="TESTONLY")
    value.update(binding or {})
    if split_binding:
        bound = [put("raw_binding_" + case, dict(value, profile_id=case), ["frame." + case + ".native.raw_binding"])
                 for case in cases]
    else:
        bound = [put("raw_binding", value, ["frame." + case + ".native.raw_binding" for case in cases])] * len(cases)
    for row, descriptor in zip(config["frames"], bound):
        case = row["case_id"]
        native = dict(contract="darsia-raw-native-frame-v1", run="ac52", acquisition={"TESTONLY": case})
        native.update(frame or {})
        row["native_task"] = dict(kind="raw_rig", processor=dict(raw_binding=descriptor),
                                  geometry=put("frame." + case + ".raw_native", native, ["frame." + case + ".native_geometry"]))
    expected = dict(rig_layout, native_task_kind="raw_rig", raw_source_binding_sha256=bound[0]["sha256"])
    return config, tuple(rows), expected, rig_layout


def test_the_ordinary_measured_route_still_refuses_raw_frames(tmp_path):
    config, rows, _, _ = raw_metadata_TESTONLY(tmp_path)
    with pytest.raises(ValueError, match="Unmeasured native processor"):
        cap._native_layout_for_task(None, "FF_AC", TASK, config, rows)


def test_the_reference_route_reads_the_raw_binding_grid_and_names_the_binding(tmp_path):
    config, rows, expected, rig_layout = raw_metadata_TESTONLY(tmp_path)
    actual = cap._native_layout_for_task(None, "FF_AC", TASK, config, rows, raw_reference=True)
    assert actual == expected
    assert actual != rig_layout


def test_rig_frames_give_the_same_layout_on_both_routes(tmp_path):
    config, rows, layout = native_metadata(tmp_path)
    assert cap._native_layout_for_task(None, "FF_AC", TASK, config, rows, raw_reference=True) == layout
    assert cap._native_layout_for_task(None, "FF_AC", TASK, config, rows) == layout


@pytest.mark.parametrize("kwargs", [
    dict(binding=dict(contract="darsia-raw-rig-source-v0")),
    dict(frame=dict(contract="darsia-raw-native-frame-v0")),
    dict(frame=dict(run="ac53")),
    dict(binding=dict(run="ac53"), frame=dict(run="ac53")),  # Another run's binding.
    dict(binding=dict(native_shape_hw=[3, 4])),
    dict(split_binding=True),
])
def test_the_reference_route_refuses_any_other_binding_frame_or_grid(tmp_path, kwargs):
    config, rows, _, _ = raw_metadata_TESTONLY(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        cap._native_layout_for_task(None, "FF_AC", TASK, config, rows, raw_reference=True)


def test_mixed_rig_and_raw_frames_are_refused_on_both_routes(tmp_path):
    config, rows, _, _ = raw_metadata_TESTONLY(tmp_path)
    config["frames"][1] = native_metadata(tmp_path / "rig")[0]["frames"][1]
    for kwargs in ({}, dict(raw_reference=True)):
        with pytest.raises(ValueError, match="Unmeasured native processor"):
            cap._native_layout_for_task(None, "FF_AC", TASK, config, rows, **kwargs)


def test_the_n43_entitlement_asks_for_the_raw_reference_route(monkeypatch):
    from .test_n43_reference_administration import entitlement_runtime_TESTONLY
    runtime, task, config, value, _ = entitlement_runtime_TESTONLY(monkeypatch)
    calls = []

    def layout(*args, **kwargs):
        calls.append(kwargs)
        return value["requested_native_layout"]

    monkeypatch.setattr(cap, "_native_layout_for_task", layout)
    cap.n43_entitlement(runtime, task, config, ())
    assert calls == [dict(raw_reference=True)]
