"""Explicit, source-exact AC44 dispatch without changing canonical modules.

The historical constructor bridge and renderer execute their frozen bytecode
with a private import graph. No source guard, numerical code, or SHA check is
relaxed. Ordinary runs retain the unmodified RigRawSourceProcessor path.
"""
from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace

from .project import sha256_file

CONTRACT = "ffac-rig-ac44-private-frozen-dispatch-v1"
GEOMETRY_SHA = "66c8df7fb12c9d8f4c3b0baad180cfb4199467b97b1706a30125b7d90f25ca4d"
ADAPTER_SHA = "6894c271153c144964c810adbab16dd9a879e9407da426ae4ea736370bb12825"
ALT_PROCESSOR_SHA = "a22dce2f0681b2cc33dc8bf054e04301c5fb322c87db0a7bbc9e1d26ccb677fe"
ALT_RENDERER_SHA = "d7d9e942a425a5d1526861737e63d9f4b1a1b0bb0e07c97db924f0f2a13d0497"
EA_SHA = "041e65a5437ecae4237f518ef08b58e2becfbb4e62d45dc9bb8ead07c194b559"
BASE_SHA = "341b239d9d1933259a34952d10532e3973f981546b2a3e6fec10585fc30aa598"
SOURCE_SHA = "fdcd7f7ff6f8718180db82fbb25ffa788b185c51effefb25d9d8b20f2f01898e"


def _bound(path, expected):
    path = Path(path)
    if not expected or sha256_file(path) != expected:
        raise ValueError("Frozen AC44 dispatch artifact changed: " + str(path))
    return path


def _importer(overrides):
    """Replace only declared from-import members in this private globals dict."""
    original = builtins.__import__
    def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
        absolute = name
        if level:
            import importlib.util
            absolute = importlib.util.resolve_name("." * level + name, globals["__package__"])
        members = overrides.get(absolute)
        if members is not None:
            if not fromlist:
                return members
            # A proxy is intentionally not installed in sys.modules.
            fallback = None
            selected = {}
            for member in fromlist:
                if hasattr(members, member):
                    selected[member] = getattr(members, member)
                else:
                    if fallback is None:
                        fallback = original(name, globals, locals, fromlist, level)
                    selected[member] = getattr(fallback, member)
            return SimpleNamespace(**selected)
        return original(name, globals, locals, fromlist, level)
    return isolated_import


def _private_module(path, name, overrides=None):
    value = ModuleType(name)
    value.__file__, value.__package__ = str(path), name.rpartition(".")[0]
    value.__dict__["__builtins__"] = dict(vars(builtins), __import__=_importer(overrides or {}))
    exec(compile(Path(path).read_bytes(), str(path), "exec"), value.__dict__)
    return value


def _frozen_function(function, overrides, extra_globals=None):
    namespace = dict(function.__globals__)
    namespace.update(extra_globals or {})
    namespace["__builtins__"] = dict(vars(builtins), __import__=_importer(overrides))
    value = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    value.__kwdefaults__ = function.__kwdefaults__
    return value


def _ac44_graph(native):
    from .tools import ffac_track1_v4r5_source_processor as base
    tools = Path(base.__file__).parent
    parent = native["v5_photometry"]["processor"]
    adapter = parent.get("ac44_constructor_forwarding_adapter")
    if (native.get("run") != "ac44" or native["v5_photometry"]["source_manifest_sha256"] != SOURCE_SHA
            or not isinstance(adapter, dict) or adapter.get("sha256") != ADAPTER_SHA
            or adapter.get("geometry_sha256") != GEOMETRY_SHA
            or parent.get("processor_module_sha256") != ALT_PROCESSOR_SHA
            or parent.get("renderer_module_sha256") != ALT_RENDERER_SHA
            or parent.get("base_processor_module_sha256") != BASE_SHA):
        raise ValueError("AC44 private dispatch requires its exact sealed parent variant")
    adapter_path = _bound(adapter["path"], ADAPTER_SHA)
    historical_plan_path = _bound(adapter["plan_path"], adapter["plan_sha256"])
    historical_plan = json.loads(historical_plan_path.read_text(encoding="utf-8-sig"))
    geometry_path = _bound(adapter_path.parent / "ac44_frozen_geometry.py", GEOMETRY_SHA)
    if historical_plan.get("geometry_module_sha256") != GEOMETRY_SHA:
        raise ValueError("Historical AC44 bridge plan does not bind the archived geometry")
    _bound(base.__file__, BASE_SHA)
    ea_path = _bound(tools / "ffac_raw_native_ea.py", EA_SHA)
    alt_path = _bound(tools / "ffac_raw_native_ea_ac44_alt.py", ALT_RENDERER_SHA)
    processor_path = _bound(tools / "ffac_track1_v4r5_ac44_alt_processor.py", ALT_PROCESSOR_SHA)
    geometry = _private_module(geometry_path, "darsia.fluidflower.tools.ffac_raw_geometry")
    tool_imports = SimpleNamespace(ffac_raw_geometry=geometry)
    ea = _private_module(ea_path, "darsia.fluidflower.tools.ffac_raw_native_ea",
                         {"darsia.fluidflower.tools": tool_imports})
    tool_imports.ffac_raw_native_ea = ea
    alt = _private_module(alt_path, "darsia.fluidflower.tools.ffac_raw_native_ea_ac44_alt",
                          {"darsia.fluidflower.tools": tool_imports})
    tool_imports.ffac_track1_v4r5_source_processor = base
    tool_imports.ffac_raw_native_ea_ac44_alt = alt
    original = _private_module(processor_path, "darsia.fluidflower.tools.ffac_track1_v4r5_ac44_alt_processor",
                               {"darsia.fluidflower.tools": tool_imports})
    tool_imports.ffac_track1_v4r5_ac44_alt_processor = original
    historical = _private_module(adapter_path, "ffac_historical_ac44_constructor_bridge")
    # Execute the original bridge bytecode. Its canonical-module operations are
    # confined to a private dict and a no-op loader for the already bound module.
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
    fake_util = SimpleNamespace(spec_from_file_location=lambda name, path: spec,
                               module_from_spec=lambda value: geometry)
    fake_importlib = SimpleNamespace(util=fake_util)
    bridge = _frozen_function(historical.install_ac44_bridge,
        {"importlib.util": fake_importlib, "darsia.fluidflower.tools": tool_imports},
        {"sys": SimpleNamespace(modules={})})
    cls = bridge(dict(ac44_geometry_module_path=str(geometry_path),
                      isolated_invalid_policy="fill_isolated_invalid_v1"), dict(adapter))
    alt._check_parent_code()
    proof = dict(contract=CONTRACT, dispatcher=dict(path=__file__,sha256=sha256_file(__file__)),
        historical_constructor_adapter=adapter,
        private_modules=[dict(path=str(path),sha256=sha256_file(path)) for path in
                         (geometry_path, ea_path, alt_path, processor_path, Path(base.__file__))],
        canonical_modules_unchanged=True, frozen_bytecode=True, correction_math_changed=False,
        source_guards_preserved=True, sha_checks_relaxed=False, cpu_only=True)
    return cls, alt, proof


def create_rig_raw_source_processor(native_manifest_path):
    """Return the ordinary processor, or the exact private AC44 frozen variant."""
    from .phase_ffac_rig_raw_source import RigRawSourceProcessor
    native = json.loads(Path(native_manifest_path).read_text(encoding="utf-8-sig"))
    if native.get("run") != "ac44":
        return RigRawSourceProcessor(native_manifest_path)
    constructor, renderer, proof = _ac44_graph(native)
    init = _frozen_function(RigRawSourceProcessor.__init__, {
        "darsia.fluidflower.tools.ffac_track1_v4r5_source_processor":
            SimpleNamespace(Track1V4R5SourceProcessor=constructor)})
    correct = _frozen_function(RigRawSourceProcessor.correct, {
        "darsia.fluidflower.tools.ffac_raw_native_ea": renderer})
    class ExactAC44RigRawSourceProcessor(RigRawSourceProcessor):
        def __init__(self, path):
            init(self, path)
            self.identity["explicit_frozen_variant_dispatch"] = proof

        def correct(self, *args, **kwargs):
            result, audit = correct(self, *args, **kwargs)
            audit["explicit_frozen_variant_dispatch"] = proof
            return result, audit
    value = ExactAC44RigRawSourceProcessor(native_manifest_path)
    value._ffac_private_geometry = renderer.geo
    value._ffac_private_alt_renderer = renderer
    if value.processor.identity().get("ac44_constructor_forwarding_adapter") != native["v5_photometry"]["processor"]["ac44_constructor_forwarding_adapter"]:
        raise ValueError("AC44 constructor adapter identity changed")
    return value


def extend_prepared_source(native_log, producer_audit, processor, ring_eligible):
    """Use the exact same sealed AC44 lens geometry on the added camera ring."""
    from . import phase_ffac_rig_raw_extension as extension
    if processor.native.get("run") != "ac44":
        return extension.extend_prepared_source(native_log, producer_audit, processor, ring_eligible)
    renderer = processor._ffac_private_alt_renderer
    renderer._check_parent_code()
    certificate = producer_audit["renderer"]
    geometry = processor._ffac_private_geometry
    if certificate["sources"]["arw_sha256"] == renderer._ALT_SHA["arw_sha256"]:
        if certificate["sources"] != renderer._ALT_SHA:
            raise ValueError("AC44 ring does not bind the exact alternate source triple")
        renderer._exact_alternate_tags(Path(producer_audit["sources"]["camera_arw"]["path"]))
        geometry = renderer._AlternateGeo()
    overrides = {"darsia.fluidflower.tools": SimpleNamespace(ffac_raw_geometry=geometry)}
    sampler = _frozen_function(extension.sample_sensor_window, overrides)
    call = _frozen_function(extension.extend_prepared_source, overrides,
                            {"sample_sensor_window": sampler})
    result, valid, audit = call(native_log, producer_audit, processor, ring_eligible)
    audit["explicit_frozen_variant_dispatch"] = processor.identity["explicit_frozen_variant_dispatch"]
    audit["ring_extension_bytecode"] = dict(path=extension.__file__, sha256=sha256_file(extension.__file__))
    return result, valid, audit


__all__ = ["create_rig_raw_source_processor", "extend_prepared_source", "CONTRACT"]
