"""Saved-map code admission locates source identities without numerical imports."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from darsia.fluidflower import phase_ffac_t4c_review as review


def test_binding_hashes_original_tuple_without_executing_targets(tmp_path, monkeypatch):
    package = tmp_path / "saved_binding_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    runner = package / "runner.py"
    numeric = package / "numeric.py"
    numeric.write_text("raise AssertionError('numerical target must not execute')\n", encoding="utf-8")
    runner.write_text(
        "raise AssertionError('runner must not execute')\n"
        "def code_binding():\n"
        "    names = (__name__, 'saved_binding_fixture.numeric')\n"
        "    return {name: sha256_file(importlib.import_module(name).__file__) for name in names}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    binding = review._declared_code_binding(SimpleNamespace(
        __file__=str(runner), __name__="saved_binding_fixture.runner",
    ))
    assert binding == {
        "saved_binding_fixture.runner": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "saved_binding_fixture.numeric": hashlib.sha256(numeric.read_bytes()).hexdigest(),
    }
    assert "saved_binding_fixture.runner" not in sys.modules
    assert "saved_binding_fixture.numeric" not in sys.modules


def test_loaded_module_keeps_literal_file_identity(tmp_path, monkeypatch):
    source = tmp_path / "loaded.py"
    source.write_text("bound source\n", encoding="utf-8")
    loaded = ModuleType("saved_binding_loaded")
    loaded.__file__ = str(source)
    monkeypatch.setitem(sys.modules, loaded.__name__, loaded)
    assert review._module_source_path(loaded.__name__) == source


def test_nonliteral_names_are_rejected_without_execution(tmp_path):
    source = tmp_path / "unsafe_runner.py"
    source.write_text(
        "raise AssertionError('no source execution')\n"
        "def code_binding():\n"
        "    names = tuple(discover_numerical_modules())\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="literal name tuple"):
        review._declared_code_binding(SimpleNamespace(
            __file__=str(source), __name__="unsafe_runner",
        ))


def test_saved_admission_guards_unchanged_except_source_lookup():
    repo = Path(__file__).resolve().parents[2]
    relative = "src/darsia/fluidflower/phase_ffac_t4c_review.py"
    original = subprocess.check_output(
        ["git", "show", "1be6b7b510233c3d35d21ed4119f8552953408a3:" + relative],
        cwd=repo, text=True, encoding="utf-8",
    )
    old = next(node for node in ast.parse(original).body
               if isinstance(node, ast.FunctionDef) and node.name == "_admitted")
    new = next(node for node in ast.parse((repo / relative).read_text()).body
               if isinstance(node, ast.FunctionDef) and node.name == "_admitted")

    class RestoreLookup(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "_declared_code_binding":
                assert len(node.args) == 1 and isinstance(node.args[0], ast.Name)
                assert node.args[0].id == "runner" and not node.keywords
                return ast.Call(func=ast.Attribute(value=ast.Name(id="runner", ctx=ast.Load()),
                                                  attr="code_binding", ctx=ast.Load()),
                                args=[], keywords=[])
            return node

    assert ast.dump(RestoreLookup().visit(new), include_attributes=False) == ast.dump(
        old, include_attributes=False,
    )
