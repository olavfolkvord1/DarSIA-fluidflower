"""Layering rules for the listed modules.

Each listed module imports only from its own layer or below (methods never import
the workflow presets or the graphical interface; presets never import the
interface), imports at module level only, and holds no experiment identifiers,
file system paths, dates or version suffixes.

"""

import ast
import re
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src"

MODULES = [
    "darsia/corrections/shape/imagemapping.py",
    "darsia/corrections/color/materialilluminationcorrection.py",
    "darsia/corrections/color/materialblocks.py",
    "darsia/corrections/color/reference_proposal.py",
    "darsia/corrections/color/sandlightfield.py",
]
"""Modules under the rules, relative to the source folder."""

LITERALS = re.compile(
    r"(?<!\w)[A-Za-z]:[\\/](?=\w)"  # drive-letter path
    r"|\\\\\\\\\w"  # network path, as written in a string literal
    r"|/(?:home|Users|mnt|media)/"  # user and mount paths
    r"|\b20\d\d-?[01]\d-?[0-3]\d\b"  # dates
    r"|\b[A-Z]{2}\d{2,3}\b"  # experiment identifiers
    r"|\bFF_?[A-Z]{2}\b"  # series names
    r"|\w_v\d+\b"  # version suffixes
)


def _forbidden(module: str) -> tuple[str, ...]:
    """Packages the module must not import, by its layer."""
    if module.startswith("darsia/gui/"):
        return ()
    if module.startswith("darsia/presets/"):
        return ("darsia.gui",)
    return ("darsia.presets", "darsia.gui")


def _imported(node: ast.AST, package: list[str]) -> list[str]:
    """Absolute names imported by an import statement."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base = package[: len(package) - node.level + 1] if node.level else []
    return [".".join(base + ([node.module] if node.module else []))]


@pytest.mark.parametrize("module", MODULES)
def test_imports_follow_the_layers(module):
    """No import of a higher layer, and no import inside a function."""
    tree = ast.parse((SOURCE / module).read_text(encoding="utf-8"))
    package = module.removesuffix(".py").split("/")[:-1]
    forbidden = _forbidden(module)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = [
                n for n in ast.walk(node) if isinstance(n, (ast.Import, ast.ImportFrom))
            ]
            assert not inner, f"{module}: import inside {node.name}()"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in _imported(node, package):
                assert not name.startswith(forbidden), f"{module} imports {name}"


@pytest.mark.parametrize("module", MODULES)
def test_no_experiment_literals(module):
    """No experiment identifiers, paths, dates or version suffixes."""
    text = (SOURCE / module).read_text(encoding="utf-8")
    assert not LITERALS.findall(text), f"{module}: {LITERALS.findall(text)}"
