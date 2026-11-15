"""Layering rules for the listed modules.

Each listed module imports only from its own layer or below (methods never import
the workflow presets or the graphical interface; presets never import the
interface), imports at module level only, and holds no experiment identifiers,
file system paths, dates or version suffixes. A private name (a leading
underscore) is imported only from a module of the same package, e.g. among the
modules of ``multiphase``, never across packages.

"""

import ast
import re
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src"

MODULES = [
    "darsia/corrections/shape/imagemapping.py",
    "darsia/corrections/shape/metricscaling.py",
    "darsia/corrections/color/materialilluminationcorrection.py",
    "darsia/corrections/color/materialblocks.py",
    "darsia/corrections/color/reference_proposal.py",
    "darsia/corrections/color/sandlightfield.py",
    "darsia/multiphase/phase_model.py",
    "darsia/multiphase/phase_rules.py",
    "darsia/signals/models/color_path_signal.py",
    "darsia/signals/color/color_path_fit.py",
    "darsia/signals/models/beer_lambert.py",
    "darsia/multi_image_analysis/run_statistics.py",
    "darsia/multi_image_analysis/gas_statistics.py",
    "darsia/multi_image_analysis/calibration_fit.py",
    "darsia/experiment/overflow.py",
    "darsia/experiment/overflow_feed.py",
    "darsia/experiment/overflow_circulation.py",
    "darsia/experiment/overflow_evaporation.py",
    "darsia/multiphase/gas_allocation.py",
    "darsia/multiphase/gas_bridge.py",
    "darsia/multiphase/gas_model_cache.py",
    "darsia/multiphase/gas_parameters.py",
    "darsia/multiphase/gas_zone.py",
    "darsia/multiphase/gas_zone_cache.py",
    "darsia/multiphase/mass_maps.py",
    "darsia/multiphase/unknown_mass.py",
    "darsia/presets/workflows/analysis/domains.py",
    "darsia/presets/workflows/analysis/domain_mass.py",
    "darsia/presets/workflows/comparison/comparison_events.py",
    "darsia/presets/workflows/comparison/ensemble.py",
    "darsia/presets/workflows/comparison/ensemble_figures.py",
    "darsia/presets/workflows/comparison/mass_wasserstein.py",
    "darsia/image/raw.py",
    "darsia/corrections/shape/registration.py",
    "darsia/corrections/shape/cameralens.py",
    "darsia/corrections/shape/radiallens.py",
    "darsia/corrections/shape/lens.py",
    "darsia/presets/workflows/setup/sand_map.py",
    "darsia/presets/workflows/setup/setup_sand_map.py",
    "darsia/presets/workflows/setup/sand_map_snapping.py",
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
    """No import of a higher layer or of another package's private names, and no
    import inside a function."""
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
        if isinstance(node, ast.ImportFrom) and any(
            alias.name.startswith("_") for alias in node.names
        ):
            (name,) = _imported(node, package)
            assert (
                not name.startswith("darsia.") or name.split(".")[:-1] == package
            ), f"{module} imports a private name from {name}"


@pytest.mark.parametrize("module", MODULES)
def test_no_experiment_literals(module):
    """No experiment identifiers, paths, dates or version suffixes."""
    text = (SOURCE / module).read_text(encoding="utf-8")
    assert not LITERALS.findall(text), f"{module}: {LITERALS.findall(text)}"
