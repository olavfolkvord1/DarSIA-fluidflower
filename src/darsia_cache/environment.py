"""Pure pre-import environment derivation for a fixed host launcher.

Importing this standalone module does not import darsia/Numba/Matplotlib, create
directories, reserve space, launch a process, or prove a hard quota. The fixed
host supervisor must first admit/create the exact scope and then pass this
environment to its unchanged standard worker/test command. No per-run program.
"""
from pathlib import PureWindowsPath
import re


def derive_cache_environment(stores_document, *, purpose, token, scope_root, parent_environment):
    """Return new child environment; metadata alone grants no allocation right."""
    if purpose not in {"jobs", "gates", "tests", "display", "logs"}:
        raise ValueError("A disposable purpose is required; Git clones are separate")
    if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise ValueError("Exact admitted process-scope token required")
    cache = stores_document.get("cache")
    if (not isinstance(cache, dict) or type(cache.get("quota_bytes")) is not int
            or cache["quota_bytes"] <= 0 or not isinstance(cache.get("profiles"), dict)
            or purpose not in cache["profiles"]):
        raise ValueError("An explicit measured purpose profile and quota are required")
    root = PureWindowsPath(cache["root"])
    if (not root.is_absolute() or re.fullmatch(r"[A-Za-z]:", root.drive) is None
            or ".." in root.parts or len(root.parts) < 2):
        raise ValueError("Normalized local non-volume-root cache required")
    expected = root / purpose / ("current" if purpose == "tests" else "") / token
    if str(PureWindowsPath(scope_root)).casefold() != str(expected).casefold():
        raise ValueError("Child cache path differs from the exact admitted scope")
    result = dict(parent_environment)
    for key in ("TEMP", "TMP", "TMPDIR"):
        result[key] = str(expected / "tmp")
    result.update(NUMBA_CACHE_DIR=str(expected / "numba"),
                  NUMBA_CACHE_LOCATOR_CLASSES="UserProvidedCacheLocator",
                  MPLCONFIGDIR=str(expected / "matplotlib"),
                  PYTHONPYCACHEPREFIX=str(expected / "pycache"),
                  DARSIA_FLUIDFLOWER_CACHE=str(expected),
                  DARSIA_PHASE_DISPLAY_CACHE=str(expected / "display"))
    # Do not change HOME/LOCALAPPDATA, JIT, numerical threads or scientific args.
    return result
