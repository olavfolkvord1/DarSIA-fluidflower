"""User-configured tool locations and lexical aliases for moved storage.

Optional registry tables: ``[tool_paths.<series>]`` and
``[storage.path_aliases]``. Relative tool paths use the series project root;
relative alias destinations use the registry directory. No drive is assumed.
"""
from __future__ import annotations

import ntpath
import os
from pathlib import Path, PureWindowsPath
import tomllib

from .series_paths import series_key, series_registry_path

_settings_stamp = None
_settings_data: dict = {}
_tool_cache: dict = {}

_DEFAULTS = {
    "calibration_logs": ("results", "calibration"),
    "analysis_v2": ("results", "analysis"),
    "combined_reviews": ("reviews", "combined"),
    "review_campaign": ("reviews", ""),
    "physics_layer_exports": ("results", "physics_layers"),
    "raw_inputs": ("data", "raw"),
    "article_inputs": ("data", "article"),
    "article_raw_inputs": ("data", "article/raw"),
    "runtime_inputs": ("data", "runtime"),
    "previous_results": ("results", ""),
    "dataset_root": ("root", ""),
}


def _settings() -> tuple[Path, dict]:
    global _settings_stamp, _settings_data
    # The registry is local configuration. Never resolve a candidate image here.
    registry = series_registry_path()
    try:
        stat = registry.stat()
        revision = (stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        revision = None
    stamp = (str(registry), revision)
    if stamp != _settings_stamp:
        if revision is None:
            data = {}
        else:
            with registry.open("rb") as stream:
                data = tomllib.load(stream)
            if data.get("schema_version") != 1:
                raise ValueError(f"Unsupported DarSIA series registry: {registry}")
        _settings_data = data
        _settings_stamp = stamp
        _tool_cache.clear()
    return registry, _settings_data


def _normalized(value) -> str:
    value = os.path.expanduser(os.fspath(value)).replace("\\", "/")
    # ntpath supports Windows spellings even in tests on a CPU-only Unix host.
    result = ntpath.normpath(value).replace("\\", "/").rstrip("/")
    if len(result) == 2 and result[1] == ":":
        result += "/"
    return result or "/"


def _is_absolute(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _alias_rows() -> list[tuple[str, str]]:
    """Return the registry's normalized (source, destination) prefix aliases.

    Sources are case-folded and sorted longest first; relative destinations are
    read against the registry folder. An empty list means no alias is set.
    """
    registry, settings = _settings()
    aliases = settings.get("storage", {}).get("path_aliases", {})
    rows = []
    for source, destination in aliases.items():
        source = _normalized(source)
        destination = _normalized(destination)
        if not _is_absolute(source):
            raise ValueError(f"Storage alias source must be absolute: {source}")
        if not _is_absolute(destination):
            destination = _normalized(registry.parent / destination)
        rows.append((source.rstrip("/").casefold(), destination.rstrip("/")))
    rows.sort(key=lambda row: len(row[0]), reverse=True)
    return rows


def storage_path_remapper():
    """Return a function that reads path strings through the storage aliases.

    The registry is read once, for callers that map many paths at a time. The
    function returns a string that an alias source prefixes as
    :func:`resolve_storage_path` would; any other string is returned
    unchanged, character for character, so a machine without aliases reads
    exactly the authored text.
    """
    rows = _alias_rows()

    def remap(value) -> str:
        text = os.fspath(value)
        if rows:
            key = _normalized(text).casefold()
            if any(key == source or key.startswith(source + "/") for source, _ in rows):
                return str(_resolve_with(text, rows))
        return text

    return remap


def resolve_storage_path(path) -> Path:
    """Apply explicitly configured prefix aliases without accessing old storage.

    Missing destinations do not fall back to an obsolete source. This changes
    only the path used by a reader, never the bytes of its frozen document.
    """
    return _resolve_with(path, _alias_rows())


def _resolve_with(path, rows) -> Path:
    """Resolve ``path`` through already read alias rows (see _alias_rows)."""
    if not rows:
        return Path(path).expanduser()
    result = _normalized(path)
    seen = set()
    applied = set()
    while True:
        key = result.casefold()
        if key in seen:
            raise ValueError(f"Cyclic storage path aliases for {path}")
        seen.add(key)
        for source, destination in rows:
            if key == source or key.startswith(source + "/"):
                if source in applied:
                    raise ValueError(f"Cyclic storage path aliases for {path}")
                applied.add(source)
                suffix = result[len(source):]
                result = destination + suffix
                if len(result) == 2 and result[1] == ":":
                    result += "/"
                break
        else:
            return Path(result)


def configured_tool_path(dataset: str, key: str) -> Path:
    """Resolve a tool default without embedding a researcher's storage layout.

    User overrides precede explicit manifest paths and legacy input locations.
    Output defaults stay inside the registered project. Unconfigured projects
    use a portable local project location; they never select someone else's
    scientific inputs. Explicit CLI paths continue to override tool defaults.
    """
    if key not in _DEFAULTS:
        raise KeyError(f"Unknown tool path: {key}")
    registry, settings = _settings()
    dataset = series_key(dataset)
    explicit_root = os.environ.get(f"DARSIA_{dataset.upper()}_PROJECT_ROOT")
    cache_key = (dataset, key, explicit_root)
    if cache_key in _tool_cache:
        return _tool_cache[cache_key]
    raw_root = explicit_root or next((row["root"] for row in settings.get("series", [])
                                     if series_key(row["id"]) == dataset), None)
    if raw_root:
        root = resolve_storage_path(raw_root)
        if not _is_absolute(str(root)):
            root = registry.parent / root
        # Accept experiment-root registration as well as project registration.
        if not (root / "series.toml").is_file() and (root / "DarSIA/series.toml").is_file():
            root = root / "DarSIA"
    else:
        from .project import default_project_root
        root = default_project_root() / dataset
    manifest_path = root / "series.toml"
    manifest = {}
    if manifest_path.is_file():
        with manifest_path.open("rb") as stream:
            manifest = tomllib.load(stream)
        if manifest.get("schema_version") != 1 or series_key(manifest["series_id"]) != dataset:
            raise ValueError(f"Wrong series manifest: {manifest_path}")
    overrides = settings.get("tool_paths", {}).get(dataset, {})
    raw = overrides.get(key, manifest.get("paths", {}).get(key,
                        manifest.get("legacy", {}).get(key)))
    if raw is None and key == "article_raw_inputs":
        article = overrides.get("article_inputs", manifest.get("paths", {}).get(
            "article_inputs", manifest.get("legacy", {}).get("article_inputs")))
        if article is not None:
            raw = str(Path(article) / "raw")
    if raw is None:
        base, suffix = _DEFAULTS[key]
        raw = str(Path(manifest.get("paths", {}).get(base, "." if base == "root" else base)) / suffix)
    result = resolve_storage_path(raw)
    if not _is_absolute(str(result)):
        result = root / result
    result = Path(os.path.abspath(result))
    _tool_cache[cache_key] = result
    return result
