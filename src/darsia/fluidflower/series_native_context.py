"""Native UI ownership of a series' explicit parameter document.

No parameter lookup takes place on module import. This adapter reads only the
selected manifest and its SHA-bound JSON; the original configuration API owns
all native parameter validation and numerical interpretation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import re
import tomllib


_ENVIRONMENT_KEYS = ("DARSIA_NATIVE_CONFIGURATION", "DARSIA_NATIVE_CONFIGURATION_SHA256")


@dataclass(frozen=True)
class SeriesNativeBinding:
    series_root: Path
    path: Path
    sha256: str
    document: Mapping


def read_series_native_binding(series_root: str | Path | None) -> SeriesNativeBinding | None:
    """Load a complete explicit binding, or return None for an absent table.

    A present malformed table/pair/digest is an error, not the no-native case.
    This function has no context or environment side effects.
    """
    if series_root is None:
        return None
    root = Path(series_root).expanduser().resolve()
    manifest = root / "series.toml"
    if not manifest.is_file():
        return None
    with manifest.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError(f"Series manifest exceeds the metadata limit: {manifest}")
    document = tomllib.loads(raw.decode("utf-8-sig"))
    if "native" not in document:
        return None
    native = document["native"]
    if not isinstance(native, Mapping):
        raise ValueError("series.toml [native] must be a table")
    path_text, digest = native.get("configuration"), native.get("sha256")
    if not isinstance(path_text, str) or not path_text.strip():
        raise ValueError("series.toml [native].configuration must name a parameter document")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("series.toml [native].sha256 must be an exact SHA-256 digest")
    path = Path(path_text).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    from .native.config import load_parameters

    parameters = load_parameters(path, digest)
    return SeriesNativeBinding(root, path, digest, parameters)


class SeriesNativeContext:
    """One native window's main-context token and owned environment pair.

    Preparation precedes switching, so a refused/invalid target cannot displace
    the currently accepted series. Background catalogue workers copy the main
    context at launch; changing the process pair does not change those copies.
    """

    def __init__(self):
        self.binding = None
        self._token = None
        self._prior_environment = None
        self._installed_environment = None

    def bind(self, series_root):
        binding = read_series_native_binding(series_root)
        if (binding is not None and self.binding is not None
                and binding.series_root == self.binding.series_root
                and binding.path == self.binding.path
                and binding.sha256 == self.binding.sha256
                and binding.document == self.binding.document):
            return self.binding
        # A bad target never reaches this point: the old accepted binding stays.
        self.close()
        if binding is None:
            return None
        from .native.config import configure_parameters

        self._prior_environment = {key: os.environ.get(key) for key in _ENVIRONMENT_KEYS}
        self._token = configure_parameters(binding.document)
        self._installed_environment = dict(zip(_ENVIRONMENT_KEYS, (str(binding.path), binding.sha256)))
        os.environ.update(self._installed_environment)
        self.binding = binding
        return binding

    def close(self):
        if self._token is not None:
            from .native.config import reset_parameters

            reset_parameters(self._token)
            self._token = None
        if self._prior_environment is not None:
            # Do not overwrite another owner's later environment update.
            if all(os.environ.get(key) == value for key, value in self._installed_environment.items()):
                for key, value in self._prior_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        self.binding = None
        self._prior_environment = self._installed_environment = None
