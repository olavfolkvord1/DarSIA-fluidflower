"""Stored arrays: one writer of a result's arrays, plain or compressed losslessly.

A result's arrays are written by :func:`save_arrays` into a NumPy archive
(``.npz``), as :func:`numpy.savez` writes them. The process's storage settings
(:class:`StorageSettings`) say whether its members are stored plain, exactly as
:func:`numpy.savez` stores them, or deflated with zlib, which loses nothing.
:func:`numpy.load` reads both, inside DarSIA and outside it, so no reader changes.
The settings are set once per process (:func:`use_storage`), before the work
starts, e.g. by a job's task from its configuration's ``[storage]`` table; without
them the arrays are stored plain.

"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StorageSettings:
    """How the arrays of results are stored (the ``[storage]`` table)."""

    compress: bool = False
    """Whether the arrays are compressed losslessly (deflated with zlib)."""
    level: int = 1
    """The zlib level of compressed arrays, from 1 (fastest) to 9 (smallest)."""


_current: list[StorageSettings] = [StorageSettings()]
"""The process's storage settings."""


def storage_settings() -> StorageSettings:
    """How this process stores the arrays of results.

    Returns
    -------
    StorageSettings
        The settings set by :func:`use_storage`, plain storage without them.

    """
    return _current[0]


def use_storage(settings: StorageSettings) -> None:
    """Store the arrays of results of this process by these settings from now on.

    Parameters
    ----------
    settings : StorageSettings
        The settings, e.g. ``StorageSettings(compress=True)``.

    """
    _current[0] = settings


def save_arrays(path: str | Path, /, **arrays) -> None:
    """Write arrays into a NumPy archive by the process's storage settings.

    The archive is that of :func:`numpy.savez`: a ZIP64 archive of one ``.npy``
    member per array, by its name, objects pickled. Stored plain, its bytes are
    those :func:`numpy.savez` writes at the same moment; compressed, each member
    is deflated at the settings' level. :func:`numpy.load` reads both.

    Parameters
    ----------
    path : str or Path
        The archive; ``.npz`` is appended if it lacks it, as :func:`numpy.savez`
        does.
    **arrays
        The arrays by name.

    """
    file = os.fspath(path)
    if not file.endswith(".npz"):
        file += ".npz"
    settings = storage_settings()
    method = zipfile.ZIP_DEFLATED if settings.compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(
        file, "w", method, allowZip64=True, compresslevel=settings.level
    ) as archive:
        for name, value in arrays.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(
                    member, np.asanyarray(value), allow_pickle=True
                )
