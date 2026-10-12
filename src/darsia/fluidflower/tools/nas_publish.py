"""Reliable directory publication to Windows-accessed SMB shares."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import UTC, datetime
import errno
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import uuid
from collections.abc import Callable

from darsia.fluidflower.project import sha256_file


_FILE_ATTRIBUTE_HIDDEN = 0x2
_INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF


def _file_inventory(root: Path) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted(
            (
                path.relative_to(root).as_posix(),
                int(path.stat().st_size),
            )
            for path in root.rglob("*")
            if path.is_file()
        )
    )


def _windows_attributes(path: Path) -> int | None:
    if os.name != "nt":
        return None
    value = getattr(path.stat(), "st_file_attributes", None)
    return None if value is None else int(value)


def is_hidden(path: str | Path) -> bool:
    """Return whether Windows reports the path with the Hidden attribute."""

    attributes = _windows_attributes(Path(path))
    return bool(attributes is not None and attributes & _FILE_ATTRIBUTE_HIDDEN)


def clear_hidden_attribute(path: str | Path) -> None:
    """Remove an inherited SMB Hidden attribute from a published path."""

    target = Path(path)
    attributes = _windows_attributes(target)
    if attributes is None or not attributes & _FILE_ATTRIBUTE_HIDDEN:
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_attributes = kernel32.GetFileAttributesW
    get_attributes.argtypes = [wintypes.LPCWSTR]
    get_attributes.restype = wintypes.DWORD
    set_attributes = kernel32.SetFileAttributesW
    set_attributes.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    set_attributes.restype = wintypes.BOOL

    path_text = str(target.absolute())
    current = int(get_attributes(path_text))
    if current == _INVALID_FILE_ATTRIBUTES:
        raise ctypes.WinError(ctypes.get_last_error())
    if current & _FILE_ATTRIBUTE_HIDDEN and not set_attributes(
        path_text, current & ~_FILE_ATTRIBUTE_HIDDEN
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if is_hidden(target):
        raise OSError(f"Published directory is still hidden: {target}")


def _unique_sibling(destination: Path, marker: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    token = uuid.uuid4().hex[:8]
    return destination.with_name(
        f"{destination.name}.{marker}_{stamp}_{socket.gethostname()}_{os.getpid()}_{token}"
    )


def _hash_inventory(root: Path) -> dict[str, str]:
    """Inventory every byte in a directory, rejecting links outside the copy."""

    inventory: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError(f"Publication cannot contain a filesystem link: {path}")
        if path.is_file():
            inventory[path.relative_to(root).as_posix()] = sha256_file(path)
    if not inventory:
        raise ValueError(f"Refusing to publish an empty directory: {root}")
    return inventory


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory without replacing a competing destination."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if os.name == "nt":
        # MoveFileExW without MOVEFILE_REPLACE_EXISTING is how os.rename works
        # on Windows. It also covers SMB shares and fails if the target appears.
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOSYS, "Atomic no-replace directory rename is unavailable")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                              ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
        return
    raise OSError(errno.ENOSYS, "Atomic no-replace directory rename is unavailable")


def remove_owned_stage(root: str | Path, child: str | Path, prefix: str) -> None:
    """Remove only this invocation's direct UUID child of its resolved stage root."""

    root, child = Path(root), Path(child)
    if not re.fullmatch(re.escape(prefix) + r"[0-9a-f]{32}", child.name):
        raise ValueError(f"Unsafe staging name: {child}")
    if (root.resolve() != root or child.parent.resolve() != root
            or child.is_symlink()
            or (hasattr(child, "is_junction") and child.is_junction())
            or child.resolve().parent != root):
        raise ValueError(f"Staging cleanup escaped its root: {child}")
    if child.exists():
        shutil.rmtree(child)


def publish_immutable_directory(
    staged: str | Path,
    destination: str | Path,
    destination_staging_root: str | Path,
    *,
    verify: Callable[[Path], None],
) -> Path:
    """Copy a checked tree to the destination volume, then expose it atomically.

    The caller's temporary tree can be on a different volume. Only a newly
    allocated folder beneath ``destination_staging_root`` is ever removed.
    ``verify`` checks model identity and any domain-specific evidence at both
    the destination-volume copy and final path before returning.
    """

    staged, destination, destination_staging_root = (
        Path(path).resolve() for path in (staged, destination, destination_staging_root)
    )
    if not staged.is_dir():
        raise NotADirectoryError(staged)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if (destination == staged or destination.is_relative_to(staged)
            or staged.is_relative_to(destination)
            or destination_staging_root == staged
            or destination_staging_root.is_relative_to(staged)
            or destination_staging_root == destination
            or destination_staging_root.is_relative_to(destination)
            or destination_staging_root.anchor != destination.anchor):
        raise ValueError("Publication source, destination and destination staging overlap")
    expected = _hash_inventory(staged)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_staging_root.mkdir(parents=True, exist_ok=True)
    own_parent = destination_staging_root / f"immutable-publication-{uuid.uuid4().hex}"
    own_parent.mkdir(exist_ok=False)
    incoming = own_parent / destination.name
    published = False
    try:
        shutil.copytree(staged, incoming)
        if _hash_inventory(incoming) != expected:
            raise OSError(f"Destination-volume staging differs from verified source: {incoming}")
        verify(incoming)
        clear_hidden_attribute(incoming)
        _rename_directory_no_replace(incoming, destination)
        published = True
        clear_hidden_attribute(destination)
        if _hash_inventory(destination) != expected:
            raise OSError(f"Published directory differs from verified source: {destination}")
        verify(destination)
        return destination
    except BaseException as error:
        if published and destination.exists():
            quarantine = destination_staging_root / f"quarantined-publication-{uuid.uuid4().hex}"
            try:
                _rename_directory_no_replace(destination, quarantine)
                error.add_note(f"Published tree moved out of the scanned library to {quarantine}")
            except OSError as quarantine_error:
                error.add_note(f"Could not quarantine {destination}: {quarantine_error}")
        raise
    finally:
        # This UUID folder was created by this invocation. Never remove the
        # caller's stage, a prior failed attempt, or a quarantine.
        remove_owned_stage(destination_staging_root, own_parent, "immutable-publication-")


def publish_directory(source: str | Path, destination: str | Path) -> Path:
    """Publish a complete directory with verification and rollback.

    Staging directories deliberately do not start with a dot. Some SMB servers
    map dot-prefixed directories to the Windows Hidden attribute, which then
    survives a rename to the final destination.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_dir():
        raise NotADirectoryError(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    expected = _file_inventory(source_path)
    if not expected:
        raise ValueError(f"Refusing to publish an empty directory: {source_path}")

    staging = _unique_sibling(destination_path, "incoming")
    archive: Path | None = None
    try:
        shutil.copytree(source_path, staging)
        if _file_inventory(staging) != expected:
            raise OSError(f"Staging inventory differs from source: {staging}")
        clear_hidden_attribute(staging)

        if destination_path.exists():
            archive = _unique_sibling(destination_path, "archived")
            os.replace(destination_path, archive)
        try:
            os.replace(staging, destination_path)
        except BaseException:
            if archive is not None and archive.exists() and not destination_path.exists():
                os.replace(archive, destination_path)
            raise

        clear_hidden_attribute(destination_path)
        if _file_inventory(destination_path) != expected:
            raise OSError(
                f"Published inventory differs from source: {destination_path}"
            )
        if is_hidden(destination_path):
            raise OSError(f"Published directory is hidden: {destination_path}")
        return destination_path
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
