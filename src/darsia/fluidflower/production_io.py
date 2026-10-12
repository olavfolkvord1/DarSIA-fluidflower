"""Content-verified local/NAS IO for production jobs.

This module imports only the standard library. It never decodes arrays or runs
scientific computations. Immutable publication never replaces an existing file;
``replace_json`` is the separate, explicitly mutable control-file operation.
Interrupted owned temporaries and local copy sources are retained.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import logging
import math
import ntpath
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
from typing import Any, Callable, Mapping
import uuid

LOGGER = logging.getLogger(__name__)
CHUNK_BYTES = 1024**2
JSON_LIMIT_BYTES = 16 * 1024**2
_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")
_TIMESTAMP_KEYS = frozenset(
    ("timestamp", "process_create_time", "pid_create_time_epoch", "create_time_epoch")
)


class IntegrityError(ValueError):
    """A content, descriptor, stability or JSON check failed; never retry it."""


def normalize_locator(path, drive_mappings: Mapping[str, str] | None = None) -> str:
    """Normalize separators/case/dots and Windows mapped drives to UNC.

    Paths locate content; this normalized value is not the content identity.
    An explicit mapping is useful for platform-independent tests. Otherwise
    Windows asks the OS for its actual mapping, rather than assuming X: is NAS.
    POSIX locators retain their case sensitivity.
    """
    value = os.fspath(path)
    windows = os.name == "nt" or bool(ntpath.splitdrive(value)[0]) or value.startswith(
        ("\\\\", "//")
    )
    if not windows:
        return os.path.normpath(os.path.abspath(value))
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    value = ntpath.abspath(value)
    drive, tail = ntpath.splitdrive(value)
    mappings = (
        {}
        if drive_mappings is None
        else {key.casefold(): value for key, value in drive_mappings.items()}
    )
    mapped = mappings.get(drive.casefold())
    if (
        mapped is None
        and drive_mappings is None
        and os.name == "nt"
        and len(drive) == 2
        and drive[1] == ":"
    ):
        buffer = ctypes.create_unicode_buffer(32768)
        length = ctypes.c_ulong(len(buffer))
        code = ctypes.windll.mpr.WNetGetConnectionW(
            drive, buffer, ctypes.byref(length)
        )
        if code == 0:
            mapped = buffer.value
        elif code not in (2250, 1200):  # Not connected / not a network device.
            raise OSError(code, "Cannot establish mapped-drive identity")
    if mapped is not None:
        if not str(mapped).startswith(("\\\\", "//")):
            raise IntegrityError("Mapped-drive endpoint must be UNC")
        value = ntpath.join(mapped, tail.lstrip("\\/"))
    return ntpath.normcase(ntpath.normpath(value))


def is_network_path(path) -> bool:
    return normalize_locator(path).startswith("\\\\")


def is_transient(error: BaseException) -> bool:
    """Only known permission/sharing/network-timeout IO errors are transient."""
    if not isinstance(error, OSError) or isinstance(
        error, (FileExistsError, FileNotFoundError)
    ):
        return False
    winerror = getattr(error, "winerror", None)
    return (
        isinstance(error, TimeoutError)
        or error.errno == errno.ETIMEDOUT
        or winerror in (5, 32, 33, 64, 121, 258, 1460)
        or (error.errno == errno.EACCES and winerror is None)
    )


class RetryPolicy:
    """Transient NAS IO retries with fresh, non-retriable guard callbacks."""

    def __init__(
        self,
        *,
        max_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        on_retry: Callable[[dict], None] | None = None,
    ):
        if (
            type(max_seconds) not in (int, float)
            or not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 60
        ):
            raise ValueError("Retry window must be positive and at most 60 seconds")
        self.max_seconds = float(max_seconds)
        self.sleep = sleep
        self.clock = clock
        self.on_retry = on_retry
        self.events: list[dict] = []

    def _log(self, event):
        self.events.append(event)
        LOGGER.warning("Production NAS IO retry", extra={"production_io": event})
        if self.on_retry is not None:
            self.on_retry(dict(event))

    def run(self, operation, path, callback, *, before_attempt=None):
        network = is_network_path(path)
        started, delay, attempt = self.clock(), 0.1, 0
        while True:
            attempt += 1
            # Authority/content guard failures are outside the IO retry catch.
            if before_attempt is not None:
                before_attempt()
            try:
                return callback()
            except OSError as error:
                if not network or not is_transient(error):
                    raise
                elapsed = self.clock() - started
                remaining = self.max_seconds - elapsed
                event = dict(
                    operation=operation,
                    locator=normalize_locator(path),
                    attempt=attempt,
                    elapsed_seconds=elapsed,
                    errno=error.errno,
                    winerror=getattr(error, "winerror", None),
                    error_type=type(error).__name__,
                    message=str(error),
                )
                if remaining <= 0:
                    event["exhausted"] = True
                    self._log(event)
                    raise
                pause = min(delay, 5.0, remaining)
                event["sleep_seconds"] = pause
                self._log(event)
                self.sleep(pause)
                delay = min(delay * 2, 5.0)


def _shape(bound):
    if (
        not isinstance(bound, Mapping)
        or set(bound) not in ({"path", "sha256"}, {"path", "sha256", "size_bytes"})
        or not isinstance(bound["path"], (str, os.PathLike))
        or not (os.path.isabs(bound["path"]) or ntpath.isabs(os.fspath(bound["path"])))
        or not isinstance(bound["sha256"], str)
        or _SHA.fullmatch(bound["sha256"]) is None
        or (
            "size_bytes" in bound
            and (type(bound["size_bytes"]) is not int or bound["size_bytes"] < 0)
        )
    ):
        raise IntegrityError("Exact compact/full absolute content descriptor required")


def _hash_file(path):
    path = Path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise IntegrityError("Regular content file required")
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            count += len(block)
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise IntegrityError("File changed during verification")
    if count != after.st_size:
        raise IntegrityError("File read was incomplete")
    return dict(path=str(path.absolute()), sha256=digest.hexdigest(), size_bytes=count)


def _match(actual, expected):
    if actual["sha256"] != expected["sha256"].lower() or (
        "size_bytes" in expected and actual["size_bytes"] != expected["size_bytes"]
    ):
        raise IntegrityError("Content SHA-256/size differs")
    return actual


def _exists(path):
    try:
        Path(path).lstat()
        return True
    except FileNotFoundError:
        return False


def _temporary(destination):
    destination = Path(destination)
    return destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")


def _write_all(stream, data):
    view, offset = memoryview(data), 0
    while offset < len(view):
        count = stream.write(view[offset : offset + CHUNK_BYTES])
        if type(count) is not int or count <= 0:
            raise OSError(errno.EIO, "Output write made no progress")
        if count > min(CHUNK_BYTES, len(view) - offset):
            raise IntegrityError("Writer returned an impossible byte count")
        offset += count


def _move_no_replace(source, destination):
    """Atomic no-overwrite move; unsupported directory platforms fail closed."""
    source, destination = Path(source), Path(destination)
    if os.name == "nt":
        return os.rename(source, destination)  # Windows rejects an existing target.
    if not source.is_dir():
        os.link(source, destination, follow_symlinks=False)
        source.unlink()
        return
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is not None:
            result = rename(
                -100, os.fsencode(source), -100, os.fsencode(destination), 1
            )  # AT_FDCWD / RENAME_NOREPLACE.
            if result == 0:
                return
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(destination))
    raise OSError(errno.ENOTSUP, "Atomic no-overwrite directory move unavailable")


def exact_timestamp(value: int | str) -> int | str:
    """Keep an identity timestamp as an integer or literal string, never float."""
    if type(value) not in (int, str):
        raise IntegrityError("Exact timestamp must be an integer or string")
    return value


def _timestamps(value, _path=()):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise IntegrityError("JSON object keys must be strings")
            if key in _TIMESTAMP_KEYS or key.endswith(("_utc", "_timestamp")):
                # This frozen dose field describes absent filename metadata,
                # not a process/control epoch. Keep its scientific null intact.
                missing_filename = (item is None and key == "filename_timestamp"
                                    and len(_path) == 3 and _path[0] == "runs"
                                    and type(_path[1]) is int
                                    and _path[2] == "measured_mfc_dose")
                if not missing_filename:
                    exact_timestamp(item)
            _timestamps(item, (*_path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _timestamps(item, (*_path, index))


def _json_bytes(document):
    _timestamps(document)
    return (json.dumps(document, allow_nan=False, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _json_load(data):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise IntegrityError("Duplicate JSON key")
            value[key] = item
        return value

    def invalid(value):
        raise IntegrityError("Non-finite JSON value: " + value)

    value = json.loads(
        data.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=invalid
    )
    _timestamps(value)
    return value


def _expected_inventory(expected_files):
    if not isinstance(expected_files, Mapping) or not expected_files:
        raise IntegrityError("Nonempty exact relative-file inventory required")
    expected = {}
    for name, bound in expected_files.items():
        if not isinstance(name, str):
            raise IntegrityError("Relative file name must be a string")
        relative = PurePosixPath(name.replace("\\", "/"))
        if (
            relative.is_absolute()
            or ntpath.splitdrive(name)[0]
            or ".." in relative.parts
            or str(relative) in (".", "")
            or str(relative) in expected
            or not isinstance(bound, Mapping)
            or set(bound) not in (
                {"sha256", "size_bytes"},
                {"path", "sha256", "size_bytes"},
            )
            or not isinstance(bound["sha256"], str)
            or _SHA.fullmatch(bound["sha256"]) is None
            or type(bound["size_bytes"]) is not int
            or bound["size_bytes"] < 0
        ):
            raise IntegrityError("Exact relative filename/SHA/integer-size required")
        expected[str(relative)] = dict(
            sha256=bound["sha256"].lower(), size_bytes=bound["size_bytes"]
        )
    return expected


def _directory_snapshot(folder, expected):
    folder = Path(folder)
    if folder.is_symlink() or not stat.S_ISDIR(folder.lstat().st_mode):
        raise IntegrityError("Ordinary directory required")
    files, directories = {}, set()
    pending = [folder]
    while pending:
        parent = pending.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink():
                    raise IntegrityError("Publication contains a symlink")
                mode = entry.stat(follow_symlinks=False).st_mode
                relative = path.relative_to(folder).as_posix()
                if stat.S_ISDIR(mode):
                    directories.add(relative)
                    pending.append(path)
                elif stat.S_ISREG(mode):
                    files[relative] = _hash_file(path)
                else:
                    raise IntegrityError("Publication contains a special file")
    required_directories = {
        str(parent)
        for name in expected
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    expected_locators = {
        normalize_locator(folder / name): name for name in expected
    }
    actual_locators = {
        normalize_locator(folder / name): name for name in files
    }
    if (
        len(expected_locators) != len(expected)
        or len(actual_locators) != len(files)
        or set(actual_locators) != set(expected_locators)
        or {normalize_locator(folder / name) for name in directories}
        != {normalize_locator(folder / name) for name in required_directories}
    ):
        raise IntegrityError("Published directory file/directory inventory differs")
    result = {}
    for locator, name in expected_locators.items():
        actual = files[actual_locators[locator]]
        result[name] = _match(actual, expected[name])
    return result


class NasIO:
    """One engine for local operations and transient-only NAS retries."""

    def __init__(self, retry_policy: RetryPolicy | None = None):
        self.retry = retry_policy if retry_policy is not None else RetryPolicy()

    def asset(self, path, *, before_attempt=None):
        return self.retry.run(
            "verify", path, lambda: _hash_file(path), before_attempt=before_attempt
        )

    def verified_asset(self, bound, *, before_attempt=None):
        _shape(bound)
        return self.retry.run(
            "verify",
            bound["path"],
            lambda: _match(_hash_file(bound["path"]), bound),
            before_attempt=before_attempt,
        )

    def verify(self, bound, *, before_attempt=None) -> Path:
        self.verified_asset(bound, before_attempt=before_attempt)
        return Path(bound["path"])

    def read_bytes(self, path, *, expected=None, max_bytes=None, before_attempt=None):
        if isinstance(path, Mapping):
            if expected is not None:
                raise IntegrityError("Supply one content descriptor")
            expected, path = path, path["path"]
        if expected is not None:
            _shape(expected)
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ValueError("Integer nonnegative read limit required")

        def read():
            p = Path(path)
            before = p.stat()
            with p.open("rb") as stream:
                data = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
            after = p.stat()
            if max_bytes is not None and len(data) > max_bytes:
                raise IntegrityError("File exceeds the read limit")
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ) or len(data) != after.st_size:
                raise IntegrityError("File changed, truncated or exceeded read limit")
            if expected is not None:
                _match(
                    dict(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)),
                    expected,
                )
            return data

        return self.retry.run("read", path, read, before_attempt=before_attempt)

    def read_json(self, path, *, expected=None, max_bytes=JSON_LIMIT_BYTES, before_attempt=None):
        # JSON/content failures remain outside retry; only complete fresh reads retry.
        return _json_load(
            self.read_bytes(
                path, expected=expected, max_bytes=max_bytes, before_attempt=before_attempt
            )
        )

    def publish_file(self, source, destination, expected, *, before_attempt=None):
        """Consume an owned temporary file into an absent immutable target."""
        source, destination = Path(source), Path(destination)
        if normalize_locator(source) == normalize_locator(destination):
            raise IntegrityError("Distinct temporary and final locator required")
        if set(expected) not in ({"sha256", "size_bytes"}, {"path", "sha256", "size_bytes"}):
            raise IntegrityError("Expected full SHA/size binding required")
        _shape(dict(path=str(source.absolute()), **{k: expected[k] for k in ("sha256", "size_bytes")}))
        state = {"attempted": False}

        def verify_destination():
            return _match(_hash_file(destination), expected)

        def move():
            if state["attempted"] and not _exists(source):
                return verify_destination()
            if _exists(destination):
                # POSIX file publication uses an atomic hard link. If its own
                # acknowledged/ambiguous link exists, finish only that same inode.
                if state["attempted"] and os.name != "nt" and os.path.samestat(
                    source.stat(), destination.stat()
                ):
                    verify_destination()
                    source.unlink()
                    return verify_destination()
                raise FileExistsError("Immutable destination already exists")
            _match(_hash_file(source), expected)
            state["attempted"] = True
            try:
                _move_no_replace(source, destination)
            except OSError as error:
                if is_transient(error) and not _exists(source):
                    return verify_destination()
                raise
            if _exists(source):
                raise IntegrityError("Publication did not consume its temporary")
            return verify_destination()

        return self.retry.run(
            "rename-file", destination, move, before_attempt=before_attempt
        )

    def _write(self, destination, producer, expected, *, allow_existing=False, before_attempt=None):
        destination = Path(destination)
        temporary = None
        move_attempted = False

        def write():
            nonlocal temporary, move_attempted
            if move_attempted and not _exists(temporary):
                return _match(_hash_file(destination), expected)
            if _exists(destination):
                if allow_existing and temporary is None:
                    return _match(_hash_file(destination), expected)
                raise FileExistsError("Immutable output conflict; preserve both sides")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = _temporary(destination)
            move_attempted = False
            with temporary.open("xb", buffering=0) as stream:
                producer(stream)
                stream.flush()
                os.fsync(stream.fileno())
            _match(_hash_file(temporary), expected)  # SHA after every complete write.
            # The outer transaction owns the retry budget; do not nest another60s.
            move_attempted = True
            try:
                _move_no_replace(temporary, destination)
            except OSError as error:
                if is_transient(error) and not _exists(temporary):
                    return _match(_hash_file(destination), expected)
                raise
            if _exists(temporary):
                raise IntegrityError("Owned output temporary remained after publication")
            return _match(_hash_file(destination), expected)

        return self.retry.run("write", destination, write, before_attempt=before_attempt)

    def write_bytes(self, destination, data, *, allow_existing=False, before_attempt=None):
        if type(data) is not bytes:
            raise TypeError("Immutable bytes required")
        expected = dict(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))

        def produce(stream):
            _write_all(stream, data)

        return self._write(
            destination, produce, expected, allow_existing=allow_existing,
            before_attempt=before_attempt,
        )

    def write_json(self, destination, document, *, allow_existing=False, before_attempt=None):
        return self.write_bytes(
            destination, _json_bytes(document), allow_existing=allow_existing,
            before_attempt=before_attempt,
        )

    def copy_file(self, source_descriptor, destination, *, allow_existing=True, before_attempt=None):
        """Stream an immutable copy; never remove its local source or failed partials."""
        full = self.verified_asset(source_descriptor, before_attempt=before_attempt)
        source = Path(full["path"])
        expected = {key: full[key] for key in ("sha256", "size_bytes")}

        def produce(stream):
            before = source.stat()
            digest, size = hashlib.sha256(), 0
            with source.open("rb") as original:
                for block in iter(lambda: original.read(CHUNK_BYTES), b""):
                    digest.update(block)
                    size += len(block)
                    if size > expected["size_bytes"]:
                        raise IntegrityError("Copy source grew")
                    _write_all(stream, block)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise IntegrityError("Copy source changed")
            _match(dict(sha256=digest.hexdigest(), size_bytes=size), expected)

        return self._write(
            destination, produce, expected, allow_existing=allow_existing,
            before_attempt=before_attempt,
        )

    def verify_directory(self, folder, expected_files, *, before_attempt=None):
        expected = _expected_inventory(expected_files)
        return self.retry.run(
            "verify-directory", folder,
            lambda: _directory_snapshot(folder, expected), before_attempt=before_attempt,
        )

    def publish_directory(self, source, destination, expected_files, *, before_attempt=None):
        """Atomically move an owned directory; exact inventory, never overwrite."""
        source, destination = Path(source), Path(destination)
        if normalize_locator(source) == normalize_locator(destination):
            raise IntegrityError("Distinct temporary and final locator required")
        expected = _expected_inventory(expected_files)
        state = {"attempted": False}

        def move():
            if state["attempted"] and not _exists(source):
                return _directory_snapshot(destination, expected)
            if _exists(destination):
                raise FileExistsError("Immutable directory already exists")
            _directory_snapshot(source, expected)
            state["attempted"] = True
            try:
                _move_no_replace(source, destination)
            except OSError as error:
                if is_transient(error) and not _exists(source):
                    return _directory_snapshot(destination, expected)
                raise
            if _exists(source):
                raise IntegrityError("Directory publication source remained")
            return _directory_snapshot(destination, expected)

        return self.retry.run(
            "rename-directory", destination, move, before_attempt=before_attempt
        )

    def copy_directory(self, source, destination, expected_files, *, allow_existing=True, before_attempt=None):
        """Copy a local finished frame to a NAS-owned temporary then publish it."""
        source, destination = Path(source), Path(destination)
        expected = _expected_inventory(expected_files)
        self.verify_directory(source, expected, before_attempt=before_attempt)
        temporary = None
        move_attempted = False

        def copy():
            nonlocal temporary, move_attempted
            if move_attempted and not _exists(temporary):
                return _directory_snapshot(destination, expected)
            if _exists(destination):
                if allow_existing and temporary is None:
                    return _directory_snapshot(destination, expected)
                raise FileExistsError("Immutable frame conflict; preserve local source")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = _temporary(destination)
            move_attempted = False
            temporary.mkdir()
            for name, bound in expected.items():
                original, target = source / name, temporary / name
                target.parent.mkdir(parents=True, exist_ok=True)
                before = original.stat()
                digest, size = hashlib.sha256(), 0
                with original.open("rb") as reader, target.open("xb") as writer:
                    for block in iter(lambda: reader.read(CHUNK_BYTES), b""):
                        digest.update(block)
                        size += len(block)
                        _write_all(writer, block)
                    writer.flush()
                    os.fsync(writer.fileno())
                after = original.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise IntegrityError("Local immutable frame changed while copying")
                _match(dict(sha256=digest.hexdigest(), size_bytes=size), bound)
                _match(_hash_file(target), bound)
            _directory_snapshot(temporary, expected)
            move_attempted = True
            try:
                _move_no_replace(temporary, destination)
            except OSError as error:
                if is_transient(error) and not _exists(temporary):
                    return _directory_snapshot(destination, expected)
                raise
            if _exists(temporary):
                raise IntegrityError("NAS temporary remained after frame publication")
            return _directory_snapshot(destination, expected)

        return self.retry.run("copy-directory", destination, copy, before_attempt=before_attempt)

    def replace_json(self, destination, document_provider, *, before_attempt=None, max_bytes=JSON_LIMIT_BYTES):
        """The sole overwrite API: fresh Python JSON for an explicitly mutable control.

        ``document_provider`` and ``before_attempt`` run outside the retry catch
        each time. A failed authority/content guard is never an IO retry.
        """
        if not callable(document_provider):
            raise TypeError("A fresh control-document provider is required")
        destination = Path(destination)
        if destination.suffix.lower() != ".json":
            raise IntegrityError("Mutable control JSON destination required")
        state = {}

        def authority():
            if before_attempt is not None:
                before_attempt()
            data = _json_bytes(document_provider())
            if type(max_bytes) is not int or max_bytes < 0 or len(data) > max_bytes:
                raise IntegrityError("Control JSON exceeds its bound")
            state["data"] = data

        def replace():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = _temporary(destination)
            data = state["data"]
            expected = dict(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
            with temporary.open("xb", buffering=0) as stream:
                _write_all(stream, data)
                stream.flush()
                os.fsync(stream.fileno())
            _match(_hash_file(temporary), expected)
            try:
                os.replace(temporary, destination)
            except OSError as error:
                if is_transient(error) and not _exists(temporary):
                    return _match(_hash_file(destination), expected)
                raise
            if _exists(temporary):
                raise IntegrityError("Control replacement temporary remained")
            return _match(_hash_file(destination), expected)

        return self.retry.run("replace-control", destination, replace, before_attempt=authority)


def content_asset(path, *, io: NasIO | None = None):
    return (io if io is not None else NasIO()).asset(path)


def verify_asset(bound, *, io: NasIO | None = None):
    return (io if io is not None else NasIO()).verified_asset(bound)


def content_identity(bound, *, io: NasIO | None = None):
    asset = verify_asset(bound, io=io)
    return asset["sha256"], asset["size_bytes"]


def same_content(left, right, *, io: NasIO | None = None) -> bool:
    engine = io if io is not None else NasIO()
    return content_identity(left, io=engine) == content_identity(right, io=engine)
