"""Bounded retries of file steps, using the queue's shared N111 decision.

These helpers never call a scientific stage. A retry serializes the same
already computed values or reads the same content-bound input. Interrupted
temporary files are retained; an existing peer's file is never replaced.
"""

from functools import wraps
from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
from secrets import token_hex

from .job_queue import rename, retry_transient

_RETRYING = ContextVar("darsia_file_step_retrying", default=False)


def retry_io(function):
    """Retry one read/hash/copy operation, retaining its original checks."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _RETRYING.get():
            return function(*args, **kwargs)
        def action():
            token = _RETRYING.set(True)
            try:
                return function(*args, **kwargs)
            finally:
                _RETRYING.reset(token)
        return retry_transient(action)
    return wrapped


@retry_io
def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def publish_file(path, writer, *, reuse_identical=False, expected_sha256=None):
    """Serialize immutable values to a fresh temporary, then commit once.

    ``writer(stream)`` performs serialization only. Each retry owns another
    temporary; partial files remain available. A completed interrupted commit
    is accepted only after its actual bytes match the newly serialized bytes.
    """
    path = Path(path)
    retry_transient(lambda: path.parent.mkdir(parents=True, exist_ok=True))
    try:
        retry_transient(path.stat)
    except FileNotFoundError as error:
        from .job_queue import transient_os_error
        if transient_os_error(error):
            raise
    else:
        if not reuse_identical:
            raise FileExistsError(path)

    @retry_io
    def serialize():
        temporary = path.with_name(f".{path.name}.{token_hex(16)}.partial")
        with temporary.open("xb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary

    temporary = serialize()
    digest = _digest(temporary)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Copied bytes differ from their frozen source; partial retained")
    try:
        rename(temporary, path)
    except FileExistsError:
        if not reuse_identical:
            raise
        if _digest(path) != digest:
            raise ValueError("Unreceipted output differs from its frozen bytes")
    return digest
