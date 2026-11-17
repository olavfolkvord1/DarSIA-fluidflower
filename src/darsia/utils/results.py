"""Stored results: published whole without replacing, read through a share's interruptions.

A result is one folder at its location: the files its method's own writer writes,
then ``result.json``, its description. The description records the publication's
token, which is the claim token of the task that made it, and what the result was
made from, so that a later request can tell whether it serves (:func:`reusable`).
The folder is written under a partial name beside its location, flushed to disk and
committed in one step that never replaces, so a reader finds a result only when it
is whole, and an interrupted write leaves only a partial, which no reader reads.

Results are often stored on a network share. Its short interruptions are retried
(:func:`retry_transient`), and a file of a complete result that reads short or is
not found is read once more after the share's cache time (:func:`read`). When the
retries run out, the result is :class:`Unavailable`, not known now, which no caller
may take for missing. A commit whose acknowledgement a dropped connection lost is
done when the location records this publication's token; another token there is
refused.

"""

import ctypes
import errno
import json
import logging
import os
import zipfile
from collections.abc import Callable, Mapping
from itertools import chain, repeat
from pathlib import Path
from time import monotonic, sleep
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

BEAT = 60.0
"""Seconds between two renewals of a task's lease."""
STALE = 600.0
"""Seconds a lease may stay unchanged, by the reader's own clock, before its task is
taken back."""
GRACE = 300.0
"""Seconds an operation is retried while the share is out."""
SHARE_CACHE = 10.0
"""Seconds the share's client may keep a file's size, a folder's contents and a
name it did not find (the file information, folder and not-found caches of an SMB
client); a retry wait, which proves nothing about ownership."""
RETRY_DELAYS = (0.5, 2.0, 5.0)
"""Seconds before each retry of a transient error; while the share is out, the last
delay repeats."""
RESULT = "result.json"
"""The file of a result's description, written last."""
REUSED = ("step", "settings", "scope", "environment", "inputs")
"""What a stored result must share with a request to serve it (:func:`reusable`)."""

# The errors of a share that is out: Windows network-path errors, which Python may
# report as a missing file or path, and the network errors of POSIX.
_NETWORK_WINERRORS = frozenset({51, 53, 54, 55, 58, 59, 64, 67, 121, 1231, 1236})
_NETWORK_ERRNOS = frozenset(
    {
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.EHOSTDOWN,
        errno.EHOSTUNREACH,
        errno.ECONNABORTED,
        errno.ECONNRESET,
        errno.ETIMEDOUT,
    }
)


class Unavailable(OSError):
    """No answer about a stored result within the retries: not known now.

    The share stayed out for longer than :data:`GRACE`, or a file of the result read
    short or was not found twice. The result is neither missing nor incomplete, so a
    caller must not compute, publish or requeue in its place; the task fails with
    this error.

    """


def share_out(error: OSError) -> bool:
    """Whether a file error says that the share is out: a network error, also one
    that Python reports as a missing file or path.

    Parameters
    ----------
    error : OSError
        The error.

    Returns
    -------
    bool
        True for a network error.

    """
    return (
        getattr(error, "winerror", None) in _NETWORK_WINERRORS
        or error.errno in _NETWORK_ERRNOS
    )


def _transient(error: OSError) -> bool:
    """Whether a file error is a short interruption, worth waiting for.

    The answers of the file system keep their meaning: an existing name (a lost
    exclusive create or rename), a missing file or folder, and a wrong file type,
    and so does :class:`Unavailable`. A network error is transient even where
    Python reports it as a missing file, and so is every other file error, e.g. a
    sharing violation.

    """
    if share_out(error):
        return True
    return not isinstance(
        error,
        (
            FileExistsError,
            FileNotFoundError,
            NotADirectoryError,
            IsADirectoryError,
            Unavailable,
        ),
    )


def retry_transient(action: Callable[[], T], *, deadline: float | None = None) -> T:
    """Run an action, retrying it through transient file errors.

    A transient error waits for the next of :data:`RETRY_DELAYS` and runs the action
    again. While the share is out (a network error), the last delay repeats as long
    as the wait ends within :data:`GRACE` seconds of the first error. Any other
    transient error, e.g. access denied or a full disk, is retried once per delay,
    so that a lasting one surfaces after about 7.5 s. A file held by another
    process (a sharing or lock violation, or a busy file) waits until the caller's
    deadline, or GRACE without one. No retry starts at or after the deadline;
    other errors are raised at once.

    Parameters
    ----------
    action : Callable[[], T]
        File operation that may run again after an interruption.
    deadline : float, optional
        Latest retry time on the monotonic clock, shared by the operations of
        a caller whose permission expires, e.g. a task's lease. Network errors
        remain bounded by GRACE as well. Without it, a busy file uses GRACE.

    Returns
    -------
    T
        What the action returns.

    """
    first, others, last = None, 0, None
    for delay in chain(RETRY_DELAYS, repeat(RETRY_DELAYS[-1])):
        if deadline is not None and monotonic() >= deadline:
            if last is not None:
                raise last
            raise TimeoutError("The file operation's deadline has expired")
        try:
            return action()
        except OSError as error:
            last = error
            first = monotonic() if first is None else first
            if not _transient(error):
                raise
            busy = getattr(error, "winerror", None) in (32, 33) or error.errno in (
                errno.EBUSY,
                errno.EAGAIN,
            )
            if share_out(error):
                if monotonic() + delay - first > GRACE:
                    raise
                if deadline is not None and monotonic() + delay >= deadline:
                    raise
            elif busy:
                until = first + GRACE if deadline is None else deadline
                if monotonic() + delay >= until:
                    raise
            else:
                others += 1
                if others > len(RETRY_DELAYS):
                    raise
                if deadline is not None and monotonic() + delay >= deadline:
                    raise
            logger.warning("File error, trying again in %.1f s: %s", delay, error)
        sleep(delay)


def _exists(path: Path) -> bool:
    """Whether a path exists; a network error is raised, not taken for absence."""
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError) as error:
        if _transient(error):
            raise
        return False
    return True


def _observe(location: Path, look: Callable[[], T]) -> T:
    """Look at a stored result through the share's interruptions; when they
    outlast :data:`GRACE`, the result is :class:`Unavailable`, never missing."""
    try:
        return retry_transient(look)
    except OSError as error:
        if not _transient(error):
            raise
        raise Unavailable(f"No answer about {location}: {error}") from error


class _RenameInfo(ctypes.Structure):
    """Windows' ``FILE_RENAME_INFO``, with room for the first character of the name."""

    _fields_ = [
        ("ReplaceIfExists", ctypes.c_ubyte),
        ("RootDirectory", ctypes.c_void_p),
        ("FileNameLength", ctypes.c_uint32),
        ("FileName", ctypes.c_wchar * 1),
    ]


def _windows_error(path: Path) -> OSError:
    """The file error of the last failed Windows call, for a path."""
    code = ctypes.get_last_error()
    return OSError(None, ctypes.FormatError(code).strip(), str(path), code)


def _windows_rename(source: Path, destination: Path) -> None:
    """Rename through one exclusive DELETE handle, never replacing the destination.

    The source is opened for DELETE without sharing it for DELETE, so a second mover
    can open it only once the first has closed it, and then finds it gone.

    """
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    wide, pointer, word = ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32
    kernel.CreateFileW.argtypes = [wide, word, word, pointer, word, word, pointer]
    kernel.CreateFileW.restype = pointer
    kernel.SetFileInformationByHandle.argtypes = [pointer, ctypes.c_int, pointer, word]
    kernel.SetFileInformationByHandle.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [pointer]
    kernel.CloseHandle.restype = ctypes.c_int
    # DELETE access, shared for reading and writing only, an existing file or a
    # folder (FILE_FLAG_BACKUP_SEMANTICS).
    handle = kernel.CreateFileW(
        os.path.abspath(source), 0x00010000, 0x00000003, None, 3, 0x02000000, None
    )
    if handle in (None, ctypes.c_void_p(-1).value):
        raise _windows_error(source)
    try:
        name = os.path.abspath(destination).encode("utf-16-le")
        buffer = ctypes.create_string_buffer(ctypes.sizeof(_RenameInfo) + len(name))
        info = _RenameInfo.from_buffer(buffer)
        info.ReplaceIfExists = False
        info.FileNameLength = len(name)
        ctypes.memmove(
            ctypes.addressof(buffer) + _RenameInfo.FileName.offset, name, len(name)
        )
        if not kernel.SetFileInformationByHandle(handle, 3, buffer, len(buffer)):
            raise _windows_error(destination)
    finally:
        kernel.CloseHandle(handle)


def _move(source: Path, destination: Path) -> None:
    """Move once, by the platform's contract: never onto a publication, one mover.

    - Windows, also on an SMB share: the source is opened through one exclusive
      DELETE handle and renamed through it with ``ReplaceIfExists`` false. The file
      system refuses an existing destination, file or folder. Of two moves of one
      source, the second cannot open it while the first holds it (a sharing
      violation, retried) and then does not find it. A plain ``os.rename`` lets
      both report success: the second moves the file on from the first's
      destination.
    - POSIX: ``rename(2)``, for a file and for a folder. Of two renames of one
      source only the first finds it; the second gets ENOENT. ``rename(2)`` refuses
      a folder with content (ENOTEMPTY or EEXIST) but replaces a file or an empty
      folder, so a file is moved only to a name of its own token (a marker), and a
      published folder holds its description. A link and unlink is not used: two
      movers could both link the source before either removes it.

    """
    try:
        if os.name == "nt":
            _windows_rename(source, destination)
        else:
            os.rename(source, destination)
    except OSError as error:
        if error.errno == errno.ENOTEMPTY:
            raise FileExistsError(
                errno.EEXIST, os.strerror(errno.EEXIST), str(destination)
            ) from error
        raise


def write_json(path: Path, value) -> None:
    """Write a stored result's JSON file, indented and with every digit of its
    numbers (Python's ``repr``); NaN is refused."""
    encoder = json.JSONEncoder(indent=1, allow_nan=False)
    # Validate before opening the target, as dumps did: invalid data must not
    # truncate an existing file. Neither pass retains the encoded chunks.
    for _ in encoder.iterencode(value):
        pass
    with Path(path).open("w", encoding="utf-8") as stream:
        for chunk in encoder.iterencode(value):
            stream.write(chunk)
        stream.write("\n")


def rename(
    source: Path,
    destination: Path,
    ours: Callable[[Path], bool] | None = None,
    *,
    deadline: float | None = None,
) -> None:
    """Move a file or folder in one step, never onto a publication, with one mover.

    Windows refuses any existing destination. POSIX refuses a folder with content
    but replaces a file or an empty folder, so a file must move to a name of the
    caller's own token, and a published folder always holds its description. Of
    two moves of one source, only one succeeds (:func:`_move`).

    Transient errors are retried (:func:`retry_transient`). A move that took effect
    before a dropped connection lost its acknowledgement, e.g. one the share's client
    sent again, answers that the source is missing or the destination exists. The
    caller then says first whether the destination is its own, which is done, or
    not, which is refused; only then is the source looked at, which the share's
    client may still show after the move. Nothing is accepted for its existence
    alone, and nothing is refused for a destination the share's client does not
    show yet: the caller looks again, as a read of a committed result does.

    Parameters
    ----------
    source : Path
        File or folder to move.
    destination : Path
        Its new path, in the same file system.
    ours : Callable[[Path], bool], optional
        Whether the destination is the caller's own, e.g. by the token it records,
        once the move answered that the source is gone or the destination exists:
        True if it is, False if it is another's; it raises if it cannot tell.
        Without it, that answer is raised.
    deadline : float, optional
        Latest retry time on the monotonic clock (:func:`retry_transient`).

    Raises
    ------
    FileExistsError
        If the destination exists and is not the caller's own.
    FileNotFoundError
        If the source is gone, e.g. moved by another mover, or the move did not
        happen.

    """
    source, destination = Path(source), Path(destination)

    def move() -> None:
        try:
            _move(source, destination)
        except (FileExistsError, FileNotFoundError) as error:
            if ours is None or _transient(error):
                raise
            if ours(destination):
                return
            if isinstance(error, FileNotFoundError) and _exists(source):
                raise  # Not ours, and the source is there: the move did not happen.
            raise FileExistsError(
                errno.EEXIST, "The destination is another's", str(destination)
            ) from error

    retry_transient(move, deadline=deadline)


def read_description(folder: Path) -> dict:
    """Read a stored result's description from its folder; a reader for :func:`read`.

    The description is one line of JSON and its newline, so a read that ends before
    the newline is short (EOFError).

    Parameters
    ----------
    folder : Path
        Folder of the result.

    Returns
    -------
    dict
        The description, with its publication's token.

    """
    text = (Path(folder) / RESULT).read_text(encoding="utf-8")
    if not text.endswith("\n"):
        raise EOFError(f"The description of {folder} reads short")
    return json.loads(text)


def _flush_folder(folder: Path) -> None:
    """On POSIX, write a folder's entries through to its disk (``fsync`` of the
    folder); Windows offers no flush of a folder."""
    if os.name == "nt":
        return
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _flush(folder: Path) -> None:
    """Write every file in a folder through to its disk, on a share to the server's,
    and on POSIX also the folders' entries."""

    def fail(error: OSError) -> None:
        raise error

    for root, _, names in os.walk(folder, onerror=fail):
        for name in names:
            with open(os.path.join(root, name), "r+b") as stream:
                os.fsync(stream.fileno())
        _flush_folder(Path(root))


def publish(
    location: Path,
    write: Callable[[Path], None],
    description: Mapping,
    token: str,
    *,
    root: Path,
) -> None:
    """Store a result: the method's own files, then its description.

    The method's writer writes into the partial folder ``.<name>.<token>.partial``
    beside the location. The description follows as ``result.json`` and records the
    token. The files are flushed to disk (``fsync``), and the partial is committed to
    the location without replacing (:func:`rename`). On POSIX the partial's folders
    are flushed before the commit, and after it every folder from the location's
    parent up to the results folder, also those an earlier publication made, so
    that a power loss leaves no committed result short or without its name; on
    Windows, also on an SMB share, the flush of the files is what the platform
    offers. A location that already
    holds a result is this publication if it records this token, e.g. after a lost
    acknowledgement, and is refused if it records another. Its description is read
    as a committed result's (:func:`read`), so a location that the share's client
    does not show yet is looked for once more, and is :class:`Unavailable` if it
    stays unreadable.

    The writer runs once. If it fails, also through a share's error, nothing is
    stored and the partial is kept, never read; a new claim publishes into a new
    partial of its own token. A transient error of the share in the create, the
    description, the flushes or the commit is retried. Only the holder of the token
    names this partial, so a create that finds it there is its own, e.g. one the
    share's client sent again.

    Parameters
    ----------
    location : Path
        Folder of the result.
    write : Callable[[Path], None]
        The method's writer of the result's files into the folder it is given.
    description : Mapping
        What the result was made from (:data:`REUSED`) and by whom, e.g. the job and
        the revision; plain JSON values without NaN, and no token.
    token : str
        The claim token of the task that publishes, a name of letters and digits.
    root : Path
        The results folder the location lies in, whose own name is durable (made
        by the Setup).

    Raises
    ------
    FileExistsError
        If the location holds a result with another token.
    Unavailable
        If the commit's outcome cannot be read within the retries.

    """
    # Lexically, without resolving links, so that «..» cannot leave the folder.
    location, root = Path(os.path.normpath(location)), Path(os.path.normpath(root))
    if "token" in description:
        raise ValueError("The token is the publication's, given apart.")
    folders = [folder for folder in location.parents if folder.is_relative_to(root)]
    if not folders:
        raise ValueError(f"{location} is not in the results folder {root}.")
    text = json.dumps(
        {**description, "token": token}, ensure_ascii=False, allow_nan=False
    )
    partial = location.with_name(f".{location.name}.{token}.partial")

    def create() -> None:
        location.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(partial)
        except FileExistsError:
            pass  # Only the token's holder names this partial: it is its own.

    def describe() -> None:
        (partial / RESULT).write_text(text + "\n", encoding="utf-8", newline="\n")
        _flush(partial)

    def ours(found: Path) -> bool:
        return read(found, read_description, committed=True).get("token") == token

    retry_transient(create)
    write(partial)
    retry_transient(describe)
    rename(partial, location, ours)
    for folder in folders:
        retry_transient(lambda folder=folder: _flush_folder(folder))


def is_complete(location: Path) -> bool:
    """Whether a complete result is stored at a location.

    A result is complete when its description is there under its final name, which
    only the commit of a whole partial gives (:func:`publish`). Interruptions of the
    share are retried, never taken for a missing result.

    Parameters
    ----------
    location : Path
        Folder of the result.

    Returns
    -------
    bool
        True if the result is stored whole, False if it is not.

    Raises
    ------
    Unavailable
        If the share gives no answer within :data:`GRACE`.

    """
    return _observe(location, lambda: _exists(Path(location) / RESULT))


def _cached(error: Exception) -> bool:
    """Whether a reader's error on a complete result can be the share's cache.

    A whole file read short: NumPy's «Failed to read all data» and «EOF:», an
    EOFError, or a truncated npz archive. Or a file of the result not found, which
    the share's client may answer for a name it did not find a moment before.

    """
    if isinstance(error, (EOFError, zipfile.BadZipFile, FileNotFoundError)):
        return True
    return isinstance(error, ValueError) and str(error).startswith(
        ("Failed to read all data", "EOF:")
    )


def _read_committed(location: Path, reader: Callable[[Path], T]) -> T:
    """Read a result whose commit is known; its description not found is a
    FileNotFoundError, which the share's client may answer for a moment."""
    if not is_complete(location):
        raise FileNotFoundError(errno.ENOENT, "No description found", str(location))
    return _observe(location, lambda: reader(location))


def read(location: Path, reader: Callable[[Path], T], committed: bool = False) -> T:
    """Read a complete result with its method's own reader.

    Transient errors of the share are retried (:func:`retry_transient`). The share's
    client may for a few seconds give a file of a just published result short, or
    not find it, so a short read or a file not found is read once more after
    :data:`SHARE_CACHE`, again through the share's interruptions. Any other error of
    the reader is raised as it is. The client may also not find the description of
    a result just committed elsewhere: a result known to be committed, e.g. by the
    token its producer's done marker names, is therefore looked for once more too.

    Parameters
    ----------
    location : Path
        Folder of the result.
    reader : Callable[[Path], T]
        The method's reader of the result's files from its folder.
    committed : bool, optional
        Whether the result is known to be committed. Without it, a result without
        its description is not complete.

    Returns
    -------
    T
        What the reader returns.

    Raises
    ------
    FileNotFoundError
        If no complete result is stored at the location, and it is not known to be
        committed.
    Unavailable
        If the share gives no answer within :data:`GRACE`, or the result's
        description or a file of it reads short or is not found twice.

    """
    location = Path(location)
    if not committed and not is_complete(location):
        raise FileNotFoundError(errno.ENOENT, "No complete result", str(location))
    try:
        return _read_committed(location, reader)
    except Exception as error:
        if not _cached(error):
            raise
        logger.warning("%s does not read whole, reading it again: %s", location, error)
    sleep(SHARE_CACHE)
    try:
        return _read_committed(location, reader)
    except Exception as error:
        if _cached(error):
            message = f"{location} does not read whole twice: {error}"
            raise Unavailable(message) from error
        raise


def reusable(description: Mapping, wanted: Mapping) -> str | None:
    """Why a stored result does not serve a request, or None if it does.

    The request holds what a result made now would record (:data:`REUSED`): the
    step, its settings, its scope (the frames or parts of the frozen selection it is
    made from), the computing environment, and the token of each input it reads, by
    the input's location. The stored description must record the same, compared as
    JSON. A result made from other frames or parts, or from another publication of
    an input, e.g. a phase model trained again with unchanged settings, does not
    serve. The revision of the code is recorded, not compared.

    Parameters
    ----------
    description : Mapping
        The stored result's description (:func:`read_description`).
    wanted : Mapping
        The request.

    Returns
    -------
    str or None
        What differs, e.g. ``"Made with other inputs (phase-model)"``, or None.

    Raises
    ------
    ValueError
        If the request lacks one of :data:`REUSED`.

    """
    missing = [key for key in REUSED if key not in wanted]
    if missing:
        raise ValueError(f"A request names its {', '.join(missing)}.")
    wanted = json.loads(json.dumps({key: wanted[key] for key in REUSED}))
    other = []
    for key in REUSED:
        stored, asked = description.get(key), wanted[key]
        if stored == asked:
            continue
        if key == "inputs" and isinstance(stored, dict) and isinstance(asked, dict):
            names = sorted(n for n in stored | asked if stored.get(n) != asked.get(n))
            key = f"inputs ({', '.join(names)})"
        other.append(key)
    return f"Made with other {', '.join(other)}" if other else None
