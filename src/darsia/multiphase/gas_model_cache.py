"""Private, bounded storage of a run's model frames before gas allocation.

The cache preserves arrays without compression or conversion. It knows neither
the phase rules nor the gas geometry; those methods may consume each frame while
it is stored, then read it again by index for allocation. A cache belongs to one
invocation and normally removes its files on exit. Explicit retained caches keep
their files and strict provenance in the local application cache, even on error.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import sys
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import numpy as np


def _local_cache() -> Path:
    """The operating system's per-user application cache, never a project path."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    if not base.is_absolute():
        raise ValueError("The local application cache must be an absolute path.")
    if os.name == "nt":
        import ctypes

        if (
            str(base).startswith("\\\\")
            or ctypes.windll.kernel32.GetDriveTypeW(str(base.anchor)) == 4
        ):
            raise ValueError("Retained model caches require a local drive.")
    return base / "DarSIA" / "gas-model"


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _identity(value, count):
    required = {
        "frame_keys",
        "dates",
        "input_tokens",
        "model",
        "settings",
        "backend",
        "source",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("A retained model cache needs exact model/input provenance.")
    value = _json(dict(value))
    for key in ("frame_keys", "dates"):
        items = value[key]
        if (
            not isinstance(items, list)
            or len(items) != count
            or any(not isinstance(v, str) or not v for v in items)
        ):
            raise ValueError(f"Model cache {key} must identify every selected frame.")
        if key == "frame_keys" and len(set(items)) != count:
            raise ValueError(f"Model cache {key} must be unique and ordered.")
    for key in required - {"frame_keys", "dates"}:
        if not isinstance(value[key], dict) or not value[key]:
            raise ValueError(f"Model cache {key} identity must be a nonempty mapping.")
    return value


def _file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path, value):
    partial = path.with_suffix(".partial")
    with partial.open("w", encoding="utf8") as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


@dataclass(frozen=True)
class GasModelFrame:
    """The model state of one frame before volume allocation.

    All arrays share a two-dimensional grid. The probability keeps its working
    floating dtype; the three label maps are uint8. The cache returns read-only
    arrays, so an allocation copies labels and rules before changing them.
    """

    gas_probability: np.ndarray
    """The frame's own gas probability, before time averaging or allocation."""
    before: np.ndarray
    """Labels before phase rules, retained for the optical reading."""
    labels: np.ndarray
    """Labels after the unchanged phase rules, before allocation."""
    rules: np.ndarray
    """The phase rules that produced the labels, before allocation."""


class GasModelReader(Sequence[GasModelFrame]):
    """A read-only view of one completed private model cache.

    Obtain this view with :meth:`GasModelCache.reader`; do not construct one
    from a result directory. Its owning cache context must remain open until
    all readers, including spawned workers, finish. Each process has its own
    bounded in-memory cache. Pickling carries only the private path and array
    contract, never cached arrays or ownership of the temporary directory.

    """

    def __init__(self, directory, count, shape, dtype, kept, files=None):
        self._directory = Path(directory)
        self.count, self.kept = count, kept
        self._shape, self._dtype = shape, dtype
        self._ready = True
        self._files = files
        self._cached: OrderedDict[int, GasModelFrame] = OrderedDict()
        self._check_inventory()

    def _check_inventory(self):
        if not self._directory.is_dir():
            raise RuntimeError("The owning model cache is no longer open.")
        expected = {f"{index:08d}.npz" for index in range(self.count)}
        if self._files is not None:
            expected.add("manifest.json")
        found = {path.name for path in self._directory.iterdir()}
        if found != expected or any(
            not path.is_file() for path in self._directory.iterdir()
        ):
            raise ValueError("The private model cache has an invalid frame inventory.")

    @classmethod
    def open(cls, directory, *, identity, kept=3):
        """Verify and reopen an explicitly selected retained cache, read-only.

        The expected identity excludes the producer's whole source revision:
        model/rule methods and their inputs must match, while later allocation
        methods may change. Every file is checked, one frame at a time; neither
        this call nor a reader deletes or modifies the cache.
        """
        directory = Path(directory)
        with (directory / "manifest.json").open(encoding="utf8") as stream:
            manifest = json.load(stream)
        fields = {
            "schema",
            "status",
            "count",
            "stored",
            "shape",
            "dtype",
            "identity",
            "producer_revision",
            "files",
        }
        if (
            set(manifest) != fields
            or manifest["schema"] != 1
            or manifest["status"] != "complete"
        ):
            raise ValueError("The retained model cache is not complete.")
        count = manifest["count"]
        if (
            type(count) is not int
            or count < 0
            or type(manifest["stored"]) is not int
            or manifest["stored"] != count
        ):
            raise ValueError("The retained model cache has an invalid count.")
        if manifest["identity"] != _identity(identity, count):
            raise ValueError("The retained model cache provenance does not match.")
        shape, dtype = manifest["shape"], manifest["dtype"]
        if not count and (shape is not None or dtype is not None):
            raise ValueError("An empty retained cache has no array shape or dtype.")
        if count and (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(v) is not int or v <= 0 for v in shape)
        ):
            raise ValueError("The retained model cache has an invalid shape.")
        if count and (not isinstance(dtype, str) or np.dtype(dtype).kind != "f"):
            raise ValueError("The retained model cache has an invalid dtype.")
        if (
            not isinstance(manifest["producer_revision"], str)
            or not manifest["producer_revision"]
        ):
            raise ValueError("The retained model cache has no producer revision.")
        files = manifest["files"]
        if not isinstance(files, dict) or set(files) != {
            f"{i:08d}.npz" for i in range(count)
        }:
            raise ValueError("The retained model cache has an invalid file manifest.")
        kept = operator.index(kept)
        if kept < 1:
            raise ValueError("A model reader needs kept >= 1.")
        reader = cls(
            directory,
            count,
            None if shape is None else tuple(shape),
            None if dtype is None else np.dtype(dtype),
            kept,
            files,
        )
        for index in range(count):
            reader._load(index)
        return reader

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cached"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cached = OrderedDict()
        self._check_inventory()

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> GasModelFrame:
        if not self._ready or self._directory is None or not self._directory.is_dir():
            raise RuntimeError("The complete model cache is not open for reading.")
        index = operator.index(index)
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        if index in self._cached:
            self._cached.move_to_end(index)
        else:
            # Evict before loading so the cache itself never holds kept+1
            # complete frames during a miss. Callers own any arrays they retain.
            if len(self._cached) >= self.kept:
                self._cached.popitem(last=False)
            self._cached[index] = self._load(index)
        return self._cached[index]

    def _validate(self, frame: GasModelFrame) -> None:
        if not isinstance(frame, GasModelFrame):
            raise TypeError("A model cache stores GasModelFrame values.")
        probability = frame.gas_probability
        if not isinstance(probability, np.ndarray) or probability.ndim != 2:
            raise ValueError("A gas probability must be a two-dimensional array.")
        if probability.dtype.kind != "f":
            raise ValueError("A gas probability must keep a floating dtype.")
        for name in ("before", "labels", "rules"):
            values = getattr(frame, name)
            if (
                not isinstance(values, np.ndarray)
                or values.shape != probability.shape
                or values.dtype != np.uint8
            ):
                raise ValueError(
                    f"The {name} map must be uint8 on the probability grid."
                )
        if self._shape is None:
            self._shape, self._dtype = probability.shape, probability.dtype
        elif probability.shape != self._shape or probability.dtype != self._dtype:
            raise ValueError("All model frames must share their grid and precision.")

    def _load(self, index: int) -> GasModelFrame:
        if self._directory is None:
            raise RuntimeError("The model cache context has closed.")
        path = self._directory / f"{index:08d}.npz"
        if self._files is not None:
            expected = self._files[path.name]
            if (
                not isinstance(expected, dict)
                or set(expected) != {"bytes", "sha256"}
                or type(expected["bytes"]) is not int
                or expected["bytes"] < 0
                or not isinstance(expected["sha256"], str)
                or len(expected["sha256"]) != 64
                or path.stat().st_size != expected["bytes"]
                or _file_digest(path) != expected["sha256"]
            ):
                raise ValueError(
                    f"Retained model cache file integrity differs: {path.name}."
                )
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != {"gas_probability", "before", "labels", "rules"}:
                raise ValueError(
                    "The private model cache has an invalid array inventory."
                )
            frame = GasModelFrame(
                saved["gas_probability"],
                saved["before"],
                saved["labels"],
                saved["rules"],
            )
        self._validate(frame)
        for values in (frame.gas_probability, frame.before, frame.labels, frame.rules):
            values.setflags(write=False)
        return frame


class GasModelCache(GasModelReader):
    """A private temporary sequence, materialized once with bounded memory.

    The operating system's temporary directory is used, so its normal environment
    configuration applies. Nothing is written to the project or a result store.
    Only a successfully exhausted :meth:`store` makes the sequence readable.
    Opting into retention instead uses the local application cache. Reopening
    then requires explicit matching model provenance and full file validation.

    Parameters
    ----------
    count : int
        Number of frames the iterator must produce, including completed frames
        when a later allocation resumes.
    kept : int, default=3
        Maximum number of complete frames cached in memory when reading. Three
        holds a current frame and its nearest earlier and later neighbours.
    retain : bool, default=False
        Keep a unique local directory, including incomplete stores on failure.
    identity : mapping, optional
        Required for retention: ordered frame keys/dates, accepted input tokens,
        model, settings, compute backend and model-source identities.
    revision : str, optional
        Required producer revision for audit when retaining. This whole-source
        revision is not used in place of the model-specific identity.

    Raises
    ------
    ValueError
        If a count is negative or the memory cache holds fewer than one frame.

    """

    _reader_type = GasModelReader

    DEFAULT_KEPT = 3
    """Current and nearest earlier/later model frames retained by default."""

    def __init__(
        self,
        count: int,
        kept: int = DEFAULT_KEPT,
        *,
        retain=False,
        identity=None,
        revision=None,
    ) -> None:
        self.count = operator.index(count)
        self.kept = operator.index(kept)
        if self.count < 0 or self.kept < 1:
            raise ValueError("A model cache needs count >= 0 and kept >= 1.")
        self._temporary: TemporaryDirectory | None = None
        self._directory: Path | None = None
        self._started = False
        self._ready = False
        self._shape: tuple[int, int] | None = None
        self._dtype: np.dtype | None = None
        self._cached: OrderedDict[int, GasModelFrame] = OrderedDict()
        self._files = {} if retain else None
        self._retain = bool(retain)
        self._identity = _identity(identity, self.count) if retain else None
        if retain and (not isinstance(revision, str) or not revision):
            raise ValueError("A retained model cache needs its producer revision.")
        self._revision = revision
        self._retained_path = None

    def __enter__(self) -> GasModelCache:
        if self._started or self._directory is not None:
            raise RuntimeError("A model cache context can be entered only once.")
        if self._retain:
            root = _local_cache()
            root.mkdir(parents=True, exist_ok=True)
            self._directory = Path(mkdtemp(prefix="model-", dir=root))
            self._retained_path = self._directory
            self._manifest("building")
        else:
            self._temporary = TemporaryDirectory(prefix="darsia-gas-model-")
            self._directory = Path(self._temporary.name)
        return self

    def __exit__(self, *_exc) -> None:
        if self._retain and not self._ready:
            self._manifest("incomplete")
        self._cached.clear()
        self._ready = False
        self._started = True
        self._directory = None
        temporary, self._temporary = self._temporary, None
        if temporary is not None:
            temporary.cleanup()

    @property
    def retained_path(self) -> Path | None:
        """Persistent local directory, also after exit; None for temporary caches."""
        return self._retained_path

    def _manifest(self, status):
        if self._retain:
            _atomic_json(
                self._directory / "manifest.json",
                {
                    "schema": 1,
                    "status": status,
                    "count": self.count,
                    "stored": len(self._files),
                    "shape": self._shape,
                    "dtype": None if self._dtype is None else self._dtype.str,
                    "identity": self._identity,
                    "producer_revision": self._revision,
                    "files": self._files,
                },
            )

    def reader(self) -> GasModelReader:
        """Create a read-only view for this process or spawned workers.

        Returns
        -------
        GasModelReader
            An independently bounded view of the completed private cache. The
            parent must keep this cache context open until all workers finish.

        Raises
        ------
        RuntimeError
            If the store has not completed or this context has closed.

        """
        if not self._ready or self._directory is None:
            raise RuntimeError("The complete model cache is not open for reading.")
        args = (self._directory, self.count, self._shape, self._dtype, self.kept)
        return (
            self._reader_type(*args)
            if self._files is None
            else self._reader_type(*args, files=self._files.copy())
        )

    def __getstate__(self):
        raise TypeError("Pass cache.reader() to workers, not the owning cache.")

    def store(self, frames: Iterable[GasModelFrame]) -> Iterator[GasModelFrame]:
        """Store each frame, then yield the original to a method consuming the run.

        The iterator must be exhausted once, inside the cache context. Too few
        or too many frames, an invalid array, or an interrupted iterator leaves
        the cache unreadable. A retained cache preserves these incomplete files
        and manifest; the default temporary cache removes them on context exit.

        Parameters
        ----------
        frames : iterable of GasModelFrame
            Model states in the complete selected run's order.

        Yields
        ------
        GasModelFrame
            The original frame, after its uncompressed file is closed.

        Raises
        ------
        RuntimeError
            If storing is attempted outside its context or more than once.
        ValueError
            If frames have an invalid grid/dtype or the count differs.
        TypeError
            If an item is not a model frame.

        """
        if self._directory is None or self._started:
            raise RuntimeError("Store model frames once inside their cache context.")
        self._started = True
        found = 0
        for index, frame in enumerate(frames):
            if self._directory is None:
                raise RuntimeError("The model cache context has closed.")
            if index >= self.count:
                raise ValueError(f"Expected {self.count} model frames; found more.")
            self._validate(frame)
            path = self._directory / f"{index:08d}.npz"
            partial = path.with_suffix(".partial")
            with partial.open("xb") as stream:
                self._write(stream, frame)
                stream.flush()
                os.fsync(stream.fileno())
            partial.replace(path)
            if self._files is not None:
                self._files[path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": _file_digest(path),
                }
                self._manifest("building")
            found += 1
            yield frame
        if found != self.count:
            raise ValueError(f"Expected {self.count} model frames; found {found}.")
        self._manifest("complete")
        self._ready = True

    def _write(self, stream, frame):
        np.savez(
            stream,
            gas_probability=frame.gas_probability,
            before=frame.before,
            labels=frame.labels,
            rules=frame.rules,
        )
