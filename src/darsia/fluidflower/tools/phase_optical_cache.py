"""Bounded, content-addressed reuse of the two optical probability heads.

Only numerical optical inputs participate in identity: a regional retraining
does not invalidate an unchanged optical computation. Entries are replaceable
acceleration data, never the scientific result or a source of phase labels.
"""
from __future__ import annotations

from darsia.fluidflower.series_paths import local_cache_root

from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
import uuid

import numpy as np

VERSION = "exact-optical-head-cache-v1"
USAGE_INDEX_VERSION = "exact-optical-cache-usage-index-v2"
HEADS = ("co2_presence_probability", "gas_probability_conditional_on_co2")
GIB = 1024 ** 3


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def array_digest(value):
    if value is None:
        return None
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def optical_identity(model, backend):
    """Fail closed for unknown model implementations; no identity by repr/path."""
    from darsia.fluidflower.phase_learning import HierarchicalPhaseModel, SerializedPhaseHead
    if type(model) is not HierarchicalPhaseModel or any(type(head) is not SerializedPhaseHead
            for head in (model.co2_presence, model.gas_within_co2)):
        raise ValueError("Persistent optical reuse requires the known serialized optical heads")
    from darsia.fluidflower import phase_features, phase_learning
    from darsia.fluidflower.phase_training import radiometry
    from darsia.fluidflower.phase_validation import predict_image_tiled
    from darsia.fluidflower.tools import phase_optical_exact_tiles, phase_inference_benchmark_backends
    from darsia.fluidflower.tools import materialize_phase_iteration, phase_studio_execution
    import cv2
    import scipy
    from threadpoolctl import threadpool_info
    modules = (phase_features, phase_learning, radiometry, phase_optical_exact_tiles,
               phase_inference_benchmark_backends)
    implementation = {module.__name__: _file_digest(module.__file__) for module in modules}
    implementation[__name__] = _file_digest(__file__)
    # Do not key unrelated UI/media/training code in the large validation module.
    for item in (predict_image_tiled, materialize_phase_iteration.CheckedCuda,
                 materialize_phase_iteration.cpu_parity_reference, phase_studio_execution.CpuPredictor):
        implementation[item.__qualname__] = hashlib.sha256(
            inspect.getsource(item).replace("\r\n", "\n").encode()).hexdigest()
    # Retain already computed heads across the explicitly audited removal of
    # diagnostic comparisons. Every unknown source revision still invalidates.
    from darsia.fluidflower.tools.phase_optical_cache_compatibility import canonical_implementation, canonical_implementation_name
    implementation = {canonical_implementation_name(name): canonical_implementation(name, digest)
                      for name, digest in implementation.items()}
    runtime = dict(python=platform.python_version(), system=platform.system(),
        machine=platform.machine(), processor=platform.processor(), numpy=np.__version__,
        scipy=scipy.__version__, opencv=cv2.__version__,
        opencv_build_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        libraries=[{key: library.get(key) for key in ("internal_api", "version", "architecture", "threading_layer")}
                   for library in threadpool_info()])
    # SIMD dispatch and BLAS builds matter to the CPU reference, even when the
    # GPU/kernel version is unchanged. NumPy records the actual enabled flags.
    runtime["cpu_features"] = dict(np._core._multiarray_umath.__cpu_features__)
    if any(head.radiometry_profile is not None for head in (model.co2_presence, model.gas_within_co2)):
        implementation["radiometry_core"] = _file_digest(radiometry.core().__file__)
    if backend == "cuda":
        import cupy as cp
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name = props.get("name", "")
        runtime["cuda"] = dict(cupy=cp.__version__, runtime=cp.cuda.runtime.runtimeGetVersion(),
            driver=cp.cuda.runtime.driverGetVersion(), compute_capability=device.compute_capability,
            name=name.decode() if isinstance(name, bytes) else str(name),
            pci_bus=str(device.pci_bus_id))
    heads = []
    for head in (model.co2_presence, model.gas_within_co2):
        heads.append(dict(contract=head.feature_contract.to_dict(),
            mean=array_digest(head.mean), scale=array_digest(head.scale),
            coefficients=array_digest(head.coefficients), intercept=float(head.intercept).hex(),
            radiometry_profile=head.radiometry_profile))
    return dict(format=VERSION, backend=backend, heads=heads, implementation=implementation,
                runtime=runtime, output_dtype="float32")


def default_root():
    explicit = os.environ.get("PHASE_OPTICAL_CACHE_ROOT")
    if explicit:
        return Path(explicit)
    if os.name == "nt":
        return local_cache_root() / 'optical_predictions'
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "darsia/optical_predictions"


class OpticalCache:
    def __init__(self, identity, baseline, *, root=None, max_bytes=None):
        self.root = Path(root or default_root()).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.identity_sha256 = json_digest(identity)
        self.baseline = array_digest(baseline)
        self.shape = tuple(baseline.shape[:2])
        if max_bytes is None:
            configured = os.environ.get("PHASE_OPTICAL_CACHE_MAX_GIB")
            max_bytes = float(configured) * GIB if configured else min(128 * GIB,
                max(0, shutil.disk_usage(self.root).free // 4))
        self.max_bytes = max(0, int(max_bytes))
        self.stats = dict(format=VERSION, root=str(self.root), max_bytes=self.max_bytes,
                          identity_sha256=self.identity_sha256,
                          hits=0, misses=0, corrupt=0, writes=0, write_skips=0,
                          usage_index_hits=0, usage_scans=0, usage_inventory_checks=0)
        self._usage = None
        self._entries = None

    def key(self, current, previous):
        return json_digest(dict(identity=self.identity_sha256, baseline=self.baseline,
            current=array_digest(current), previous=array_digest(previous)))

    def _entry(self, key):
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("Invalid optical cache key")
        return self.root / key

    @contextmanager
    def _lock(self):
        """OS-owned advisory lock: process death cannot leave a stale writer."""
        with (self.root / ".writer.lock").open("a+b") as stream:
            stream.seek(0)
            if stream.read(1) == b"":
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def load(self, key):
        """Load only complete, byte-verified float32 heads; corruption is a miss."""
        entry = self._entry(key)
        try:
            receipt = json.loads((entry / "receipt.json").read_text(encoding="utf-8"))
            content = {name: value for name, value in receipt.items() if name != "receipt_sha256"}
            if (receipt["receipt_sha256"] != json_digest(content) or receipt["key"] != key
                    or receipt["identity_sha256"] != self.identity_sha256
                    or json_digest(receipt["identity"]) != self.identity_sha256):
                raise ValueError("Optical cache receipt identity differs")
            payload = entry / "heads.npy"
            if _file_digest(payload) != receipt["payload_sha256"]:
                raise ValueError("Optical cache payload digest differs")
            values = np.load(payload, allow_pickle=False)
            if (values.shape != (*self.shape, 2) or values.dtype != np.float32
                    or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
                raise ValueError("Optical cache payload is not two float32 probability heads")
            # Best-effort LRU touch; a concurrent eviction never affects loaded arrays.
            try:
                os.utime(entry / "receipt.json", None)
            except OSError:
                pass
            self.stats["hits"] += 1
            return {name: values[..., index] for index, name in enumerate(HEADS)}, receipt
        except (OSError, ValueError, KeyError, TypeError):
            self.stats["misses"] += 1
            if entry.exists():
                self.stats["corrupt"] += 1
            return None

    def _scan_entries_locked(self):
        """Reconcile usage and LRU from actual entries when the index is stale.

        The parent directory's mtime changes whenever a complete entry is
        installed/removed or a pending directory appears. Receipt touches do
        not change it; those are read afresh if eviction becomes necessary.
        """
        entries = []
        for item in self.root.iterdir():
            if item.is_dir() and item.name.startswith(".pending-"):
                if item.resolve().parent != self.root:
                    raise RuntimeError("Refusing optical cache cleanup outside cache root")
                shutil.rmtree(item)
            elif (item.is_dir() and len(item.name) == 64
                  and all(c in "0123456789abcdef" for c in item.name)):
                files = list(item.iterdir())
                size = sum(path.stat().st_size for path in files if path.is_file())
                receipt = item / "receipt.json"
                entries.append((receipt.stat().st_mtime_ns if receipt.exists() else item.stat().st_mtime_ns,
                                size, item))
        self.stats["usage_scans"] += 1
        return entries

    def _rescan_usage_locked(self):
        entries = self._scan_entries_locked()
        self._entries = {item.name: (mtime, size) for mtime, size, item in entries}
        return sum(size for _, size, _ in entries)

    def _root_inventory_locked(self):
        """Check entry names and interrupted writes without stat-ing payloads.

        Windows can report the old parent mtime immediately after a directory
        is created. A name fingerprint makes warm-index reuse safe even in that
        window, while a pending directory always triggers cleanup.
        """
        names = []
        pending = False
        with os.scandir(self.root) as items:
            for item in items:
                name = item.name
                if name.startswith(".pending-") and item.is_dir(follow_symlinks=False):
                    pending = True
                elif (len(name) == 64 and all(c in "0123456789abcdef" for c in name)
                      and item.is_dir(follow_symlinks=False)):
                    names.append(name)
        self.stats["usage_inventory_checks"] += 1
        return dict(entry_count=len(names), entry_names_sha256=json_digest(sorted(names)),
                    has_pending=pending)

    def _used_bytes_locked(self):
        """Use a lock-protected, crash-detectable byte ledger on warm writes.

        The index lives in a subdirectory so replacing it cannot change the
        cache root's mtime. A crash between entry commit and index commit, or
        another writer that does not use this index, forces a real rescan.
        """
        root_mtime = self.root.stat().st_mtime_ns
        inventory = self._root_inventory_locked()
        if (self._usage is not None and self._entries is not None
                and not inventory["has_pending"]
                and self._usage["root_mtime_ns"] == root_mtime
                and self._usage["entry_count"] == inventory["entry_count"]
                and self._usage["entry_names_sha256"] == inventory["entry_names_sha256"]):
            self.stats["usage_index_hits"] += 1
            return self._usage["used_bytes"]
        index = self.root / ".usage-index" / "state.json"
        try:
            state = json.loads(index.read_text(encoding="utf-8"))
            content = {name: value for name, value in state.items() if name != "sha256"}
            if (state["format"] == USAGE_INDEX_VERSION and state["root_mtime_ns"] == root_mtime
                    and not inventory["has_pending"]
                    and state["entry_count"] == inventory["entry_count"]
                    and state["entry_names_sha256"] == inventory["entry_names_sha256"]
                    and type(state["used_bytes"]) is int and state["used_bytes"] >= 0
                    and state["sha256"] == json_digest(content)
                    and isinstance(state["entries"], dict)
                    and all(len(key) == 64 and all(c in "0123456789abcdef" for c in key)
                            and isinstance(value, list) and len(value) == 2
                            and all(type(number) is int and number >= 0 for number in value)
                            for key, value in state["entries"].items())
                    and sum(value[1] for value in state["entries"].values()) == state["used_bytes"]):
                self._usage = content
                self._entries = {key: tuple(value) for key, value in state["entries"].items()}
                self.stats["usage_index_hits"] += 1
                return state["used_bytes"]
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return self._rescan_usage_locked()

    def _evict_oldest_locked(self):
        """Honor receipt touches without stat-ing every entry per write.

        A cache hit may advance a receipt mtime without changing the parent
        directory. We refresh the oldest candidate until its on-disk time
        matches the ledger; only then is it truly the oldest remaining entry.
        """
        while self._entries:
            key = min(self._entries, key=lambda name: (*self._entries[name], name))
            cached_mtime, cached_size = self._entries[key]
            item = self._entry(key)
            receipt = item / "receipt.json"
            try:
                actual_mtime = receipt.stat().st_mtime_ns if receipt.exists() else item.stat().st_mtime_ns
            except OSError:
                # An external writer changed the root during this lock.
                # Skip this replaceable cache write; next writer rescans.
                self._usage = self._entries = None
                raise
            if actual_mtime != cached_mtime:
                self._entries[key] = (actual_mtime, cached_size)
                continue
            if item.resolve().parent != self.root:
                raise RuntimeError("Refusing optical cache eviction outside cache root")
            shutil.rmtree(item)
            del self._entries[key]
            return cached_size
        return 0

    def _record_usage_locked(self, used_bytes):
        directory = self.root / ".usage-index"
        directory.mkdir(exist_ok=True)
        state = dict(format=USAGE_INDEX_VERSION,
                     root_mtime_ns=self.root.stat().st_mtime_ns,
                     used_bytes=int(used_bytes),
                     entries={key: list(value) for key, value in self._entries.items()})
        if sum(value[1] for value in self._entries.values()) != used_bytes:
            raise ValueError("Optical cache usage ledger is inconsistent")
        inventory = self._root_inventory_locked()
        if inventory["has_pending"] or inventory["entry_count"] != len(self._entries):
            raise ValueError("Optical cache inventory changed during the write")
        state.update(entry_count=inventory["entry_count"],
                     entry_names_sha256=inventory["entry_names_sha256"])
        state["sha256"] = json_digest(state)
        temporary = directory / f".{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
            os.replace(temporary, directory / "state.json")
            self._usage = {name: value for name, value in state.items() if name != "sha256"}
        finally:
            temporary.unlink(missing_ok=True)

    def store(self, key, heads, parity=None):
        values = np.stack([heads[name] for name in HEADS], axis=-1)
        if values.dtype != np.float32 or values.shape != (*self.shape, 2):
            raise ValueError("Persistent optical cache requires unchanged float32 heads")
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError("Cannot cache invalid optical probabilities")
        # Never consume the last disk space or exceed the configured payload budget.
        required = values.nbytes + 32768
        if required > self.max_bytes or required > shutil.disk_usage(self.root).free // 2:
            self.stats["write_skips"] += 1
            return
        temporary = None
        try:
            with self._lock():
                used = self._used_bytes_locked()
                target = self._entry(key)
                if target.exists():
                    # Replacement also repairs corrupt entries whose files
                    # may have changed in place without a parent mtime change.
                    used = self._rescan_usage_locked()
                if used + required > self.max_bytes:
                    while used + required > self.max_bytes:
                        removed = self._evict_oldest_locked()
                        if not removed:
                            break
                        used -= removed
                temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
                np.save(temporary / "heads.npy", values, allow_pickle=False)
                receipt = dict(format=VERSION, key=key, identity_sha256=self.identity_sha256,
                    identity=self.identity,
                    payload_sha256=_file_digest(temporary / "heads.npy"), shape=list(values.shape),
                    dtype="float32", parity=parity, created_unix=time.time())
                receipt["receipt_sha256"] = json_digest(receipt)
                (temporary / "receipt.json").write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
                if target.exists():
                    # This is only a cache entry under our checked content-key root.
                    old_size = sum(path.stat().st_size for path in target.iterdir() if path.is_file())
                    shutil.rmtree(target)
                    used -= old_size
                    self._entries.pop(key, None)
                os.replace(temporary, target)
                temporary = None
                entry_size = sum(path.stat().st_size for path in target.iterdir() if path.is_file())
                used += entry_size
                self._entries[key] = ((target / "receipt.json").stat().st_mtime_ns, entry_size)
                try:
                    self._record_usage_locked(used)
                except OSError:
                    # The complete entry is already committed. A missing
                    # index simply forces a scan on the next writer.
                    self._usage = None
                    self._entries = None
                self.stats["writes"] += 1
        except OSError:
            # An unwritable/full cache never blocks a valid scientific result.
            self.stats["write_skips"] += 1
        finally:
            if temporary is not None and temporary.exists() and temporary.resolve().parent == self.root:
                shutil.rmtree(temporary)

    def invalidate(self, key):
        """Withdraw previously unchecked heads if a later reference check fails."""
        with self._lock():
            entry = self._entry(key)
            if entry.exists():
                if entry.resolve().parent != self.root:
                    raise RuntimeError("Refusing optical cache deletion outside cache root")
                shutil.rmtree(entry)


def create_cache(model, baseline, backend, *, root=None, max_bytes=None):
    try:
        return OpticalCache(optical_identity(model, backend), baseline, root=root, max_bytes=max_bytes), None
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return None, f"{type(error).__name__}: {error}"


def verified_parity(receipt):
    """Old, partial or failed receipts can never suppress a new CPU check."""
    parity = receipt.get("parity") if receipt else None
    if not isinstance(parity, dict) or set(parity.get("heads", {})) != set(HEADS):
        return None
    for check in parity["heads"].values():
        if not (0 <= check.get("q99_abs_error", float("inf")) <= 1e-3
                and 0 <= check.get("disagreement", float("inf")) <= 1e-4
                and 0 <= check.get("max_abs_error", float("inf")) < float("inf")):
            return None
    reference = parity.get("cpu_reference")
    if not isinstance(reference, dict) or not isinstance(reference.get("seconds"), (float, int)):
        return None
    return parity if 0 <= reference["seconds"] < float("inf") and reference.get("method") else None
