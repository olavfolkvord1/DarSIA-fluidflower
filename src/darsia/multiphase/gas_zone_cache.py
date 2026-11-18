"""Private storage of per-frame gas zones, independent of their geometry."""

import json
import operator

import numpy as np

from darsia.multiphase.gas_model_cache import GasModelCache, GasModelReader


def _encode(value):
    """Preserve numeric diagnostics and tuple keys without object arrays/pickle."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return ["value", value]
    if isinstance(value, dict):
        return ["dict", [[_encode(key), _encode(item)] for key, item in value.items()]]
    if isinstance(value, (list, tuple)):
        return [
            "tuple" if isinstance(value, tuple) else "list",
            [_encode(item) for item in value],
        ]
    raise TypeError(
        "Gas-zone diagnostics must contain numbers and JSON-like containers."
    )


def _decode(node):
    kind, value = node
    if kind == "value":
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise ValueError("Invalid gas-zone diagnostic scalar.")
        return value
    if kind == "dict":
        result = {_decode(key): _decode(item) for key, item in value}
        if len(result) != len(value):
            raise ValueError("Duplicate gas-zone diagnostic keys.")
        return result
    if kind in {"list", "tuple"}:
        result = [_decode(item) for item in value]
        return tuple(result) if kind == "tuple" else result
    raise ValueError("Invalid gas-zone diagnostic type.")


class GasZoneReader(GasModelReader):
    """A read-only, picklable view of a completed private gas-zone sequence.

    Obtain it from :meth:`GasZoneCache.reader`. The owning context must outlive
    every reader and spawned worker. The view never writes or removes files.

    """

    def __init__(self, directory, count, shape, dtype, kept, available=None):
        self._available = count if available is None else available
        super().__init__(directory, count, shape, dtype, kept)

    def _check_inventory(self):
        if self._available == self.count:
            return super()._check_inventory()
        if not self._directory.is_dir():
            raise RuntimeError("The owning gas-zone cache is no longer open.")
        if any(
            not (self._directory / f"{index:08d}.npz").is_file()
            for index in range(self._available)
        ):
            raise ValueError("The published gas-zone prefix is incomplete.")

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += self.count
        if 0 <= index < self.count and index >= self._available:
            raise RuntimeError(
                "This gas-zone index is outside the published reader prefix."
            )
        return super().__getitem__(index)

    def _validate(self, zone):
        from darsia.multiphase.gas_zone import GasLink, GasZone

        if not isinstance(zone, GasZone):
            raise TypeError("A gas-zone cache stores GasZone values.")
        if (
            not isinstance(zone.contour, np.ndarray)
            or zone.contour.ndim != 2
            or zone.contour.dtype != bool
        ):
            raise ValueError("A gas-zone contour must be a two-dimensional bool array.")
        if self._shape is None:
            self._shape, self._dtype = zone.contour.shape, zone.contour.dtype
        elif zone.contour.shape != self._shape:
            raise ValueError("All gas zones must share their grid.")
        if not isinstance(zone.links, dict):
            raise TypeError("Gas-zone links must be a port mapping.")
        for port, link in zone.links.items():
            if (
                not isinstance(port, tuple)
                or len(port) != 2
                or any(not np.isfinite(value) for value in port)
            ):
                raise ValueError("Gas-zone ports must be finite coordinate pairs.")
            if not isinstance(link, GasLink):
                raise TypeError("Gas-zone links must be GasLink values.")
            for name, dtype in (
                ("path", np.dtype(bool)),
                ("blend", np.dtype(np.float32)),
            ):
                value = getattr(link, name)
                if (
                    not isinstance(value, np.ndarray)
                    or value.shape != self._shape
                    or value.dtype != dtype
                ):
                    raise ValueError(
                        f"A gas-zone {name} must be {dtype} on the contour grid."
                    )
            coordinates = link.path_coordinates
            if (
                not isinstance(coordinates, np.ndarray)
                or coordinates.dtype != np.int32
                or coordinates.ndim != 2
                or coordinates.shape[1] != 2
                or len(coordinates) == 0
                or np.any(coordinates < 0)
                or np.any(coordinates >= self._shape)
            ):
                raise ValueError(
                    "Gas-zone path coordinates must be nonempty int32 grid pairs."
                )
            steps = np.abs(np.diff(coordinates, axis=0))
            if (
                np.any(np.max(steps, axis=1) != 1)
                or not link.path[tuple(coordinates.T)].all()
            ):
                raise ValueError(
                    "Gas-zone path coordinates must follow the path in pixel order."
                )

    def _load(self, index):
        from darsia.multiphase.gas_zone import GasLink, GasZone

        with np.load(self._directory / f"{index:08d}.npz", allow_pickle=False) as saved:
            metadata = saved["metadata"]
            if metadata.ndim != 1 or metadata.dtype != np.uint8:
                raise ValueError("Invalid gas-zone metadata array.")
            value = _decode(json.loads(metadata.tobytes().decode("utf-8")))
            if (
                set(value) != {"schema", "ports", "link_diagnostics", "diagnostics"}
                or value["schema"] != 2
            ):
                raise ValueError("Invalid gas-zone metadata contract.")
            ports = value["ports"]
            expected = {"metadata", "contour"} | {
                f"{kind}_{index:08d}"
                for index in range(len(ports))
                for kind in ("path", "blend", "path_coordinates")
            }
            if (
                set(saved.files) != expected
                or len(ports) != len(value["link_diagnostics"])
                or len(set(ports)) != len(ports)
            ):
                raise ValueError("Invalid gas-zone link inventory.")
            links = {
                port: GasLink(
                    saved[f"path_{index:08d}"],
                    saved[f"blend_{index:08d}"],
                    diagnostic,
                    saved[f"path_coordinates_{index:08d}"],
                )
                for index, (port, diagnostic) in enumerate(
                    zip(ports, value["link_diagnostics"], strict=True)
                )
            }
            zone = GasZone(saved["contour"], links, value["diagnostics"])
        self._validate(zone)
        zone.contour.setflags(write=False)
        for link in zone.links.values():
            link.path.setflags(write=False)
            link.blend.setflags(write=False)
            link.path_coordinates.setflags(write=False)
        return zone


class GasZoneCache(GasModelCache):
    """Store one private, uncompressed gas zone for every selected frame.

    Storage, exact-count completion, atomic publication, cleanup and bounded
    reads follow :class:`GasModelCache`. One zone is retained by default;
    primary and held-out allocations share this sequence. It knows no physics.

    """

    DEFAULT_KEPT = 1
    _reader_type = GasZoneReader
    _validate = GasZoneReader._validate
    _load = GasZoneReader._load

    def __init__(self, count: int, kept: int = DEFAULT_KEPT):
        super().__init__(count, kept)
        self._written = 0

    def store(self, zones):
        """Store zones once and yield originals after atomic file publication.

        Parameters
        ----------
        zones : iterable of GasZone
            Prefix zones in the complete selected run's order.

        Yields
        ------
        GasZone
            The original zone, already available to an explicit prefix reader.

        """
        for index, zone in enumerate(super().store(zones)):
            self._written = index + 1
            yield zone

    def reader(self, through=None):
        """Read the complete cache, or an explicit already-published prefix.

        Parameters
        ----------
        through : int, optional
            Last readable frame index. Omit to require complete storage. A
            prefix reader keeps the full sequence length but refuses all future
            indices, even after the parent publishes more zones.

        Returns
        -------
        GasZoneReader
            A read-only view whose owner must outlive every spawned worker.

        Raises
        ------
        RuntimeError
            If the context is closed or the requested prefix is not published.

        """
        if through is None:
            return super().reader()
        through = operator.index(through)
        if self._directory is None or not 0 <= through < self._written:
            raise RuntimeError("The requested gas-zone prefix has not been published.")
        return GasZoneReader(
            self._directory,
            self.count,
            self._shape,
            self._dtype,
            self.kept,
            available=through + 1,
        )

    def _write(self, stream, zone):
        metadata = {
            "schema": 2,
            "ports": list(zone.links),
            "link_diagnostics": [link.diagnostics for link in zone.links.values()],
            "diagnostics": zone.diagnostics,
        }
        encoded = json.dumps(
            _encode(metadata),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        arrays = {
            "contour": zone.contour,
            "metadata": np.frombuffer(encoded, dtype=np.uint8),
        }
        for index, link in enumerate(zone.links.values()):
            arrays[f"path_{index:08d}"] = link.path
            arrays[f"blend_{index:08d}"] = link.blend
            arrays[f"path_coordinates_{index:08d}"] = link.path_coordinates
        np.savez(stream, **arrays)
