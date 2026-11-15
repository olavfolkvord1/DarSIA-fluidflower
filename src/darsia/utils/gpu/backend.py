"""Compute backends: the few operations of the methods that a device can take.

A method computes its heaviest operations on the process's backend
(:func:`compute_backend`), never on a backend it names itself; everything else stays
on the CPU. The CPU backend (:class:`darsia.CpuBackend`) is the reference and the
default. The backend is set once per process (:func:`use_compute_backend`), before
the work starts. A backend's label names its device class, which a job records and
locks; its device names the device and its driver, which each result records.

"""

from __future__ import annotations

from typing import Protocol

import numpy as np

import darsia
from darsia.utils.gpu.cpu import CpuBackend


class ComputeBackend(Protocol):
    """The operations a backend gives, with the numbers of the CPU backend.

    A new backend is one class with these members.

    """

    name: str
    """Name of the backend: ``"cpu"`` or ``"opencl"``."""
    label: str
    """The backend and its device class, which a job records and locks: ``"cpu"``,
    or ``"opencl:"`` and the device's name."""
    device: str
    """The device the operations run on, with its driver; recorded with each
    result."""

    def remap(
        self, planes: np.ndarray, map_x: np.ndarray, map_y: np.ndarray
    ) -> np.ndarray:
        """Bilinear samples of image planes at source coordinates
        (:meth:`darsia.CpuBackend.remap`)."""
        ...

    def sand_means(
        self, values: np.ndarray, masks: list[np.ndarray], rev: darsia.REV
    ) -> np.ndarray:
        """Each mask's mean of the values over a square, on the mask's pixels
        (:meth:`darsia.CpuBackend.sand_means`)."""
        ...


_current: list[ComputeBackend] = [CpuBackend()]
"""The process's backend."""


def compute_backend() -> ComputeBackend:
    """The backend the methods of this process compute on.

    Returns
    -------
    ComputeBackend
        The backend set by :func:`use_compute_backend`, the CPU without one.

    """
    return _current[0]


def use_compute_backend(backend: ComputeBackend) -> None:
    """Compute the methods of this process on a backend from now on.

    Parameters
    ----------
    backend : ComputeBackend
        The backend, e.g. :class:`darsia.CpuBackend`.

    """
    _current[0] = backend
