"""Module with interface to compute and visualize Wasserstein distances."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import darsia


@lru_cache(maxsize=8)
def _matching_grid(shape, voxel_sizes) -> darsia.Grid:
    """Reuse topology for repeated solves on the same tensor grid."""
    return darsia.Grid(list(shape), list(voxel_sizes))


def wasserstein_distance(
    mass_src: darsia.Image,
    mass_dst: darsia.Image,
    method: Literal["newton", "bregman", "gprox", "cv2.emd"] = "newton",
    weight: Optional[darsia.Image] = None,
    **kwargs,
) -> float | tuple[float, dict]:
    """Unified access to Wasserstein distance computation between images with same mass.

    Parameters
    ----------
    mass_src : darsia.Image
        Source distribution.
    mass_dst : darsia.Image
        Destination distribution.
    method : Literal["newton", "bregman", "gprox", "cv2.emd"]
        Method to use.
    **kwargs
        Additional arguments (only for "newton", "bregman", "gprox").

        - options (dict): options for the method. Set ``profile_memory`` to
          trace Python allocations. ``return_pressure`` controls pressure
          recovery and defaults to the value of ``return_info``.

    Returns
    -------
    float | tuple[float, dict
        Wasserstein distance or (distance, info).
    """
    # Define method for computing 1-Wasserstein distance
    method_name = method.lower()

    if method_name in ["newton", "bregman", "gprox"]:
        # Use Finite Volume Iterative Method (Newton or Bregman)

        options = dict(kwargs.get("options", {}))
        options.setdefault("return_pressure", options.get("return_info", False))
        # Callbacks may inspect and retain the solver's public grid.
        shared_grid = not options.get("callbacks")
        grid = (
            _matching_grid(tuple(mass_dst.num_voxels), tuple(mass_dst.voxel_size))
            if shared_grid
            else darsia.generate_grid(mass_dst)
        )

        # Define method
        match method_name:
            case "newton":
                w1 = darsia.BeckmannNewtonSolver(grid, weight, options)
            case "bregman":
                w1 = darsia.BeckmannBregmanSolver(grid, weight, options)
            case "gprox":
                w1 = darsia.BeckmannGproxPGHDSolver(grid, weight, options)
        result = w1(mass_src, mass_dst)
        if shared_grid and options.get("return_info"):
            result[1]["grid"] = deepcopy(grid)
        return result

    elif method_name == "cv2.emd":
        # Use Earth Mover's Distance from CV2
        assert weight is None, "Weighted EMD not supported by cv2."
        preprocess = kwargs.get("preprocess")
        w1 = darsia.EMD(preprocess)

    else:
        raise NotImplementedError(f"Method {method_name} not implemented.")

    # Compute and return Wasserstein distance
    return w1(mass_src, mass_dst)


def wasserstein_distance_to_vtk(
    path: Path,
    info: dict,
) -> None:
    """Write the output of the Wasserstein distance to a VTK file.

    Parameters
    ----------
    path : Path
        Path to the VTK file.
    info : dict
        Information dictionary output of :func:`wasserstein_distance`.

    Notes
    -----
    Requires ``pyevtk`` to be installed.
    """
    data = [
        (key, info[key], format)
        for key, format in [
            ("src", darsia.Format.SCALAR),
            ("dst", darsia.Format.SCALAR),
            ("mass_diff", darsia.Format.SCALAR),
            ("flux", darsia.Format.VECTOR),
            ("weighted_flux", darsia.Format.VECTOR),
            ("pressure", darsia.Format.SCALAR),
            ("transport_density", darsia.Format.SCALAR),
            ("weight", darsia.Format.TENSOR),
            ("weight_inv", darsia.Format.TENSOR),
        ]
    ]
    darsia.plotting.to_vtk(Path(path), data)
