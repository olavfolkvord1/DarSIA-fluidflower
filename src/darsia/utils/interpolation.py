"""Module containing interpolation functionality for scalar data."""

from __future__ import annotations

from math import isqrt
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares
from scipy.spatial import Delaunay

import darsia
from darsia.utils.linalg import solve


def interpolate_measurements_2d(
    measurements: tuple[np.ndarray, np.ndarray, np.ndarray],
    coordinate_system: darsia.CoordinateSystem,
) -> np.ndarray:
    """Build a voxelised spatial map from scattered measurements by RBF interpolation.

    Parameters
    ----------
    measurements : tuple of numpy.ndarray
        ``(x, y, data)`` arrays providing the interpolation input.
    coordinate_system : darsia.CoordinateSystem
        Coordinate system of the corresponding physical image; sets the output
        shape.

    Returns
    -------
    numpy.ndarray
        The interpolated map.
    """
    assert len(measurements) == 3, "Measurements must be a tuple of (x, y, data)."
    # Create an interpolation object from data.
    interpolator = RBFInterpolator(
        np.transpose(
            np.vstack(
                (
                    measurements[0],
                    measurements[1],
                )
            )
        ),
        measurements[2],
    )

    # Create a mesh of points at which the interpolator shall be evaluated.
    Ny, Nx = coordinate_system.shape
    coords_vector = coordinate_system.coordinates

    # Evaluate interpolation
    interpolated_data_vector = interpolator(coords_vector)
    return interpolated_data_vector.reshape((Ny, Nx), order="F")


def interpolate_linear_in_hull(
    measurements: tuple[np.ndarray, np.ndarray, np.ndarray],
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate scattered measurements linearly inside their convex hull, and
    take the nearest measurement outside it.

    The positions of the measurements are triangulated (Delaunay, in double
    precision). A point inside a triangle, up to 100 double precision machine
    epsilons in its barycentric coordinates, takes the linear interpolation of
    the triangle's three values. Each point is looked up among the triangles whose
    bounding boxes share its cell of a regular grid over the measurements, about
    one cell per triangle, in the order of the triangulation; the first that holds
    the point gives its value, e.g., on an edge of two triangles. A point outside
    every triangle takes the value of the nearest measurement, the first one on
    ties. All arithmetic runs in the floating dtype of the coordinates (at least
    single precision); the measurements are converted to it.

    Parameters
    ----------
    measurements : tuple of numpy.ndarray
        ``(x, y, data)`` of the measurements, at least three not on one line.
    coordinates : numpy.ndarray
        Points (N, 2) of x and y at which to interpolate.

    Returns
    -------
    values : numpy.ndarray
        The interpolated value of each point, in the coordinates' dtype.
    outside : numpy.ndarray
        Boolean flag of the points outside the convex hull, which take the
        nearest measurement's value.

    """
    x, y, data = (np.ravel(measurement) for measurement in measurements)
    triangles = Delaunay(np.column_stack((x, y)).astype(np.float64)).simplices
    points = np.asarray(coordinates)
    dtype = np.result_type(points, np.float32)
    points = points.astype(dtype, copy=False)
    xy = np.column_stack((x, y)).astype(dtype)
    data = data.astype(dtype)
    vertices = xy[triangles]

    # The triangles by the cells of a regular grid over the measurements.
    origin = xy.min(axis=0)
    span = xy.max(axis=0) - origin
    count = max(1, isqrt(len(triangles)))

    def cell(position: np.ndarray) -> np.ndarray:
        """Column and row of the grid cell of each position (N, 2)."""
        return np.clip((position - origin) / span * count, 0, count - 1).astype(np.intp)

    cells = [[] for _ in range(count**2)]
    for index, ((x0, y0), (x1, y1)) in enumerate(
        zip(cell(vertices.min(axis=1)), cell(vertices.max(axis=1)))
    ):
        for row in range(y0, y1 + 1):
            for col in range(x0, x1 + 1):
                cells[row * count + col].append(index)

    # Barycentric coordinates relative to the third vertex of each triangle.
    edges = np.transpose(vertices[:, :2] - vertices[:, 2:3], (0, 2, 1))
    inverses = np.stack([solve(edge, np.eye(2, dtype=dtype)) for edge in edges])
    grid = cell(points)
    ids = grid[:, 1] * count + grid[:, 0]
    order = np.argsort(ids, kind="stable")
    ordered = ids[order]
    stops = np.r_[np.flatnonzero(np.diff(ordered)) + 1, len(ordered)]
    values = np.empty(len(points), dtype)
    outside = np.ones(len(points), dtype=bool)
    eps = dtype.type(100 * np.finfo(np.float64).eps)
    start = 0
    for stop in stops:
        group = order[start:stop]
        for triangle in cells[ordered[start]]:
            todo = group[outside[group]]
            if not len(todo):
                break
            dx, dy = (points[todo] - vertices[triangle, 2]).T
            inverse = inverses[triangle]
            u = inverse[0, 0] * dx + inverse[0, 1] * dy
            v = inverse[1, 0] * dx + inverse[1, 1] * dy
            w = 1 - u - v
            inside = (u >= -eps) & (v >= -eps) & (w >= -eps)
            corner = data[triangles[triangle]]
            values[todo[inside]] = (
                u[inside] * corner[0] + v[inside] * corner[1] + w[inside] * corner[2]
            )
            outside[todo[inside]] = False
        start = stop

    # Outside the hull, the nearest measurement; the first one on ties.
    far = points[outside]
    distance = np.full(len(far), np.inf, dtype)
    nearest = np.empty(len(far), dtype)
    for (x0, y0), value in zip(xy, data):
        squared = (far[:, 0] - x0) ** 2 + (far[:, 1] - y0) ** 2
        closer = squared < distance
        distance[closer] = squared[closer]
        nearest[closer] = value
    values[outside] = nearest
    return values, outside


def polynomial_interpolation(
    measurements: tuple[np.ndarray, ...],
    coordinate_system: darsia.CoordinateSystem,
    degree: int = 2,
) -> np.ndarray:
    """Determine a voxeled spatial map from measurements through polynomial interpolation.

    Parameters
    ----------
    measurements : tuple[np.ndarray, ...]
        Tuple of x, y and data measurements providing the interpolation input.
    coordinate_system : darsia.CoordinateSystem
        Coordinate system of the corresponding physical image; sets the output
        shape.
    degree : int
        Degree of the polynomial interpolation.

    Returns
    -------
    np.ndarray
        The interpolated map.
    """
    # Deterimine dimension of space of polynomial coefficients
    dimension = sum(
        [1 for i in range(degree + 1) for j in range(degree + 1) if i + j <= degree]
    )

    k = -np.ones((degree + 1, degree + 1), dtype=int)
    counter = 0
    for i in range(degree + 1):
        for j in range(degree + 1):
            if i + j <= degree:
                k[i, j] = counter
                counter += 1

    def polynomial_interpolator(
        coords: np.ndarray,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """Polynomial interpolation function."""
        output = np.zeros_like(coords[:, 0])
        for i in range(degree + 1):
            for j in range(degree + 1):
                if i + j <= degree:
                    output += coefficients[k[i, j]] * np.multiply(
                        coords[:, 0] ** i, coords[:, 1] ** j
                    )
        return output

    # Determine coefficients as Least-squares solution
    measurements_coordinates = np.transpose(
        np.vstack(
            (
                measurements[0],
                measurements[1],
            )
        )
    )
    measurements_data = measurements[2]

    def objective_function(coefficients: np.ndarray) -> np.ndarray:
        """Objective function for least squares optimization."""
        return (
            polynomial_interpolator(measurements_coordinates, coefficients)
            - measurements_data
        )

    opt_result = least_squares(objective_function, np.ones(dimension))
    coefficients = opt_result.x

    # Create a mesh of points at which the interpolator shall be evaluated.
    Ny, Nx = coordinate_system.shape
    coords_vector = coordinate_system.coordinates
    return polynomial_interpolator(coords_vector, coefficients).reshape(
        (Ny, Nx), order="F"
    )


def illumination_interpolation(
    measurements: tuple[np.ndarray, ...],
    coordinate_system: darsia.CoordinateSystem,
) -> np.ndarray:
    """Determine a voxeled spatial map from measurements through polynomial interpolation.

    Parameters
    ----------
    measurements : tuple[np.ndarray, ...]
        Tuple of x, y and data measurements providing the interpolation input.
    coordinate_system : darsia.CoordinateSystem
        Coordinate system of the corresponding physical image; sets the output
        shape.

    Returns
    -------
    np.ndarray
        The interpolated map.
    """

    def interpolator(
        coords: np.ndarray,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """Polynomial interpolation function."""
        output = np.zeros_like(coords[:, 0])
        coord0 = coefficients[0]
        coord1 = coefficients[1]
        coord2 = coefficients[2]
        p = coefficients[4]  # 2 : classical choice
        dist = (
            np.sqrt(
                (coords[:, 0] - coord0) ** 2
                + (coords[:, 1] - coord1) ** 2
                + coord2**2  # 2d case
            )
            ** p
        )
        i0 = coefficients[3]
        output = i0 / dist
        return output

    # Determine coefficients as Least-squares solution
    measurements_coordinates = np.transpose(
        np.vstack(
            (
                measurements[0],
                measurements[1],
            )
        )
    )
    measurements_data = measurements[2]

    def objective_function(coefficients: np.ndarray) -> np.ndarray:
        """Objective function for least squares optimization."""
        return interpolator(measurements_coordinates, coefficients) - measurements_data

    opt_result = least_squares(objective_function, np.ones(5))
    coefficients = opt_result.x

    # Create a mesh of points at which the interpolator shall be evaluated.
    Ny, Nx = coordinate_system.shape
    coords_vector = coordinate_system.coordinates
    return interpolator(coords_vector, coefficients).reshape((Ny, Nx), order="F")


def interpolate_to_image(
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
    image: darsia.Image,
    method: Literal[
        "rbf", "polynomial", "linear", "quadratic", "cubic", "quartic", "triangulation"
    ] = "rbf",
) -> darsia.Image:
    """Interpolate data to image.

    Parameters
    ----------
    data : np.ndarray
        (x,y,measurements) data to be interpolated.
    image : darsia.Image
        Image to which data shall be interpolated.
    method : str
        Interpolation method to use. Options are:

        - "rbf": Radial Basis Function interpolation (default).

        - "polynomial": Polynomial interpolation.

        - "linear": Linear interpolation.

        - "quadratic": Quadratic interpolation.

        - "cubic": Cubic interpolation.

        - "quartic": Quartic interpolation.

        - "triangulation": Linear on the Delaunay triangles of the measurements,
          the nearest measurement outside their convex hull
          (:func:`interpolate_linear_in_hull`).

    Returns
    -------
    darsia.Image
        Interpolated image.
    """
    # Initialize image
    interpolated_image = image.copy()

    # Convert data to 1D columns if provided in mesh format
    assert len(data) == 3, "Data must be a tuple of (x, y, data)."
    if all([len(d.shape) == 2 for d in data]):
        data = (
            np.ravel(data[0]),
            np.ravel(data[1]),
            np.ravel(data[2]),
        )

    if method.lower() == "rbf":
        # Define array through RBF interpolation
        interpolated_image.img = interpolate_measurements_2d(
            data,
            interpolated_image.coordinatesystem,
        )
    elif method.lower() == "triangulation":
        # Linear inside the measurements' convex hull, the nearest one outside.
        Ny, Nx = interpolated_image.coordinatesystem.shape
        values, _ = interpolate_linear_in_hull(
            data, interpolated_image.coordinatesystem.coordinates
        )
        interpolated_image.img = values.reshape((Ny, Nx), order="F")
    elif method.lower() == "illumination":
        # Define array through polynomial interpolation
        interpolated_image.img = illumination_interpolation(
            data,
            interpolated_image.coordinatesystem,
        )
    elif method.lower() in ["linear", "quadratic", "cubic", "quartic"]:
        degrees = {
            "linear": 1,
            "quadratic": 2,
            "cubic": 3,
            "quartic": 4,
        }
        degree = degrees[method.lower()]
        interpolated_image.img = polynomial_interpolation(
            data,
            interpolated_image.coordinatesystem,
            degree,
        )
    else:
        raise NotImplementedError(
            f"""Interpolation method "{method}" is not supported."""
        )

    return interpolated_image


def interpolate_to_image_from_csv(
    csv_file: Path,
    key: str,
    image: darsia.Image,
    method: Literal[
        "rbf", "polynomial", "linear", "quadratic", "cubic", "quartic", "triangulation"
    ] = "rbf",
) -> darsia.Image:
    """Interpolate data from CSV to image.

    Parameters
    ----------
    csv_file : Path
        Path to the CSV file containing the data.
    key : str
        Key to identify the data in the CSV file.
    image : darsia.Image
        Image to which data shall be interpolated.
    method : str
        Interpolation method to use. Options are:

        - "rbf": Radial Basis Function interpolation (default).

        - "polynomial": Polynomial interpolation.

        - "linear": Linear interpolation.

        - "quadratic": Quadratic interpolation.

        - "cubic": Cubic interpolation.

        - "quartic": Quartic interpolation.

        - "triangulation": Linear on the Delaunay triangles of the measurements,
          the nearest measurement outside their convex hull
          (:func:`interpolate_linear_in_hull`).

    Returns
    -------
    darsia.Image
        Interpolated image.
    """
    # Convert data to the format expected by interpolate_to_image
    data = pd.read_csv(csv_file)
    x_key = "x" if "x" in data.columns else "X"
    y_key = "y" if "y" in data.columns else "Y"
    x = data[x_key].to_numpy()
    y = data[y_key].to_numpy()
    mean = data[key].to_numpy()

    return interpolate_to_image(
        (x, y, mean),
        image,
        method=method,
    )
