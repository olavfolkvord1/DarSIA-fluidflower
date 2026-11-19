"""Physical coordinate systems for Image objects."""

from __future__ import annotations

from typing import Union

import numpy as np

import darsia


class CoordinateSystem:
    """Class for coordinate system for general space-time images.

    A coordinate system has knowledge about the conversion of voxels (in standard format),
    i.e., ijk format with (0,0,0) being identified with the top left front corner.
    This implicitly concludes that the coordinate system is directly related
    to a certain fixed indexing, provided at initialization. Conversion maps
    from voxels to coordinates and vice-versa are provided.
    """

    def __init__(self, img: darsia.Image):
        """Generate a coordinate system based on the metadata of an existing image.

        Parameters
        ----------
        img : darsia.Image
            Image for which a coordinate system shall be constructed.
        """

        assert img.indexing in ["i", "ij", "ijk"], f"Indexing not supported."
        self.indexing = img.indexing
        """Indexing of the underlying image."""

        self.dim = img.space_dim
        """Dimension of the underlying space."""

        self.shape = img.img.shape[: self.dim]
        """Dimensions of the underlying images in terms of voxels."""

        self.dimensions = img.dimensions
        """Physical dimensions in each index direction."""

        self.axes = "xyz"[: self.dim]
        """Axes of the underlying space."""

        self.voxel_size = {}
        """Voxel size in each dimension/axis."""
        for axis in self.axes:
            pos, _ = darsia.interpret_indexing(axis, self.indexing)
            self.voxel_size[axis] = img.voxel_size[pos]

        self._coordinate_of_origin_voxel: darsia.Coordinate = img.origin
        """Coordinate of origin voxel."""

        opposite_corner_voxel = img.img.shape[: self.dim]
        self._coordinate_of_opposite_voxel: darsia.Coordinate = self.coordinate(
            opposite_corner_voxel
        )
        """Coordinate of opposite voxel."""

        corners = np.vstack(
            (self._coordinate_of_origin_voxel, self._coordinate_of_opposite_voxel)
        )
        self.domain = {}
        """Extremal points/bounding box of the active coordinate system / the image."""
        for i, axis in enumerate(self.axes):
            self.domain[axis + "min"] = np.min(corners[:, i])
            self.domain[axis + "max"] = np.max(corners[:, i])

        self.max_coordinate = np.max(
            np.vstack(
                (self._coordinate_of_origin_voxel, self._coordinate_of_opposite_voxel)
            ),
            axis=0,
        )
        """Extremal coordinate (max) of the active coordinate system / the image."""

        self.min_coordinate = np.min(
            np.vstack(
                (self._coordinate_of_origin_voxel, self._coordinate_of_opposite_voxel)
            ),
            axis=0,
        )
        """Extremal coordinate (min) of the active coordinate system / the image."""

    @property
    def voxels(self) -> darsia.VoxelArray:
        """Voxel array of image, collecting all voxels.

        Returns
        -------
        VoxelArray
            Voxel array of image.
        """
        if not hasattr(self, "_voxels"):
            self._voxels = darsia.make_voxel(
                np.indices(self.shape, dtype=int).reshape((self.dim, -1), order="F").T
            )
        return self._voxels

    @property
    def coordinates(self) -> darsia.CoordinateArray:
        """Coordinate array of image, collecting all coordinates.

        Returns
        -------
        CoordinateArray
            Coordinate array of image.
        """
        if not hasattr(self, "_coordinates"):
            self._coordinates = self.coordinate(self.voxels)
        return self._coordinates

    def length(self, num: Union[int, np.ndarray], axis: str) -> float:
        """
        Convert number of voxels/voxels to metric units, when interpreting
        in some given axis.

        Parameters
        ----------
        num : int or array
            Number(s) of voxels/voxels.
        axis : str
            Axis "x", "y", or "z".

        Returns
        -------
        float or array
            Length in metric units.
        """
        assert axis in self.axes
        return num * self.voxel_size[axis]

    def num_voxels(
        self, length: Union[float, np.ndarray], axis: str
    ) -> Union[int, np.ndarray]:
        """
        Convert length in metric units to number of voxels, when interpreting
        the length in some given axis.

        Parameters
        ----------
        length : float or 1d array of floats
            Length(s) in metric units.
        axis : str
            Axis "x", "y", or "z".

        Returns
        -------
        int or array
            Number(s) of voxels.
        """
        # Include all touched voxels; use therefore ceil.
        assert axis in self.axes
        return np.ceil(length / self.voxel_size[axis]).astype(int)

    def coordinate(
        self,
        voxel: Union[
            np.ndarray,
            list[int],
            tuple[int],
            darsia.Voxel,
            darsia.VoxelArray,
        ],
    ) -> Union[darsia.Coordinate, darsia.CoordinateArray]:
        """
        Conversion from voxel to Cartesian coordinate, i.e., from (row,col) to (x,y)
        format plus scaling for a 2d image.

        Handles both single and multiple voxels.

        Parameters
        ----------
        voxel : np.ndarray, list, tuple, Coordinate, or Voxel
            Voxel location in the
            same format as the indexing of the underlying baseline image (see __init__);
            one voxel per row.

        Returns
        -------
        np.ndarray
            Corresponding coordinate in (x,y) format.
        """
        # Convert list to array
        if isinstance(voxel, tuple):
            voxel = list(voxel)
        if isinstance(voxel, list):
            voxel = np.array(voxel)
        assert isinstance(voxel, np.ndarray)

        # Aim at handling both single voxels stored in a 1d array as well as
        # multiple voxels stored in a 2d array. Convert to the more general
        # case. Cache the original size for the later output.
        voxel_array = np.atleast_2d(voxel)

        # Determine coordinates with the help of the origin
        coordinate = np.empty_like(voxel_array, dtype=float)
        for i, axis in enumerate(self.axes):
            pos, revert = darsia.interpret_indexing(axis, self.indexing)
            scaling = -1 if revert else 1
            coordinate[:, i] = (
                self._coordinate_of_origin_voxel[i]
                + scaling * voxel_array[:, pos] * self.voxel_size[axis]
            )

        # Return in same format as the input
        return darsia.make_coordinate(coordinate.reshape(voxel.shape))

    def voxel(
        self,
        coordinate: Union[np.ndarray, list[float], darsia.Voxel, darsia.Coordinate],
    ) -> np.ndarray:
        """
        Conversion from Cartesian coordinate to voxel in matrix indexing format.

        Handles both single and multiple coordinates.

        Parameters
        ----------
        coordinate : np.ndarray, list, Coordinate or Voxel
            Coordinate in Cartesian
            format, i.e., [x,y,z]; one coordinate per row.

        Returns
        -------
        np.ndarray
            Corresponding pixels in "ij"/"ijk" format.
        """
        # Convert list to array
        if isinstance(coordinate, list):
            coordinate = np.array(coordinate)
        assert isinstance(coordinate, np.ndarray)

        # Aim at handling both single coordinates stored in a 1d array as well as
        # multiple coordinates stored in a 2d array. Convert to the more general
        # case.
        coordinate_array = np.atleast_2d(coordinate)

        # Inverse of self.coordinate().
        pixel = np.empty_like(coordinate_array, dtype=int)
        for i, axis in enumerate(self.axes):
            pos, revert = darsia.interpret_indexing(axis, self.indexing)
            scaling = -1 if revert else 1
            pixel[:, pos] = np.floor(
                scaling
                * (coordinate_array[:, i] - self._coordinate_of_origin_voxel[i])
                / self.voxel_size[axis]
            )

        # Return in same format as the input, and force int dtype.
        return darsia.make_voxel(np.round(pixel.reshape(coordinate.shape)).astype(int))

    def coordinate_vector(self, pixel_vector: np.ndarray) -> np.ndarray:
        """
        Conversion from vectors (relative distances) in terms of pixels to coordinates.

        Parameters
        ----------
        pixel_vector : np.ndarray
            Vector(s) in pixel plane (one vector per row).

        Returns
        -------
        np.ndarray
            Coordinate vector(s) in Cartesian format.
        """
        # Aim at handling both single coordinates stored in a 1d array as well as
        # multiple coordinates stored in a 2d array. Convert to the more general
        # case.
        pixel_vector_array = np.atleast_2d(pixel_vector)

        coordinate_vector = np.empty_like(pixel_vector_array, dtype=float)
        for i, axis in enumerate(self.axes):
            pos, revert = darsia.interpret_indexing(axis, self.indexing)
            scaling = -1 if revert else 1
            coordinate_vector[:, i] = (
                scaling * pixel_vector_array[:, pos] * self.voxel_size[axis]
            )

        # Reshape needed if only a single vector has been used in the argument.
        return coordinate_vector.reshape(pixel_vector.shape)

    def pixel_vector(self, coordinate_vector: np.ndarray) -> np.ndarray:
        """
        Analogue to self.coorindate_vector().

        """
        raise NotImplementedError


def check_equal_coordinatesystems(
    coordinatesystem1: CoordinateSystem,
    coordinatesystem2: CoordinateSystem,
    exclude_size: bool = False,
) -> tuple[bool, dict]:
    """Check whether two coordinate systems are equivalent, i.e., they share basic
    attributes.

    Parameters
    ----------
    coordinatesystem1 : CoordinateSystem
        First coordinate system.
    coordinatesystem2 : CoordinateSystem
        Second coordinate system.
    exclude_size : bool
        Flag controlling whether the size quantities are exluded.

    Returns
    -------
    bool
        True iff the two coordinate systems are equivalent.
    dict
        Log of the failed checks.
    """
    success = True
    failure_log = []

    if not (coordinatesystem1.indexing == coordinatesystem2.indexing):
        failure_log.append("indexing")

    if not (coordinatesystem1.dim == coordinatesystem2.dim):
        failure_log.append("space_dim")

    if not exclude_size:
        if not (np.allclose(coordinatesystem1.shape, coordinatesystem2.shape)):
            failure_log.append("shape")

    if not (np.allclose(coordinatesystem1.dimensions, coordinatesystem2.dimensions)):
        failure_log.append("dimensions")

    if not (coordinatesystem1.axes == coordinatesystem2.axes):
        failure_log.append("axes")

    if not exclude_size:
        voxel_size_equal = True
        for axis in coordinatesystem1.axes:
            voxel_size_equal = voxel_size_equal and np.isclose(
                coordinatesystem1.voxel_size[axis], coordinatesystem2.voxel_size[axis]
            )
        if not voxel_size_equal:
            failure_log.append("voxel_size")

    if not (
        np.allclose(
            coordinatesystem1._coordinate_of_origin_voxel,
            coordinatesystem2._coordinate_of_origin_voxel,
        )
    ):
        failure_log.append("coordinate_of_origin_voxel")

    if not (
        np.allclose(
            coordinatesystem1._coordinate_of_opposite_voxel,
            coordinatesystem2._coordinate_of_opposite_voxel,
        )
    ):
        failure_log.append("coordinate_of_opposite_voxel")

    success = len(failure_log) == 0

    return success, failure_log


def pixel_centres(image: darsia.Image) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of the pixel centres of a 2d image: x per column, y per row.

    The centres lie half a pixel inside the image's top left corner (its origin),
    one voxel size apart, x to the right and y upwards. Every position on a run
    grid (boxes, ports, heights, distances) is taken from them.

    Parameters
    ----------
    image : darsia.Image
        A 2d image with its physical dimensions and origin.

    Returns
    -------
    tuple of arrays
        x of each column's centres (cols,) and y of each row's (rows,), in the
        units of the image's dimensions (metres on a run grid).

    """
    rows, cols = image.img.shape[:2]
    dy, dx = image.voxel_size
    x = image.origin[0] + (np.arange(cols) + 0.5) * dx
    y = image.origin[1] - (np.arange(rows) + 0.5) * dy
    return x, y


def optical_pitch_m(
    pixel_area: darsia.Image, domain: np.ndarray
) -> tuple[float, float]:
    """The pitch [m] on the sand plane of one pixel step down a column and along a
    row.

    Every optical length given in millimetres (a signal mean, a background block or
    margin, a shadow rule, a measurement's tiles) is converted to pixels with it.
    The sand lies behind the plane of the rig's coordinates, and the measured pixel
    areas carry that offset as the optical area factor: their median in the domain
    over the coordinate cell. The pitch is the coordinate pitch, the grid's
    dimensions over its pixels, times the factor's square root, so that lengths
    agree with the areas.

    Parameters
    ----------
    pixel_area : darsia.Image
        Measured area [m2] of each pixel of a grid (e.g. ``Rig.pixel_area``), with
        the grid's physical dimensions.
    domain : array
        Pixels whose areas give the factor, e.g. the rig's domain.

    Returns
    -------
    tuple of float
        The pitch down a column and along a row.

    """
    pitch = [
        float(size) / count
        for size, count in zip(pixel_area.dimensions, pixel_area.img.shape[:2])
    ]
    area = np.broadcast_to(np.asarray(pixel_area.img, np.float64), domain.shape)
    factor = float(np.median(area[domain])) / (pitch[0] * pitch[1])
    return tuple(step * float(np.sqrt(factor)) for step in pitch)


def optical_grid(
    image: darsia.Image, optical_pitch: tuple[float, float], dtype: type
) -> darsia.Image:
    """The image's grid with the dimensions of its pixels at the pitch on the sand
    plane, in ``dtype``.

    Every optical length given in millimetres is sized on it, e.g. the square of a
    mean (:class:`darsia.REV`) or the blocks of a background, as our production
    sized them.

    Parameters
    ----------
    image : darsia.Image
        An image on the grid, e.g. the sand map.
    optical_pitch : tuple of float
        Pitch [m] on the sand plane down a column and along a row
        (:func:`optical_pitch_m`).
    dtype : type
        Floating type of the dimensions, that of the images measured.

    Returns
    -------
    darsia.Image
        A scalar image of the image's values with the optical dimensions.

    """
    rows, cols = image.img.shape
    return darsia.ScalarImage(
        image.img,
        dimensions=[dtype(rows * optical_pitch[0]), dtype(cols * optical_pitch[1])],
    )
