"""Image mapping correction: sample an image onto a target grid at stored coordinates.

The mapping replays a geometric transform that has been determined elsewhere, for
instance by a registration of the camera image to the rig. For each pixel of the
target grid, the mapping stores the source pixel coordinates at which the input
image is sampled. Nothing is estimated here.

"""

from pathlib import Path

import numpy as np

import darsia


class ImageMappingCorrection(darsia.BaseCorrection):
    """Bilinear sampling of an image at stored source coordinates.

    The sampling maps hold, for every pixel of the target grid, the source column
    (``map_x``) and source row (``map_y``) in pixel units, with pixel centres at
    integer coordinates. Maps of shape (rows, cols) apply to every channel; maps of
    shape (rows, cols, channels) give each channel its own coordinates, e.g., to
    replay a lateral chromatic aberration correction.

    Non-finite input values mark invalid source pixels. A target pixel is invalid,
    and set to NaN in all channels, if any channel draws on an invalid source pixel
    with a nonzero interpolation weight, or if its 2 x 2 interpolation footprint
    reaches outside the source image (this includes samples exactly on the last
    source row or column). Values are sampled in the image's floating dtype, at
    least single precision, on the process's compute backend
    (:func:`darsia.compute_backend`).

    """

    def __init__(
        self,
        map_x: np.ndarray | None = None,
        map_y: np.ndarray | None = None,
        source_shape: tuple[int, int] | None = None,
        dimensions: list[float] | None = None,
        origin: list[float] | None = None,
    ) -> None:
        """Initialize the mapping, or an empty object to be loaded from file.

        Parameters
        ----------
        map_x : array, optional
            Source column of each target pixel, shape (rows, cols) or
            (rows, cols, channels).
        map_y : array, optional
            Source row of each target pixel, same shape as ``map_x``.
        source_shape : tuple of int, optional
            Number of rows and columns of the images to be mapped.
        dimensions : list of float, optional
            Physical height and width of the target grid.
        origin : list of float, optional
            Cartesian coordinates of the top-left target voxel. Defaults to
            ``[0, height]``, the convention of :class:`darsia.Image`.

        """
        if map_x is None and map_y is None:
            return
        self.map_x = np.asarray(map_x, dtype=np.float32)
        """Source column of each target pixel (float32, as OpenCV takes maps)."""
        self.map_y = np.asarray(map_y, dtype=np.float32)
        """Source row of each target pixel (float32, as OpenCV takes maps)."""
        if self.map_x.shape != self.map_y.shape or self.map_x.ndim not in (2, 3):
            raise ValueError("map_x and map_y must share the shape (rows, cols[, c]).")
        self.source_shape = tuple(int(n) for n in source_shape)
        """Number of rows and columns of the images to be mapped."""
        self.dimensions = [float(d) for d in dimensions]
        """Physical height and width of the target grid."""
        self.origin = np.array(
            [0.0, self.dimensions[0]] if origin is None else origin, dtype=float
        )
        """Cartesian coordinates of the top-left target voxel."""

    @property
    def target_shape(self) -> tuple[int, int]:
        """Number of rows and columns of the target grid."""
        return self.map_x.shape[:2]

    def correct_array(self, image: np.ndarray) -> np.ndarray:
        """Sample an image array on the target grid.

        Parameters
        ----------
        image : array
            Image on the source grid, shape (rows, cols) or (rows, cols, channels).

        Returns
        -------
        array
            Image on the target grid in the image's floating dtype, NaN where
            invalid.

        Raises
        ------
        ValueError
            If the image is not on the source grid, e.g., if it has been mapped
            already.

        """
        if image.shape[:2] != self.source_shape:
            raise ValueError(
                f"Image of shape {image.shape[:2]} is not on the source grid "
                f"{self.source_shape} of the mapping; a mapped image is not mapped "
                "again."
            )
        planes = np.atleast_3d(image)
        planes = planes.astype(np.result_type(planes, np.float32), copy=False)
        if self.map_x.ndim == 3 and self.map_x.shape[2] != planes.shape[2]:
            raise ValueError("Channel-wise maps require one map per image channel.")

        # Sample the values and, at the same coordinates, the indicator of invalid
        # source pixels; a target pixel is valid only if the indicator samples to 0.
        invalid = ~np.isfinite(planes)
        values = np.where(invalid, 0, planes)
        indicator = invalid.astype(planes.dtype)
        backend = darsia.compute_backend()
        result = backend.remap(values, self.map_x, self.map_y)
        support = backend.remap(indicator, self.map_x, self.map_y)
        valid = (
            np.isfinite(result).all(axis=-1)
            & np.isfinite(support).all(axis=-1)
            & (support == 0).all(axis=-1)
        )
        result[~valid] = np.nan
        return result if image.ndim == 3 else result[..., 0]

    def correct_metadata(self, metadata: dict = {}) -> dict:
        """Physical extent of the target grid.

        Parameters
        ----------
        metadata : dict, optional
            Metadata of the input image (not used).

        Returns
        -------
        dict
            Dimensions and origin of the target grid.

        """
        return {
            "dimensions": list(self.dimensions),
            "origin": darsia.make_coordinate(self.origin),
        }

    # ! ---- I/O ----

    def save(self, path: Path) -> None:
        """Save the mapping to a npz file.

        Parameters
        ----------
        path : Path
            Path to the file.

        """
        np.savez(
            path,
            class_name=type(self).__name__,
            map_x=self.map_x,
            map_y=self.map_y,
            source_shape=np.array(self.source_shape),
            dimensions=np.array(self.dimensions),
            origin=self.origin,
        )

    def load(self, path: Path) -> None:
        """Load the mapping from a npz file.

        Parameters
        ----------
        path : Path
            Path to the file.

        """
        with np.load(path) as data:
            self.__init__(
                map_x=data["map_x"],
                map_y=data["map_y"],
                source_shape=tuple(data["source_shape"]),
                dimensions=list(data["dimensions"]),
                origin=data["origin"],
            )
