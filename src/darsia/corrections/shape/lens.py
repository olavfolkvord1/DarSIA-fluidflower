"""The lens model of a raw file: the first configured model that applies to it.

A lens model is a settings class with a method ``coordinates(path)``
(:class:`LensModel`). For a raw file it gives the function from camera to sensor
coordinates that :func:`darsia.registration_mapping` takes, or None when the file
lacks what the model needs. Two models exist: the tables a camera records in each
raw file (:class:`darsia.CameraLensSettings`) and a model given in configuration
(:class:`darsia.RadialLensSettings`) for cameras that record none. Another camera's
model is one more module with such a settings class.

"""

from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence


class LensModel(Protocol):
    """A lens model of raw files, e.g. :class:`darsia.CameraLensSettings`.

    A new lens model is one settings class with this method.

    """

    def coordinates(self, path: Path) -> Optional[Callable]:
        """The function from camera columns and rows to per-channel sensor columns
        and rows for a raw file, or None if the file lacks what the model needs
        (e.g. recorded tables)."""


def lens_coordinates(path: Path, models: Sequence[LensModel]) -> Callable:
    """The coordinate function of the first lens model that applies to a raw file.

    Parameters
    ----------
    path : Path
        Raw file.
    models : sequence of LensModel
        Configured lens models in order of preference, e.g. the recorded tables
        first and a configured model second.

    Returns
    -------
    callable
        Maps camera columns and rows to per-channel sensor columns and rows.

    Raises
    ------
    ValueError
        If no configured model applies, e.g. the file records no lens tables and
        no model is given for its camera.

    """
    for model in models:
        coordinates = model.coordinates(path)
        if coordinates is not None:
            return coordinates
    raise ValueError(
        f"{path} records no lens tables and no other lens model is configured."
    )
