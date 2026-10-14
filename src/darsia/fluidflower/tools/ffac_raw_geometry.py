"""Compatibility entry points for explicitly configured sensor registration."""
from pathlib import Path

import numpy as np

from darsia.fluidflower.native import registration as _core
from darsia.fluidflower.native.config import require_parameters
from darsia.fluidflower.native.io import sha256_file as _sha256


def _parameters():
    return require_parameters("registration")


def __getattr__(name):
    if name not in {"_PROFILE", "_DISTORTION", "_CA_RED", "_CA_BLUE", "_KNOTS",
                    "_CONTENT_WH", "_CAMERA_WH", "_RAW_WH"}:
        raise AttributeError(name)
    p = _parameters()["common"]
    values = {
        "_PROFILE": lambda: {key: value["coefficients"] for key, value in p["profiles"].items()},
        "_DISTORTION": lambda: {key: tuple(value["distortion"]) for key, value in p["profiles"].items()},
        "_CA_RED": lambda: tuple(p["lens"]["ca_red"]),
        "_CA_BLUE": lambda: tuple(p["lens"]["ca_blue"]),
        "_KNOTS": lambda: np.asarray(p["lens"]["knots"]),
        "_CONTENT_WH": lambda: tuple(p["sensor"]["content_wh"]),
        "_CAMERA_WH": lambda: tuple(p["sensor"]["camera_wh"]),
        "_RAW_WH": lambda: tuple(p["sensor"]["raw_wh"]),
    }
    return values[name]()


def _legacy_scope(path, parameters):
    policy = parameters["legacy"]
    prefix = policy["profile_path_prefix"]
    source_id = next((part.upper() for part in Path(path).parts
                      if part.upper().startswith(prefix) and part[len(prefix):].isdigit()), None)
    if source_id is None:
        raise ValueError("Cannot identify source run from original sensor path")
    profile_id = (policy["lower_profile"] if int(source_id[len(prefix):]) <= policy["profile_number_split"]
                  else policy["upper_profile"])
    return source_id, profile_id


def _sony_tags(arw_path):
    return _core.read_lens_tables(arw_path, parameters=_parameters()["common"]["lens"])


def _bandpass(image):
    return _core.bandpass(image, parameters=_parameters()["common"]["matching"])


def _matched_rows(a, b, points, *, template_half, search_half, guess, min_correlation):
    return _core.match_rows(a, b, points, template_half=template_half, search_half=search_half,
        guess=guess, min_correlation=min_correlation, parameters=_parameters()["common"]["matching"])


_metrics = _core.match_metrics


def _review_to_camera(review_g, camera_g):
    return _core.review_to_camera(review_g, camera_g, parameters=_parameters()["common"])


def _raw_xy(u, v, h, params, distortion, ca=None):
    return _core.raw_coordinates(u, v, h, params, distortion, ca,
                                parameters=_parameters()["common"]["lens"])


def _raw_review_residual(g_plane, native_green, h, params, distortion):
    return _core.raw_review_residual(g_plane, native_green, h, params, distortion,
                                   parameters=_parameters()["common"])


def registered_raw_linear(arw_path, camera_jpeg_path, review_native_jpeg_path, *,
                          expected_source_hashes, content_box):
    parameters = _parameters()
    source_id, profile_id = _legacy_scope(arw_path, parameters)
    return _core.registered_sensor_linear(arw_path, camera_jpeg_path, review_native_jpeg_path,
        expected_source_hashes=expected_source_hashes, content_box=content_box,
        parameters=parameters, source_id=source_id, profile_id=profile_id)
