"""Compatibility entry points for source-only chart registration."""
from darsia.fluidflower.native import registration as _core
from darsia.fluidflower.native.config import require_parameters
from darsia.fluidflower.native.io import sha256_file as _sha256


def _gray_roi(path, box):
    return _core.gray_roi(path, box, parameters=require_parameters("registration")["common"]["sensor"])


def register_frame_chart_xy8(*, baseline_camera_jpeg, baseline_camera_sha256,
        frame_camera_jpeg, frame_camera_sha256, baseline_chart_xy8, baseline_chart_log_rgb,
        baseline_review_to_camera_h, frame_review_to_camera_h, frame_native_linear_rgb):
    return _core.register_chart(baseline_camera_jpeg=baseline_camera_jpeg,
        baseline_camera_sha256=baseline_camera_sha256, frame_camera_jpeg=frame_camera_jpeg,
        frame_camera_sha256=frame_camera_sha256, baseline_chart_xy8=baseline_chart_xy8,
        baseline_chart_log_rgb=baseline_chart_log_rgb,
        baseline_review_to_camera_h=baseline_review_to_camera_h,
        frame_review_to_camera_h=frame_review_to_camera_h,
        frame_native_linear_rgb=frame_native_linear_rgb, parameters=require_parameters("registration"))
