"""Apply the frozen RAW correction to camera-visible pixels outside the old crop.

The prepared native centre is reused exactly. Only the peripheral ring is
decoded, using the source-certified camera transform and existing light
coefficients. This neither fits a new correction nor promotes hardware.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .project import sha256_file

CONTRACT = "ffac-rig-frozen-raw-camera-ring-v1"
NATIVE_HW = (2979, 5472)
CANVAS_HW = (3546, 6304)
ORIGIN_XY = (-416, -496)


def frozen_light_from_receipt(info, grid):
    """Evaluate the recorded polynomial; do not fit on the added ring."""
    from .tools import ffac_v5_photometry as v5
    if info.get("mode") == "exact-source-bound-B" and info.get("zero_field") is True:
        return None
    if info.get("central_frozen_v5_parity") is not True:
        raise ValueError("Ring requires the existing certified frame-light coefficients")
    if tuple(info.get("grid_origin_cell_xy", ())) != tuple(grid.origin_cell_xy):
        raise ValueError("Frame-light receipt belongs to another reference grid")
    degree = info["degree"]
    coef = np.asarray(info["coefficients"], np.float64)
    if degree not in (1, 2) or coef.shape != ((degree + 1)*(degree + 2)//2,) or not np.isfinite(coef).all():
        raise ValueError("Invalid frozen frame-light polynomial")
    yy, xx = np.mgrid[:grid.shape_hw[0], :grid.shape_hw[1]]
    return (v5._light_basis((xx + grid.origin_cell_xy[0] - 1.5)/4,
                           (yy + grid.origin_cell_xy[1] - 1.5)/4, degree) @ coef).astype(np.float32)


def peripheral_rectangles(mask, native_hw=NATIVE_HW, origin_xy=ORIGIN_XY):
    """Disjoint minimal camera-ring windows, excluding the reused native centre."""
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.ndim != 2:
        raise TypeError("Camera-ring eligibility requires a boolean canvas mask")
    height, width = mask.shape
    x0, y0 = (-int(x) for x in origin_xy)
    nh, nw = native_hw
    if x0 < 0 or y0 < 0 or x0+nw > width or y0+nh > height:
        raise ValueError("Native centre lies outside the requested canvas")
    rectangles = []
    for ax, ay, bx, by in ((0,0,width,y0), (0,y0+nh,width,height),
                           (0,y0,x0,y0+nh), (x0+nw,y0,width,y0+nh)):
        ys, xs = np.nonzero(mask[ay:by,ax:bx])
        if len(xs):
            rectangles.append((ax+int(xs.min()),ay+int(ys.min()),
                               ax+int(xs.max())+1,ay+int(ys.max())+1))
    return rectangles


def sample_sensor_window(demosaiced, renderer, rectangle, *, origin_xy=ORIGIN_XY,
                         float_planes=None):
    """Same frozen Sony inverse map and bilinear RAW interpolation as the centre."""
    from .tools import ffac_raw_geometry as geo
    image = np.asarray(demosaiced)
    if image.dtype != np.uint16 or image.shape != (*geo._RAW_WH[::-1],3):
        raise TypeError("Ring must use the full-resolution uint16 RAW demosaic")
    h = np.asarray(renderer["review_to_camera_h"],np.float64)
    if h.shape != (3,3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-10:
        raise ValueError("Invalid source-certified camera transform")
    profile = renderer["profile"]
    distortion = tuple(renderer["sony_distortion_table"])
    if distortion != geo._DISTORTION[profile]:
        raise ValueError("Ring differs from its frozen Sony distortion profile")
    a,b,c,d = rectangle
    result = np.empty((3,d-b,c-a),np.float32)
    ca = (renderer["sony_ca_red_table"],None,renderer["sony_ca_blue_table"])
    planes = ([image[...,channel].astype(np.float32) for channel in range(3)]
              if float_planes is None else float_planes)
    if (len(planes)!=3 or any(plane.dtype!=np.float32 or plane.shape!=image.shape[:2] for plane in planes)):
        raise TypeError("RAW sampler planes must preserve full-sensor float32 geometry")
    for first in range(b,d,128):
        last = min(d,first+128)
        yy,xx = np.mgrid[first:last,a:c].astype(np.float32)
        xx += origin_xy[0]
        yy += origin_xy[1]
        for channel in range(3):
            rx,ry = geo._raw_xy(xx,yy,h,geo._PROFILE[profile],distortion,ca[channel])
            result[channel,first-b:last-b] = cv2.remap(planes[channel],
                rx.astype(np.float32),ry.astype(np.float32),cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan)
    return result


def extend_prepared_source(native_log, producer_audit, processor, ring_eligible):
    """Return full-canvas scientific float data and an explicit extension audit.

    ``ring_eligible`` is the photo-certified rig domain minus actual hardware,
    intersected with the camera footprint. Sensor/photometry-invalid pixels
    remain NaN and are filled categorically by the downstream caller.
    """
    from .tools.ffac_raw_native_ea import _check_opencv_cfa_order
    from .tools import ffac_v5_observed_sand as observed
    from .tools.ffac_v5_old_field_native import apply_old_field_native
    from .tools import ffac_raw_geometry as geo

    native = np.asarray(native_log)
    eligible = np.asarray(ring_eligible)
    if native.dtype != np.float32 or native.shape != (3,*NATIVE_HW):
        raise TypeError("Ring centre must be the prepared CHW float32 scientific source")
    if (eligible.dtype != bool or eligible.ndim != 2 or
            eligible.shape[0] < CANVAS_HW[0] or eligible.shape[1] < CANVAS_HW[1]):
        raise ValueError("Camera-ring domain does not share the phase canvas")
    output = np.full((3,*eligible.shape),np.nan,np.float32)
    ox,oy = (-x for x in ORIGIN_XY)
    output[:,oy:oy+NATIVE_HW[0],ox:ox+NATIVE_HW[1]] = native
    rectangles = peripheral_rectangles(eligible,native_hw=NATIVE_HW,origin_xy=ORIGIN_XY)
    audit = dict(contract=CONTRACT,cpu_only=True,training=False,new_photometry_fit=False,
                 origin_xy=list(ORIGIN_XY),canvas_hw=list(eligible.shape),native_centre_bytes_preserved=True,
                 requested_ring_pixels=sum(int(eligible[b:d,a:c].sum()) for a,b,c,d in rectangles),
                 model_input_encoding="chw-float32-log-camera-linear-rgb", windows=[])
    if not rectangles:
        audit["finite_ring_pixels"] = 0
        return output,np.isfinite(output).all(0),audit
    if processor.identity != producer_audit.get("processor"):
        raise ValueError("Peripheral correction differs from the prepared source's frozen calibration")
    raw_source = producer_audit["sources"]["camera_arw"]
    raw_path = Path(raw_source["path"])
    if sha256_file(raw_path) != raw_source["sha256"]:
        raise ValueError("Peripheral RAW source changed after scientific preparation")
    renderer = producer_audit["renderer"]
    if renderer["sources"]["arw_sha256"] != raw_source["sha256"]:
        raise ValueError("Peripheral camera transform belongs to another RAW source")
    p = processor.processor
    if renderer["run"].casefold() != p.run.casefold():
        raise ValueError("Peripheral RAW sensor belongs to another run")
    h = np.asarray(producer_audit["H_ac28_to_frame_native"],np.float64)
    light = frozen_light_from_receipt(producer_audit["photometry"]["per_frame_light"],p.grid)
    _check_opencv_cfa_order()
    import rawpy
    with rawpy.imread(str(raw_path)) as raw:
        if (raw.raw_image_visible.shape != geo._RAW_WH[::-1] or
            tuple(raw.black_level_per_channel) != (800,800,800,800) or
            raw.white_level != renderer["white_level"] or
            not np.array_equal(raw.raw_pattern,[[0,1],[3,2]]) or raw.color_desc != b"RGBG"):
            raise ValueError("Peripheral RAW sensor metadata differs from the frozen source")
        black = np.maximum(raw.raw_image_visible.astype(np.int32)-800,0).astype(np.uint16)
    demosaic = cv2.demosaicing(black,cv2.COLOR_BayerRG2BGR_EA)
    planes = [demosaic[...,channel].astype(np.float32) for channel in range(3)]
    for rectangle in rectangles:
        a,b,c,d = rectangle
        linear = sample_sensor_window(demosaic,renderer,rectangle,float_planes=planes)
        with np.errstate(divide="ignore",invalid="ignore"):
            log = np.where(linear>2,np.log(linear),np.nan).astype(np.float32)
        corrected = observed.apply_native(log,p.fit,p.reference,p.grid,h,
            source_eligible_mask=eligible[b:d,a:c],light_field_ref8=light,
            destination_origin_xy=(a+ORIGIN_XY[0],b+ORIGIN_XY[1]))
        if p.old_field is not None:
            corrected = apply_old_field_native(corrected,p.old_field,p.grid,h,
                destination_origin_xy=(a+ORIGIN_XY[0],b+ORIGIN_XY[1]))
        valid = np.isfinite(corrected).all(0)
        corrected[:,~valid] = np.nan
        output[:,b:d,a:c] = corrected
        audit["windows"].append(dict(canvas_xyxy=list(rectangle),
            eligible_pixels=int(eligible[b:d,a:c].sum()),finite_scientific_pixels=int(valid.sum())))
    audit["finite_ring_pixels"] = sum(row["finite_scientific_pixels"] for row in audit["windows"])
    audit["raw_source"] = raw_source
    audit["frozen_frame_light"] = producer_audit["photometry"]["per_frame_light"]
    audit["unsupported_ring_pixels"] = audit["requested_ring_pixels"]-audit["finite_ring_pixels"]
    return output,np.isfinite(output).all(0),audit
