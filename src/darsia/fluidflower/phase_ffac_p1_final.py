"""Fixed P1-final categorical post-processing; no model evaluation.

Raster cutoffs follow the locked top_band_width_compare reference: round the
60 mm band on the nominal AC28 grid (365 rows), floor the 30 mm column gate.
Measured per-pixel areas are used for reports, never to change these cutoffs.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import cv2
import numpy as np

CONTRACT = "ffac-g3-t4c-p1-final-candidate-v1"
SPEC_SHA256 = "b421872c3b400376f8c0d65af2db9ab10c329dc751a8956ea212020be8aaa5e8"
NOMINAL_AREA_MM2 = .027053
MM_PER_PIXEL = NOMINAL_AREA_MM2 ** .5
BAND_PIXELS = round(60 / MM_PER_PIXEL)
COLUMN_GATE_PIXELS = int(30 / MM_PER_PIXEL)
RULE = dict(order=["restore_flagged_floor_labels", "remove_detached_top_components"],
    band_mm=60, band_pixels=BAND_PIXELS, column_gate_mm=30, column_gate_pixels=COLUMN_GATE_PIXELS,
    nominal_pixel_area_mm2=NOMINAL_AREA_MM2, connectivity=8, passes=1,
    top_line="first label !=255 per column", depth="row minus own-column top line",
    component_removal="touches acting-column band and nowhere deeper than band",
    deeper_component="keep whole, including pixels in the band", bottom_edge_changed=False)


@dataclass
class FinalResult:
    labels: np.ndarray
    after_floor_removal: np.ndarray
    removed_top: np.ndarray
    previous_top_labels: np.ndarray
    qa: dict


def select_backend(requested="auto"):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("P1-final backend must be auto, cuda or cpu")
    if requested == "cpu": return "cpu"
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            # Initialize now: a missing driver must not fail midway through a batch.
            cp.zeros(1).sum().item()
            return "cuda"
    except (ImportError, RuntimeError, OSError) as error:
        if requested == "cuda": raise RuntimeError("Requested CUDA backend is unavailable") from error
    if requested == "cuda": raise RuntimeError("Requested CUDA backend is unavailable")
    return "cpu"


def _top_mask(labels, backend):
    if backend == "cuda":
        import cupy as xp
        from cupyx.scipy.ndimage import label
        source = xp.asarray(labels)
    else:
        xp = np; source = labels
    domain = source != 255
    has = domain.any(axis=0)
    top = xp.where(has, domain.argmax(axis=0), labels.shape[0]).astype(xp.int32)
    median = xp.median(top[has]) if bool(has.any()) else xp.asarray(labels.shape[0])
    acting = has & (xp.abs(top-median) <= COLUMN_GATE_PIXELS)
    depth = xp.arange(labels.shape[0], dtype=xp.int32)[:, None]-top[None, :]
    co2 = (source == 1) | (source == 2)
    if backend == "cuda":
        components, count = label(co2, structure=xp.ones((3,3),dtype=xp.int8))
    else:
        count_plus_one, components = cv2.connectedComponents(co2.astype(np.uint8), connectivity=8)
        count = count_plus_one-1
    inside = co2 & acting[None,:] & (depth >= 0) & (depth <= BAND_PIXELS)
    deeper = co2 & (depth > BAND_PIXELS)
    def present(mask):
        selected=components[mask]
        # CuPy's bincount reduces max() even for an empty input; negative
        # control frames and components entirely below the band are valid.
        return (xp.bincount(selected,minlength=int(count)+1)>0 if selected.size
                else xp.zeros(int(count)+1,dtype=bool))
    in_ids = present(inside)
    deep_ids = present(deeper)
    remove_ids = in_ids & ~deep_ids; remove_ids[0] = False
    removed = co2 & remove_ids[components]
    audit = dict(removed_top_components=int(remove_ids.sum()), excluded_columns=int((~acting).sum()),
                 acting_columns=int(acting.sum()), top_median_row=float(median))
    if backend == "cuda":
        return xp.asnumpy(removed), xp.asnumpy(top), xp.asnumpy(acting), audit
    return removed, top, acting, audit


def apply_p1_final(p1_labels, base_labels, floor_mask, pixel_area_mm2, *, backend="auto"):
    """Return new labels plus exact provenance and independent identity checks."""
    wall, cpu = time.perf_counter(), time.process_time()
    p1, base, floor, area = map(np.asarray,(p1_labels,base_labels,floor_mask,pixel_area_mm2))
    if (p1.dtype != np.uint8 or p1.ndim != 2 or base.dtype != np.uint8 or base.shape != p1.shape
        or floor.dtype != bool or floor.shape != p1.shape or area.shape != p1.shape
        or not np.isfinite(area).all() or np.any(area < 0)
        or not set(np.unique(p1)).issubset({0,1,2,255})):
        raise ValueError("P1-final requires exact categorical labels, source-bound flags and measured areas")
    if np.any(floor & ((p1 != 1) | (base != 0))):
        raise ValueError("Floor provenance is not exclusively original no-CO2 to aqueous fill")
    after_a = p1.copy(); after_a[floor] = base[floor]
    union_exact = np.array_equal(np.isin(after_a,(1,2)),np.isin(base,(1,2)))
    gas_exact = np.array_equal(after_a == 2,p1 == 2)
    if not union_exact or not gas_exact:
        raise ValueError("Floor removal failed the exact BASE union / P1 gas identity")
    actual = select_backend(backend)
    gpu_seconds = 0.
    if actual == "cuda":
        import cupy as cp
        begin, end = cp.cuda.Event(),cp.cuda.Event(); begin.record()
    removed, top, acting, audit = _top_mask(after_a,actual)
    if actual == "cuda":
        end.record(); end.synchronize(); gpu_seconds=float(cp.cuda.get_elapsed_time(begin,end))/1000
    final = after_a.copy(); final[removed] = 0
    previous = np.where(removed,after_a,0).astype(np.uint8)
    # Independent CPU checks: complete components, fixed depth and exact flags.
    depth = np.arange(p1.shape[0],dtype=np.int32)[:,None]-top[None,:]
    changed = final != after_a
    closed = not np.any(cv2.dilate(removed.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
                        & np.isin(after_a,(1,2)) & ~removed)
    components = cv2.connectedComponents(removed.astype(np.uint8),connectivity=8)[0]-1
    exact_outside = np.array_equal(final[~(floor|removed)],p1[~(floor|removed)])
    passed = (np.array_equal(changed,removed) and closed and components == audit["removed_top_components"]
              and not np.any(removed & (depth > BAND_PIXELS)) and exact_outside
              and np.array_equal(final == 255,p1 == 255))
    if not passed: raise ValueError("P1-final fixed component/flag/outside-mask QA failed")
    measure = lambda mask:float(area[mask].sum(dtype=np.float64)/100)
    nominal = lambda mask:float(np.count_nonzero(mask)*NOMINAL_AREA_MM2/100)
    remains = np.isin(final,(1,2)) & acting[None,:] & (depth >= 0)
    qa = dict(audit, all_pass=True, after_a_co2_union_equals_base=union_exact, after_a_gas_equals_p1=gas_exact,
        changed_pixels_exactly_top_flags=True, removed_components_closed=True,
        removed_components_within_fixed_band=True, labels_outside_masks_identical=True,
        rig_domain_unchanged=True, removed_floor_pixels=int(floor.sum()), removed_top_pixels=int(removed.sum()),
        removed_floor_cm2=measure(floor), removed_top_aqueous_cm2=measure(removed & (after_a == 1)),
        removed_top_gas_cm2=measure(removed & (after_a == 2)),
        nominal_removed_top_aqueous_cm2=nominal(removed & (after_a == 1)),
        nominal_removed_top_gas_cm2=nominal(removed & (after_a == 2)),
        remaining_co2_60mm_cm2=measure(remains & (depth <= BAND_PIXELS)),
        remaining_co2_90mm_cm2=measure(remains & (depth <= round(90/MM_PER_PIXEL))),
        retained_60mm_reason="connected to a component deeper than 60 mm" if np.any(remains & (depth<=BAND_PIXELS)) else "none",
        backend=actual, gpu_seconds=gpu_seconds, cpu_seconds=time.process_time()-cpu,
        wall_seconds=time.perf_counter()-wall, training=False, inference=False)
    return FinalResult(final,after_a,removed,previous,qa)
