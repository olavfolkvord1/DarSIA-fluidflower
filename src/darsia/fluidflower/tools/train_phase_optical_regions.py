"""Fit regional optical heads from every effective native review revision.

Artifacts live in the series project and disposable arrays in a local cache.
This first-stage experiment does not replace a selected Studio model.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import cv2
import numpy as np
from PIL import Image
from darsia.fluidflower import phase_optical_regions as regional
from darsia.fluidflower.phase_learning import HierarchicalPhaseModel
from darsia.fluidflower.phase_training import logistic, joint_aqueous
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools.train_phase_model_further import read, native_source
from darsia.fluidflower.tools.phase_optical_cleanup_preview import write, digest
from darsia.fluidflower.tools.phase_coupled_constraint_comparison import annotation_window
from darsia.fluidflower.tools.train_series_b_completion import pool_annotation


# Native RGB arrays and the regional feature arrays are acyclic NumPy objects.
# Refcounts release them as soon as a completed frame leaves the pipeline; a
# full cyclic-GC scan after every frame adds substantial CPU-only latency.
GC_COLLECT_INTERVAL = 16


def _v5_linear_asset(path, native):
    if (native.get("v5_photometry") or {}).get("format") != (
            "ffac-v5-native-source-materialization-v1"):
        return None
    path = resolve_review_path(path)
    if path.name == "pre_injection.jpg":
        row = native.get("reference") or {}
    else:
        match = re.fullmatch(r"frame_(\d{3})\.jpg", path.name)
        if match is None:
            raise ValueError(f"Unrecognized V5 native frame: {path}")
        index = int(match.group(1))
        frames = native["frame_sources"]
        if index >= len(frames):
            raise ValueError(f"V5 native frame index outside timeline: {path}")
        row = frames[index]
    sidecar = resolve_review_path(row["v5_log_linear_rgb_path"])
    if sidecar.resolve() != path.with_suffix(".log_linear_rgb.npy").resolve():
        raise ValueError(f"V5 linear input differs from displayed frame: {path}")
    if sha256_file(sidecar) != row["v5_log_linear_rgb_sha256"]:
        raise ValueError(f"V5 linear sidecar SHA-256 mismatch: {sidecar}")
    policy = native.get("v5_float_support_policy")
    from darsia.fluidflower.phase_training.v5_linear_input import EXACT_VALID_MASK_POLICY
    if policy == EXACT_VALID_MASK_POLICY:
        valid_path = resolve_review_path(row["v5_log_linear_valid_mask_path"])
        valid_sha = row["v5_log_linear_valid_mask_sha256"]
        if (valid_path.resolve() != path.with_suffix(
                ".log_linear_valid_mask.npy").resolve()
                or sha256_file(valid_path) != valid_sha):
            raise ValueError(f"V5 finite RAW mask differs from native frame: {path}")
    elif policy is None:
        if (row.get("v5_log_linear_valid_mask_path") is not None
                or row.get("v5_log_linear_valid_mask_sha256") is not None):
            raise ValueError("V5 finite RAW mask lacks an explicit policy")
        valid_path = valid_sha = None
    else:
        raise ValueError("Unknown V5 finite RAW support policy")
    return sidecar, row["v5_log_linear_rgb_sha256"], valid_path, valid_sha


def crop(path, box, native=None):
    x, y, w, h = box
    asset = _v5_linear_asset(path, native or {})
    if asset is not None:
        from darsia.fluidflower.phase_training.v5_linear_input import (
            V5_UNIT_DENOMINATOR, decode_log_linear,
        )
        bound_box = (native or {}).get('content_box_xywh')
        if bound_box is None or tuple(map(int, bound_box)) != (x, y, w, h):
            raise ValueError('V5 float sidecar must use its exact native content box')
        rgb = decode_log_linear(asset[0], expected_sha256=asset[1],
                                content_shape_hw=(h, w),
                                unit_denominator=V5_UNIT_DENOMINATOR,
                                invalid_policy=(native or {}).get(
                                    "v5_float_support_policy"),
                                valid_mask_path=asset[2],
                                valid_mask_sha256=asset[3])
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.dtype("float32")
                or rgb.shape != (h, w, 3)):
            raise ValueError("V5 regional model input must be native float32 RGB")
        return rgb
    with Image.open(resolve_review_path(path)) as im:
        return np.asarray(im.convert("RGB").crop((x, y, x+w, y+h)))


def native_domain(job, native):
    descriptor = native.get("domain") or job.get("phase_transfer_context")
    if descriptor and descriptor.get("path"):
        path = resolve_review_path(descriptor["path"])
        if sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"Native domain changed: {path}")
        with np.load(path, allow_pickle=False) as a:
            domain = a["domain"].astype(bool)
            spacing = (abs(float(a['y_m'][1]-a['y_m'][0])), abs(float(a['x_m'][1]-a['x_m'][0])))
    else:
        if job["dataset"] != "ffum":
            raise ValueError("A series-specific native domain is required")
        # Historical FF_UM manifests explicitly name serialized rig facies.
        # The original rig supplies both the domain and physical pixel size.
        path = Path(r"\\UNAS-Pro-8\UiB_FluidFlower\FF_UM\Analysis\V1\Results") / job["run"].capitalize() / "results/setup/rig/facies.npz"
        with np.load(path, allow_pickle=True) as a:
            domain = a["array"] != 1
            meta = a["metadata"].item()
            spacing = tuple(np.asarray(meta["dimensions"])/domain.shape)
    if tuple(domain.shape) != tuple(native["analysis_image_shape_hw"]):
        raise ValueError("Historical facies and native photo geometry differ")
    return domain, spacing, dict(path=str(path), sha256=sha256_file(path))


def v5_frame_domain(native, cohort, frame):
    """Keep observation/overflow separate from supported sand physics."""
    arrays, binding = cohort.context(frame, fields=(
        "observable_phase_domain", "physical_sand_domain",
        "physics_supported_domain", "facies_safe_interior",
        "photo_certified_sand_domain", "photo_boundary_uncertain_domain"))
    descriptor = native.get("domain") or {}
    path = resolve_review_path(descriptor.get("path") or "")
    if (not path.is_file() or not descriptor.get("sha256")
            or sha256_file(path) != descriptor["sha256"]):
        raise ValueError("V5 nominal physical spacing context changed")
    with np.load(path, allow_pickle=False) as archive:
        x_m = np.asarray(archive["x_m"], np.float64)
        y_m = np.asarray(archive["y_m"], np.float64)
    if x_m.ndim != 1 or y_m.ndim != 1 or min(len(x_m), len(y_m)) < 2:
        raise ValueError("V5 nominal optical spacing requires 1D axes")
    spacing = (abs(float(y_m[1] - y_m[0])),
               abs(float(x_m[1] - x_m[0])))
    supported = arrays["physics_supported_domain"]
    physical = arrays["physical_sand_domain"]
    observable = arrays["observable_phase_domain"]
    safe = arrays["facies_safe_interior"]
    photo_sand = arrays["photo_certified_sand_domain"]
    photo_uncertain = arrays["photo_boundary_uncertain_domain"]
    if (supported.shape != tuple(native["analysis_image_shape_hw"])
            or physical.shape != supported.shape or observable.shape != supported.shape
            or np.any(supported & ~physical) or np.any(physical & ~observable)
            or np.any(safe & ~physical) or np.any(photo_sand & photo_uncertain)
            or not np.isfinite(spacing).all() or min(spacing) <= 0):
        raise ValueError("V5 frame domain or nominal optical spacing is invalid")
    return arrays, spacing, dict(
        policy="v5-physical-supported-and-observable-domains-v1",
        sha256=binding["receipt_sha256"],
        frame_geometry_receipt_sha256=binding["receipt_sha256"],
        context_sha256=binding["reference_context_sha256"],
        photo_sand_mask_sha256=binding["photo_certified_sand_mask_sha256"],
        nominal_spacing_context_sha256=descriptor["sha256"])


def track1_frame_domain(native, cohort, frame, *, nominal_profile=None,
                        nominal_profile_root=None):
    """A source-photo sand core and nominal spacing, without physical facies."""
    sand, unresolved, binding = cohort.photo(frame)
    descriptor = (native.get("v5_photometry") or {}).get("original_domain")
    if descriptor is None:
        # Six older source revisions have no domain NPZ. Their nominal spacing
        # comes from the immutable optical-only parent context for this run;
        # neither facies nor physical coordinates are transferred from it.
        scope = (nominal_profile or {}).get("v5_track1_review_scope") or {}
        run = str(native.get("run") or "").casefold()
        entry = ((nominal_profile or {}).get("runs") or {}).get(f"ffac/{run}") or {}
        marker = (entry.get("provenance") or {}).get("v5_track1_photo_geometry") or {}
        if (scope.get("contract") != "ffac-v5-track1-photo-only-review-context-v1"
                or scope.get("mass_analysis_allowed") is not False
                or marker.get("contract") !=
                    "ffac-v5-track1-photo-geometry-cohort-v1"
                or marker.get("validation_only") is not True
                or marker.get("nominal_axes_only") is not True
                or not entry.get("path") or not entry.get("sha256")
                or nominal_profile_root is None):
            raise ValueError("Track1 historical nominal spacing lacks a frozen optical-only parent")
        parent = Path(nominal_profile_root).resolve()
        path = (parent / entry["path"]).resolve()
        if not path.is_relative_to(parent):
            raise ValueError("Track1 nominal context escaped its frozen parent")
        descriptor = {"sha256": entry["sha256"]}
        spacing_source = "frozen-parent-optical-only-nominal-context"
    else:
        path = resolve_review_path(descriptor.get("path") or "")
        spacing_source = "original-source-domain-nominal-axes"
    if (not path.is_file() or not descriptor.get("sha256") or
            sha256_file(path) != descriptor["sha256"]):
        raise ValueError("Track1 nominal source spacing context changed")
    with np.load(path, allow_pickle=False) as archive:
        x_m = np.asarray(archive["x_m"], np.float64)
        y_m = np.asarray(archive["y_m"], np.float64)
    if (x_m.ndim != 1 or y_m.ndim != 1 or min(len(x_m), len(y_m)) < 2
            or sand.shape != tuple(native["analysis_image_shape_hw"])):
        raise ValueError("Track1 nominal axes or native photo geometry changed")
    spacing = (abs(float(y_m[1] - y_m[0])),
               abs(float(x_m[1] - x_m[0])))
    if not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Track1 nominal optical pixel spacing is invalid")
    return dict(observable_phase_domain=np.ones_like(sand),
                photo_certified_sand_domain=sand,
                photo_boundary_uncertain_domain=unresolved), spacing, dict(
                    policy="ffac-v5-track1-source-photo-optical-only-v1",
                    sha256=binding["receipt_sha256"],
                    frame_geometry_receipt_sha256=binding["receipt_sha256"],
                    photo_sand_mask_sha256=binding[
                        "photo_sand_reference_mask_sha256"],
                    nominal_spacing_context_sha256=descriptor["sha256"],
                    nominal_spacing_source=spacing_source,
                    physical_scale_qa_pass=False, mass_eligible=False)


def groups(evidence, *, excluded_runs=(), fit_partition_rows=None):
    excluded = {str(run).casefold() for run in excluded_runs}
    fitted = [record for record in evidence["regions"]
              if record["run"].casefold() not in excluded
              and (fit_partition_rows is None or
                   fit_partition_rows[record["region_id"]]["role"] == "fit")]
    logistic.assert_safe_training_regions(fitted)
    origins = read(evidence["audit"]["annotation_origins"]["path"])["regions"]
    manifests, native_cache = {}, {}
    grouped = {}
    for index, record in enumerate(evidence["regions"]):
        # Chemistry-shifted runs remain frozen review evidence, but none of
        # their image sources may be opened for this light-only fit.
        if record["run"].casefold() in excluded:
            continue
        if (fit_partition_rows is not None and
                fit_partition_rows[record["region_id"]]["role"] != "fit"):
            continue
        origin = origins[record["region_id"]]
        document = origin["document_index"]
        if document not in manifests:
            manifests[document] = read(evidence["pairs"][document][1])
        jobs = [j for j in manifests[document]["jobs"] if j["run"].lower() == record["run"]]
        source_key = str(jobs[0].get("analysis_source_review_root") or jobs[0]["review_output_dir"]) + ":" + record["run"]
        if source_key not in native_cache:
            native_cache[source_key] = native_source(jobs[0])
        source, path, native = native_cache[source_key]
        if (fit_partition_rows is not None and
                (native.get("v5_photometry") or {}).get("source_manifest_sha256") !=
                fit_partition_rows[record["region_id"]]["source_manifest_sha256"]):
            raise ValueError("Track1 regional fit record belongs to another source revision")
        frame = record["frame_index"]
        if not 0 <= frame < native["frame_count"]:
            raise ValueError("Annotation frame is outside its original timeline")
        timeline = native["frame_sources"][frame]
        if abs(float(timeline["elapsed_h"])-float(record["elapsed_h"])) > 1e-5:
            matches = [i for i, item in enumerate(native['frame_sources'])
                       if abs(float(item['elapsed_h'])-float(record['elapsed_h'])) <= 1e-5]
            if len(matches) != 1:
                raise ValueError(f"Annotation has no unique original timestamp: {record['region_id']}")
            # Legacy review timelines omitted an initial reference frame.
            # Keep the original canvas and use its unique exact elapsed time.
            frame = matches[0]
        key = (str(path), frame)
        if key not in grouped:
            grouped[key] = dict(source=source, native_path=path, native=native,
                                job=jobs[0], frame=frame, records=[], timeline_alignments=[])
        if frame != record['frame_index']:
            grouped[key]['timeline_alignments'].append(dict(region_id=record['region_id'],
                original_index=record['frame_index'], resolved_index=frame,
                elapsed_h=record['elapsed_h'], policy='unique-exact-time-on-original-native-canvas'))
        grouped[key]["records"].append((index, record))
    return [grouped[k] for k in sorted(grouped)]


def small_region_samples(prepared, record, native, native_domain, windows, restrictions):
    """Sample inside the original native mask when no coarse cell is selected.

    Interpolate image features at native pixel centers, never enlarge labels to
    the surrounding 1 mm cell. Check contradictory annotations at those same
    native pixels before assigning their original region owner.
    """
    bounds, mask = annotation_window(record['payload'], canvas_wh=native['canvas_wh'],
                                     content_box=native['content_box_xywh'])
    x0,y0,x1,y1=bounds
    mask &= native_domain[y0:y1,x0:x1]
    allowed=np.full(mask.shape,7,np.uint8)
    for other, (ox0,oy0,ox1,oy1) in windows:
        if other['confidence']=='low' or other['phase_class'] not in restrictions:
            continue
        ax,ay,bx,by=max(x0,ox0),max(y0,oy0),min(x1,ox1),min(y1,oy1)
        if ax>=bx or ay>=by:
            continue
        _,other_mask=annotation_window(other['payload'],canvas_wh=native['canvas_wh'],
                                       content_box=native['content_box_xywh'])
        view=allowed[ay-y0:by-y0,ax-x0:bx-x0]
        view[other_mask[ay-oy0:by-oy0,ax-ox0:bx-ox0]] &= restrictions[other['phase_class']]
    conflict=mask & (allowed==0)
    rows,cols=np.nonzero(mask & ~conflict)
    h,w=prepared['domain'].shape;nh,nw=native_domain.shape
    yy=np.clip((rows+y0+.5)*h/nh-.5,0,h-1)
    xx=np.clip((cols+x0+.5)*w/nw-.5,0,w-1)
    # Features next to an excluded domain must not interpolate across it.
    ya,xa=np.floor(yy).astype(int),np.floor(xx).astype(int)
    yb,xb=np.minimum(ya+1,h-1),np.minimum(xa+1,w-1)
    valid=(prepared['domain'][ya,xa] & prepared['domain'][ya,xb]
           & prepared['domain'][yb,xa] & prepared['domain'][yb,xb])
    yy,xx,ya,xa,yb,xb=(a[valid] for a in (yy,xx,ya,xa,yb,xb))
    if len(yy)>512:
        seed=int.from_bytes(hashlib.sha256(record['region_id'].encode()).digest()[:8],'little')
        take=np.random.default_rng(seed).choice(len(yy),512,replace=False)
        yy,xx,ya,xa,yb,xb=(a[take] for a in (yy,xx,ya,xa,yb,xb))
    dy,dx=(yy-ya).astype(np.float32)[:,None],(xx-xa).astype(np.float32)[:,None]
    features=prepared['features']
    values=((1-dy)*((1-dx)*features[ya,xa]+dx*features[ya,xb])
            +dy*((1-dx)*features[yb,xa]+dx*features[yb,xb]))
    return values,dict(native_pixels_in_domain=int(mask.sum()),
                       conflicting_native_pixels=int(conflict.sum()),
                       sampling_method='native-points-with-interpolated-regional-features')


def region_samples(prepared, records, native, *, native_domain=None,
                   observable_domain=None, physical_domain=None,
                    feature_support_domain=None, photo_sand_domain=None,
                    photo_uncertain_domain=None):
    """Mask only pixels with contradictory dense/absence phase constraints."""
    shape = prepared["domain"].shape
    if native_domain is not None and tuple(native_domain.shape)!=tuple(native['analysis_image_shape_hw']):
        raise ValueError('Native sampling domain and annotation geometry differ')
    allowed = np.full(shape, 7, np.uint8)  # water, aqueous, gas bits
    restrictions = dict(water=1, aqueous=2, gas=4, co2_absent_region=1,
                        current_experiment_co2_artifact=1, gas_absent_region=3,
                        aqueous_absent_region=5, aqueous_present_gas_absent_region=3,
                        mixed_aqueous_water=3)
    windows, native_windows = [], []
    for index, record in records:
        bounds, mask = annotation_window(record["payload"], canvas_wh=native["canvas_wh"],
                                          content_box=native["content_box_xywh"])
        pooled, (x0,y0,x1,y1) = pool_annotation(mask, bounds, native["analysis_image_shape_hw"], shape)
        native_windows.append((record,bounds))
        selected = (pooled > .5) & prepared["domain"][y0:y1,x0:x1]
        windows.append((index, record, (x0,y0,x1,y1), selected, not (pooled>.5).any()))
        if record["confidence"] != "low" and record["phase_class"] in restrictions:
            view = allowed[y0:y1,x0:x1]
            view[selected] &= restrictions[record["phase_class"]]
    features, owners, coverage = [], [], []
    for index, record, (x0,y0,x1,y1), selected, below_grid in windows:
        conflict = selected & (allowed[y0:y1,x0:x1] == 0)
        rows, cols = np.nonzero(selected & ~conflict)
        if record["confidence"] == "low":
            rows, cols = rows[:0], cols[:0]
        if len(rows) > 512:
            seed = int.from_bytes(hashlib.sha256(record["region_id"].encode()).digest()[:8], "little")
            take = np.random.default_rng(seed).choice(len(rows), 512, replace=False)
            rows, cols = rows[take], cols[take]
        values=prepared["features"][rows+y0, cols+x0]
        sampling=dict(sampling_method='majority-covered-regional-cells')
        # Fallback applies only to geometric loss, never to low confidence,
        # domain exclusion or a coarse-grid contradiction.
        if below_grid and record['confidence']!='low' and native_domain is not None:
            values,sampling=small_region_samples(prepared,record,native,native_domain,
                                                native_windows,restrictions)
        if len(values):
            features.append(values)
            owners.append(np.full(len(values), index, np.int32))
        support = {}
        if observable_domain is not None:
            (nx0, ny0, nx1, ny1), native_mask = annotation_window(
                record["payload"], canvas_wh=native["canvas_wh"],
                content_box=native["content_box_xywh"])
            observable_part = observable_domain[ny0:ny1, nx0:nx1]
            supported_part = (physical_domain if physical_domain is not None
                              else native_domain)[ny0:ny1, nx0:nx1]
            if np.any(native_mask & ~observable_part):
                raise ValueError(f"{record['region_id']}: source label leaves observable image")
            support = dict(label_pixels_observable=int(np.count_nonzero(native_mask)),
                label_pixels_physics_supported=int(np.count_nonzero(
                    native_mask & supported_part)),
                label_pixels_optical_only=int(np.count_nonzero(
                    native_mask & ~supported_part)))
            if photo_sand_domain is not None:
                sand_part = photo_sand_domain[ny0:ny1, nx0:nx1]
                support.update(label_pixels_photo_certified_sand=int(np.count_nonzero(
                    native_mask & sand_part)),
                    label_pixels_off_certified_sand=int(np.count_nonzero(
                    native_mask & ~sand_part)))
                if photo_uncertain_domain is not None:
                    support["label_pixels_photo_boundary_uncertain"] = int(
                        np.count_nonzero(native_mask & photo_uncertain_domain[
                            ny0:ny1, nx0:nx1]))
            if feature_support_domain is not None:
                feature_part = feature_support_domain[ny0:ny1, nx0:nx1]
                support.update(label_pixels_feature_supported=int(np.count_nonzero(
                    native_mask & feature_part)),
                    label_pixels_regional_supported=int(np.count_nonzero(
                    native_mask & feature_part & supported_part)))
                if photo_sand_domain is not None:
                    support["label_pixels_photo_and_feature_supported"] = int(
                        np.count_nonzero(native_mask & feature_part & sand_part))
        coverage.append(dict(region_id=record["region_id"], run=record["run"],
            frame_index=record["frame_index"], phase_class=record["phase_class"],
            selected_working_pixels=int(selected.sum()), conflicting_working_pixels=int(conflict.sum()),
            training_rows=len(values), confidence=record["confidence"], **sampling,
            **support,
            status="included" if len(values) else "no-unambiguous-supervision"))
    return (np.concatenate(features) if features else np.empty((0,len(regional.FEATURE_NAMES)),np.float32),
            np.concatenate(owners) if owners else np.empty(0,np.int32), coverage)


def extract(root, cache, *, threads=None, feature_cache_root=None, backend='cuda'):
    from darsia.fluidflower.tools.phase_inference_benchmark_backends import (
        CudaFeaturePredictor, CpuFeaturePredictor)
    from darsia.fluidflower.tools.ffac_v5_frame_context import V5FrameGeometryCohort
    from darsia.fluidflower.tools.ffac_v5_track1_photo_context import (
        Track1FramePhotoCohort,
    )
    from darsia.fluidflower.tools.ffac_v5_temporal_alignment import align_temporal_inputs
    from darsia.fluidflower.phase_validation import _model_identity
    from darsia.fluidflower.phase_execution_resources import available_resources
    from darsia.fluidflower.tools.phase_region_feature_cache import FeatureCache
    from darsia.fluidflower.tools.phase_studio_execution import CpuPredictor
    if backend not in ('cpu', 'cuda'):
        raise ValueError('Choose cpu or cuda for extraction')
    threads = threads or int(os.environ.get('PHASE_CPU_LIMIT', available_resources()[0]))
    cv2.setNumThreads(threads)
    evidence = read(root/"evidence.json")
    chemistry_scope = evidence.get("audit", {}).get("v5_chemistry_fit_scope")
    v5_binding = evidence.get("audit", {}).get("v5_raw_source_binding")
    partition_binding = evidence.get("audit", {}).get("v5_review_fit_partition")
    fit_partition_rows = None
    if partition_binding is not None:
        if v5_binding is None:
            raise ValueError("Track1 review partition requires V5 float source binding")
        from darsia.fluidflower.phase_training.v5_review_fit_partition import (
            verify_partition_rows,
        )
        fit_partition_rows = verify_partition_rows(
            partition_binding, evidence["regions"])
    if v5_binding is not None:
        if chemistry_scope is None:
            raise ValueError("V5 regional extraction omitted frozen chemistry exclusion")
        from darsia.fluidflower.phase_training.v5_chemistry_fit_scope import verify_scope
        excluded_runs = verify_scope(chemistry_scope, evidence["regions"])
    elif chemistry_scope is not None:
        raise ValueError("Chemistry fit exclusion without V5 source binding")
    else:
        excluded_runs = set()
    model = HierarchicalPhaseModel.load(Path(evidence["parent"]))
    settings = regional.Settings()
    code = {str(p):sha256_file(p) for p in (Path(__file__),Path(regional.__file__))}
    plan = groups(evidence, excluded_runs=excluded_runs,
                  fit_partition_rows=fit_partition_rows)
    if not plan:
        raise ValueError("Chemistry exclusion leaves no regional source frames to extract")
    v5_frames = sum((group["native"].get("v5_photometry") or {}).get("format") ==
                    "ffac-v5-native-source-materialization-v1" for group in plan)
    track1_frames = any(group["native"].get("v5_track1_photo_geometry")
                        for group in plan)
    nominal_profile_root = Path(evidence["parent"]).resolve()
    nominal_profile_path = nominal_profile_root / "phase_optical_regions.json"
    nominal_profile = read(nominal_profile_path) if track1_frames else None
    from darsia.fluidflower import phase_features, phase_learning
    from darsia.fluidflower.tools import phase_inference_benchmark_backends as optical
    image_cache = FeatureCache(feature_cache_root or cache.parent / 'regional_image_features')
    import PIL, scipy
    runtime = dict(numpy=np.__version__, opencv=cv2.__version__, pillow=PIL.__version__, scipy=scipy.__version__)
    if backend == 'cuda':
        import cupy as cp
        runtime.update(cupy=cp.__version__, cuda=cp.cuda.runtime.runtimeGetVersion(),
                       driver=cp.cuda.runtime.driverGetVersion(),
                       device=str(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']))
    shared_identity = dict(settings=asdict(settings), feature_names=list(regional.FEATURE_NAMES),
        preprocessing=hashlib.sha256((inspect.getsource(crop)+inspect.getsource(native_domain)).encode()).hexdigest(),
        weights={head:sha256_file(Path(evidence['parent'])/'models/generic'/f'{head}_signal.json')
                 for head in ('co2_presence','gas_within_co2')},
        code={Path(m.__file__).name:sha256_file(Path(m.__file__))
              for m in (regional, optical, phase_features, phase_learning)},
        libraries=runtime, backend=backend)
    if backend == 'cpu':
        from darsia.fluidflower.tools import phase_studio_execution, phase_optical_exact_tiles
        from darsia.fluidflower import phase_validation
        for module in (phase_studio_execution, phase_optical_exact_tiles, phase_validation):
            shared_identity['code'][Path(module.__file__).name] = sha256_file(Path(module.__file__))
    model_id, model_sha = _model_identity(Path(evidence["parent"]))
    protocol = dict(format=regional.VERSION, settings=asdict(settings), feature_names=list(regional.FEATURE_NAMES),
        parent_model_id=model_id,parent_model_sha256=model_sha,evidence_sha256=sha256_file(root/"evidence.json"),
        code_sha256=code,frames=len(plan),regions=len(evidence["regions"]),
        validation_scope="Regional head experiment. Frozen optical base has seen historical training runs; no independent generalization claim.",
        native_revision_policy="Each annotation remains on its own source revision; no last-per-run image lookup",
        context_maps_used=bool(v5_frames),raw_predictions_are_labels=False,
        backend=backend+" shared optical features",
        v5_frame_geometry_policy=("source-sha-bound-native-domain-and-temporal-refs-v1"
                                  if v5_frames else None),
        conflict_policy="Defer ambiguous revisions and mask contradictory dense overlap; preserve source labels")
    if chemistry_scope is not None:
        protocol["v5_chemistry_fit_scope"] = chemistry_scope
    if partition_binding is not None:
        protocol["v5_review_fit_partition"] = partition_binding
    if track1_frames:
        protocol["v5_track1_nominal_profile_sha256"] = sha256_file(
            nominal_profile_path)
    if (root/"protocol.json").exists() and read(root/"protocol.json") != protocol:
        raise ValueError("Extraction protocol changed; start a new experiment")
    write(root/"protocol.json",protocol)
    all_x, all_owners, all_coverage, receipts = [], [], [], []

    def finish_frame(*, prepared, current, baseline, scores, domain,
                     observable_domain, physical_domain, feature_support_domain,
                      photo_sand_domain, photo_uncertain_domain,
                     spacing,
                     image_identity, feature_cache_hit, group, native, run, frame,
                     inputs, samples, receipt_path, started, domain_ref):
        # Only CPU work runs here. The caller remains the sole CUDA owner.
        if prepared is None:
            prepared = regional.prepare(current, baseline, scores['co2_presence_probability'],
                scores['gas_probability_conditional_on_co2'], domain, spacing, settings)
            image_cache.save(image_identity, prepared)
        x, owners, coverage = region_samples(prepared, group['records'], native,
                                             native_domain=domain,
                                             observable_domain=observable_domain,
                                             physical_domain=physical_domain,
                                             feature_support_domain=feature_support_domain,
                                             photo_sand_domain=photo_sand_domain,
                                             photo_uncertain_domain=photo_uncertain_domain)
        if ((native.get('v5_photometry') or {}).get('format') ==
                'ffac-v5-native-source-materialization-v1'
                and x.dtype != np.dtype('float32')):
            raise ValueError('V5 regional model features must be float32')
        np.savez_compressed(samples, features=x, owners=owners)
        receipt = dict(inputs_sha256=inputs, domain=domain_ref, run=run,
            frame_index=frame, timeline_alignments=group.get('timeline_alignments', []),
            native_manifest=str(group['native_path']),
            source_name=native['frame_sources'][frame].get('source_name'),
            records=coverage, samples=str(samples), samples_sha256=sha256_file(samples),
            change_floor=prepared['change_floor'], seconds=time.monotonic()-started,
            image_feature_cache_hit=feature_cache_hit,
            image_feature_cache=str(image_cache.folder(image_identity)))
        write(receipt_path, receipt)
        del prepared, domain
        return x, owners, receipt

    completed_uncached = 0
    def accept(result, index, run, frame, collect=False):
        nonlocal completed_uncached
        x, owners, receipt = result
        all_x.append(x); all_owners.append(owners)
        all_coverage.extend(receipt['records']); receipts.append(receipt)
        write(root/'status.json', dict(stage='extracting', frames_done=index+1,
                                       frames_total=len(plan), run=run, frame=frame))
        print(json.dumps(dict(stage='extracting', done=index+1, total=len(plan),
                              run=run, frame=frame, rows=len(x))), flush=True)
        if collect:
            # Explicit collections remain on the CUDA-owning caller thread,
            # but NumPy array refcounts handle the ordinary per-frame release.
            completed_uncached += 1
            if completed_uncached % GC_COLLECT_INTERVAL == 0:
                gc.collect()

    predictor = None
    previous_source = None
    baseline = None
    last_decoded_current_path = None
    last_decoded_current = None
    pending = None
    v5_cohorts = {}
    v5_inputs = {}
    extended_baseline_cache = {}
    pool = (ThreadPoolExecutor(max_workers=1, thread_name_prefix='regional-cpu')
            if backend == 'cuda' else nullcontext(None))
    with pool as cpu_worker:
      for n, group in enumerate(plan):
        source, native, frame, job = (group[k] for k in ("source","native","frame","job"))
        box=native["content_box_xywh"]; run=job["run"]
        current_path=source/"native_frames"/run/f"frame_{frame:03d}.jpg"
        baseline_path=source/"native_references"/run/"pre_injection.jpg"
        excluded=set(job.get("independent_reference_frames") or [])
        earlier=[i for i in range(frame) if i not in excluded]
        previous_path=source/"native_frames"/run/f"frame_{earlier[-1]:03d}.jpg" if earlier else None
        v5 = (native.get("v5_photometry") or {}).get("format") == "ffac-v5-native-source-materialization-v1"
        temporal_mode = (native.get("v5_temporal_reference") or {}).get("contract") if v5 else None
        masked_temporal = temporal_mode in (
            "ffac-v5-native-crop-source-supported-v1",
            "ffac-v5-source-backed-extended-ring-v1")
        if v5 and temporal_mode not in (None,
                "ffac-v5-native-crop-source-supported-v1",
                "ffac-v5-source-backed-extended-ring-v1"):
            raise ValueError("Unknown V5 regional temporal source contract")
        cohort = None
        frame_binding = None
        previous_binding = None
        feature_support_domain = None
        feature_support_receipt = None
        linear_input = None
        if v5:
            native_path = Path(group["native_path"])
            cohort = v5_cohorts.get(native_path)
            if cohort is None:
                cohort = (Track1FramePhotoCohort(native_path)
                          if native.get("v5_track1_photo_geometry") else
                          V5FrameGeometryCohort(native_path))
                v5_cohorts[native_path] = cohort
            if isinstance(cohort, Track1FramePhotoCohort) and fit_partition_rows is None:
                raise ValueError("Track1 regional fit omitted 4048-ID partition")
            frame_binding = cohort.frame(frame)
            previous_binding = cohort.frame(earlier[-1]) if earlier else None
            if masked_temporal:
                from darsia.fluidflower.phase_training.v5_linear_input import RunV5LinearInput
                linear_input = v5_inputs.get(native_path)
                if linear_input is None:
                    linear_input = RunV5LinearInput(
                        native_manifest_path=native_path, run_id=run,
                        content_shape_hw=tuple(native["analysis_image_shape_hw"]))
                    v5_inputs[native_path] = linear_input
                feature_support_domain, feature_support_receipt = cohort.feature_support(
                    frame, earlier[-1] if earlier else None, linear_input)
        inputs={str(p):sha256_file(p) for p in (group["native_path"],baseline_path,current_path,*([previous_path] if previous_path else []))}
        if v5:
            for photo in (baseline_path, current_path,
                          *([previous_path] if previous_path else [])):
                sidecar, sidecar_sha, valid_path, valid_sha = _v5_linear_asset(
                    photo, native)
                inputs[str(sidecar)] = sidecar_sha
                if valid_path is not None:
                    inputs[str(valid_path)] = valid_sha
            inputs[frame_binding["receipt_path"]] = frame_binding["receipt_sha256"]
            if isinstance(cohort, Track1FramePhotoCohort):
                inputs[frame_binding["photo_sand_reference_mask_path"]] = (
                    frame_binding["photo_sand_reference_mask_sha256"])
                inputs[frame_binding["photo_sand_reference_receipt_path"]] = (
                    frame_binding["photo_sand_reference_receipt_sha256"])
            else:
                inputs[frame_binding["reference_context_path"]] = (
                    frame_binding["reference_context_sha256"])
            if previous_binding is not None:
                inputs[previous_binding["receipt_path"]] = previous_binding["receipt_sha256"]
            if feature_support_receipt is not None:
                inputs["v5_feature_support_receipt_sha256"] = feature_support_receipt[
                    "receipt_sha256"]
        key=digest(dict(protocol=protocol,inputs=inputs,records=group["records"]))
        folder=cache/key;folder.mkdir(parents=True,exist_ok=True)
        samples=folder/"samples.npz";receipt_path=folder/"receipt.json"
        if samples.is_file() and receipt_path.is_file():
            receipt=read(receipt_path)
            if sha256_file(samples)!=receipt["samples_sha256"]:raise ValueError("Training cache changed")
            with np.load(samples) as a:x,owners=a["features"],a["owners"]
            if v5 and x.dtype != np.dtype('float32'):
                raise ValueError('V5 cached regional model features must be float32')
            if pending is not None:
                accept(pending[0].result(), *pending[1:])
                pending = None
            accept((x, owners, receipt), n, run, frame)
        else:
            started=time.monotonic()
            if v5:
                if isinstance(cohort, Track1FramePhotoCohort):
                    domains, spacing, domain_ref = track1_frame_domain(
                        native, cohort, frame,
                        nominal_profile=nominal_profile,
                        nominal_profile_root=nominal_profile_root)
                    photo_sand_domain = domains["photo_certified_sand_domain"]
                    photo_uncertain_domain = domains["photo_boundary_uncertain_domain"]
                    physical_domain = photo_sand_domain
                else:
                    domains, spacing, domain_ref = v5_frame_domain(
                        native, cohort, frame)
                    photo_sand_domain = domains["photo_certified_sand_domain"]
                    photo_uncertain_domain = domains["photo_boundary_uncertain_domain"]
                    physical_domain = (domains["physics_supported_domain"]
                                       & domains["facies_safe_interior"]
                                       & photo_sand_domain)
                domain = (physical_domain & feature_support_domain
                          if feature_support_domain is not None else physical_domain)
                observable_domain = domains["observable_phase_domain"]
            else:
                domain, spacing, domain_ref = native_domain(job, native)
                observable_domain = domain
                physical_domain = domain
                photo_sand_domain = None
                photo_uncertain_domain = None
            if v5:
                for _owner, record in group["records"]:
                    (x0,y0,x1,y1), label_mask = annotation_window(
                        record["payload"], canvas_wh=native["canvas_wh"],
                        content_box=native["content_box_xywh"])
                    if np.any(label_mask & ~observable_domain[y0:y1,x0:x1]):
                        raise ValueError(
                            f"{record['region_id']}: V5 label leaves its observable frame")
                    if feature_support_domain is not None:
                        original = int(np.count_nonzero(label_mask))
                        sand_eligible = int(np.count_nonzero(
                            label_mask & photo_sand_domain[y0:y1,x0:x1]))
                        retained = int(np.count_nonzero(
                            label_mask & photo_sand_domain[y0:y1,x0:x1]
                            & feature_support_domain[y0:y1,x0:x1]))
                        if sand_eligible == 0:
                            raise ValueError(
                                f"{record['region_id']}: no photo-certified sand "
                                f"inside {original} reviewed pixels")
                        if retained < .90 * sand_eligible:
                            raise ValueError(
                                f"{record['region_id']}: V5 regional source evidence "
                                f"retains {retained}/{sand_eligible} certified sand "
                                f"pixels (<90%); whole annotation={original}")
            image_identity = dict(shared_identity, inputs=inputs, content_box=box,
                domain_sha256=domain_ref['sha256'], spacing_m=list(spacing))
            if v5:
                image_identity["v5_frame_alignment"] = dict(
                    input_encoding="v5-unit-float32",
                    frame_geometry_receipt_sha256=frame_binding["receipt_sha256"],
                    previous_geometry_receipt_sha256=(
                        None if previous_binding is None else
                        previous_binding["receipt_sha256"]),
                    source_code_sha256=sha256_file(Path(__file__)),
                    frame_context_code_sha256=sha256_file(
                        Path(sys.modules[type(cohort).__module__].__file__)),
                    temporal_alignment_code_sha256=sha256_file(
                        Path(sys.modules[align_temporal_inputs.__module__].__file__)),
                    temporal_reference_contract=temporal_mode,
                    feature_support_receipt_sha256=(None if feature_support_receipt is None
                        else feature_support_receipt["receipt_sha256"]),
                )
            prepared = image_cache.load(image_identity)
            feature_cache_hit = prepared is not None
            current = scores = None
            if not feature_cache_hit:
                predictor_source = (str(source), run, inputs[str(baseline_path)],
                                    tuple(box), "v5-float" if v5 else "legacy-review")
                if previous_source != predictor_source:
                    if predictor is not None:
                        del predictor
                        if backend == 'cuda':
                            cp.get_default_memory_pool().free_all_blocks()
                    baseline=crop(baseline_path,box,native)
                    if v5:
                        predictor=(CudaFeaturePredictor(model,baseline,
                                   input_encoding='v5-unit-float32') if backend == 'cuda'
                                   else CpuFeaturePredictor(model,baseline,
                                   input_encoding='v5-unit-float32'))
                    else:
                        predictor=(CudaFeaturePredictor(model,baseline) if backend == 'cuda'
                                   else CpuPredictor(model,baseline,{},threads=threads))
                    previous_source=predictor_source
                current=crop(current_path,box,native)
                # Most reviewed frames are consecutive. The prior frame's
                # exact corrected RGB was already decoded as `current`; keep
                # one frame only, keyed by its full source path. This also
                # works on CPU-only hosts and cannot cross source revisions.
                previous=(last_decoded_current if previous_path == last_decoded_current_path
                          else crop(previous_path,box,native) if previous_path else None)
                if v5:
                    for role, value in (("baseline", baseline),
                                        ("current", current),
                                        ("previous", previous)):
                        if value is not None and (not isinstance(value, np.ndarray)
                            or value.dtype != np.dtype('float32')
                            or value.ndim != 3 or value.shape[2] != 3):
                            raise ValueError(
                                f'V5 regional {role} model input must be native float32 RGB')
                    baseline_reference = baseline
                    previous_reference = previous
                    baseline_origin = previous_origin = (0, 0)
                    baseline_source_valid = previous_source_valid = None
                    if temporal_mode == "ffac-v5-source-backed-extended-ring-v1":
                        cache_key = (native_path, str(baseline_path))
                        cached = extended_baseline_cache.get(cache_key)
                        if cached is None:
                            cached = linear_input.load_extended(
                                baseline_path, "pre_injection")
                            extended_baseline_cache.clear()
                            extended_baseline_cache[cache_key] = cached
                        baseline_reference, baseline_origin, baseline_source_valid = cached
                        if previous_path is not None:
                            previous_reference, previous_origin, previous_source_valid = (
                                linear_input.load_extended(
                                    previous_path, f"frame_{earlier[-1]:03d}"))
                    aligned_baseline, aligned_previous, temporal_qa = align_temporal_inputs(
                        baseline=baseline_reference, current=current,
                        previous=previous_reference,
                        h_baseline_to_current=cohort.H_v5_B_to_frame(frame),
                        h_baseline_to_previous=(cohort.H_v5_B_to_frame(earlier[-1])
                                                if earlier else None),
                        required_valid_mask=(feature_support_domain
                            if feature_support_domain is not None else domain),
                        baseline_origin_xy=baseline_origin,
                        previous_origin_xy=previous_origin,
                        baseline_source_valid_mask=baseline_source_valid,
                        previous_source_valid_mask=previous_source_valid)
                    for role, value in (("aligned baseline", aligned_baseline),
                                        ("aligned previous", aligned_previous)):
                        if value is not None and (not isinstance(value, np.ndarray)
                            or value.dtype != np.dtype('float32')
                            or value.ndim != 3 or value.shape[2] != 3):
                            raise ValueError(
                                f'V5 regional {role} model input must be native float32 RGB')
                    predictor.set_baseline(aligned_baseline)
                    scores=predictor.predict(current,aligned_previous,
                                             current_key=frame,previous_key=None)
                    baseline_for_frame=aligned_baseline
                else:
                    temporal_qa=None
                    baseline_for_frame=baseline
                    scores=(predictor.predict(current,previous,current_key=frame,
                                              previous_key=earlier[-1] if earlier else None)
                            if backend == 'cuda' else predictor(current, previous))
                last_decoded_current_path, last_decoded_current = current_path, current
            else:
                baseline_for_frame=baseline
                temporal_qa=None
            work = dict(prepared=prepared, current=current,
                baseline=baseline_for_frame, scores=scores,
                domain=domain, observable_domain=observable_domain,
                physical_domain=physical_domain,
                feature_support_domain=feature_support_domain,
                photo_sand_domain=photo_sand_domain,
                photo_uncertain_domain=photo_uncertain_domain,
                spacing=spacing, image_identity=image_identity,
                feature_cache_hit=feature_cache_hit, group=group, native=native, run=run,
                frame=frame, inputs=inputs, samples=samples, receipt_path=receipt_path,
                started=started, domain_ref={**domain_ref,
                                             **({} if feature_support_receipt is None else
                                                {"feature_support": feature_support_receipt}),
                                             **({} if temporal_qa is None else
                                                {"temporal_alignment":temporal_qa})})
            if backend == 'cuda':
                # Compute the next frame on the sole CUDA thread while one CPU
                # worker finishes the previous frame; commit in plan order.
                if pending is not None:
                    accept(pending[0].result(), *pending[1:])
                pending = (cpu_worker.submit(finish_frame, **work), n, run, frame, True)
            else:
                accept(finish_frame(**work), n, run, frame, True)
      if pending is not None:
          accept(pending[0].result(), *pending[1:])
    if completed_uncached % GC_COLLECT_INTERVAL:
        gc.collect()
    for record in evidence["regions"]:
        if (record["run"].casefold() in excluded_runs or
                (fit_partition_rows is not None and
                 fit_partition_rows[record["region_id"]]["role"] != "fit")):
            reason = ("chemistry-evaluation-only-out-of-distribution"
                      if record["run"].casefold() in excluded_runs else
                      "track1-evaluation-only-" + str(
                          fit_partition_rows[record["region_id"]]["reason"]))
            all_coverage.append(dict(region_id=record["region_id"], run=record["run"],
                frame_index=record["frame_index"], phase_class=record["phase_class"],
                confidence=record["confidence"], selected_working_pixels=0,
                conflicting_working_pixels=0, training_rows=0,
                status=reason))
    destination=cache/"all_samples.npz"
    np.savez_compressed(destination,features=np.concatenate(all_x),owners=np.concatenate(all_owners))
    result=dict(samples=str(destination),samples_sha256=sha256_file(destination),frames=receipts,
                coverage=all_coverage,classes=dict(Counter(r['phase_class'] for r in all_coverage if r['training_rows'])))
    if chemistry_scope is not None:
        result["v5_chemistry_fit_scope"] = chemistry_scope
    if partition_binding is not None:
        result["v5_review_fit_partition"] = partition_binding
    write(root/"extraction.json",result)
    return result


def fit(root):
    evidence=read(root/"evidence.json");extraction=read(root/"extraction.json")
    if sha256_file(Path(extraction["samples"]))!=extraction["samples_sha256"]:raise ValueError("Samples changed")
    with np.load(extraction["samples"]) as a:x,owners=a["features"],a["owners"]
    if (evidence.get('audit', {}).get('v5_raw_source_binding') is not None
            and x.dtype != np.dtype('float32')):
        raise ValueError('V5 regional fit requires float32 model features')
    regions=[logistic.ReviewedRegion(**r) for r in evidence["regions"]]
    partition_binding = evidence.get("audit", {}).get("v5_review_fit_partition")
    fit_regions = regions
    if partition_binding is not None:
        from darsia.fluidflower.phase_training.v5_review_fit_partition import (
            verify_partition_rows,
        )
        partition_rows = verify_partition_rows(
            partition_binding, evidence["regions"])
        if extraction.get("v5_review_fit_partition") != partition_binding:
            raise ValueError("Track1 regional samples lost their frozen fit partition")
        fit_indices = [index for index, region in enumerate(regions)
                       if partition_rows[region.region_id]["role"] == "fit"]
        source_index_to_fit = np.full(len(regions), -1, dtype=np.int32)
        source_index_to_fit[fit_indices] = np.arange(len(fit_indices), dtype=np.int32)
        if (owners.size == 0 or np.any(owners < 0) or
                np.any(owners >= len(regions)) or
                np.any(source_index_to_fit[owners] < 0)):
            raise ValueError("Track1 regional sample belongs to evaluation-only review")
        owners = source_index_to_fit[owners]
        fit_regions = [regions[index] for index in fit_indices]
    logistic.assert_safe_training_regions(fit_regions)
    fit_runs = None
    chemistry_scope = evidence.get("audit", {}).get("v5_chemistry_fit_scope")
    if evidence.get("audit", {}).get("v5_raw_source_binding") is not None:
        if chemistry_scope is None:
            raise ValueError("V5 regional fit omitted frozen chemistry exclusion")
        from darsia.fluidflower.phase_training.v5_chemistry_fit_scope import verify_scope
        excluded = verify_scope(chemistry_scope, regions)
        fit_runs = {r.run for r in fit_regions if r.run.casefold() not in excluded}
        if len(fit_runs) < 2:
            raise ValueError("V5 chemistry exclusion leaves too few regional fit runs")
    elif chemistry_scope is not None:
        raise ValueError("Chemistry exclusion without V5 source binding")
    seeds={};reports={}
    for head in joint_aqueous.HEADS:
        write(root/"status.json",dict(stage="fitting",head=head,rows=len(x)))
        seeds[head],reports[head]=logistic.fit_weighted_logistic_mil(x,owners,fit_regions,
            head=head,runs=fit_runs,regularization=.001,backend="cpu",max_iter=1000)
        print(json.dumps(dict(stage="seed_fitted",head=head,converged=seeds[head].converged)),flush=True)
    fits,report=joint_aqueous.fit_joint_aqueous_mil(x,owners,fit_regions,seed_fits=seeds,
        regularization={h:.001 for h in seeds},runs=fit_runs,
        backend="cpu",max_iter=1500,tolerance=1e-6)
    model=dict(format=regional.VERSION,feature_names=list(regional.FEATURE_NAMES),
        settings=read(root/"protocol.json")["settings"],heads={h:{k:v.tolist() if isinstance(v,np.ndarray) else v
            for k,v in asdict(f).items()} for h,f in fits.items()},
        parent_model=evidence["parent"],protocol_sha256=sha256_file(root/"protocol.json"),
        evidence_sha256=sha256_file(root/"evidence.json"),fit_report=report,seed_reports=reports,
        scientific_acceptance=False,
        training_scope=("V5 Track1 source-supported optical review subset"
                        if partition_binding is not None else
                        "V5 light-only chemistry-evaluated review subset"
                        if fit_runs is not None else
                        "all effective FF_UM native review revisions"),
        v5_chemistry_fit_scope=chemistry_scope,
        v5_review_fit_partition=partition_binding,
        original_review_ids_preserved=len(regions),
        fit_review_ids=len(fit_regions))
    write(root/"regional_model.json",model)
    write(root/"status.json",dict(stage="fitted",converged=report["converged"],model=str(root/"regional_model.json")))
    print(json.dumps(dict(stage="fitted",converged=report["converged"],regions=len(regions))),flush=True)
    return model


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--fit-only",action="store_true")
    args=parser.parse_args()
    try:
        if not args.fit_only:extract(args.root,args.cache)
        fit(args.root)
    except BaseException as exc:
        write(args.root/"status.json",dict(stage="failed",error=f"{type(exc).__name__}: {exc}"))
        raise
