"""Exact source-bound 20 mm photographic carryover proxy for Step C.

Uses the unchanged primary bottom-band diagnostic; labels and annotations do
not determine the proxy. Its reference is the closest saved frame strictly
before injection, exactly as in the frozen diagnostic reporter.
"""
from pathlib import Path
import time

import numpy as np

from . import phase_ffac_bottom_band_diagnostics as diagnostic
from .tools import report_ffac_bottom_band_diagnostics as reporter
from .tools import report_ffac_p1_final as saved

CONTRACT = "ffac-p1-final2-source-bound-carryover-proxy-v1"


def closest_preinjection(frames):
    candidates=[row for row in frames if row.get("elapsed_h") is not None and float(row["elapsed_h"])<0]
    return max(candidates,key=lambda row:float(row["elapsed_h"])) if candidates else None


def generate_carryover(candidate, run, output=None):
    """Return full phase mask, whether any floor is observable, and provenance.

    The optional output is a new directory for the mask and immutable receipt;
    existing files are never overwritten. Runtime consumers transport from the
    returned source homography, never from a reused frame number alone.
    """
    started,cpu=time.perf_counter(),time.process_time()
    candidate=Path(candidate).resolve();run=run.casefold()
    manifest_path,manifest=saved.native_manifest(candidate,run)
    manifest_asset=saved.artifact(manifest_path)
    source=closest_preinjection(manifest["frame_sources"])
    if source is None:
        shape=tuple(manifest["reference"]["rig_domain_revision"]["phase_shape_hw"])
        return np.zeros(shape,bool),False,dict(contract=CONTRACT,run=run,known=False,
            status="unknown: no saved frame strictly before injection",native_manifest=manifest_asset,
            method=diagnostic.method_document(),annotations_read=False)
    index=int(source["frame_index"])
    row_path=saved.frame_path(candidate,run,index);row=saved.read(row_path)
    if (row.get("complete") is not True or row["source_sha256"]!=source["source_sha256"]
            or float(row["elapsed_h"])!=float(source["elapsed_h"]) or row["frame_index"]!=index):
        raise ValueError("Carryover pre-injection source/time binding changed")
    revision=row["rig_domain_revision"];parameters=revision["parameters"]
    scale=parameters["physical_scale_revision"]
    if not scale.get("physical_scale_approved") or scale["inner_width_mm"]!=900:
        raise ValueError("The exact carryover diagnostic requires the approved 900 mm scale")
    provenance=dict(contract=CONTRACT,run=run,frame_index=index,elapsed_h=float(row["elapsed_h"]),source_sha256=row["source_sha256"],
        source_frame_receipt=saved.artifact(row_path),native_manifest=manifest_asset,
        classes=revision["classes"],measured_area=revision["area_map"],physical_scale=scale,
        geometry=row["binding"]["source"]["rig_geometry"],native_origin_xy=revision["native_origin_xy"],
        method=diagnostic.method_document(),primary_search_height_mm=20.,
        diagnostic_code=saved.artifact(diagnostic.__file__),unchanged_reporter_code=saved.artifact(reporter.__file__),
        helper_code=saved.artifact(__file__),annotations_read=False,source_labels_used_for_proxy=False)
    with reporter._load(revision["classes"]) as classes, reporter._load(revision["area_map"]) as areas:
        observed=reporter._raw_observation(row,source,manifest,provenance)
        geometry=provenance["geometry"]
        reference_polygon=np.asarray(geometry["reference_rig"]["polygon_reference_native"],float)
        H=np.asarray(geometry["H_reference_rig_to_frame_native"],float)
        projected=np.c_[reference_polygon,np.ones(len(reference_polygon))]@H.T
        projected=projected[:,:2]/projected[:,2:]
        if not np.allclose(projected,geometry["frame_inner_polygon_native"],atol=1e-6,rtol=0):
            raise ValueError("Carryover floor polygon differs from exact bound transport")
        segments,indices=diagnostic.bottom_edge_segments(reference_polygon)
        projected_segments=np.concatenate([segments,np.ones((*segments.shape[:2],1))],axis=2)@H.T
        projected_segments=projected_segments[...,:2]/projected_segments[...,2:]
        zone=diagnostic.prepare_floor_zone(polygon_native=geometry["frame_inner_polygon_native"],
            origin_xy=revision["native_origin_xy"],H_native_to_metric_pixel=parameters["H_frame_native_to_metric_pixel"],
            metric_pixel_area_mm2=parameters["metric_pixel_area_mm2"],linear_scale=scale["linear_factor"],
            rig_domain=classes["rig_domain"],hardware_excluded=classes["hardware_excluded"],raw_observed=observed,
            pixel_area_mm2=areas[revision["area_map"].get("array_key","pixel_area_mm2")],modelled=classes["modelled"],
            bottom_segments_native=projected_segments)
        signal=reporter._raw_roi(row,manifest,zone,provenance)
        rgb=diagnostic.fixed_natural_rgb(signal,zone.raw_observed)
        positive=diagnostic.colour_proxy_mask(rgb)
        summary,band=diagnostic.colour_in_floor_connected_sand(zone,positive,height_mm=20.)
        mask=np.zeros(tuple(revision["phase_shape_hw"]),bool);mask[zone.y0:zone.y1]=band
    inputs_asset=manifest["ffac_rig_camera_sources"]["input_manifest"]
    inputs=saved.read(saved.bound(inputs_asset,base=manifest_path.parent))
    provenance.update(input_manifest=inputs_asset,H_ac28_to_frame_native=inputs["frames"][index]["H_ac28_to_frame_native"],
        reference_bottom_segment_indices=indices.tolist(),roi_xyxy=[0,zone.y0,mask.shape[1],zone.y1],
        known=bool(summary["observable_photographic_floor_columns"]),proxy_summary=summary,
        status=summary["status"],mask_pixels=int(mask.sum()),bottom_mass_allowed=False,mass_eligible=False,
        timing=dict(wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,gpu_seconds=0.,backend="cpu-frozen-photographic-proxy"))
    if output is not None:
        folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
        with (folder/"carryover_20mm.packbits.npy").open("xb") as f:np.save(f,np.packbits(mask.reshape(-1),bitorder="little"),allow_pickle=False)
        provenance["mask"]=dict(**saved.artifact(folder/"carryover_20mm.packbits.npy"),shape_hw=list(mask.shape),bitorder="little",array_encoding="numpy-packed-bool-flat")
        saved.write_new(folder/"carryover_receipt.json",provenance)
    return mask,provenance["known"],provenance
