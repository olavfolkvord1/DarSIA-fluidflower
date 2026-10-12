"""New, read-only P1-final bottom-edge and section 4d photographic reports.

Reads saved source-bound frames only. Writes exclusively to a new report
directory. Measurements never authorize mass, change domain or modify labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .. import phase_ffac_p1_final_bottom_report as measure
from ..review_storage import resolve_review_path
from .ffac_track1_display_modes import render_display_modes

SPEC_SHA256 = "b421872c3b400376f8c0d65af2db9ab10c329dc751a8956ea212020be8aaa5e8"
CONTRACT = "ffac-p1-final-bottom-figures-report-v1"
EXPECTATIONS_SHA256 = "401cd86b8971fdbdc9be044d78af5df63c4120d4f402e5c2a6a7c304d83f6b9a"
SELECTED = (("ac17", 5), ("ac17", 6), ("ac17", 20), ("ac19", 13), ("ac28", 13), ("ac40", 13), ("ac51", 19))
_VERIFIED = {}


def _init_worker():
    cv2.setNumThreads(1)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def artifact(path):
    path = Path(path)
    digest = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def bound(asset, *, base="."):
    p = Path(asset["path"])
    p = resolve_review_path(p if p.is_absolute() else Path(base)/p).resolve()
    stat = p.stat()
    key = (str(p), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    digest = _VERIFIED.get(key)
    if digest is None:
        digest = artifact(p)["sha256"]
        _VERIFIED[key] = digest
    if digest != asset["sha256"]:
        raise ValueError("Bottom/figure input SHA changed: " + str(p))
    return p


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def frame_path(root, run, index):
    choices = [Path(root)/"rig_camera_frames"/case/f"frame_{index:03d}.json" for case in (run, run.upper())]
    paths = [p for p in choices if p.is_file()]
    if not paths:
        raise FileNotFoundError(str(choices[0]))
    return paths[0]


def native_manifest(root, run):
    for prefix in ("source_alias", ""):
        p = Path(root)/prefix/"native_frames"/run/"manifest.json"
        if p.is_file():
            return p, read(p)
    raise FileNotFoundError("No exact source native manifest for " + run)


def arrays(revision):
    with np.load(bound(revision["classes"]), allow_pickle=False) as z:
        domain = z["rig_domain"].copy()
    with np.load(bound(revision["area_map"]), allow_pickle=False) as z:
        area = z[revision["area_map"].get("array_key", "pixel_area_mm2")].copy()
    if domain.dtype != bool or area.shape != domain.shape or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Bottom measurement classes and measured area differ")
    return domain, area


def raw_roi(record, manifest, y0, y1):
    descriptor = record["full_extended_scientific"]
    revision = record["rig_domain_revision"]
    h, w = revision["phase_shape_hw"]
    if descriptor["phase_shape_hw"] != [h, w] or descriptor["source_sha256"] != record["source_sha256"]:
        raise ValueError("Bottom RAW does not belong to its exact source/grid")
    roi = np.full((3, y1-y0, w), np.nan, np.float32)
    centre = descriptor["native_centre"]
    signal = np.load(bound(centre["log_linear_rgb"]), mmap_mode="r", allow_pickle=False)
    valid = np.load(bound(centre["valid_mask"]), mmap_mode="r", allow_pickle=False)
    x, y, cw, ch = manifest["content_box_xywh"]
    if signal.dtype != np.float32 or signal.shape != (3, ch, cw) or valid.dtype != bool or valid.shape != (ch, cw):
        raise ValueError("Bottom RAW centre encoding changed")
    a, b = max(y, y0), min(y+ch, y1)
    if a < b:
        roi[:, a-y0:b-y0, x:x+cw] = signal[:, a-y:b-y]
        good = valid[a-y:b-y].copy()
        fill = record.get("v5_filled_invalid_mask_path")
        if fill:
            filled = np.load(bound(dict(path=fill, sha256=record["v5_filled_invalid_mask_sha256"])), mmap_mode="r", allow_pickle=False)
            good &= ~filled[a-y:b-y]
        roi[:, a-y0:b-y0, x:x+cw][:, ~good] = np.nan
    used = [centre["log_linear_rgb"], centre["valid_mask"]]
    for window in descriptor["ring_windows"]:
        left, top, right, bottom = window["canvas_xyxy"]
        a, b = max(top, y0), min(bottom, y1)
        if a < b:
            signal = np.load(bound(window["log_linear_rgb"]), mmap_mode="r", allow_pickle=False)
            roi[:, a-y0:b-y0, left:right] = signal[:, a-top:b-top]
            used.append(window["log_linear_rgb"])
    observed = np.isfinite(roi).all(axis=0)
    return roi, observed, used


def photo_canvas(photo_asset, shape, manifest):
    rgb = np.asarray(Image.open(bound(photo_asset)).convert("RGB"))
    result = np.zeros((*shape, 3), np.uint8)
    valid = np.zeros(shape, bool)
    if rgb.shape[:2] == tuple(manifest["analysis_image_shape_hw"]):
        x, y, w, h = manifest["content_box_xywh"]
        result[y:y+h, x:x+w] = rgb
        valid[y:y+h, x:x+w] = True
    elif rgb.shape[1] == shape[1] and rgb.shape[0] <= shape[0]:
        result[:rgb.shape[0]] = rgb
        # JPEG display outside the native acquisition is background, not data.
        x, y, w, h = manifest["content_box_xywh"]
        valid[y:y+h, x:x+w] = True
    else:
        raise ValueError("Native lift photograph has another canvas")
    return result, valid


def _texture_photo(rgb, valid):
    # Native JPEG is a corroboration channel only; no radiometric agreement is
    # claimed with RAW. Logs make multiplicative brightness harmless to texture.
    signal = np.moveaxis(np.log(rgb.astype(np.float32)+1.), -1, 0)
    return measure.texture_luminance(signal, valid)


def measure_image(record, manifest, input_row, *, name, output):
    revision = record["rig_domain_revision"]
    domain, area = arrays(revision)
    parameters = revision["parameters"]
    H = np.asarray(parameters["H_frame_native_to_metric_pixel"], float)
    scale = parameters.get("physical_scale_revision") or revision.get("physical_scale_revision")
    if not scale or scale["physical_scale_approved"] is not True or scale["inner_width_mm"] != 900.:
        raise ValueError("Bottom report needs the approved measured 900 mm scale")
    factor = math.sqrt(parameters["metric_pixel_area_mm2"]) * scale["linear_factor"]
    origin = revision["native_origin_xy"]
    columns, bottom, span = measure.sample_columns(domain, H, origin, factor)
    mmx, mmy = measure.local_metric_mm(H, columns+origin[0], bottom[columns]+origin[1], factor)
    median_bottom = float(np.median(bottom[domain.any(0)]))
    floor_columns = np.abs(bottom[columns]-median_bottom)*mmy <= measure.METHOD["non_floor_column_guard_mm"]
    if not floor_columns.any():
        raise ValueError("Bottom report has no photographed floor-region columns")
    margin = math.ceil(35./float(np.min(mmy)))+20
    context_below = math.ceil(5./float(np.min(mmy)))+2
    y0, y1 = max(0, int(np.min(bottom[columns[floor_columns]]))-margin), min(domain.shape[0], int(np.max(bottom[columns[floor_columns]]))+context_below)
    signal, observed, raw_assets = raw_roi(record, manifest, y0, y1)
    texture, L, valid = measure.texture_luminance(signal, observed)
    photo, photo_valid = photo_canvas(input_row["photographs"]["lift1"], domain.shape, manifest)
    nt, nl, nv = _texture_photo(photo[y0:y1], photo_valid[y0:y1])
    samples = []
    for column, mx, my, is_floor in zip(columns, mmx, mmy, floor_columns):
        if not is_floor:
            samples.append(dict(column=int(column), bottom_canvas_y=float(bottom[column]),sand_bottom_canvas_y=None,uncertainty_px=None,
                                status="unknown: column ends on a sidewall/hardware cutout, more than 30 mm above median floor",native_check="not a floor column"))
            continue
        s = measure.measure_profile(texture, L, valid, x=int(column), bottom_y=bottom[column], roi_y0=y0, pixel_x_mm=float(mx), pixel_y_mm=float(my))
        native = measure.measure_profile(nt, nl, nv, x=int(column), bottom_y=bottom[column], roi_y0=y0, pixel_x_mm=float(mx), pixel_y_mm=float(my))
        found = s.get("sand_bottom_canvas_y")
        native_available = found is not None and 0 <= round(found) < domain.shape[0] and photo_valid[round(found), column]
        s = measure.native_check(s, native, pixel_y_mm=float(my), native_available=bool(native_available))
        s["local_pixel_y_mm"] = float(my)
        s["local_pixel_x_mm"] = float(mx)
        samples.append(s)
    summary = measure.distance_summary(samples, H, origin, factor)
    curve = measure.interpolate_supported(samples, domain.shape[1])
    band = measure.measured_band(domain[y0:y1], curve, row_offset=y0)
    summary.update(band_domain_area_cm2=float(area[y0:y1][band].sum(dtype=np.float64))/100.,
                   measured_columns=int(np.isfinite(curve).sum()), unknown_domain_columns=int((domain.any(0) & ~np.isfinite(curve)).sum()), bottom_span_mm=float(span),
                   raw_only_samples=sum(s["native_check"].startswith("RAW-only") for s in samples), non_floor_samples=int((~floor_columns).sum()), hump=measure.hump_summary(samples, H, origin, factor))
    summary["area_status"] = "measured-support area only; full band area unknown" if summary["unknown_domain_columns"] else "complete observed-column band area"
    summary["full_band_domain_area_cm2"] = None if summary["unknown_domain_columns"] else summary["band_domain_area_cm2"]
    images, _ = render_display_modes(signal, domain[y0:y1])
    raw_lift = np.rint(images["lift1"] * 255).astype(np.uint8)
    comparison = raw_lift.copy()
    support = photo_valid[y0:y1] & observed
    comparison[support] = photo[y0:y1][support]
    overlay = comparison.copy()
    for x in range(domain.shape[1]):
        b = int(bottom[x])-y0
        if domain[:, x].any() and 0 <= b < overlay.shape[0]:
            overlay[max(0,b-1):min(overlay.shape[0],b+2), x] = (255, 70, 65)
        if np.isfinite(curve[x]):
            r = round(curve[x])-y0
            if 0 <= r < overlay.shape[0]:
                overlay[max(0,r-1):min(overlay.shape[0],r+2), x] = (40, 255, 85)
    image_path = output/(name+".bottom.png")
    Image.fromarray(overlay).save(image_path)
    raw_path = output/(name+".raw_bottom.png")
    Image.fromarray(raw_lift).save(raw_path)
    profile_path = output/(name+".profiles.png")
    profile = Image.new("RGB", (1200, 460), "#16191e")
    draw = ImageDraw.Draw(profile)
    profiled = [i for i,s in enumerate(samples) if "texture_profile" in s]
    chosen = [profiled[i] for i in np.linspace(0,len(profiled)-1,min(4,len(profiled))).astype(int)]
    for n, i in enumerate(chosen):
        s = samples[i]
        vals = np.asarray(s["texture_profile"], float)
        rows = np.asarray(s["profile_rows"], float)
        good = vals >= 0
        left = n*300
        maximum = float(np.percentile(vals[good], 98)) if good.any() else 1.
        pts = [(left+30+min(250., float(v)/max(maximum,1e-5)*250.), 60+(float(r)-y0)/(y1-y0)*360) for r,v in zip(rows[good],vals[good])]
        if len(pts)>1:
            draw.line(pts, fill=(120, 190, 255), width=1)
        draw.text((left+10,10), f"x={s['column']}  {s['status'][:28]}", fill="white")
        if s.get("sand_bottom_canvas_y") is not None:
            yy = 60+(s["sand_bottom_canvas_y"]-y0)/(y1-y0)*360
            draw.line((left+20,yy,left+285,yy), fill=(40,255,85), width=2)
    profile.save(profile_path)
    for i,s in enumerate(samples):
        if i not in chosen:
            for key in ("profile_rows","texture_profile","luminance_profile"):
                s.pop(key,None)
    result = dict(name=name, source_sha256=record["source_sha256"], elapsed_h=record["elapsed_h"], summary=summary, samples=samples,
                  curve_canvas_y=[float(v) if np.isfinite(v) else None for v in curve], roi_xyxy=[0,y0,domain.shape[1],y1],
                  provenance=dict(raw=raw_assets, native_lift=input_row["photographs"]["lift1"], classes=revision["classes"], area_map=revision["area_map"], source_registration=input_row["geometry_receipt"],
                                  H_ac28_to_frame_native=input_row["H_ac28_to_frame_native"], physical_scale=scale,
                                  H_frame_native_to_metric_pixel=H.tolist(),metric_mm_factor=float(factor),native_origin_xy=list(origin),
                                  metric_pixel_area_mm2=float(parameters["metric_pixel_area_mm2"]),coordinate_conversion=measure.METHOD["coordinate_conversion"]),
                  figures=[artifact(image_path),artifact(raw_path),artifact(profile_path)], mass_eligible=False, bottom_mass_allowed=False)
    write_new(output/(name+".measurement.json"), result)
    return result, curve


def _phase_overlay(photo, labels):
    rgb = photo.copy()
    for value, colour in ((1,(72,235,135)), (2,(255,105,155))):
        selected = labels == value
        rgb[selected] = np.rint(rgb[selected].astype(float)*.5+np.asarray(colour)*.5).astype(np.uint8)
    return rgb


def unpack_mask(asset, shape):
    packed = np.load(bound(asset), allow_pickle=False)
    if packed.dtype != np.uint8:
        raise ValueError("Removal mask must have the recorded packbits encoding")
    return np.unpackbits(packed, bitorder=asset.get("bitorder", "big"), count=int(np.prod(shape))).reshape(shape).astype(bool)


def removal_mask(row, shape):
    return unpack_mask(row["top_surface_rule_masks"], shape)


def frame_figure(parent, candidate, run, index, record, manifest, input_row, output):
    p = read(frame_path(parent,run,index))
    f = read(frame_path(candidate,run,index))
    before = np.asarray(Image.open(bound(p["labels_map"])))
    after = np.asarray(Image.open(bound(f["labels_map"])))
    shape = before.shape
    photo, valid = photo_canvas(input_row["photographs"]["lift1"], shape, manifest)
    domain, _ = arrays(record["rig_domain_revision"])
    occupied = domain.any(axis=0)
    top = np.argmax(domain, axis=0)[occupied]
    median_top = float(np.median(top))
    top = top[np.abs(top-median_top) <= math.floor(30./math.sqrt(.027053))]
    bottom = shape[0]-1-np.argmax(domain[::-1],axis=0)[occupied]
    boxes = {"top": (0,max(0,int(np.percentile(top,10))-30),shape[1],min(shape[0],int(np.percentile(top,90))+580)),
             "bottom": (0,max(0,int(np.percentile(bottom,10))-360),shape[1],min(shape[0],int(np.percentile(bottom,90))+30))}
    for name, box in boxes.items():
        x0,y0,x1,y1 = box
        raw, support, _ = raw_roi(record,manifest,y0,y1)
        rgb, _ = render_display_modes(raw,domain[y0:y1])
        view = np.rint(rgb["lift1"] * 255).astype(np.uint8)
        existing = photo[y0:y1]
        seen = valid[y0:y1] & support
        view[seen] = existing[seen]
        pictures = (view,_phase_overlay(view,before[y0:y1]),_phase_overlay(view,after[y0:y1]))
        cell_width=1400
        scale=cell_width/shape[1]
        cell_height=max(1,round((y1-y0)*scale))
        sheet=Image.new("RGB",(cell_width*3,cell_height+50),"#171a1f")
        draw=ImageDraw.Draw(sheet)
        for i,(title,rgb) in enumerate(zip(("Photo lift x4.3","P1","P1-final"),pictures)):
            draw.text((i*cell_width+15,10),f"{run.upper()} frame {index} t={record['elapsed_h']:.3f} h {name} - {title}",fill="white")
            sheet.paste(Image.fromarray(rgb).resize((cell_width,cell_height),Image.Resampling.LANCZOS),(i*cell_width,50))
        path=output/f"{run.upper()}_frame_{index:03d}_{name}_photo_P1_final.png"
        sheet.save(path)
    removed=removal_mask(f,shape)
    # Outline after removing only downsampled display aliases, never labels.
    boundary=removed & ~cv2.erode(removed.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    boundary=cv2.dilate(boundary.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    photo[boundary]=(255,205,35)
    image=Image.fromarray(photo)
    image.thumbnail((800,470),Image.Resampling.LANCZOS)
    path=output/f"{run.upper()}_frame_{index:03d}_removed_outline.png"
    image.save(path)
    return path


def report_run(parent, candidate, run, output, *, measure_only=False):
    started=time.perf_counter(); cpu=time.process_time()
    output=Path(output)/run; output.mkdir(parents=True,exist_ok=False)
    manifest_path, manifest=native_manifest(parent,run)
    source_key=manifest["ffac_rig_camera_sources"]
    input_path=bound(source_key["input_manifest"],base=manifest_path.parent)
    inputs=read(input_path)
    frames=manifest["frame_sources"]
    baseline=manifest["reference"]
    reference,curve=measure_image(baseline,manifest,inputs["reference"],name=run.upper()+"_reference",output=output)
    last=frames[-1]
    last_measure,last_curve=measure_image(last,manifest,inputs["frames"][-1],name=run.upper()+"_last",output=output)
    frame_results=[]; figures=[]
    if not measure_only:
        previous_area=None; area=None
        Href=np.asarray(inputs["reference"]["H_ac28_to_frame_native"],float)
        ref_origin=baseline["rig_domain_revision"]["native_origin_xy"]
        maximum=max(range(len(frames)),key=lambda i: sum(read(frame_path(candidate,run,i))["qa"].get(k,0.) for k in ("removed_top_aqueous_cm2","removed_top_gas_cm2")))
        figure_indices={i for r,i in SELECTED if r==run}; figure_indices.add(maximum)
        for index,record in enumerate(frames):
            row_path=frame_path(candidate,run,index); row=read(row_path)
            if row.get("complete") is not True or row.get("bottom_mass_allowed") is not False or row["source_sha256"]!=record["source_sha256"] or float(row["elapsed_h"])!=float(record["elapsed_h"]):
                raise ValueError("P1-final report frame identity or bottom mass block changed")
            labels=np.asarray(Image.open(bound(row["labels_map"])))
            rev=row["rig_domain_revision"]
            with np.load(bound(rev["classes"]),allow_pickle=False) as z: domain=z["rig_domain"].copy()
            if previous_area!=rev["area_map"]["sha256"]:
                with np.load(bound(rev["area_map"]),allow_pickle=False) as z: area=z[rev["area_map"].get("array_key","pixel_area_mm2")].copy()
                previous_area=rev["area_map"]["sha256"]
            H=np.asarray(inputs["frames"][index]["H_ac28_to_frame_native"],float) @ np.linalg.inv(Href)
            transported=measure.transport_curve(curve,H_source_to_target=H,source_origin_xy=ref_origin,target_origin_xy=rev["native_origin_xy"],target_width=labels.shape[1])
            measured=measure.measured_band(domain,transported)
            frame_results.append(dict(run=run.upper(),frame_index=index,elapsed_h=float(row["elapsed_h"]),source_sha256=row["source_sha256"],
                band_domain_area_cm2=float(area[measured].sum(dtype=np.float64))/100.,aqueous_cm2=float(area[measured & (labels==1)].sum(dtype=np.float64))/100.,
                gas_cm2=float(area[measured & (labels==2)].sum(dtype=np.float64))/100.,measured_columns=int(np.isfinite(transported).sum()),
                unknown_domain_columns=int((domain.any(0) & ~np.isfinite(transported)).sum()),
                area_status="measured photographic support only; unmeasured columns remain unknown",
                curve_source="pre-injection photographic curve transported through the unchanged source homography", labels=row["labels_map"],
                frame_receipt=artifact(row_path),mass_eligible=False,bottom_mass_allowed=False))
            if index in figure_indices:
                path=frame_figure(parent,candidate,run,index,record,manifest,inputs["frames"][index],output)
                if index==maximum: figures.append(artifact(path))
    result=dict(contract=CONTRACT,run=run.upper(),reference=reference,last=last_measure,per_frame=frame_results,maximum_removal_thumbnail=figures,
                source_manifest=artifact(manifest_path),input_manifest=source_key["input_manifest"],measurement_method=measure.method_document(),
                timing=dict(wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,backend="cpu-report-only"),mass_eligible=False,bottom_mass_allowed=False)
    write_new(output/"run_report.json",result)
    return result


def montage(output,results):
    for kind in ("bottom_edges","maximum_removed"):
        if kind=="bottom_edges":
            cells=[(r["run"]+" reference",r["reference"]["figures"][0]["path"]) for r in results]+[(r["run"]+" last",r["last"]["figures"][0]["path"]) for r in results]
            cellw,cellh,columns=700,140,2
        else:
            cells=[(r["run"],r["maximum_removal_thumbnail"][0]["path"]) for r in results if r["maximum_removal_thumbnail"]]
            cellw,cellh,columns=600,390,6
        if not cells: continue
        sheet=Image.new("RGB",(columns*cellw,math.ceil(len(cells)/columns)*cellh+70),"#181b20")
        draw=ImageDraw.Draw(sheet)
        draw.text((15,12),"P1-final descriptive report; maps unchanged by measurement; bottom remains blocked for mass",fill="white")
        draw.text((15,34),"Red: rig-domain bottom. Green: supported photographic sand/bright-band transition. Missing green = unknown." if kind=="bottom_edges" else "Photo lift x4.3; yellow outline = CO2 removed by the top-surface rule",fill="#c2cedb")
        for n,(title,path) in enumerate(cells):
            rgb=Image.open(path).convert("RGB");rgb.thumbnail((cellw-10,cellh-28),Image.Resampling.LANCZOS)
            x=(n%columns)*cellw;y=70+(n//columns)*cellh
            draw.text((x+10,y),title,fill="white");sheet.paste(rgb,(x+5,y+25))
        path=Path(output)/(kind+"_all_runs.png");sheet.save(path)


def expectation_comparison(candidate, csv_path):
    """Compare the already locked 126-frame scan in its nominal area units."""
    csv_asset=artifact(csv_path)
    if csv_asset["sha256"] != EXPECTATIONS_SHA256:
        raise ValueError("The descriptive expectation table differs from the table named in the approved spec")
    with Path(csv_path).open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    compared=[]
    for old in rows:
        run,index=old["run"].casefold(),int(old["frame"])
        row=read(frame_path(candidate,run,index)); qa=row["qa"]
        nominal=sum(float(qa[k]) for k in ("nominal_removed_top_aqueous_cm2","nominal_removed_top_gas_cm2"))
        measured=sum(float(qa[k]) for k in ("removed_top_aqueous_cm2","removed_top_gas_cm2"))
        expected=float(old["removed60_cm2"])
        compared.append(dict(run=run.upper(),frame_index=index,elapsed_h=row["elapsed_h"],source_sha256=row["source_sha256"],
            expected_nominal_removed_cm2=expected,actual_nominal_removed_cm2=nominal,actual_measured_removed_cm2=measured,
            nominal_minus_expected_cm2=nominal-expected,measured_minus_nominal_cm2=measured-nominal))
    values=[r["actual_nominal_removed_cm2"] for r in compared]
    measured=[r["actual_measured_removed_cm2"] for r in compared]
    return dict(source=csv_asset,frame_count=len(compared),nominal_median_cm2=float(np.median(values)),nominal_max_cm2=float(max(values)),
                measured_median_cm2=float(np.median(measured)),measured_max_cm2=float(max(measured)),
                maximum_absolute_nominal_difference_from_rounded_descriptive_scan_cm2=max(abs(r["nominal_minus_expected_cm2"]) for r in compared),
                explanation="Claude's descriptive scan used fixed nominal 0.027053 mm2/pixel; production area reports use the unchanged approved per-pixel 900 mm maps. The scan precedes floor removal, but its top components are compared after mandated step A. CSV values were rounded to two decimal places; it is descriptive and not an acceptance gate.",per_frame=compared)


def ac51_top_clip_report(parent,candidate):
    """Explain old boundary-only gas clipping using saved masks, not inference."""
    run,index="ac51",19
    old_path=frame_path(parent,run,index); final_path=frame_path(candidate,run,index)
    old=read(old_path); final=read(final_path)
    before=np.asarray(Image.open(bound(old["labels_map"])))
    after=np.asarray(Image.open(bound(final["labels_map"])))
    removed=removal_mask(final,before.shape)
    gas=removed & (before==2)
    descriptor=old["binding"]["source"]["rig_geometry"]["frame_photo_top_keep"]
    with np.load(bound(descriptor),allow_pickle=False) as archive:
        top=archive[descriptor.get("array_key","top_keep")].copy()
    if top.shape!=before.shape or top.dtype!=bool:
        raise ValueError("AC51 old top clip mask differs from its source canvas")
    with np.load(bound(old["rig_domain_revision"]["area_map"]),allow_pickle=False) as archive:
        weights=archive[old["rig_domain_revision"]["area_map"].get("array_key","pixel_area_mm2")]
        gas_area=float(weights[gas].sum(dtype=np.float64))/100.
    domain=before!=255; has=domain.any(axis=0); line=np.argmax(domain,axis=0)
    depth=np.arange(before.shape[0])[:,None]-line[None,:]
    d=depth[gas]*math.sqrt(.027053)
    return dict(run="AC51",frame_index=index,elapsed_h=old["elapsed_h"],source_sha256=old["source_sha256"],
        removed_gas_pixels=int(gas.sum()),removed_gas_measured_cm2=gas_area,removed_gas_nominal_cm2=float(gas.sum()*.027053/100),
        removed_gas_inside_old_top_keep_pixels=int((gas & top).sum()),removed_gas_outside_old_top_keep_pixels=int((gas & ~top).sum()),
        old_declared_top_clip_removed_pixels=old["gas_inference"]["top_clip_removed_pixels"],
        depth_below_domain_top_min_nominal_mm=float(d.min()) if len(d) else None,depth_below_domain_top_max_nominal_mm=float(d.max()) if len(d) else None,
        exact_removed_gas_is_label_zero=bool(np.all(after[gas]==0)),
        cause="The old top clip only demoted gas outside the photographed sand-surface keep-mask. These gas pixels are inside that old keep-mask, below the sand surface. It had no detached-component rejection band inside sand. P1-final's distinct 60 mm connected-component rule removes the shallow detached CO2 component, including gas; this is an added rule, not failure of the old clip's above-surface check.",
        provenance=dict(parent_receipt=artifact(old_path),final_receipt=artifact(final_path),old_top_keep=descriptor,removed_components=final["top_surface_rule_masks"]),
        inference_performed=False,labels_changed_by_report=False)


def run_report(parent,candidate,output,*,workers=8,runs=None,measure_only=False,expectations_csv=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    parent,candidate=Path(parent).resolve(),Path(candidate).resolve()
    seal=read(parent/"rig_camera_candidate_receipt.json")
    if seal.get("complete") is not True or seal.get("placement") != "P1" or (seal.get("run_count"),seal.get("frame_count"))!=(43,864) or seal.get("bottom_mass_allowed") is not False:
        raise ValueError("The photographic report requires the complete source-bound P1 43/864 bottom-blocked candidate")
    if not measure_only:
        final=read(candidate/"rig_camera_candidate_receipt.json")
        if final.get("contract") != "ffac-g3-t4c-p1-final-candidate-v1" or final.get("complete") is not True or (final.get("run_count"),final.get("frame_count"))!=(43,864) or final.get("bottom_mass_allowed") is not False:
            raise ValueError("Section 2/4d must follow the completed and sealed P1-final label revision")
    selected=sorted(set(runs or [item["run"].casefold() for item in seal["frames"]]))
    started=time.perf_counter(); coordinator_cpu=time.process_time(); results=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker) as pool:
        futures={pool.submit(report_run,parent,candidate,r,output,measure_only=measure_only):r for r in selected}
        for future in as_completed(futures):
            value=future.result();results.append(value)
            print(json.dumps(dict(run=value["run"],complete=True,reference=value["reference"]["summary"],last=value["last"]["summary"],wall_seconds=value["timing"]["wall_seconds"])),flush=True)
    results.sort(key=lambda r:r["run"])
    montage(output,results)
    compact=[]
    for r in results:
        for key in ("reference","last"):
            summary=r[key]["summary"]
            compact.append(dict(run=r["run"],image=key,**{k:summary[k] for k in ("accepted_samples","sample_count","unknown_samples","median_mm","p10_mm","p90_mm","band_domain_area_cm2","measured_columns","unknown_domain_columns","raw_only_samples")},hump_height_mm=summary["hump"].get("height_mm"),hump_x_range=summary["hump"].get("x_range_metric_mm")))
    def csv_new(name,rows):
        if not rows:return
        with (output/name).open("x",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    csv_new("bottom_edge_by_run_reference_last.csv",compact)
    frame_rows=[{k:v for k,v in row.items() if k not in ("labels","frame_receipt")} for r in results for row in r["per_frame"]]
    csv_new("bottom_band_P1_final_by_frame.csv",frame_rows)
    expectations=expectation_comparison(candidate,expectations_csv) if expectations_csv and not measure_only else None
    ac51=ac51_top_clip_report(parent,candidate) if not measure_only and "ac51" in selected else None
    if expectations:
        write_new(output/"top_removal_expectation_comparison.json",expectations)
        csv_new("top_removal_expectation_comparison.csv",expectations["per_frame"])
    if ac51:write_new(output/"AC51_frame_019_old_top_clip_cause.json",ac51)
    document=dict(contract=CONTRACT,complete=True,spec_sha256=SPEC_SHA256,measurement_method=measure.method_document(),parent=artifact(parent/"rig_camera_candidate_receipt.json"),
                  candidate=artifact(candidate/"rig_camera_candidate_receipt.json") if not measure_only else None,run_count=len(results),frame_count=len(frame_rows),per_run=results,
                  report_code=dict(measurement=artifact(measure.__file__),driver=artifact(__file__)),
                  timing=dict(wall_seconds=time.perf_counter()-started,cpu_seconds=sum(r["timing"]["cpu_seconds"] for r in results)+time.process_time()-coordinator_cpu,
                              worker_cpu_seconds=sum(r["timing"]["cpu_seconds"] for r in results),coordinator_cpu_seconds=time.process_time()-coordinator_cpu,
                              gpu_seconds=0.,backend="cpu-report-only"),
                  expectation_comparison=expectations,ac51_top_clip=ac51,
                  scientific_assets_changed=False,labels_changed=False,mass_eligible=False,bottom_mass_allowed=False)
    write_new(output/"bottom_report.json",document)
    return document


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",required=True);parser.add_argument("--candidate",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--workers",type=int,default=8);parser.add_argument("--runs");parser.add_argument("--measure-only",action="store_true")
    parser.add_argument("--expectations-csv")
    args=parser.parse_args(argv)
    cv2.setNumThreads(1)
    run_report(args.parent,args.candidate,args.output,workers=args.workers,runs=args.runs.split(",") if args.runs else None,measure_only=args.measure_only,expectations_csv=args.expectations_csv)


if __name__=="__main__":
    main()
