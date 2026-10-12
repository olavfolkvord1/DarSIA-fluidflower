"""Read-only display-mode QA and six contact sheets for source-bound Track1.

All statistics use native-resolution assets. Thumbnails are confined to the
contact sheets. No correction, colour fitting, inference or existing-file
write occurs here. Checker geometry is measured independently of sand H.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtGui import QImage

from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.phase_view_overlay import display_phase_overlay
from darsia.fluidflower.tools.qa_ffac_track1_display_mask_rerun import _phase_folder

MODES = ("natural", "lift1", "lift2")
GROUP_RUNS = ("ac23", "ac28", "ac41", "ac56")
CHEMISTRY_FLAGS = {"ac20", "ac21", "ac22", "ac24", "ac27", "ac29", "ac32", "ac44", "ac51"}
REF = np.array([[115,82,68],[194,150,130],[98,122,157],[87,108,67],[133,128,177],[103,189,170],
                [214,126,44],[80,91,166],[193,90,99],[94,60,108],[157,188,64],[224,163,46],
                [56,61,150],[70,148,73],[175,54,60],[231,199,31],[187,86,149],[8,133,161],
                [243,243,242],[200,200,200],[160,160,160],[122,122,121],[85,85,85],[52,52,52]], float)
Y_WEIGHTS = np.array([.2126, .7152, .0722])


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _bound(path, digest, label):
    path = Path(str(path))
    if not isinstance(digest, str) or len(digest) != 64 or not path.is_file() or sha256_file(path) != digest.lower():
        raise ValueError(f"Display QA {label} missing or SHA changed: {path}")
    return path


def _exclusive_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return payload


def _csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sources(batch_root, *, expected_runs=43, expected_frames=864):
    receipt = Path(batch_root) / "materialization_receipt.json"
    batch = _read(receipt)
    rows = batch.get("canonical_mapping") or []
    if len(rows) != expected_runs or len({str(row["run"]).casefold() for row in rows}) != expected_runs:
        raise ValueError("Display QA source batch has an incomplete or duplicate current run cohort")
    sources = []
    for row in sorted(rows, key=lambda item: int(str(item["run"])[2:])):
        path = _bound(row["materialized_manifest_path"], row["materialized_manifest_sha256"], "native manifest")
        if row.get("receipt_path"):
            _bound(row["receipt_path"], row.get("receipt_sha256"), "source revision seal")
        manifest = _read(path)
        frames = manifest.get("frame_sources") or []
        if (manifest.get("run") != str(row["run"]).casefold() or len(frames) != manifest.get("frame_count")
                or [item.get("frame_index") for item in frames] != list(range(len(frames)))):
            raise ValueError("Display QA native timeline disagrees with its sealed cohort")
        sources.append((manifest["run"], manifest, path))
    if sum(len(item[1]["frame_sources"]) for item in sources) != expected_frames:
        raise ValueError("Display QA frame count disagrees with its sealed cohort")
    return sources, {"path": str(receipt), "sha256": sha256_file(receipt)}


def _analysis(rgb, manifest):
    h, w = map(int, manifest["analysis_image_shape_hw"])
    if rgb.shape[:2] == (h, w):
        return rgb
    canvas = tuple(reversed(tuple(map(int, manifest["canvas_wh"]))))
    x, y, bw, bh = map(int, manifest["content_box_xywh"])
    if rgb.shape[:2] != canvas or (bh, bw) != (h, w) or x < 0 or y < 0 or x+bw > canvas[1] or y+bh > canvas[0]:
        raise ValueError("Display QA photographic canvas disagrees with its manifest")
    return rgb[y:y+bh, x:x+bw]


def _mode_rgb(row, mode, manifest):
    declaration = (row.get("v5_display_modes") or {}).get(mode) or {}
    path = _bound(declaration.get("jpeg_path"), declaration.get("jpeg_sha256"), f"{mode} JPEG")
    suffix = "" if mode == "natural" else f".{mode}"
    expected = Path(row["frame_path"]).with_name(f"frame_{int(row['frame_index']):03d}{suffix}.jpg")
    if path.resolve() != expected.resolve(): raise ValueError("Display QA mode JPEG points outside its exact native frame path")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    _analysis(rgb, manifest)
    return rgb, path


def _mask(row, key, manifest, *, compressed=False):
    path = _bound(row.get(key + "_path"), row.get(key + "_sha256"), key)
    label = "display_invalid_mask.npz" if compressed else "photo_sand_mask.npy"
    expected = Path(row["frame_path"]).with_name(f"frame_{int(row['frame_index']):03d}.{label}")
    if path.resolve() != expected.resolve(): raise ValueError("Display QA mask points outside its exact native frame path")
    if compressed:
        with np.load(path, allow_pickle=False) as payload:
            mask = payload["invalid_mask"]
    else:
        mask = np.load(path, allow_pickle=False)
    if mask.dtype != np.bool_ or mask.shape != tuple(manifest["analysis_image_shape_hw"]):
        raise ValueError(f"Display QA {key} is not a native boolean mask")
    return mask


def _small_invalid(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(mask, dtype=np.uint8), connectivity=8)
    lookup = np.zeros(count, bool)
    lookup[1:] = stats[1:, cv2.CC_STAT_AREA] <= 64
    return lookup[labels]


def gray_dot_counts(rgb, original_small_invalid):
    """Count marker-like outliers only at exact original invalid coordinates.

    A legitimate grey card/patch is never selected by colour alone. A filled
    pixel that naturally agrees with its grey neighbours is also not a dot.
    No heuristic modifies the photograph.
    """
    near128 = np.all(np.abs(rgb.astype(np.int16) - 128) <= 20, axis=-1) & original_small_invalid
    luminance = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    local = cv2.medianBlur(luminance, 5)
    dots = near128 & (np.abs(luminance.astype(np.int16) - local.astype(np.int16)) > 20)
    return int(dots.sum()), int(near128.sum())


def sand_median_y(rgb, sand):
    samples = rgb[sand].astype(np.float64) / 255.
    if not len(samples):
        raise ValueError("Display QA photo-sand support is empty")
    linear = np.where(samples <= .04045, samples / 12.92, ((samples + .055) / 1.055) ** 2.4)
    return float(np.median(linear @ Y_WEIGHTS))


def patch_centres(p11=(5510., 788.), p41=(5795., 788.), p14=(5511., 1067.)):
    p11, p41, p14 = map(np.asarray, (p11, p41, p14))
    return {c*6 + (5-r): p11 + c*(p41-p11)/3 + r*(p14-p11)/3 for c in range(4) for r in range(6)}


def _warp_template(gray, scale, angle):
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    width = int(np.ceil(abs(matrix[0,0])*w + abs(matrix[0,1])*h))
    height = int(np.ceil(abs(matrix[0,1])*w + abs(matrix[0,0])*h))
    matrix[0,2] += (width-w)/2
    matrix[1,2] += (height-h)/2
    warped = cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_AREA, borderValue=0)
    return warped, matrix


def _uniform_grid_offset(rgb, found):
    """Evaluate all 625 fixed offsets together in a small chart ROI."""
    centres = np.rint(np.array(list(found.values()))).astype(int)
    offsets_x,offsets_y = np.meshgrid(np.arange(-24,25,2),np.arange(-24,25,2))
    offsets_x,offsets_y = offsets_x.ravel(),offsets_y.ravel()
    cx,cy = centres[None,:,0]+offsets_x[:,None],centres[None,:,1]+offsets_y[:,None]
    h,w = rgb.shape[:2]
    eligible = np.all((cx>=12)&(cy>=12)&(cx+13<=w)&(cy+13<=h),axis=1)
    if not eligible.any(): raise ValueError("Display QA checker patch cores fall outside the image")
    x0,y0 = max(0,int(cx.min())-12),max(0,int(cy.min())-12)
    x1,y1 = min(w,int(cx.max())+13),min(h,int(cy.max())+13)
    crop = rgb[y0:y1,x0:x1]
    left,right = np.clip(cx-12-x0,0,crop.shape[1]),np.clip(cx+13-x0,0,crop.shape[1])
    top,bottom = np.clip(cy-12-y0,0,crop.shape[0]),np.clip(cy+13-y0,0,crop.shape[0])
    objective = np.zeros(len(offsets_x),np.float64)
    for channel in range(3):
        integral,squared = cv2.integral2(np.ascontiguousarray(crop[...,channel]),sdepth=cv2.CV_64F,sqdepth=cv2.CV_64F)
        total = integral[bottom,right]-integral[top,right]-integral[bottom,left]+integral[top,left]
        total_sq = squared[bottom,right]-squared[top,right]-squared[bottom,left]+squared[top,left]
        objective += (total_sq/625-(total/625)**2).mean(axis=1)/3
    objective += .0001*(offsets_x*offsets_x+offsets_y*offsets_y)
    objective[~eligible] = np.inf
    best = int(np.argmin(objective))
    return int(offsets_x[best]),int(offsets_y[best])


def locate_checker(rgb, template, centres, *, template_origin=(5462.,704.), expected_xy=(5462.,704.),
                   search_radius=700, scales=None, angles=None):
    """Template-derived chart-plane geometry; sand homographies are not used."""
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(np.ascontiguousarray(template), cv2.COLOR_RGB2GRAY)
    x0, y0 = max(0, int(expected_xy[0]-search_radius)), max(0, int(expected_xy[1]-search_radius))
    x1 = min(gray.shape[1], int(expected_xy[0]+search_radius+template.shape[1]))
    y1 = min(gray.shape[0], int(expected_xy[1]+search_radius+template.shape[0]))
    scale_down = .25
    search = cv2.resize(gray[y0:y1,x0:x1], None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_AREA)
    reference = cv2.resize(template_gray, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_AREA)
    best = None
    for scale in (np.linspace(.85,1.10,26) if scales is None else scales):
        for angle in ((-1.5,-.75,0.,.75,1.5) if angles is None else angles):
            warped, matrix = _warp_template(reference, float(scale), float(angle))
            if warped.shape[0] >= search.shape[0] or warped.shape[1] >= search.shape[1]:
                continue
            scores = cv2.matchTemplate(search, warped, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(scores)
            if best is None or score > best[0]:
                best = (float(score), matrix, loc, float(scale), float(angle))
    if best is None:
        raise ValueError("Display QA checker template cannot fit the declared search region")
    score, matrix, loc, scale, angle = best
    found = {}
    for key, centre in centres.items():
        point = (np.asarray(centre)-template_origin)*scale_down
        out = matrix @ np.r_[point,1.]
        found[key] = ((out[0]+loc[0])/scale_down+x0, (out[1]+loc[1])/scale_down+y0)
    # Fixed-grid ROI refinement uses only patch-core uniformity; no colour or
    # exposure transform is fitted to the QA reference. Bounds are fixed.
    dx,dy = _uniform_grid_offset(rgb,found)
    found = {key:(point[0]+dx,point[1]+dy) for key,point in found.items()}
    return found, {"template_score":score,"template_scale":scale,"template_angle_deg":angle,
                   "grid_refinement_dx":dx,"grid_refinement_dy":dy,
                   "grid_refinement_objective":"patch-core variance, fixed ±24px; no colour refit"}


def checker_scores(rgb, centres):
    samples, clipped, per_patch = [], [], []
    for key in sorted(centres):
        x,y = (int(round(v)) for v in centres[key])
        core = rgb[y-12:y+13,x-12:x+13]
        if core.shape != (25,25,3):
            raise ValueError("Display QA checker core is outside the photographic grid")
        mean = core.reshape(-1,3).mean(0)
        is_clipped = bool((mean >= 250).any())
        (clipped if is_clipped else samples).append(key)
        per_patch.append({"patch_index":key,"centre_xy":[x,y],"mean_rgb8":mean.tolist(),
                          "observed_display_clipped":is_clipped,"mean_abs_error_rgb8":float(np.abs(mean-REF[key]).mean())})
    error = float(np.mean([per_patch[key]["mean_abs_error_rgb8"] for key in samples])) if samples else None
    white = per_patch[18]["mean_rgb8"]
    return {"patches_used":len(samples),"patches_observed_clipped":clipped,"mean_abs_error_rgb8":error,
            "white_patch_abs_R_minus_B_rgb8":abs(white[0]-white[2]),"white_patch_rgb8":white,
            "raw_spatial_clipping_mask_available":False,
            "clipping_exclusion":"observed displayed core mean ≥250/255 in any channel; RAW patch clipping is not independently located",
            "per_patch":per_patch}


def _run_display(source, template, expected_contract):
    run, manifest, manifest_path = source
    frames = manifest["frame_sources"]
    nearest = min(range(len(frames)), key=lambda i: abs(float(frames[i]["elapsed_h"])-2.25))
    rows, thumbnails, baseline = [], {}, None
    for index, frame in enumerate(frames):
        sand = _mask(frame,"v5_photo_sand_mask",manifest)
        invalid = _mask(frame,"v5_display_invalid_mask",manifest,compressed=True)
        small = _small_invalid(invalid)
        receipt = _read(_bound(frame["v5_frame_receipt_path"],frame["v5_frame_receipt_sha256"],"frame receipt"))
        metadata = (receipt.get("certificate") or {}).get("display_modes_metadata") or {}
        if metadata.get("contract") != expected_contract:
            raise ValueError("Display QA frame differs from the frozen shared display contract")
        expected_invalid = (metadata.get("invalid_cleanup") or {}).get("invalid_input_pixels")
        if expected_invalid != int(invalid.sum()):
            raise ValueError("Display QA original-invalid mask disagrees with the frozen rendering receipt")
        for mode in MODES:
            rgb, path = _mode_rgb(frame,mode,manifest)
            content = _analysis(rgb,manifest)
            dots, signature = gray_dot_counts(content,small)
            rows.append({"run":run,"frame_index":index,"elapsed_h":float(frame["elapsed_h"]),"mode":mode,
                         "original_invalid_pixels":int(invalid.sum()),"original_small_invalid_pixels":int(small.sum()),
                         "gray128_marker_outlier_pixels":dots,"gray128_signature_pixels_at_original_invalid":signature,
                         "jpeg_path":str(path),"jpeg_sha256":frame["v5_display_modes"][mode]["jpeg_sha256"],"passed":dots==0})
            if index in {0,nearest}:
                thumbnail = Image.fromarray(rgb)
                thumbnail.thumbnail((450,254),Image.Resampling.LANCZOS)
                if index == 0: thumbnails[(mode,"frame000")] = thumbnail.copy()
                if index == nearest: thumbnails[(mode,"nearest2p25")] = thumbnail.copy()
            if index == 0 and mode == "natural":
                value = sand_median_y(content,sand)
                centres, geometry = locate_checker(rgb,template,patch_centres())
                scores = checker_scores(rgb,centres)
                baseline = {"run":run,"natural_sand_median_Y":value,"target_Y":.0141,
                            "relative_difference":value/.0141-1,"sand_luminance_pass":abs(value/.0141-1)<=.05,
                            "chemistry_or_protocol_sensitivity":run in CHEMISTRY_FLAGS,
                            "checker":dict(**geometry,**scores),
                            "white_patch_neutral_pass":scores["white_patch_abs_R_minus_B_rgb8"]<=10,
                            "group_checker_pass":(geometry["template_score"]>=.35 and scores["patches_used"]>=12
                                and scores["mean_abs_error_rgb8"] is not None and scores["mean_abs_error_rgb8"]<=8)}
    return {"run":run,"rows":rows,"baseline":baseline,"thumbnails":thumbnails,
            "nearest_frame":nearest,"nearest_elapsed_h":float(frames[nearest]["elapsed_h"]),
            "manifest_path":str(manifest_path)}


def _contact_sheets(results, output):
    try: font = ImageFont.truetype("arial.ttf",20)
    except OSError: font = ImageFont.load_default()
    paths = []
    for mode in MODES:
        for selection in ("frame000","nearest2p25"):
            sheet = Image.new("RGB",(7*450,((len(results)+6)//7)*286),(24,24,24))
            draw = ImageDraw.Draw(sheet)
            for index,row in enumerate(results):
                x,y = index%7*450,index//7*286
                sheet.paste(row["thumbnails"][(mode,selection)],(x,y+28))
                frame,time = (0,float(row["rows"][0]["elapsed_h"])) if selection=="frame000" else (row["nearest_frame"],row["nearest_elapsed_h"])
                draw.text((x+5,y+3),f"{row['run'].upper()} f{frame:03d} {time:.3f} h · {mode}",fill="white",font=font)
            path = output/f"contact_{mode}_{selection}.jpg"
            with path.open("xb") as stream: sheet.save(stream,format="JPEG",quality=95,subsampling=0)
            paths.append(str(path))
    return paths


def inspect_display_modes(batch_root,output_dir,*,workers=4,expected_runs=43,expected_frames=864):
    """§4a/b/c/g: full-quality numbers, source SHAs, six thumbnail sheets."""
    output = Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    report_path = output/"qa_display_modes.json"
    if report_path.exists(): raise FileExistsError(report_path)
    sources,binding = _sources(batch_root,expected_runs=expected_runs,expected_frames=expected_frames)
    reference = next((item for item in sources if item[0]=="ac23"),None)
    if reference is None: raise ValueError("Display QA fixed AC23 chart template source is absent")
    reference_rgb,_ = _mode_rgb(reference[1]["frame_sources"][0],"natural",reference[1])
    reference_frame = reference[1]["frame_sources"][0]
    reference_receipt = _read(_bound(reference_frame["v5_frame_receipt_path"],reference_frame["v5_frame_receipt_sha256"],"AC23 reference frame receipt"))
    expected_contract = ((reference_receipt.get("certificate") or {}).get("display_modes_metadata") or {}).get("contract")
    if not isinstance(expected_contract,dict) or expected_contract.get("contract") != "ffac-track1-display-modes-v1":
        raise ValueError("Display QA source is not the frozen three-mode contract")
    template = reference_rgb[704:1274,5462:5852].copy()
    if template.shape != (570,390,3): raise ValueError("Display QA fixed AC23 template has another native grid")
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix="ffac-display-qa") as pool:
        results = list(pool.map(lambda source:_run_display(source,template,expected_contract),sources))
    frames = [frame for item in results for frame in item["rows"]]
    baseline = [item["baseline"] for item in results]
    csv_path = output/"display_modes_per_frame.csv"; _csv(csv_path,frames)
    contacts = _contact_sheets(results,output)
    checks = {"a_gray128_marker_outliers":all(row["passed"] for row in frames),
              "b_group_checker_error":all(row["group_checker_pass"] for row in baseline if row["run"] in GROUP_RUNS)
                  and all(run in {row["run"] for row in baseline} for run in GROUP_RUNS),
              "b_white_neutral_all_runs":all(row["white_patch_neutral_pass"] for row in baseline),
              "c_natural_sand_median":all(row["sand_luminance_pass"] for row in baseline),
              "g_all43_six_contact_sheets":len(contacts)==6 and len(results)==expected_runs}
    return _exclusive_json(report_path,{"contract":"ffac-track1-display-modes-qa-v1","created_utc":utc_now(),
        "batch":binding,"batch_root":str(batch_root),"runs":len(results),"frames":expected_frames,"workers":workers,
        "scientific_or_existing_assets_changed":False,"adaptive_gain_or_colour_refit":False,
        "frozen_shared_display_contract":expected_contract,
        "acceptance_checks":checks,"passed":all(checks.values()),
        "gray128_marker_outlier_pixels":sum(row["gray128_marker_outlier_pixels"] for row in frames),
        "dot_scope":"original SHA-bound invalid components ≤64px; grey128-like deviation >20 from local 5x5 median; no photograph modification",
        "checker_template":"AC23 natural f0 fixed native crop x5462:5852,y704:1274; per-run chart-plane template matching, no sand H",
        "checker_limits":{"mean_abs_error_rgb8":8,"white_abs_R_minus_B_rgb8":10,"template_score":.35,"minimum_unclipped_display_patches":12},
        "checker_limitation":"No registered RAW saturation mask exists. Patch exclusions use observed display clipping, not a claim of guaranteed RAW-unclipped patches.",
        "sand_measurement":"Median per-pixel EOTF(decoded natural JPEG) luminance on exact safe_filled; comparison target0.0141±5%; chemical flags do not alter gains",
        "per_run_frame0":baseline,"per_frame_csv":str(csv_path),"contact_sheets":contacts})


def _qimage_rgba(image):
    rows = np.frombuffer(image.constBits(),np.uint8).reshape(image.height(),image.bytesPerLine())
    return rows[:,:image.width()*4].reshape(image.height(),image.width(),4)


def _overlay_run(source,candidate):
    run,manifest,_ = source
    folder = _phase_folder(Path(candidate),run)
    rows = []
    x,y,w,h = map(int,manifest["content_box_xywh"])
    cw,ch = map(int,manifest["canvas_wh"])
    for index,frame in enumerate(manifest["frame_sources"]):
        path = folder/f"frame_{index:03d}.png"
        digest = sha256_file(path)
        image = QImage(str(path)).convertToFormat(QImage.Format.Format_RGBA8888)
        if image.isNull() or (image.width(),image.height()) != (cw,ch): raise ValueError("Display QA phase map has another native grid")
        full_sand = np.zeros((ch,cw),bool); full_sand[y:y+h,x:x+w] = _mask(frame,"v5_photo_sand_mask",manifest)
        original = _qimage_rgba(image)
        display = display_phase_overlay(image,"all",photo_sand=full_sand)
        rendered = _qimage_rgba(display)
        cyan = np.all(original[...,:3]==(30,210,255),axis=-1)&(original[...,3]>0)
        rendered_cyan = np.all(rendered[...,:3]==(30,210,255),axis=-1)&(rendered[...,3]>0)
        inside_delta = int(np.count_nonzero(np.any(original[cyan&full_sand]!=rendered[cyan&full_sand],axis=-1)))
        outside = int(np.count_nonzero(rendered_cyan&~full_sand))
        immutable = digest==sha256_file(path)
        rows.append({"run":run,"frame_index":index,"elapsed_h":frame.get("elapsed_h"),
                     "stored_cyan_outside_sand_pixels":int(np.count_nonzero(cyan&~full_sand)),
                     "rendered_cyan_outside_sand_pixels":outside,"cyan_inside_changed_pixels":inside_delta,
                     "cyan_inside_pixels":int(np.count_nonzero(cyan&full_sand)),
                     "phase_png_sha256":digest,"classification_file_unchanged":immutable,
                     "passed":not outside and not inside_delta and immutable})
    return rows


def inspect_overlay_rendering(batch_root,candidate,output_dir,*,workers=4,expected_runs=43,expected_frames=864):
    """§4d uses the actual Qt scene filter, retaining full classification files."""
    output = Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    report = output/"qa_display_overlay.json"
    if report.exists(): raise FileExistsError(report)
    sources,binding = _sources(batch_root,expected_runs=expected_runs,expected_frames=expected_frames)
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix="ffac-overlay-qa") as pool:
        rows = [row for group in pool.map(lambda source:_overlay_run(source,candidate),sources) for row in group]
    csv_path = output/"display_overlay_per_frame.csv"; _csv(csv_path,rows)
    return _exclusive_json(report,{"contract":"ffac-track1-display-overlay-qa-v1","created_utc":utc_now(),
        "candidate":str(candidate),"batch":binding,"runs":len(sources),"frames":len(rows),"workers":workers,
        "rendered_cyan_outside_sand_pixels":sum(row["rendered_cyan_outside_sand_pixels"] for row in rows),
        "cyan_inside_changed_pixels":sum(row["cyan_inside_changed_pixels"] for row in rows),
        "classification_files_unchanged":all(row["classification_file_unchanged"] for row in rows),
        "passed":all(row["passed"] for row in rows),"per_frame_csv":str(csv_path)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("display","overlays"))
    parser.add_argument("--batch-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--candidate",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16: parser.error("workers must be 1–16")
    if args.command == "display": result = inspect_display_modes(args.batch_root,args.output,workers=args.workers)
    else:
        if args.candidate is None: parser.error("overlays requires --candidate")
        result = inspect_overlay_rendering(args.batch_root,args.candidate,args.output,workers=args.workers)
    print(json.dumps({"passed":result["passed"],"runs":result["runs"],"frames":result["frames"]}))


if __name__ == "__main__":
    main()
