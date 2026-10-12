"""Round-four retained NPZ products, English previews, films and mass plots.

The caller owns the immutable job, fresh temporary directory, live guard,
calibration/check decision and verified NAS publication. No computation,
backend qualification, worker launch, or production permission is granted here.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import uuid


PRODUCT_NAMES = ("concentration_aqueous", "saturation_free_gas", "mass_total",
                 "mass_aqueous", "mass_free_gas", "phase_labels")
FLOAT_PRODUCTS = PRODUCT_NAMES[:-1]
UREST_AMENDMENT_SHA256 = "f1567b3f8724b41c6401e8d72d139c45910cd4a6aeba09dba75436da2a0882d0"


def anchor_from_geometry(frame, *, selected_target_h):
    """Exact original2.25h affected-volume formula without an RGB/model call.

    `frame` may be a geometry-only object from the already admitted phase,
    measured area and corrected physical map. The caller retains the original
    anchor metadata/phase/source admission; this function supplies no admission.
    Its multiply, where and sum order is the literal R2 volume prefix.
    """
    import numpy as np
    if selected_target_h != 2.25:
        raise ValueError("The exact indexed end-of-injection target is required")
    domain, labels, known = frame.domain, frame.phase_labels, frame.known_physical_domain
    shape = domain.shape
    if (domain.dtype != bool or domain.ndim != 2 or known.dtype != bool or known.shape != shape
            or labels.dtype != np.uint8 or labels.shape != shape or np.any(known & ~domain)
            or not np.isin(labels, (0,1,2,255)).all()):
        raise ValueError("Original admitted categorical domain and measured physical support are required")
    positive = domain & np.isin(labels, (1,2))
    area = np.broadcast_to(np.asarray(frame.pixel_area_m2, np.float64), shape)
    depth = np.broadcast_to(np.asarray(frame.depth_m, np.float64), shape)
    phi = np.broadcast_to(np.asarray(frame.porosity, np.float64), shape)
    if (not np.isfinite(area[domain]).all() or np.any(area[domain] <= 0)
            or not np.isfinite(depth[known]).all() or np.any(depth[known] <= 0)
            or not np.isfinite(phi[known]).all() or np.any(phi[known] <= 0) or np.any(phi[known] > 1)):
        raise ValueError("Measured area/depth/porosity must satisfy original physical guards")
    volume = np.where(known, area*depth*phi, 0.)
    affected = float(np.sum(volume[positive]))
    if affected <= 0: raise ValueError("The end-of-injection image has no measured affected pore volume")
    return dict(elapsed_h=frame.elapsed_h, affected_pore_volume_m3=affected,
        frame_index=frame.frame_index, source_binding=dict(frame.provenance),
        affected_volume_includes_excluded_known_physical=True,
        affected_volume_imputes_missing_geometry=False,
        unknown_geometry_phase_pixels=int(np.count_nonzero(positive & ~known)))


def geometry_provider(context, case_id, prepared, *, correction_matrix=None):
    """Actual original-receipt geometry and optical-support sandmap callback.

    A correction, if used, is the single measured AC28/reference-grid shift and
    scale. With no supplied correction this retains the current registration;
    it makes no claim that photographic offsets have passed the measurement.
    No other run's source, phase, floor, or physical table is substituted.
    """
    import numpy as np
    from darsia.fluidflower.tools.build_ffac_p1_final2_candidate import transported_floor
    case = context.cases[case_id]
    entry, reader, session = case["entry"], context.reader, context.session
    if entry["run"] != prepared.run or entry["frame_index"] != prepared.frame_index:
        raise ValueError("Round4 geometry callback has a different source acquisition")
    receipt = reader.json(entry["receipt"])
    revision = receipt["rig_domain_revision"]
    if tuple(revision["phase_shape_hw"]) != prepared.domain.shape:
        raise ValueError("Bound physical coordinates differ from the selected phase grid")
    geometry = receipt["full_extended_scientific"]["binding"]["geometry"]
    coordinate_module, source = context.modules["geometry"], context.modules["source"]
    x, y = coordinate_module.frame_coordinates(session["matrix"], geometry, revision)
    floor = transported_floor(session["floor_curve"], dict(
        H_ac28_to_frame_native=geometry["H_ac28_to_frame_native"], rig_domain_revision=revision))
    # This ORIGINAL function retains actual RAW validity, registered baseline
    # coverage and clear-water support. RGB-finiteness alone is insufficient.
    optical = coordinate_module.measured_photo_support(prepared, receipt, reader)
    correction = np.eye(3) if correction_matrix is None else np.asarray(correction_matrix, np.float64)
    facies = None
    if not np.array_equal(correction, np.eye(3)):
        if (correction.shape != (3, 3) or not np.isfinite(correction).all()
                or correction[0, 0] <= 0 or correction[0, 0] != correction[1, 1]
                or correction[0, 1] != 0 or correction[1, 0] != 0
                or not np.array_equal(correction[2], (0., 0., 1.))):
            raise ValueError("Only the measured single AC28 shift/scale is permitted")
        reference = source.load_physical_reference(session["parts"]["physical_reference"], reader)
        matrix = source.matrix_ac28_to_canvas(geometry, revision["native_origin_xy"])
        cv2 = context.modules["adapter"].cv2
        facies = cv2.warpPerspective(np.asarray(reference["facies"], np.uint8), matrix @ correction,
            prepared.domain.shape[::-1], flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return dict(x_m=x, y_m=y, floor_curve_canvas_y=floor, depth_points=session["depth_points"],
        porosity_by_facies=session["porosity"], swi_by_facies=session["swi"],
        optical_support=optical, transported_facies=facies,
        source_binding=dict(phase_receipt=entry["receipt"], source=entry["source_binding"],
                            physical_reference=session["parts"]["physical_reference"],
                            correction_matrix=correction.tolist()))


def _array_digest(value):
    """C-order byte identity without an additional full contiguous array."""
    import numpy as np
    value = np.asarray(value)
    digest = hashlib.sha256()
    for chunk in np.nditer(value, flags=["external_loop", "buffered", "zerosize_ok"],
                          op_flags=["readonly"], order="C", buffersize=65536):
        digest.update(chunk.tobytes(order="C"))
    return dict(dtype=value.dtype.str, shape=list(value.shape), sha256=digest.hexdigest())


def _write_npz(path, arrays, *, io, guard):
    """Standard NumPy compressed NPZ, chosen data types and no overwrite."""
    import numpy as np
    guard()
    # Lossless standard NPY members; level1 avoids spending CPU on level6
    # compression while preserving every value, dtype, shape and field name.
    import zipfile
    with zipfile.ZipFile(path, mode="x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for name, value in arrays.items():
            with archive.open(name+".npy", mode="w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(value), allow_pickle=False)
    guard()
    return io.asset(path)


def _relative(bound):
    return dict(path=Path(bound["path"]).name, sha256=bound["sha256"], size_bytes=bound["size_bytes"])


def _static(response, frame_folder, *, run, static_root, registry, io, guard):
    """One static pack, with explicit immutable per-frame geometry if it varies."""
    import numpy as np
    frame = response["frame"]
    shape = frame.domain.shape
    fields = dict(pixel_area_m2=np.broadcast_to(np.asarray(frame.pixel_area_m2), shape),
        depth_m=np.broadcast_to(np.asarray(frame.depth_m), shape),
        porosity=np.broadcast_to(np.asarray(frame.porosity), shape), facies=frame.facies,
        transport_weight_l=np.asarray(response["analysis"].geometry.cached_voxel_volume, np.float64)*1000.)
    identities = {name:_array_digest(value) for name,value in fields.items()}
    units = dict(pixel_area_m2="m2", depth_m="m", porosity="1", facies="category",
                 transport_weight_l="L pore volume per pixel")
    if run not in registry:
        parent = Path(static_root)
        guard();parent.mkdir(parents=True, exist_ok=True)
        target = parent/run
        if target.exists(): raise FileExistsError(target)
        temporary = parent/("."+run+".partial."+uuid.uuid4().hex)
        temporary.mkdir()
        pack = _write_npz(temporary/"static.npz", fields, io=io, guard=guard)
        metadata = dict(run=run, arrays=identities, units=units)
        io.write_json(temporary/"static.json", metadata)
        expected = {"static.npz":pack, "static.json":io.asset(temporary/"static.json")}
        guard();io.publish_directory(temporary, target, expected)
        registry[run] = dict(asset=io.asset(target/"static.npz"), arrays=identities, units=units)
    base = registry[run]
    if base["arrays"] == identities:
        return dict(base, per_frame_geometry=False)
    # Actual original AC19 grids can vary. Never replace a measured frame's
    # area/depth/volume by the first frame merely to call them static.
    pack = _write_npz(Path(frame_folder)/"native_geometry.npz", fields, io=io, guard=guard)
    return dict(asset=_relative(pack), arrays=identities, units=units,
                per_frame_geometry=True, original_run_static=base["asset"])


def _preview_photo(frame, registered_photo, shape, *, exposure, gamma):
    import numpy as np
    import cv2
    if registered_photo is not None:
        photo = np.asarray(registered_photo)
        if photo.dtype != np.uint8 or photo.shape != (*frame.domain.shape, 3):
            raise ValueError("The source-bound display photo must be registered to this exact frame grid")
        policy = "source-bound existing registered display; analysis arrays unchanged"
    else:
        if not np.isfinite(exposure) or exposure <= 0 or not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("Explicit fixed display exposure/gamma are required")
        # Downsample BEFORE display conversion. This is only a thumbnail and
        # is never fed back to calibration, analysis, masks or scientific NPZ.
        values = cv2.resize(frame.current_rgb, shape[::-1], interpolation=cv2.INTER_AREA)
        values = np.nan_to_num(values, nan=0., posinf=0., neginf=0.)
        photo = np.rint(np.clip(values*exposure, 0., 1.)**gamma*255.).astype(np.uint8)
        policy = f"registered scientific current_rgb preview only; fixed exposure={exposure}, gamma={gamma}"
        return photo, policy
    return cv2.resize(photo, shape[::-1], interpolation=cv2.INTER_AREA), policy


def _provisional_banner(image, renderer, failed_check):
    """Display-only banner; the caller's explicit failed check is retained."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 78), fill=(110, 30, 20))
    label = "PROVISIONAL — " + (failed_check if failed_check else "checks not accepted")
    draw.text((65, 15), label + "; not for scientific use",
              fill=(255,255,255), font=renderer.font(38, True))


def write_products(temporary, *, response, result, export, prediction, bindings,
                   io, guard, static_root, static_registry, renderer,
                   frame_position, total_frames, solubility_limit_g_l,
                   registered_photo=None, preview_exposure=1., preview_gamma=1/2.2,
                   provisional=False, failed_check=None, delegation=None, write_record=True):
    """Save final frozen products and ONE English preview while arrays are alive.

    `bindings` explicitly contains source_sha256, phase_sha256,
    calibration_sha256 and code_sha256. The caller then commits this folder
    with its existing LocalProductionJob and publishes via its existing copier.
    No product is downcast: locked float32 OR float64 is saved honestly.
    """
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw
    if not callable(guard) or type(provisional) is not bool or type(write_record) is not bool:
        raise ValueError("The actual live guard and explicit provisional state are required")
    if failed_check is not None and (not isinstance(failed_check, str) or not failed_check.strip()):
        raise ValueError("An explicit failed-check name must be nonempty")
    if failed_check is not None: provisional = True
    if (type(total_frames) is not int or total_frames < 1 or type(frame_position) is not int
            or not 0 <= frame_position < total_frames):
        raise ValueError("The exact selected-frame count and chronological position are required")
    folder, frame = Path(temporary), response["frame"]
    existing = list(folder.iterdir()) if folder.is_dir() else None
    if existing is None or any(path.name != "native_statistics.json.gz" or not path.is_file() for path in existing):
        raise ValueError("Fresh owned frame temporary may contain only its original native_statistics.json.gz")
    precision = result.get("precision", result.get("working_pixel_dtype"))
    if precision not in ("float32", "float64"):
        raise ValueError("The locked true working precision must be explicit")
    products = export["products"]
    if set(products) != set(PRODUCT_NAMES):
        raise ValueError("Exactly the five required floating products and locked phase labels are required")
    shape, dtype = frame.domain.shape, np.dtype(precision)
    for name in FLOAT_PRODUCTS:
        if products[name].shape != shape or products[name].dtype != dtype:
            raise ValueError("Saved product dtype/grid differs from the locked working precision: "+name)
    if products["phase_labels"].dtype != np.uint8 or not np.array_equal(products["phase_labels"], frame.phase_labels):
        raise ValueError("Scientific products must preserve the original categorical phase map")
    main = result["fields"]["main_mass_eligible_mask"]
    for name in ("mass_total", "mass_aqueous", "mass_free_gas"):
        if np.any(products[name][~main] != 0):
            raise ValueError("Main mass maps must be zero outside M")
    for name in ("source_sha256", "phase_sha256", "calibration_sha256", "code_sha256"):
        value = bindings.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Explicit actual SHA-256 binding absent: "+name)
    urest = prediction["U_rest_g"]
    if isinstance(urest, (bool, np.bool_)):
        raise ValueError("U_rest must be its actual numerical capacity or undefined")
    urest_defined = urest is not None and bool(np.isfinite(urest))
    urest_value = float(urest) if urest_defined else None
    outside = dict(frame.excluded_masks or {})
    outside["unquantified_facies"] = frame.domain & np.isin(frame.facies, (0,3,4,6))
    packed = dict(products, main_mass_eligible_mask=main)
    for name,mask in outside.items():
        if np.asarray(mask).dtype != bool or mask.shape != shape:
            raise ValueError("Excluded mask differs from the native grid")
        packed[name+"_mask"] = mask
    for name in ("E_lo_g", "E_hi_g"):
        value = prediction[name]
        if not np.isfinite(value): raise ValueError("Outside-M estimate is not a finite reported scalar")
        packed[name] = np.asarray(value, np.float64)
    packed["U_rest_g"] = np.asarray(urest_value if urest_defined else np.nan, np.float64)
    packed["U_rest_defined"] = np.asarray(urest_defined, bool)
    packed["U_rest_amendment_sha256"] = np.asarray(UREST_AMENDMENT_SHA256)
    for name in ("source_sha256", "phase_sha256", "calibration_sha256", "code_sha256"):
        packed[name] = np.asarray(bindings[name])
    static = _static(response, folder, run=frame.run_id, static_root=static_root,
                     registry=static_registry, io=io, guard=guard)
    product_asset = _write_npz(folder/"products.npz", packed, io=io, guard=guard)
    record = dict(contract="ffac-round4-retained-products-v1", complete=True, run=frame.run_id,
        frame_index=frame.frame_index, elapsed_h=float(frame.elapsed_h),
        selected_frame_count=total_frames, chronological_position=frame_position,
        selected_target_h=prediction["selected_target_h"], working_pixel_dtype=precision,
        reduction_dtype="float64", products=_relative(product_asset),
        units=export["units"], static=static, bindings=dict(bindings),
        total_mass_g=float(result["total_mass_g"]), aqueous_mass_g=float(result["aqueous_mass_g"]),
        gas_mass_g=float(result["gas_mass_g"]), injected_mass_g=prediction.get("injected_mass_g"),
        domains=result["domains"], excluded_mass=result["excluded_mass"],
        E_lo_g=float(prediction["E_lo_g"]), E_hi_g=float(prediction["E_hi_g"]),
        U_rest_g=urest_value, U_rest_defined=urest_defined,
        U_rest_amendment_sha256=UREST_AMENDMENT_SHA256,
        W_g=prediction.get("W_g"), W_over_d=prediction.get("W_over_d"),
        outside_M_parts=prediction.get("excluded_parts", {}),
        sandmap=frame.provenance.get("round4_sandmap"), provisional=provisional,
        failed_check=failed_check, provisional_delegation=delegation,
        phase_labels_changed=False, field_precision_changed=False, Wasserstein_performed=False)
    if not np.isfinite(solubility_limit_g_l) or solubility_limit_g_l <= 0:
        raise ValueError("One fixed solubility colour limit for every run is required")
    ratio = min(1690/shape[1], 922/shape[0], 1.)
    preview_shape = (max(1, round(shape[0]*ratio)), max(1, round(shape[1]*ratio)))
    photo, policy = _preview_photo(frame, registered_photo, preview_shape,
                                   exposure=preview_exposure, gamma=preview_gamma)
    concentration = cv2.resize(products["concentration_aqueous"], preview_shape[::-1], interpolation=cv2.INTER_NEAREST)
    phase = cv2.resize(products["phase_labels"], preview_shape[::-1], interpolation=cv2.INTER_NEAREST)
    aqueous = (phase == 1) & np.isfinite(concentration)
    quantified_gas = ((frame.phase_labels == 2) & frame.known_physical_domain
                      & np.isfinite(response["swi"]) & np.isfinite(products["saturation_free_gas"]))
    gas = cv2.resize(quantified_gas.astype(np.uint8),preview_shape[::-1],interpolation=cv2.INTER_NEAREST).astype(bool)
    background = renderer.make_background("ffac", frame.run_id, solubility_limit_g_l, language="en")
    preview = renderer.compose(background, photo, concentration, aqueous, gas, record,
                               frame_position, total_frames, solubility_limit_g_l, language="en")
    image = Image.fromarray(preview).copy()
    ignored = response.get("metadata", {}).get("ignore_labels", ())
    skipped_native = frame.domain & np.isin(frame.facies, ignored) & ~quantified_gas
    skipped = cv2.resize(skipped_native.astype(np.uint8), preview_shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    skipped |= (phase != 2) & ~np.isfinite(concentration)
    skipped |= (phase == 2) & ~gas
    if skipped.any():
        display = renderer.colors(concentration, aqueous, gas, solubility_limit_g_l)
        display[skipped] = (145,145,145)
        renderer.put_panel(image, display, (1900,159,1690,922), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    draw.text((65, 1100), f"t = {frame.elapsed_h:.2f} h", fill=renderer.FG, font=renderer.font(32, True))
    # The old renderer assumes72h; retain its panels/legend but replace only
    # that decorative timeline, since selected runs can extend to96h.
    draw.rectangle((60, 1300, 3780, 1435), fill=renderer.BG)
    draw.text((65, 1340), "Selected acquisitions · fixed colour scale · no temporal interpolation",
              fill=renderer.MUTED, font=renderer.font(25))
    if skipped.any():
        draw.rectangle((1900,1100,1926,1126), fill=(145,145,145))
        draw.text((1938,1095), "Unquantified facies / unavailable concentration", fill=renderer.MUTED, font=renderer.font(25))
    if provisional:
        _provisional_banner(image, renderer, failed_check)
    guard();image.save(folder/"preview.png")
    record.update(preview=_relative(io.asset(folder/"preview.png")), preview_policy=policy,
                  preview_language="en", fixed_concentration_limit_g_l=float(solubility_limit_g_l),
                  skipped_facies_display="gray; scientific products and phase labels unchanged")
    sandmap=record["sandmap"]
    if sandmap is not None:
        for part in ("added_to_M","filled_without_optical_support","below_sand_floor"):
            for quantity in ("area_cm2","co2_area_cm2"):
                record[part+"_"+quantity]=sandmap[part][quantity]
        at24=float(record["selected_target_h"]) == 24.
        last=frame_position == total_frames-1
        record["required_sandmap_area_time"]="24h_and_last" if at24 and last else "24h" if at24 else "last" if last else None
    if frame.run_id == "ac19" and float(record["selected_target_h"]) in (24.,96.):
        # Display-only edges on the SAME registered thumbnail already prepared
        # above. No scientific image decode, model call or array modification.
        facies=cv2.resize(frame.facies,preview_shape[::-1],interpolation=cv2.INTER_NEAREST)
        domain=cv2.resize(frame.domain.astype(np.uint8),preview_shape[::-1],interpolation=cv2.INTER_NEAREST).astype(bool)
        layer=np.zeros(domain.shape,bool);boundary=np.zeros(domain.shape,bool)
        layer[1:]|=facies[1:] != facies[:-1];layer[:-1]|=facies[1:] != facies[:-1]
        layer[:,1:]|=facies[:,1:] != facies[:,:-1];layer[:,:-1]|=facies[:,1:] != facies[:,:-1]
        boundary[1:]|=domain[1:] != domain[:-1];boundary[:-1]|=domain[1:] != domain[:-1]
        boundary[:,1:]|=domain[:,1:] != domain[:,:-1];boundary[:,:-1]|=domain[:,1:] != domain[:,:-1]
        edge_photo=photo.copy();edge_photo[layer & domain]=(255,230,0);edge_photo[boundary]=(0,240,255)
        figure=Image.new("RGB",(preview_shape[1]+80,preview_shape[0]+150),renderer.BG)
        figure.paste(Image.fromarray(edge_photo),(40,100))
        label=ImageDraw.Draw(figure)
        label.text((40,16),f"AC19 corrected sand map · t = {frame.elapsed_h:.2f} h",fill=renderer.FG,font=renderer.font(26,True))
        label.text((40,57),"Yellow: facies edges · Cyan: rig domain edge · Registered photograph",fill=renderer.MUTED,font=renderer.font(20))
        if provisional:_provisional_banner(figure,renderer,failed_check)
        guard();figure.save(folder/"physical_edges.png")
        record.update(physical_edges=_relative(io.asset(folder/"physical_edges.png")),physical_edges_language="en",
                      physical_edges_display_only=True)
    if write_record:
        guard();io.write_json(folder/"result.json", record)
    return record


def render_run(records, *, frame_roots, output_dir, renderer, io, guard, fps=3,
               failed_check=None, delegation=None):
    """One full movie and every PNG; reuse existing Encoder, no NPZ decoding."""
    from PIL import Image
    import numpy as np
    rows = list(records)
    if failed_check is not None:
        if not isinstance(failed_check, str) or not failed_check.strip() or delegation is None:
            raise ValueError("A provisional failed-check film requires its actual name and caller-supplied delegation")
    if not rows or fps != 3 or len({row["run"] for row in rows}) != 1:
        raise ValueError("One complete chronological run at the fixed3fps is required")
    times = [row["elapsed_h"] for row in rows]
    if any(b <= a for a,b in zip(times,times[1:])) or any(row.get("complete") is not True for row in rows):
        raise ValueError("Every selected acquisition must have completed in strict camera-time order")
    if (len(frame_roots) != len(rows) or any(row.get("selected_frame_count") != len(rows) for row in rows)
            or [row.get("chronological_position") for row in rows] != list(range(len(rows)))):
        raise ValueError("A partial selected run cannot be presented as a full film")
    limits = {row["fixed_concentration_limit_g_l"] for row in rows}
    if len(limits) != 1 or any(row["preview_language"] != "en" for row in rows):
        raise ValueError("English and one fixed concentration scale are required")
    target = Path(output_dir)
    guard();target.mkdir(parents=True, exist_ok=False)
    run = rows[0]["run"]
    encoder = renderer.Encoder(target/(run+".mp4"), fps)
    frames = []
    try:
        for index,row in enumerate(rows):
            guard()
            source = dict(row["preview"], path=str(Path(frame_roots[index])/row["preview"]["path"]))
            path = io.verify(source)
            with Image.open(path) as image:
                image = image.convert("RGB")
                if image.size != renderer.SIZE: raise ValueError("Compact preview size changed")
                if failed_check is not None: _provisional_banner(image, renderer, failed_check)
                encoder.add(np.asarray(image), 1)
                if failed_check is not None:
                    # A NEW display PNG has the same failure banner as the new
                    # film. The original immutable preview stays byte-identical.
                    guard();image.save(target/(f"{index:05d}.png"))
                    output = io.asset(target/(f"{index:05d}.png"))
                else:
                    output = io.copy_file(source, target/(f"{index:05d}.png"))
            frames.append(dict(frame_index=row["frame_index"], elapsed_h=row["elapsed_h"], preview=output))
        guard();encoder.close()
    except BaseException:
        if encoder.process.poll() is None: encoder.abort()
        raise
    verification = renderer.validate_video(encoder.output, len(rows), fps)
    value = dict(contract="ffac-round4-full-selected-run-film-v1", complete=True, run=run,
        movie=io.asset(encoder.output), frames=frames, source_frame_count=len(rows), fps=fps,
        language="en", chronological_all_selected_frames=True, fixed_scale_g_l=next(iter(limits)),
        temporal_interpolation=False, scientific_NPZ_decoded=False,
        provisional=failed_check is not None or any(row["provisional"] for row in rows),
        failed_check=failed_check, provisional_delegation=delegation, verification=verification)
    guard();io.write_json(target/"film.json", value)
    return value


def mass_tables(records, output, *, io, guard):
    """Masses per domain and excluded parts, including the fixed outside bounds."""
    rows = list(records)
    target = Path(output)
    fields = ("run","frame_index","elapsed_h","selected_target_h","domain","total_mass_g",
              "aqueous_mass_g","gas_mass_g","injected_mass_g","E_lo_g","E_hi_g","U_rest_g",
              "U_rest_defined","W_g","W_over_d","required_sandmap_area_time",
              "added_to_M_area_cm2","added_to_M_co2_area_cm2",
              "filled_without_optical_support_area_cm2","filled_without_optical_support_co2_area_cm2",
              "below_sand_floor_area_cm2","below_sand_floor_co2_area_cm2")
    values = []
    for row in rows:
        for domain,masses in {**row["domains"], **{"excluded:"+k:v for k,v in row["excluded_mass"].items()}}.items():
            value = {name:row[name] for name in fields if name in row}
            value.update(domain=domain, **{k:masses.get(k) for k in ("total_mass_g","aqueous_mass_g","gas_mass_g")})
            values.append(value)
    guard()
    with target.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader();writer.writerows(values)
    guard()
    return io.asset(target)


def plot_masses(records, output, *, io, guard, aggregate=False):
    """English injected/detected/dissolved/free-gas plots; no fitted adjustments."""
    import numpy as np
    import matplotlib.pyplot as plt
    rows = list(records)
    if not rows: raise ValueError("Actual completed mass records are required")
    target = Path(output)
    if target.exists(): raise FileExistsError(target)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    try:
        if aggregate:
            times = sorted({row["selected_target_h"] for row in rows})
            for key,label,color in (("injected_mass_g","Injected","black"),("total_mass_g","Detected total","C0"),
                    ("aqueous_mass_g","Dissolved CO2","C1"),("gas_mass_g","Free gas CO2","C2")):
                medians, low, high = [], [], []
                for t in times:
                    sample = [row[key] for row in rows if row["selected_target_h"] == t and row.get(key) is not None]
                    if not sample: medians.append(np.nan);low.append(np.nan);high.append(np.nan)
                    else:
                        q = np.percentile(sample, (25,50,75));low.append(q[0]);medians.append(q[1]);high.append(q[2])
                ax.plot(times, medians, label=label+" median", color=color)
                ax.fill_between(times, low, high, color=color, alpha=.13)
            ax.set_title("All selected runs · median and interquartile range")
        else:
            if len({row["run"] for row in rows}) != 1: raise ValueError("Per-run plot requires exactly one run")
            ordered = sorted(rows,key=lambda row:row["elapsed_h"])
            x = [row["elapsed_h"] for row in ordered]
            for key,label,color in (("injected_mass_g","Injected","black"),("total_mass_g","Detected total","C0"),
                    ("aqueous_mass_g","Dissolved CO2","C1"),("gas_mass_g","Free gas CO2","C2")):
                ax.plot(x, [np.nan if row.get(key) is None else row[key] for row in ordered], label=label, color=color)
            ax.fill_between(x, [row["total_mass_g"]+row["E_lo_g"] for row in ordered],
                              [row["total_mass_g"]+row["E_hi_g"]+row["U_rest_g"]
                               if row.get("U_rest_defined") else np.nan for row in ordered],
                              color="C0",alpha=.12,label="Main plus outside-M interval (where defined)")
            run = ordered[0]["run"].upper();ax.set_title(run+(" · descriptive" if run == "AC38" else ""))
        if any(row["provisional"] for row in rows):
            names = sorted({row["failed_check"] for row in rows if row.get("failed_check")})
            label = "PROVISIONAL — " + ("; ".join(names) if names else "checks not accepted")
            ax.text(.5,.98,label,transform=ax.transAxes,ha="center",va="top",color="darkred")
        ax.set_xlabel("Time since first injection (h)");ax.set_ylabel("CO2 mass (g)")
        ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        guard();fig.savefig(target,dpi=160)
    finally: plt.close(fig)
    guard();return io.asset(target)


def publish_media(local, destination, *, io, guard):
    """Verified NAS publication through the one existing NasIO implementation."""
    root = Path(local)
    files = {path.relative_to(root).as_posix():io.asset(path) for path in root.rglob("*") if path.is_file()}
    guard()
    result = io.copy_directory(root, destination, files)
    guard()
    return result
