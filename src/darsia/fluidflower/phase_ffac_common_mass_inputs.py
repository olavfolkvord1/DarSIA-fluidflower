"""Source-bound inputs for one FF_AC native calibration, without fitting.

All paths are supplied by the caller. Metadata discovery never opens a phase
image, a RAW array or a held-out run's pixels. Scientific loading is explicit
and requires a frozen calibration before a held-out run can be evaluated.
Historical JPEG color paths and historical unregistered facies are not inputs.
"""
from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image

CONTRACT = "ffac-common-native-p1-final3-inputs-v1"
PHASE_CONTRACT = "ffac-g3-t4c-p1-final3-candidate-v1"
EXCLUDED = ("ac20", "ac24", "ac27", "ac51")
HELD_OUT = ("ac19", "ac25", "ac35", "ac41", "ac47", "ac52", "ac56",
            "ac60", "ac21", "ac22", "ac32", "ac44")
UNQUANTIFIED_FACIES = (0, 3, 4, 6)
CAVEATS = {
    "ac21": "Lower pH or acid residue after AC20; documented recipe retained.",
    "ac22": "Acid residue at the bottom after AC20; documented recipe retained.",
    "ac32": "Protocol probably interchanges BTB and NaOH; no independent weighing log.",
    "ac44": "Protocol gives 58% BTB; images reportedly look standard. Protocol dose retained.",
    "ac38": "Injection time reconstructed (~30 s); pressure/temperature copied from AC39; original chemistry workbook unavailable.",
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset(path: str | Path) -> dict:
    path = Path(path)
    return dict(path=str(path), sha256=sha256(path), size_bytes=path.stat().st_size)


def read_bound_json(descriptor: Mapping) -> dict:
    path = Path(descriptor["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != descriptor["sha256"]:
        raise ValueError(f"Source JSON SHA-256 differs: {path}")
    result = json.loads(data.decode("utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError("Expected source-bound JSON object")
    return result


def write_new_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def split_runs(runs) -> dict:
    normalized = tuple(sorted(str(r).casefold() for r in runs))
    if len(normalized) != 43 or len(set(normalized)) != 43:
        raise ValueError("The locked candidate must contain all 43 unique runs")
    if not set(EXCLUDED).issubset(normalized) or not set(HELD_OUT).issubset(normalized):
        raise ValueError("Locked split is not contained in phase candidate")
    production = tuple(r for r in normalized if r not in EXCLUDED)
    calibration = tuple(r for r in production if r not in HELD_OUT)
    if len(production) != 39 or len(calibration) != 27:
        raise ValueError("Exact 39 production / 27 calibration / 12 holdout split required")
    return dict(production_runs=list(production), calibration_runs=list(calibration),
                held_out_runs=list(HELD_OUT), excluded_runs=list(EXCLUDED))


def require_evaluation_access(run: str, *, frozen_calibration: Mapping | None) -> None:
    run = str(run).casefold()
    if run in EXCLUDED:
        raise ValueError("Excluded chemical outlier has no production maps in this step")
    if run in HELD_OUT:
        if not frozen_calibration or frozen_calibration.get("frozen") is not True:
            raise ValueError("Freeze calibration before any held-out pixel computation")
        files = frozen_calibration.get("calibration_files") or {}
        frozen = [dict(path=path, sha256=value) if isinstance(value, str) else value
                  for path, value in files.items()]
        if not frozen:
            frozen = frozen_calibration.get("files") or []
        if not frozen:
            raise ValueError("Frozen calibration needs its immutable SHA-bound files")
        for row in frozen:
            if sha256(row["path"]) != row["sha256"]:
                raise ValueError("Frozen calibration changed before holdout evaluation")


def _scientific_assets(descriptor: Mapping) -> list[dict]:
    if (descriptor.get("contract") != "ffac-rig-frozen-raw-camera-ring-v1"
            or descriptor.get("complete") is not True
            or descriptor.get("unit_denominator") != 65535
            or descriptor.get("storage") != "exact-native-centre-plus-disjoint-frozen-raw-ring-windows"):
        raise ValueError("Exact frozen float RAW centre/ring contract required")
    centre = descriptor["native_centre"]
    rows = [dict(centre["log_linear_rgb"], role="log_float32_native_centre"),
            dict(centre["valid_mask"], role="valid_native_centre")]
    rows.extend(dict(window["log_linear_rgb"], role="log_float32_ring",
                     canvas_xyxy=window["canvas_xyxy"])
                for window in descriptor["ring_windows"])
    return rows


def metadata_inventory(candidate_root: str | Path, *, workers: int = 8) -> dict:
    """Inspect exact saved source/time/grid bindings without loading pixels."""
    root = Path(candidate_root)
    seal_path = root / "rig_camera_candidate_receipt.json"
    seal = read_bound_json(asset(seal_path))
    if seal.get("complete") is not True or seal.get("contract") != PHASE_CONTRACT:
        raise ValueError("Completed P1 final 3 candidate required")
    paths = sorted((root / "rig_camera_frames").glob("*/frame_*.json"))
    split = split_runs({p.parent.name for p in paths})
    if len(paths) != 864:
        raise ValueError("Locked candidate needs 864 committed frame receipts")

    def inspect(path):
        descriptor = asset(path)
        row = read_bound_json(descriptor)
        source = row["binding"]["source"]
        run, index = path.parent.name, int(path.stem.removeprefix("frame_"))
        if (row.get("contract") != PHASE_CONTRACT or row.get("complete") is not True
                or row["run"] != run or row["frame_index"] != index
                or source["run"] != run or source["frame_index"] != index
                or row["source_sha256"] != source["source_sha256"]
                or row["elapsed_h"] != source["elapsed_h"]):
            raise ValueError("Committed phase receipt has ambiguous source/time identity")
        rig = row["rig_domain_revision"]
        if rig.get("physical_scale_approved") is not True:
            raise ValueError("Measured pixel area has not been approved")
        current, baseline = row["full_extended_scientific"], row["baseline_extended_scientific"]
        if current["source_sha256"] != source["source_sha256"]:
            raise ValueError("RAW and locked phase source differ")
        if current["phase_shape_hw"] != rig["phase_shape_hw"]:
            raise ValueError("RAW and locked phase grid differ")
        missing = [a["path"] for a in (*_scientific_assets(current), *_scientific_assets(baseline),
                                     row["labels_map"], rig["area_map"], rig["classes"])
                   if not Path(a["path"]).is_file()]
        return dict(run=run, frame_index=index, elapsed_h=float(source["elapsed_h"]),
                    source_sha256=source["source_sha256"], receipt=descriptor,
                    labels=row["labels_map"], current_scientific_assets=_scientific_assets(current),
                    baseline_scientific_assets=_scientific_assets(baseline), area_map=rig["area_map"],
                    rig_classes=rig["classes"], phase_shape_hw=rig["phase_shape_hw"],
                    native_origin_xy=rig["native_origin_xy"], missing_assets=missing,
                    source_changed=bool(source.get("source_changed")),
                    held_out=run in HELD_OUT, excluded=run in EXCLUDED,
                    caveat=CAVEATS.get(run, ""))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(inspect, paths))
    return dict(contract=CONTRACT, candidate_root=str(root), candidate_seal=asset(seal_path),
                split=split, frame_count=len(frames), frames=frames,
                dense_arrays_read=False, heldout_pixel_computation=False,
                missing_asset_count=sum(len(r["missing_assets"]) for r in frames),
                inherited_mass_flags_unchanged=True,
                locked_phase_approval="Olav, 9 Oct 2026, common mass specification")


def clearwater_commands(inventory: Mapping, *, output_root: str | Path,
                        raw_root: str | Path) -> dict:
    """Predeclare middle-of-five RAW wash inputs; never inspect their pixels."""
    from .tools.clearwater_phase_reference import select_clear_reference_paths, CLEAR_REFERENCE_SPECS
    root, output = Path(inventory["candidate_root"]), Path(output_root)
    rows = []
    for run in inventory["split"]["production_runs"]:
        native_path = root / "source_alias/native_frames" / run / "manifest.json"
        native = read_bound_json(asset(native_path))
        processor = native["v5_photometry"]["processor"]
        configured = select_clear_reference_paths(Path(raw_root), run)
        selected = configured[2]
        raw = selected.with_suffix(".ARW")
        if not raw.is_file():
            raise FileNotFoundError(f"Configured clearwater capture lacks RAW: {raw}")
        args = dict(run=run, native_manifest_path=str(native_path),
                    photo_aggregate_path=processor["photo_source_aggregate_path"],
                    calibration_manifest_path=processor["calibration_manifest_path"],
                    raw_root=str(raw_root), output=str(output / run))
        rows.append(dict(run=run, calibration_run=run not in HELD_OUT,
                         execution_after_freeze=run in HELD_OUT, prepare_one_args=args,
                         native_manifest=asset(native_path),
                         processing_module="darsia.fluidflower.tools.ffac_v5_normalized_clearwater",
                         frozen_processor=processor,
                         configured_jpegs=[str(p) for p in configured],
                         selected_jpeg=asset(selected), selected_raw=asset(raw),
                         selection="middle of five predeclared clearwater captures",
                         borrowed_reference=Path(CLEAR_REFERENCE_SPECS[run].folder).parts[0].casefold() != run,
                         borrowed_reference_note=("Historical predeclared clearwater source is from another run; same packing and registration require explicit receipt." if run in ("ac23", "ac41") else "")))
    return dict(contract=CONTRACT, rows=rows, pixels_inspected=False,
                holdout_commands_not_executed=True)


def recipe_inputs(overview_path: str | Path, *, raw_root: str | Path,
                  runs, common_default_mM=(1.25, .726)) -> dict:
    """Check original workbook bytes and preserve measured dose versus proxy."""
    overview = read_bound_json(asset(overview_path))
    by_run = {r["run"]: r for r in overview["records"]}
    result = []
    for run in runs:
        record = by_run.get(run)
        baseline = dict(run=run, caveat=CAVEATS.get(run, ""))
        if record is not None:
            # Resolve only this archived basename beneath the supplied run folder.
            # A drive letter from the archive is never used as an executable path.
            basename = re.split(r"[\\/]", record["protocol_path"])[-1]
            path = Path(raw_root) / run.upper() / basename
            original_path = path
            expected = record["protocol_sha256"]
            recipe_equivalent = False
            if not path.is_file():
                alternate = Path(raw_root) / basename
                if alternate.is_file():
                    path = alternate
            if not path.is_file():
                raise ValueError(f"Original recipe workbook is missing: {original_path}")
            actual = sha256(path)
            # Verify the actual source cell even when the whole workbook SHA
            # equals the archived input. Never substitute a parsed recipe from
            # a similarly named but chemically different workbook.
            import openpyxl
            sheet, cell = record["composition_source"].split("!", 1)
            workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
            try:
                observed = workbook[sheet][cell].value
            finally:
                workbook.close()
            if str(observed).strip() != str(record["composition_raw"]).strip():
                raise ValueError(f"Recipe cell changed since parsed source: {path}:{sheet}!{cell}")
            recipe_equivalent = True
            baseline.update(protocol=asset(path), source_cell=record["composition_source"],
                            original_archive_protocol_sha256=expected,
                            recipe_cell_equivalence_verified=recipe_equivalent,
                            measured_source_cell_value=observed,
                            cell_comparison="exact text after trimming surrounding whitespace only",
                            composition_raw=record["composition_raw"],
                            fluid_volume_l=record.get("fluid_volume_l"),
                            ph_values=record.get("ph_values"),
                            chemistry_notes=record.get("chemistry_notes", []))
        alkalinity = None if record is None else record.get("naoh_recipe_proxy_mM")
        btb = None if record is None else record.get("btb_recipe_mM")
        measured = alkalinity is not None and btb is not None
        if measured:
            mode = "documented recipe; NaOH concentration is alkalinity proxy"
        else:
            alkalinity, btb = map(float, common_default_mM)
            mode = "explicit common DarSIA default; missing or ambiguous recipe"
        if not np.isfinite([alkalinity, btb]).all() or min(alkalinity, btb) <= 0:
            raise ValueError("Recipe concentrations must be finite and positive")
        result.append(dict(baseline, alkalinity_mM=float(alkalinity), btb_mM=float(btb),
                           alkalinity_M=float(alkalinity)/1000, btb_M=float(btb)/1000,
                           measured_recipe_available=measured, mode=mode,
                           alkalinity_source="Nominal documented NaOH dose is not measured in-cell alkalinity.",
                           common_default_mM=list(common_default_mM)))
    return dict(contract=CONTRACT, overview=asset(overview_path), runs=result,
                default_runs=[r["run"] for r in result if not r["measured_recipe_available"]],
                per_run_fitted_parameters=False)


def read_atmosphere(injection_path: str | Path, state_path: str | Path) -> dict:
    """Read explicit measured/registered observations; never clamp their range."""
    with Path(injection_path).open(encoding="utf-8-sig", newline="") as stream:
        injection = list(csv.DictReader(stream))
    start = min(datetime.fromisoformat(row["start"]) for row in injection)
    stages = []
    for row in injection:
        begin, end = (datetime.fromisoformat(row[key]) for key in ("start", "end"))
        rate, density = float(row["rate_sccm"]), float(row["density kg/m3"])
        if end < begin or not np.isfinite([rate, density]).all() or rate < 0 or density <= 0:
            raise ValueError("Injection CSV has invalid measured/nominal stage inputs")
        stages.append(dict(start_h=(begin-start).total_seconds()/3600,
                           end_h=(end-start).total_seconds()/3600,
                           start=begin.isoformat(), end=end.isoformat(),
                           rate_sccm=rate, density_kg_m3=density,
                           mass_rate_g_s=rate*density*1e-3/60,
                           location_x_m=float(row["location_x"]),
                           location_y_m=float(row["location_y"])))
    with Path(state_path).open(encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    rows = []
    for row in source_rows:
        if "PLACEHOLDER" in str(row).upper():
            raise ValueError("Pressure/temperature placeholder is not a measurement")
        values = dict(time_hours=(datetime.fromisoformat(row["datetime"])-start).total_seconds()/3600,
                      pressure_bar=float(row["pressure_bar"]),
                      temperature_c=float(row["temperature_celsius"]),
                      pressure_gradient_bar_per_m=float(row["pressure_gradient_bar"]),
                      temperature_gradient_c_per_m=float(row["temperature_gradient_celsius"]))
        if not np.isfinite(list(values.values())).all() or values["pressure_bar"] <= 0:
            raise ValueError("Invalid atmospheric observation")
        rows.append(values)
    if not rows or np.any(np.diff([r["time_hours"] for r in rows]) <= 0):
        raise ValueError("Atmospheric observations must be strictly chronological")
    return dict(injection=asset(injection_path), pressure_temperature=asset(state_path),
                origin_datetime=start.isoformat(), rows=rows,
                injection_stages=stages, injection_source="protocol nominal setpoints; not an MFC measurement",
                nominal_final_injected_mass_g=nominal_mass_at({"injection_stages": stages}, max(s["end_h"] for s in stages)),
                observations_rewritten=False, extrapolation_allowed=False)


def nominal_mass_at(record: Mapping, time_hours: float) -> float:
    """Integrate source protocol rates, independently of the image or fit."""
    if not np.isfinite(time_hours):
        raise ValueError("Dose requires an explicit finite camera-relative time")
    return float(sum(max(0., min(time_hours, row["end_h"])-row["start_h"])*3600*row["mass_rate_g_s"]
                     for row in record["injection_stages"]))


def measured_mfc_dose(flowplot_path: str | Path, atmosphere: Mapping, *,
                      script_start_datetime: str, clock_source: Mapping) -> dict:
    """Integrate the original measured stream; do not clamp or repair spikes.

    The percent-to-sccm factor is obtained from the injection CSV, not inferred
    from images or mass. The controller clock must be explicitly bound to a
    protocol timestamp. This remains a measurement with clock, delivery and
    gas-density assumptions; it is not proof of CO2 retained inside the rig.
    """
    data = np.loadtxt(flowplot_path, delimiter=";", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 6 or not np.isfinite(data).all():
        raise ValueError("Measured flowplot must contain six finite original columns")
    if np.any(np.diff(data[:, 0]) <= 0) or np.any(data[:, 1] < 0):
        raise ValueError("Measured MFC clock must increase and measured flow be nonnegative")
    with Path(atmosphere["injection"]["path"]).open(encoding="utf-8-sig", newline="") as stream:
        original = list(csv.DictReader(stream))
    factors = [float(row["rate_sccm"])/float(row["flow_percent"]) for row in original if float(row["flow_percent"]) > 0]
    if not factors or not np.allclose(factors, factors[0], rtol=0, atol=1e-12):
        raise ValueError("Controller percent-to-sccm mapping is not source-consistent")
    density = np.asarray([stage["density_kg_m3"] for stage in atmosphere["injection_stages"]])
    if not np.allclose(density, density[0], rtol=0, atol=1e-12):
        raise ValueError("Measured stream requires explicit per-time varying gas-density integration")
    offset = (datetime.fromisoformat(atmosphere["origin_datetime"])-datetime.fromisoformat(script_start_datetime)).total_seconds()
    times = data[:, 0]-offset
    begin = min(stage["start_h"] for stage in atmosphere["injection_stages"])*3600
    end = max(stage["end_h"] for stage in atmosphere["injection_stages"])*3600
    if times[0] > begin or times[-1] < end-2:
        raise ValueError("Measured flow stream does not cover the complete declared injection")
    # The last protocol end can lie within one sample period of acquisition
    # stop. Admit only observed time; never invent a final flow or extend it.
    good = (times >= begin) & (times <= end)
    selected = times[good]
    selected = np.r_[begin, selected] if selected[0] > begin else selected
    rate = np.interp(selected, times, data[:, 1])*factors[0]*density[0]*1e-3/60
    cumulative = np.r_[0., np.cumsum(np.diff(selected)*(rate[:-1]+rate[1:])*.5)]
    return dict(source=asset(flowplot_path), clock_source=dict(clock_source),
        script_start_datetime=script_start_datetime, gas_start_offset_s=offset,
        elapsed_h=(selected/3600).tolist(), cumulative_mass_g=cumulative.tolist(),
        percent_to_sccm=float(factors[0]), density_kg_m3=float(density[0]),
        measured_final_mass_g=float(cumulative[-1]),
        nominal_final_mass_g=nominal_mass_at(atmosphere, end/3600),
        readings_above_full_scale=int((data[good, 1] > 100).sum()),
        max_measured_percent=float(data[good, 1].max()),
        max_acquisition_gap_s=float(np.diff(data[:, 0]).max()),
        last_observed_h=float(selected[-1]/3600), protocol_end_h=float(end/3600),
        source_values_clamped=False,
        assumptions=["MFC measured percent is multiplied by its protocol percent-to-sccm calibration.",
                     "Protocol script-start clock anchors elapsed controller seconds.",
                     "Protocol standard gas density converts standard flow to mass.",
                     "Measured controller flow is assumed delivered to the connected rig; no independent delivery meter.",
                     "Transient readings over full scale are retained, not silently clipped.",
                     "A final unobserved fraction of a sampling interval is not filled."])


def injected_mass_at(record: Mapping, time_hours: float) -> float:
    """Choose only a documented dose source; never infer one from phase maps."""
    measured = record.get("measured_mfc_dose")
    if measured is None or record.get("dose_target_source") != "source-bound measured MFC integral":
        return nominal_mass_at(record, time_hours)
    if not np.isfinite(time_hours):
        raise ValueError("Dose requires explicit finite camera-relative time")
    times = np.asarray(measured["elapsed_h"])
    values = np.asarray(measured["cumulative_mass_g"])
    return 0. if time_hours <= 0 else float(np.interp(time_hours, times, values))


def atmosphere_at(record: Mapping, time_hours: float) -> dict:
    rows = record["rows"]
    times = np.asarray([r["time_hours"] for r in rows], np.float64)
    if not np.isfinite(time_hours) or not times[0] <= time_hours <= times[-1]:
        raise ValueError("Atmosphere is outside observed range; no silent carry-forward")
    return {name: float(np.interp(time_hours, times, [r[name] for r in rows]))
            for name in ("pressure_bar", "temperature_c", "pressure_gradient_bar_per_m", "temperature_gradient_c_per_m")}


def matrix_ac28_to_canvas(geometry: Mapping, origin_xy) -> np.ndarray:
    h = np.asarray(geometry["H_ac28_to_frame_native"], np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("Invalid registered frame homography")
    x, y = map(float, origin_xy)
    return np.asarray([[1., 0., -x], [0., 1., -y], [0., 0., 1.]]) @ h


def transport_physical_reference(reference: Mapping[str, np.ndarray], matrix,
                                 shape_hw) -> dict[str, np.ndarray]:
    """Transport measured fields; outside verified reference support is unknown."""
    h, w = map(int, shape_hw)
    source_domain = np.asarray(reference["domain"])
    if source_domain.ndim != 2 or source_domain.dtype != bool:
        raise ValueError("Physical reference needs exact boolean domain")
    result = {}
    for name in ("facies", "domain", "porosity", "depth_m"):
        values = np.asarray(reference[name])
        if values.shape != source_domain.shape:
            raise ValueError("Physical reference fields use different geometries")
        categorical = name in ("facies", "domain")
        if name == "facies" and not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Facies is categorical")
        data = values.astype(np.uint8) if categorical else values.astype(np.float32)
        transformed = cv2.warpPerspective(data, np.asarray(matrix, np.float64), (w, h),
            flags=cv2.INTER_NEAREST if categorical else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        result[name] = transformed.astype(bool) if name == "domain" else transformed
    coverage = cv2.warpPerspective(source_domain.astype(np.float32), np.asarray(matrix, np.float64),
        (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = result["domain"] & (coverage >= 1-1e-6)
    valid &= np.isfinite(result["depth_m"]) & (result["depth_m"] > 0)
    valid &= np.isfinite(result["porosity"]) & (result["porosity"] > 0) & (result["porosity"] <= 1)
    result["physical_supported_mask"] = valid
    result["unknown_physical_mask"] = ~valid
    return result


class VerifiedAssetReader:
    """Verify actual bytes once per unchanged path/size/mtime, then mmap."""
    def __init__(self, *, asset_reader=None):
        self.asset_reader = asset_reader
        self.verified = {}
        self.raw_baselines = {}
        self.physical_references = {}
        self.json_documents = {}

    def path(self, descriptor: Mapping) -> Path:
        if self.asset_reader is not None:
            return self.asset_reader.path(descriptor)
        path = Path(descriptor["path"])
        before = path.stat()
        key = (str(path.resolve()).casefold(), before.st_size, before.st_mtime_ns)
        expected = descriptor["sha256"]
        cached = self.verified.get(key)
        if cached is not None and cached != expected:
            raise ValueError("Asset has conflicting SHA bindings")
        if cached is None:
            digest = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Scientific input changed during verification")
            if digest != expected:
                raise ValueError(f"Scientific asset SHA differs: {path}")
            self.verified[key] = digest
        return path

    def array(self, descriptor: Mapping) -> np.ndarray:
        if self.asset_reader is not None:
            return self.asset_reader.array(descriptor)
        return np.load(self.path(descriptor), mmap_mode="r", allow_pickle=False)

    def json(self, descriptor: Mapping) -> dict:
        if self.asset_reader is not None:
            return self.asset_reader.json(descriptor)
        path = self.path(descriptor)
        key = (str(path.resolve()).casefold(), descriptor["sha256"])
        if key not in self.json_documents:
            self.json_documents[key] = json.loads(path.read_text(encoding="utf-8-sig"))
        return self.json_documents[key]


def compose_scientific_raw(descriptor: Mapping, reader: VerifiedAssetReader):
    """Exact frozen RAW centre+ring to float32 camera-linear RGB, no tone/gamma."""
    _scientific_assets(descriptor)
    shape = tuple(descriptor["phase_shape_hw"])
    centre = descriptor["native_centre"]
    hc, wc = tuple(centre["native_shape_hw"])
    ox, oy = descriptor["native_origin_xy"]
    x, y = -int(ox), -int(oy)
    if min(x, y) < 0 or x+wc > shape[1] or y+hc > shape[0]:
        raise ValueError("Scientific RAW centre lies outside the phase canvas")
    rgb = np.zeros((*shape, 3), np.float32)
    valid = np.zeros(shape, bool)
    covered = np.zeros(shape, bool)
    log = reader.array(centre["log_linear_rgb"])
    mask = reader.array(centre["valid_mask"])
    if log.dtype != np.float32 or log.shape != (3, hc, wc) or mask.dtype != bool or mask.shape != (hc, wc):
        raise ValueError("Scientific centre encoding is not exact float32 log RGB")
    def place(values, box, support=None):
        a, b, c, d = map(int, box)
        if (not 0 <= a < c <= shape[1] or not 0 <= b < d <= shape[0]
                or covered[b:d, a:c].any() or values.dtype != np.float32
                or values.shape != (3, d-b, c-a)):
            raise ValueError("Scientific windows overlap or use another native grid")
        good = np.isfinite(values).all(0)
        if support is not None:
            good &= support
        for channel in range(3):
            target = rgb[b:d, a:c, channel]
            target[good] = np.exp(values[channel][good])/np.float32(65535)
        valid[b:d, a:c] = good
        covered[b:d, a:c] = True
    place(log, (x, y, x+wc, y+hc), mask)
    for window in descriptor["ring_windows"]:
        place(reader.array(window["log_linear_rgb"]), window["canvas_xyxy"])
    if not np.isfinite(rgb[valid]).all() or np.any(rgb[valid] < 0):
        raise ValueError("Scientific RAW overflows; no clipping is permitted")
    return rgb, valid


@dataclass(frozen=True)
class CommonMassFrameInputs:
    run: str
    frame_index: int
    elapsed_h: float
    current_rgb: np.ndarray
    baseline_rgb: np.ndarray
    clear_water_rgb: np.ndarray
    clear_water_supported_mask: np.ndarray
    phase_labels: np.ndarray
    facies: np.ndarray
    domain: np.ndarray
    supported_mask: np.ndarray
    mass_eligible_mask: np.ndarray
    bottom_strip_mask: np.ndarray
    below_photo_filled_mask: np.ndarray
    unknown_physical_mask: np.ndarray
    unquantified_facies_mask: np.ndarray
    depth_m: np.ndarray
    porosity: np.ndarray
    pixel_area_m2: np.ndarray
    recipe: Mapping
    atmosphere: Mapping
    nominal_injected_mass_g: float
    dimensions_m: tuple[float, float]
    provenance: Mapping


def load_frame_inputs(metadata: Mapping, *, physical_reference: Mapping,
                      physical_reference_asset: Mapping, recipe: Mapping,
                      atmosphere: Mapping, nominal_injected_mass_g: float,
                      clear_water_reference: Mapping,
                      original_domain_reference: Mapping | None = None,
                      session_reader: VerifiedAssetReader | None = None,
                      frozen_calibration: Mapping | None = None) -> CommonMassFrameInputs:
    """Load scientific pixels only after the fixed split/freeze access guard."""
    run = metadata["run"]
    require_evaluation_access(run, frozen_calibration=frozen_calibration)
    receipt = read_bound_json(metadata["receipt"])
    reader = session_reader if session_reader is not None else VerifiedAssetReader()
    reader.path(physical_reference_asset)
    revision = receipt["rig_domain_revision"]
    shape = tuple(revision["phase_shape_hw"])
    current, current_valid = compose_scientific_raw(receipt["full_extended_scientific"], reader)
    base_desc = receipt["baseline_extended_scientific"]
    baseline_key = (run, base_desc["source_sha256"], tuple(base_desc["phase_shape_hw"]),
                    tuple(base_desc["native_origin_xy"]))
    # Recheck source stamps on cached arrays; changed bytes invalidate admission.
    for bound in _scientific_assets(base_desc):
        reader.path(bound)
    if baseline_key not in reader.raw_baselines:
        reader.raw_baselines[baseline_key] = compose_scientific_raw(base_desc, reader)
    baseline, baseline_valid = reader.raw_baselines[baseline_key]
    current_matrix = matrix_ac28_to_canvas(receipt["full_extended_scientific"]["binding"]["geometry"], revision["native_origin_xy"])
    base_matrix = matrix_ac28_to_canvas(base_desc["binding"]["geometry"], base_desc["native_origin_xy"])
    warp = current_matrix @ np.linalg.inv(base_matrix)
    baseline = cv2.warpPerspective(baseline, warp, (shape[1], shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    coverage = cv2.warpPerspective(baseline_valid.astype(np.float32), warp, (shape[1], shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    optical_valid = current_valid & (coverage >= 1-1e-6)
    native_geometry = receipt["full_extended_scientific"]["binding"]["geometry"]
    native_manifest = reader.json(dict(path=native_geometry["native_manifest_path"],
                                      sha256=native_geometry["native_manifest_sha256"]))
    photometry_sha = hashlib.sha256(json.dumps(native_manifest["v5_photometry"],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    clear_water, clear_supported, clear_provenance = registered_clearwater(
        clear_water_reference, current_matrix, shape, reader,
        expected_run=run,
        expected_native_manifest_sha256=native_geometry["native_manifest_sha256"],
        expected_photometry_sha256=photometry_sha)
    optical_valid &= clear_supported
    physical = transport_physical_reference(physical_reference, current_matrix, shape)
    with Image.open(reader.path(receipt["labels_map"])) as image:
        labels = np.asarray(image).copy()
    if labels.dtype != np.uint8 or labels.shape != shape or not np.isin(labels, (0, 1, 2, 255)).all():
        raise ValueError("Fixed categorical production labels are invalid")
    with np.load(reader.path(revision["classes"]), allow_pickle=False) as archive:
        classes = {key: archive[key].copy() for key in ("rig_domain", "modelled", "filled_observed", "filled_unobserved")}
    if any(a.dtype != bool or a.shape != shape for a in classes.values()):
        raise ValueError("Rig class masks use a different grid")
    domain = classes["rig_domain"]
    if not np.array_equal(domain, labels != 255):
        raise ValueError("Frozen phase labels and rig domain disagree")
    with np.load(reader.path(revision["area_map"]), allow_pickle=False) as archive:
        area = np.asarray(archive[revision["area_map"].get("array_key", "pixel_area_mm2")], np.float64)/1e6
    if area.shape != shape or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Measured per-pixel areas must be finite and positive")
    if original_domain_reference is None:
        raise ValueError("Mass requires the source-bound original domain to retain the bottom-strip block")
    bottom_strip, bottom_provenance = blocked_added_bottom_strip(receipt, domain,
        current_matrix, original_domain_reference, reader)
    supported = domain & optical_valid & physical["physical_supported_mask"]
    below_photo = classes["filled_unobserved"]
    unknown = domain & ~physical["physical_supported_mask"]
    unquantified = domain & np.isin(physical["facies"], UNQUANTIFIED_FACIES)
    # Eligibility describes the explicitly blocked geometry, not a successful
    # optical response. Keep unsupported facies and missing optical/physical
    # support here so the native engine reports unknown capacity/area and NaN,
    # instead of silently removing them from the main uncertainty bounds.
    eligible = domain & ~bottom_strip & ~below_photo
    provenance = dict(contract=CONTRACT, source_receipt=metadata["receipt"],
        physical_reference=dict(physical_reference_asset), physical_reference_provisional=True,
        physical_boundary_error_px="10-20 px, Olav-accepted provisional v3 reference",
        baseline_to_current=warp.tolist(), raw_encoding="float32-camera-linear-rgb-exp(log)/65535-no-clipping",
        phase_labels_modified=False, original_excluded_label=255,
        bottom_mass_allowed=False, unobserved_mass_allowed=False,
        bottom_strip_source=bottom_provenance,
        undefined_facies_not_borrowed=True, measured_area=revision["area_map"],
        current_rgb_max=float(current[current_valid].max()) if current_valid.any() else None,
        clear_water_reference=clear_provenance,
        physical_unknown_pixels=int(unknown.sum()), supported_pixels=int(supported.sum()),
        mass_eligible_pixels=int(eligible.sum()), optical_unresolved_pixels=int((domain & ~optical_valid).sum()),
        caveat=CAVEATS.get(run, ""))
    provenance.update(dose_target_source=atmosphere.get("dose_target_source", "protocol nominal setpoints"),
        nominal_protocol_injected_mass_g=nominal_mass_at(atmosphere, metadata["elapsed_h"]),
        measured_mfc_dose_receipt=atmosphere.get("measured_mfc_dose_receipt"),
        dose_timing_uncertainty_seconds=atmosphere.get("timing_uncertainty_seconds"))
    return CommonMassFrameInputs(run, int(metadata["frame_index"]), float(metadata["elapsed_h"]),
        current, baseline, clear_water, clear_supported, labels, physical["facies"], domain, supported, eligible,
        bottom_strip, below_photo, unknown, unquantified, physical["depth_m"],
        physical["porosity"], area, recipe, atmosphere_at(atmosphere, metadata["elapsed_h"]),
        float(nominal_injected_mass_g), (.49, .90), provenance)


def registered_clearwater(clear_water_reference, matrix_ac28_to_frame,
                         shape_hw, reader, *, expected_run=None,
                         expected_native_manifest_sha256=None,
                         expected_photometry_sha256=None):
    """Carry an exact native-B normalized RAW wash into the current rig grid."""
    receipt_descriptor = clear_water_reference["receipt"]
    receipt = read_bound_json(receipt_descriptor)
    if receipt.get("encoding") != "float32-camera-linear-unit-rgb":
        raise ValueError("Clearwater must use the frozen scientific RAW encoding")
    if expected_run is not None and receipt.get("run") != expected_run:
        raise ValueError("Clearwater scientific receipt belongs to another run")
    photometry_equal = (expected_photometry_sha256 is not None
                       and receipt.get("v5_photometry_sha256") == expected_photometry_sha256)
    if expected_photometry_sha256 is not None and not photometry_equal:
        raise ValueError("Clearwater and production RAW have different frozen photometry bindings")
    if expected_native_manifest_sha256 is not None and receipt.get("native_manifest_sha256") != expected_native_manifest_sha256 and not photometry_equal:
        raise ValueError("Clearwater and production RAW have different frozen photometry bindings")
    if "H_ac28_to_clearwater_grid" not in clear_water_reference:
        raise ValueError("Clearwater requires explicit source-bound native-B geometry")
    source = reader.array(dict(path=receipt["rgb_path"], sha256=receipt["rgb_sha256"]))
    valid = reader.array(dict(path=receipt["valid_mask_path"], sha256=receipt["valid_mask_sha256"]))
    if source.dtype != np.float32 or source.shape != (*valid.shape, 3) or valid.dtype != bool:
        raise ValueError("Scientific clearwater shape/dtype differs from its native grid")
    source_valid = valid & np.isfinite(source).all(-1) & (source > 0).all(-1)
    matrix = np.asarray(clear_water_reference["H_ac28_to_clearwater_grid"], np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Clearwater native-B registration is invalid")
    warp = np.asarray(matrix_ac28_to_frame, np.float64) @ np.linalg.inv(matrix)
    h, w = map(int, shape_hw)
    values = cv2.warpPerspective(np.where(source_valid[..., None], source, 0), warp,
        (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    coverage = cv2.warpPerspective(source_valid.astype(np.float32), warp, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    supported = (coverage >= 1-1e-6) & np.isfinite(values).all(-1) & (values > 0).all(-1)
    return values, supported, dict(receipt=receipt_descriptor, native_B_to_current=warp.tolist(),
        unsupported_not_extrapolated=True, source_bound_geometry=clear_water_reference.get("geometry_binding"),
        production_native_manifest_sha256=expected_native_manifest_sha256,
        reference_native_manifest_sha256=receipt.get("native_manifest_sha256"),
        exact_photometry_sha256=expected_photometry_sha256,
        alias_metadata_equivalence=photometry_equal)


def load_common_frame(metadata: Mapping, input_plan: Mapping, *,
                      clearwater_transport: Mapping | None = None,
                      session_reader: VerifiedAssetReader | None = None,
                      frozen_calibration: Mapping | None = None):
    """Public campaign adapter; inputs remain immutable, held-out access guarded.

    ``input_plan`` contains a physical_reference descriptor, already loaded
    recipe/atmosphere manifests (or their descriptors), and the reference result
    folder. A recovered clearwater transport can be supplied explicitly.
    """
    run = metadata["run"]
    require_evaluation_access(run, frozen_calibration=frozen_calibration)
    reader = session_reader if session_reader is not None else VerifiedAssetReader()
    def document(value):
        return reader.json(value) if "path" in value and "sha256" in value else value
    recipes = document(input_plan["recipes"])
    atmospheres = document(input_plan["atmosphere"])
    recipe = next(row for row in recipes["runs"] if row["run"] == run)
    atmosphere = next(row for row in atmospheres["runs"] if row["run"] == run)
    if clearwater_transport is None:
        clearwater_transport = input_plan.get("clearwater_transports", {}).get(run)
    if clearwater_transport is None:
        result = read_bound_json(asset(Path(input_plan["clearwater_results_folder"]) / f"{run}.json"))
        if result.get("complete") is not True:
            raise ValueError(f"Scientific clearwater preparation is incomplete for {run}")
        transport_descriptor = result["transport"]
        transport = read_bound_json(transport_descriptor)
    else:
        transport_descriptor = clearwater_transport if "path" in clearwater_transport else clearwater_transport.get("geometry_binding")
        transport = document(clearwater_transport)
    if transport.get("run") != run or transport.get("contract") != "ffac-common-native-clearwater-transport-v1":
        raise ValueError("Source-bound clearwater transport must match this run")
    clearwater = dict(receipt=dict(path=transport["clearwater_receipt"], sha256=transport["clearwater_receipt_sha256"]),
                     H_ac28_to_clearwater_grid=transport["H_ac28_to_clearwater_grid"],
                     geometry_binding=transport_descriptor)
    physical_descriptor = input_plan["physical_reference"]
    physical = load_physical_reference(physical_descriptor, reader)
    return load_frame_inputs(metadata, physical_reference=physical,
        physical_reference_asset=physical_descriptor, recipe=recipe,
        atmosphere=atmosphere,
        nominal_injected_mass_g=injected_mass_at(atmosphere, metadata["elapsed_h"]),
        clear_water_reference=clearwater,
        original_domain_reference=input_plan["original_domain_reference"],
        session_reader=reader,
        frozen_calibration=frozen_calibration)


def l1_protocol_proxy_neighbourhood(metadata: Mapping, reference_asset: Mapping,
                                   *, radius_mm: float = 20.,
                                   session_reader: VerifiedAssetReader | None = None):
    """Source-bound nominal L1 diagnostic, explicitly not photo certification.

    Coordinates come from the declared AC28 protocol proxy, and distances use
    the already approved metric transform. No image/plume is used to learn an
    injector position. This cannot enter the main mass calibration objective.
    """
    reader = session_reader if session_reader is not None else VerifiedAssetReader()
    reference = read_bound_json(reference_asset)
    point = np.asarray(reference["L1"][:2], np.float64)
    if point.shape != (2,) or not np.isfinite(point).all() or radius_mm <= 0:
        raise ValueError("Finite source-bound nominal L1 point and radius required")
    receipt = read_bound_json(metadata["receipt"])
    parameters = receipt["rig_domain_revision"]["parameters"]
    approval = parameters["physical_scale_revision"]
    if approval.get("physical_scale_approved") is not True:
        raise ValueError("L1 sensitivity needs the measured physical scale")
    metric = np.asarray(parameters["H_frame_native_to_metric_pixel"], np.float64)
    linear = float(approval["linear_factor"])
    unit_mm = float(parameters["metric_pixel_area_mm2"])**.5
    target_metric = point * linear
    h, w = map(int, metadata["phase_shape_hw"])
    ox, oy = map(float, metadata["native_origin_xy"])
    x = np.arange(w, dtype=np.float32)[None, :] + ox
    y = np.arange(h, dtype=np.float32)[:, None] + oy
    denominator = metric[2, 0]*x+metric[2, 1]*y+metric[2, 2]
    dx = ((metric[0, 0]*x+metric[0, 1]*y+metric[0, 2])/denominator-target_metric[0])*unit_mm
    dy = ((metric[1, 0]*x+metric[1, 1]*y+metric[1, 2])/denominator-target_metric[1])*unit_mm
    mask = np.isfinite(dx) & np.isfinite(dy) & (dx*dx+dy*dy <= radius_mm**2)
    return mask, dict(l1_position_source_bound=True,
        l1_position_status="protocol_proxy_unverified", provisional_l1_location=True,
        l1_position_verified=False, location_source=dict(reference_asset),
        source_sha256=reference_asset["sha256"], reference_point_ac28_xy=point.tolist(),
        radius_mm=float(radius_mm), uncertainty="No photo verification; nominal protocol proxy only",
        role="separate sensitivity, not main mass or fitted position")


def load_physical_reference(descriptor: Mapping, reader: VerifiedAssetReader):
    """Load one SHA-verified full-resolution physical reference per session."""
    path = reader.path(descriptor)
    key = (str(path.resolve()).casefold(), descriptor["sha256"])
    if key not in reader.physical_references:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in
                      ("facies", "domain", "porosity", "depth_m")}
        for value in arrays.values():
            value.flags.writeable = False
        reader.physical_references[key] = arrays
    return reader.physical_references[key]


def to_native_frame(frame: CommonMassFrameInputs, *, l1_neighbourhood_mask=None):
    """Keep unknown physical pixels outside native volume, reported by area."""
    from .phase_ffac_common_mass import CommonFrameArrays
    known = frame.domain & ~frame.unknown_physical_mask
    # Masks for numeric excluded-mass diagnostics partition known geometry;
    # unknown geometry has no justified volume or mass and is reported as area.
    bottom = frame.bottom_strip_mask & known
    below = frame.below_photo_filled_mask & known & ~bottom
    return CommonFrameArrays(run_id=frame.run, frame_index=frame.frame_index,
        current_rgb=frame.current_rgb, baseline_rgb=frame.baseline_rgb,
        clear_water_rgb=frame.clear_water_rgb,
        clear_water_supported_mask=frame.clear_water_supported_mask,
        phase_labels=frame.phase_labels, facies=frame.facies, domain=frame.domain,
        supported_mask=frame.supported_mask, mass_eligible_mask=frame.mass_eligible_mask,
        depth_m=frame.depth_m, porosity=frame.porosity,
        pixel_area_m2=frame.pixel_area_m2, dimensions_m=frame.dimensions_m,
        pressure_bar=frame.atmosphere["pressure_bar"], temperature_c=frame.atmosphere["temperature_c"],
        pressure_gradient_bar_per_m=frame.atmosphere.get("pressure_gradient_bar_per_m", 0.),
        temperature_gradient_c_per_m=frame.atmosphere.get("temperature_gradient_c_per_m", 0.),
        elapsed_h=frame.elapsed_h, nominal_injected_mass_g=frame.nominal_injected_mass_g,
        recipe=frame.recipe, provenance=dict(frame.provenance,
            unknown_physical_domain_cm2=float(frame.pixel_area_m2[frame.unknown_physical_mask].sum()*1e4),
            unknown_physical_co2_cm2=float(frame.pixel_area_m2[frame.unknown_physical_mask & np.isin(frame.phase_labels, (1, 2))].sum()*1e4),
            unknown_physical_mass_g=None),
        excluded_masks={"bottom_strip": bottom, "below_photo_filled": below},
        l1_neighbourhood_mask=l1_neighbourhood_mask,
        unknown_physical_mask=frame.unknown_physical_mask)


def _bottom_added_mask(domain, modelled):
    """Conservative lower addition mask; every unsupported column is blocked."""
    occupied = modelled.any(0)
    lowest = modelled.shape[0]-1-modelled[::-1].argmax(0)
    rows = np.arange(domain.shape[0])[:, None]
    return domain & ~modelled & ((rows > lowest[None]) | ~occupied[None])


def blocked_added_bottom_strip(receipt: Mapping, domain: np.ndarray,
                              current_ac28_to_canvas: np.ndarray,
                              original_domain_reference: Mapping,
                              reader: VerifiedAssetReader):
    """Retain the historical bottom strip, including newly inferred RAW pixels.

    Current ``modelled`` means newly available RAW support and is not the old
    domain. Carry the original categorical domain by its source registration,
    then apply the established nearest inner-boundary side partition. Only
    domain membership is read from the ancestor; its categories never change
    or override P1 final 3.
    """
    manifest = reader.json(original_domain_reference)
    source = receipt["binding"]["source"]
    old_index = source["frame_index"] if source["source_changed"] else source["parent_source_frame_index"]
    candidates = [row for row in manifest["frames"] if row["run"] == source["run"] and row["frame_index"] == old_index]
    if len(candidates) != 1:
        raise ValueError("Original bottom block has no unique source-bound geometry")
    old = candidates[0]
    old_label_descriptor = old["parent_labels_map"]
    with Image.open(reader.path(old_label_descriptor)) as image:
        old_domain = np.asarray(image) != 255
    if tuple(old_domain.shape) != tuple(old["native_shape_hw"]):
        raise ValueError("Original categorical domain grid differs from its registration")
    old_h = np.asarray(old["H_ac28_to_frame_native"], np.float64)
    warp = current_ac28_to_canvas @ np.linalg.inv(old_h)
    h, w = domain.shape
    carried = cv2.warpPerspective(old_domain.astype(np.uint8), warp, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    geometry = source["rig_geometry"]
    top_descriptor = geometry["frame_photo_top_keep"]
    with np.load(reader.path(top_descriptor), allow_pickle=False) as archive:
        photo_top = archive[top_descriptor.get("array_key", "keep")].copy()
    if photo_top.dtype != bool or photo_top.shape != domain.shape:
        raise ValueError("Original bottom strip needs this frame's photographed surface")
    polygon = np.asarray(geometry["frame_inner_polygon_native"], np.float64)
    origin = np.asarray(receipt["rig_domain_revision"]["native_origin_xy"], np.float64)
    outer = np.zeros(domain.shape, np.uint8)
    cv2.fillPoly(outer, [np.rint((polygon-origin)*256).astype(np.int32)], 1, shift=8)
    outer = outer.astype(bool)
    left = outer.argmax(1)
    right = w-1-outer[:, ::-1].argmax(1)
    floor = h-1-outer[::-1].argmax(0)
    surface = photo_top.argmax(0)
    xs = np.arange(w, dtype=np.float32)[None]
    result = np.zeros(domain.shape, bool)
    for first in range(0, h, 128):
        last = min(h, first+128)
        ys = np.arange(first, last, dtype=np.float32)[:, None]
        nearest = np.stack(np.broadcast_arrays(ys-surface[None], floor[None]-ys,
            xs-left[first:last,None], right[first:last,None]-xs)).argmin(0)
        result[first:last] = domain[first:last] & ~carried[first:last] & (nearest == 1)
    return result, dict(original_domain_manifest=dict(original_domain_reference),
        original_labels=old_label_descriptor, original_domain_to_current=warp.tolist(),
        method="historical added-strip nearest raster-axis inner boundary, tie order top,bottom,left,right",
        old_frame_index=int(old_index), source_changed=bool(source["source_changed"]),
        prediction_categories_used=False, current_modelled_class_used=False)
