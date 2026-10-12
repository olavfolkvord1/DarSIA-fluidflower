"""Rule B: keep free gas only in injector-connected or CO2-breached open sand.

This is a CPU-only categorical operation after the existing gas minimum.  It
never refits a classifier, changes the CO2 union, filters gas again, or changes
gas inside barrier facies.  Reference geometry is derived separately from the
series entry-pressure table and cap observations, then transported by exact H.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import tomllib

import cv2
import numpy as np


VERSION = "ffac-gas-barrier-rule-b-v1"
GEOMETRY_VERSION = "ffac-gas-barrier-rule-b-geometry-v1"
SPEC_SHA256 = "41fa22e6bbf91b10086eaf9eb5a198039d4453c1687370338221ca010322a3f8"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_identity(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + np.asarray(array.shape, "<i8").tobytes() + array.tobytes()).hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _asset(path, expected=None):
    path = Path(path).resolve()
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Rule B bound source changed: {path}")
    return dict(path=str(path), sha256=digest)


def _write(path, encoded):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"Refusing to overwrite a Rule B artifact: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded)
    temporary.replace(path)


def _write_json(path, value):
    _write(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")


def _save_geometry(path, components, barrier, domain):
    encoded = io.BytesIO()
    np.savez_compressed(encoded, open_components=components, barrier_mask=barrier, reference_domain=domain)
    _write(path, encoded.getvalue())
    return _asset(path)


def _validate_components(open_components, shape=None, barrier_mask=None):
    components = np.asarray(open_components)
    if (components.ndim != 2 or not components.size or components.dtype.kind not in "ui" or
            (shape is not None and components.shape != tuple(shape)) or components.min() < 0):
        raise ValueError("Rule B requires nonnegative integer open components on the native grid")
    # IDs are geometric components, not unconstrained image class values.
    if int(components.max()) > 255:
        raise ValueError("Rule B supports at most 255 reference component IDs")
    if barrier_mask is not None:
        barrier = np.asarray(barrier_mask)
        if barrier.dtype != np.bool_ or barrier.shape != components.shape or np.any(barrier & (components > 0)):
            raise ValueError("Barrier and open sand must be disjoint native boolean/integer grids")
    return components


def apply_barrier_rule_b(labels, open_components, source_component_ids, *, pixel_area_mm2,
                         barrier_mask=None):
    """Return categorical labels and the repeated four-connected bridge audit.

    Each CO2 connected component induces a hyperedge between the open regions
    it touches.  Reachability starts at protocol injector regions.  A newly
    reached open region can activate a separate CO2 hyperedge on the next pass.
    The CC is computed once; graph closure operates on its small touch table.
    """
    labels = np.asarray(labels)
    if labels.dtype != np.uint8 or labels.ndim != 2 or not labels.size or not np.isin(labels, (0, 1, 2, 255)).all():
        raise ValueError("Rule B labels must be uint8 water0/aqueous1/gas2/unresolved255")
    components = _validate_components(open_components, labels.shape, barrier_mask)
    area = float(pixel_area_mm2)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Rule B requires a positive finite nominal pixel area")
    sources = sorted(set(int(value) for value in source_component_ids))
    if not sources or any(value <= 0 or value > 255 for value in sources):
        raise ValueError("Rule B requires positive protocol source component IDs")
    co2 = (labels == 1) | (labels == 2)
    count, cc, stats, _centroids = cv2.connectedComponentsWithStats(np.asarray(co2, np.uint8), connectivity=4)
    stride = max(int(components.max()), max(sources)) + 1
    touched = co2 & (components > 0)
    encoded = cc[touched].astype(np.int64) * stride + components[touched].astype(np.int64)
    # FF_AC has only two open regions; this table is much smaller than the native
    # bitmap.  Refuse pathological allocation instead of consuming unbounded RAM.
    if count * stride > 32_000_000:
        raise ValueError("Rule B touch graph exceeds its bounded CPU allocation")
    touches = np.bincount(encoded, minlength=count * stride).reshape(count, stride) > 0
    touches[0] = False
    allowed = np.zeros(stride, bool)
    allowed[sources] = True
    bridges, bridge_ids, breach_events = [], set(), []
    iteration = 0
    while True:
        active_cc = np.any(touches[:, allowed], axis=1)
        reached = np.any(touches[active_cc], axis=0) if np.any(active_cc) else np.zeros(stride, bool)
        new = np.flatnonzero(reached & ~allowed)
        new = new[new != 0]
        if not new.size:
            break
        iteration += 1
        for component in new.tolist():
            triggers = np.flatnonzero(active_cc & touches[:, component])
            breach_events.append(dict(component_id=component, iteration=iteration,
                bridging_co2_component_ids=triggers.tolist(),
                touched_previously_allowed_components=sorted({int(value) for identifier in triggers
                    for value in np.flatnonzero(touches[identifier] & allowed)})))
            for identifier in triggers.tolist():
                if identifier not in bridge_ids:
                    bridge_ids.add(identifier)
                    stat = stats[identifier]
                    bridges.append(dict(co2_component_id=identifier,
                        co2_component_pixels=int(stat[cv2.CC_STAT_AREA]),
                        open_component_ids=np.flatnonzero(touches[identifier]).tolist(),
                        bbox_xywh=[int(value) for value in stat[:4]]))
        allowed[new] = True
    removed = (labels == 2) & (components > 0) & ~allowed[components]
    revised = labels.copy()
    revised[removed] = 1
    union_after = (revised == 1) | (revised == 2)
    if not np.array_equal(co2, union_after) or np.any((revised != labels) & ~removed):
        raise AssertionError("Rule B must change only gas to aqueous, preserving the CO2 union")
    gas_before, gas_after = labels == 2, revised == 2
    audit = dict(contract=VERSION, source_component_ids=sources,
        allowed_component_ids=np.flatnonzero(allowed).tolist(),
        breached_components=sorted(set(np.flatnonzero(allowed).tolist()) - set(sources)),
        breach_events=breach_events, bridging_co2_components=sorted(bridges, key=lambda row: row["co2_component_id"]),
        co2_connected_components=count - 1, closure_iterations=iteration,
        before_gas_pixels=int(gas_before.sum()), after_gas_pixels=int(gas_after.sum()),
        removed_gas_pixels=int(removed.sum()), removed_gas_cm2=float(removed.sum() * area / 100.),
        co2_union_changed_pixels=0, only_gas_to_aqueous=True, gas_minimum_reapplied=False,
        open_components_array_sha256=array_identity(components),
        component_gas_before={str(identifier): int(np.count_nonzero(gas_before & (components == identifier)))
                              for identifier in np.unique(components)},
        component_gas_after={str(identifier): int(np.count_nonzero(gas_after & (components == identifier)))
                             for identifier in np.unique(components)})
    if barrier_mask is not None:
        audit["barrier_gas_before"] = int(np.count_nonzero(gas_before & barrier_mask))
        audit["barrier_gas_after"] = int(np.count_nonzero(gas_after & barrier_mask))
        audit["barrier_gas_changed_pixels"] = 0
    return revised, audit


def derive_barrier_labels(facies, domain, entry_pressure_pa, cap_top_columns,
                          *, pressure_contrast=10., minimum_cap_fraction=.01):
    """Intersect measured pressure-contrast facies with observed cap holders."""
    facies, domain = np.asarray(facies), np.asarray(domain)
    if facies.shape != domain.shape or facies.ndim != 2 or domain.dtype != np.bool_:
        raise ValueError("Barrier derivation needs matching native facies and boolean domain")
    total = sum(int(value) for value in cap_top_columns.values())
    if total <= 0:
        raise ValueError("Barrier derivation requires observed cap-top columns")
    pressures = np.full(256, np.nan, dtype=np.float64)
    for label, value in entry_pressure_pa.items():
        if 0 <= int(label) <= 255:
            pressures[int(label)] = float(value)
    upper, lower = facies[:-1], facies[1:]
    up, lo = pressures[upper], pressures[lower]
    contrast = domain[:-1] & domain[1:] & np.isfinite(up) & np.isfinite(lo) & (lo > 0) & (up >= pressure_contrast * lo)
    candidates, edge_counts = np.unique(upper[contrast], return_counts=True)
    ratios = {int(label): int(cap_top_columns.get(int(label), 0)) / total for label in candidates}
    labels = sorted(int(label) for label in candidates if ratios[int(label)] >= minimum_cap_fraction)
    return labels, dict(entry_pressure_contrast=float(pressure_contrast), minimum_cap_top_fraction=float(minimum_cap_fraction),
        total_cap_top_columns=total, cap_top_columns_by_facies={str(k): int(v) for k, v in sorted(cap_top_columns.items())},
        pressure_contrast_edge_pixels_by_facies={str(int(k)): int(v) for k, v in zip(candidates, edge_counts)},
        cap_top_fraction_by_pressure_candidate={str(k): value for k, value in ratios.items()},
        barrier_facies_ids=labels, gas_side_extension_pixels=0,
        rule="entry pressure >=10x directly below AND >=1% observed cap-top columns")


def reference_open_components(facies, domain, barrier_labels, injector_points_xy):
    barrier = np.isin(facies, barrier_labels) & domain
    open_mask = domain & ~barrier
    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.asarray(open_mask, np.uint8), connectivity=4)
    if count > 256:
        raise ValueError("More than 255 reference open-sand components cannot be stored as uint8")
    components = labels.astype(np.uint8)
    injectors = []
    for point in injector_points_xy:
        x, y = map(int, point["xy_px"])
        if not (0 <= y < domain.shape[0] and 0 <= x < domain.shape[1]):
            raise ValueError("Protocol injector lies outside reference grid")
        identifier = int(components[y, x])
        if identifier == 0:
            raise ValueError("Protocol injector is not inside open reference sand")
        injectors.append(dict(point, component_id=identifier))
    summary = dict(connectivity=4, open_component_count=count - 1,
        components=[dict(component_id=i, pixels=int(stats[i, cv2.CC_STAT_AREA]),
                         bbox_xywh=[int(value) for value in stats[i, :4]]) for i in range(1, count)],
        injectors=injectors, source_component_ids=sorted({point["component_id"] for point in injectors}),
        barrier_pixels=int(barrier.sum()), domain_pixels=int(domain.sum()),
        fault9_open=bool(np.all(components[(facies == 9) & domain] > 0)))
    return components, barrier, summary


def transport_components(components, barrier, domain, homography, shape_hw):
    components = _validate_components(components, barrier_mask=barrier)
    matrix = np.asarray(homography, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Rule B requires a finite nonsingular native homography")
    if np.asarray(domain).dtype != np.bool_ or domain.shape != components.shape:
        raise ValueError("Reference domain differs from open-component grid")
    size = tuple(map(int, shape_hw))[::-1]
    def warp(array):
        return cv2.warpPerspective(np.asarray(array, np.uint8), matrix, size, flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped = warp(components)
    barriers = warp(barrier).astype(bool)
    domains = warp(domain).astype(bool)
    if np.any((warped > 0) & ~domains) or np.any(barriers & (warped > 0)):
        raise AssertionError("Nearest-neighbour transport broke reference partition")
    return warped, barriers, domains


def _pressure_table(path, overrides):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        table = {int(row["id"]): row for row in csv.DictReader(stream)}
    override = tomllib.loads(Path(overrides).read_text(encoding="utf-8"))
    for label, record in override.get("facies", {}).items():
        table[int(label)] = dict(table[int(record["properties_like"])])
    pressures = {}
    for label, row in table.items():
        value = float(row["pc [mbar]"])
        if np.isfinite(value):
            pressures[label] = 100. * value
    return pressures


def _cap_columns(facies, folder):
    sampled = facies[4::8, 4::8][:372, :684]
    excluded = {"ac20", "ac24", "ac27", "ac29", "ac44", "ac51"}
    counts, sources = {}, []
    for path in sorted(Path(folder).glob("ac*_pass1.npz")):
        run = path.name.split("_")[0]
        if run in excluded:
            continue
        with np.load(path, allow_pickle=False) as archive:
            for stage in (0, 1):
                key = f"C{stage}"
                if key not in archive.files:
                    continue
                cap = np.unpackbits(archive[key], axis=-1)[..., :684].astype(bool)
                if cap.shape != sampled.shape:
                    raise ValueError("Cap-top profile grid differs from recorded facies sampling")
                valid = cap.any(axis=0)
                first = cap.argmax(axis=0)
                columns = np.flatnonzero(valid & (first > 0))
                identifiers, numbers = np.unique(sampled[first[columns] - 1, columns], return_counts=True)
                for identifier, number in zip(identifiers, numbers):
                    counts[int(identifier)] = counts.get(int(identifier), 0) + int(number)
        sources.append(dict(run=run, cap_artifact=_asset(path)))
    if len(sources) != 37:
        raise ValueError("Rule B cap evidence must use the exact 37-run series cohort")
    return counts, sources, sorted(excluded)


def _frame_geometry(task, reference, output, derivation_asset):
    original = task["registration_receipt"]
    _asset(original["path"], original["sha256"])
    geometry = _read(original["path"])
    components, barrier, domain = reference
    warped, barriers, domains = transport_components(components, barrier, domain, task["H_ac28_to_frame_native"], task["native_shape_hw"])
    path = Path(output) / "frames" / task["run"] / f"frame_{task['frame_index']:03d}.npz"
    asset = _save_geometry(path, warped, barriers, domains)
    return dict(task, components=asset, component_array_sha256=array_identity(warped),
        barrier_array_sha256=array_identity(barriers), reference_domain_array_sha256=array_identity(domains),
        barrier_derivation_sha256=derivation_asset["sha256"], interpolation="cv2.INTER_NEAREST",
        border_value=0, relabelled_after_transport=False,
        open_component_pixels={str(int(i)): int(n) for i, n in zip(*np.unique(warped, return_counts=True))},
        barrier_pixels=int(barriers.sum()), reference_domain_pixels=int(domains.sum()))


def build_barrier_geometry(parent_candidate, physics_folder, output, *, spec_path, workers=8):
    """Derive the Rule B reference, then materialize all 864 native transports."""
    started = time.perf_counter()
    cv2.setNumThreads(1)
    parent, physics, output = Path(parent_candidate).resolve(), Path(physics_folder).resolve(), Path(output).resolve()
    spec = _asset(spec_path, SPEC_SHA256)
    profile_path = parent / "phase_optical_regions.json"
    profile = _read(profile_path)
    top_manifest_path = parent / profile["photo_sand_top_revision"]["manifest"]["path"]
    top_asset = _asset(top_manifest_path, profile["photo_sand_top_revision"]["manifest"]["sha256"])
    top = _read(top_manifest_path)
    facies_path = physics / "results/ac28_facies_v3b_fault9.npz"
    with np.load(facies_path, allow_pickle=False) as archive:
        facies, domain = archive["facies"], archive["domain"]
    common = Path(r"\\UNAS-Pro-8\UiB_FluidFlower\FF_AC\DarSIA\config\common.toml")
    common_cfg = tomllib.loads(common.read_text(encoding="utf-8"))
    table_path = Path(common_cfg["facies"]["props"])
    overrides_path = physics / "ffac_physics_overrides.toml"
    pressures = _pressure_table(table_path, overrides_path)
    cap_counts, cap_sources, excluded = _cap_columns(facies, physics / "results/v5c")
    barrier_labels, derivation = derive_barrier_labels(facies, domain, pressures, cap_counts)
    if barrier_labels != [1, 2]:
        raise ValueError(f"Series-derived barriers differ from expected ESF1/C2: {barrier_labels}")
    # Physical x/y arrays are the source of the protocol points quoted in §2.3.
    grid_path = physics.parent / "ffac_geometry_facies_20260930/results/ac28_physical_reference_v3_provisional.npz"
    with np.load(grid_path, allow_pickle=False) as archive:
        xm, ym = archive["x_m"], archive["y_m"]
    stages = profile["runs"]["ffac/ac28"]["injection_stages"]
    points = []
    for index, stage in enumerate(stages):
        distance = (xm - float(stage["x_m"])) ** 2 + (ym - float(stage["y_m"])) ** 2
        y, x = np.unravel_index(np.argmin(distance), distance.shape)
        points.append(dict(name=f"L{index+1}", xy_m=[float(stage["x_m"]), float(stage["y_m"])],
                           xy_px=[int(x), int(y)], reference_grid_distance_m=float(np.sqrt(distance[y, x]))))
    if [point["xy_px"] for point in points] != [[2786, 2897], [4465, 1969]]:
        raise ValueError("Protocol/reference-grid injector points differ from the frozen specification")
    components, barrier, summary = reference_open_components(facies, domain, barrier_labels, points)
    if summary["open_component_count"] != 2 or summary["source_component_ids"] != [2] or not summary["fault9_open"]:
        raise ValueError("Reference open topology differs from the specified two-region conduit topology")
    derivation.update(contract="ffac-gas-barrier-rule-b-derivation-v1", spec=spec,
        reference_facies=_asset(facies_path), reference_facies_metadata=_asset(facies_path.with_suffix(".json")),
        protocol_profile=_asset(profile_path), protocol_reference_grid=_asset(grid_path),
        series_common=_asset(common), series_facies_table=_asset(table_path), physics_overrides=_asset(overrides_path),
        source_derivation_script=_asset(physics / "scripts/make_barrier_zone.py"),
        source_physics_reader=_asset(physics / "scripts/physics_setup.py"),
        entry_pressure_pa={str(k): value for k, value in pressures.items()},
        entry_pressure_source="measured series pc [mbar] table ×100 Pa/mbar; fault9 properties_like5",
        cap_sources=cap_sources, cap_cohort_runs=[row["run"] for row in cap_sources], cap_excluded_runs=excluded,
        cap_sampling="facies[4::8,4::8][:372,:684]; unpack C0/C1; count facies one row above first cap in each occupied column",
        reference_topology=summary, rule_is_after_gas_minimum=True, gas_minimum_reapplied=False,
        mass_eligible=False, physical_scale_verified=False, code=_asset(__file__))
    derivation_path = output / "barrier_derivation.json"
    _write_json(derivation_path, derivation)
    derivation_asset = _asset(derivation_path)
    reference_asset = _save_geometry(output / "reference_components_ac28.npz", components, barrier, domain)
    rows = top["frames"]
    if len(rows) != 864 or len({(row["run"], row["frame_index"]) for row in rows}) != 864 or len({row["run"] for row in rows}) != 43:
        raise ValueError("Parent surface revision must bind exact 43-run /864-frame inventory")
    request = dict(contract=GEOMETRY_VERSION, spec_sha256=SPEC_SHA256, parent_candidate=str(parent),
        parent_surface_manifest=top_asset, parent_profile=_asset(profile_path), barrier_derivation=derivation_asset,
        reference_components=reference_asset, source_component_ids=summary["source_component_ids"], code=_asset(__file__))
    _write_json(output / "request.json", request)
    tasks, native_hashes = [], {}
    for row in rows:
        path = Path(row["shadow_native_manifest_path"])
        if row["run"] not in native_hashes:
            native_hashes[row["run"]] = _asset(path, row["shadow_native_manifest_sha256"])
        geo = _read(row["registration_receipt"]["path"])
        if geo["run"] != row["run"] or geo["frame_index"] != row["frame_index"] or geo["H_ac28_to_frame_native"] != row["H_ac28_to_frame_native"]:
            raise ValueError("Parent frame geometry differs from its exact transported surface")
        task = dict(run=row["run"], frame_index=int(row["frame_index"]), elapsed_h=float(row["elapsed_h"]),
            native_manifest_path=str(path), native_manifest_sha256=row["shadow_native_manifest_sha256"],
            native_shape_hw=row["native_shape_hw"], content_box_xywh=row["content_box_xywh"],
            H_ac28_to_frame_native=row["H_ac28_to_frame_native"], registration_receipt=row["registration_receipt"],
            source_component_ids=summary["source_component_ids"])
        tasks.append(task)
    result_rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_frame_geometry, task, (components, barrier, domain), output, derivation_asset) for task in tasks]
        for future in as_completed(futures):
            result_rows.append(future.result())
            if len(result_rows) % 50 == 0:
                print(json.dumps(dict(complete=len(result_rows), total=864, seconds=round(time.perf_counter()-started, 2))), flush=True)
    result_rows.sort(key=lambda row: (row["run"], row["frame_index"]))
    manifest = dict(request, complete=True, frames=result_rows, run_count=43, frame_count=864,
        reference_shape_hw=list(facies.shape), reference_component_summary=summary,
        barrier_facies_ids=barrier_labels, transport="exact H_ac28_to_frame_native, nearest neighbour, outside0; no per-frame relabelling",
        scope="FF_AC only; studio_validation_only; no retraining or new area filtering", physical_scale_verified=False,
        known_limits=["FF_UM seal outside analysis domain needs separate capillary breakthrough rule",
                      "Disconnected late CO2 above seal remains aqueous, physical interpretation unresolved",
                      "Residual false gas below seal is unchanged"])
    path = output / "manifest.json"
    _write_json(path, manifest)
    result = dict(contract=GEOMETRY_VERSION, complete=True, manifest=_asset(path), runs=43, frames=864,
                  source_component_ids=summary["source_component_ids"], barrier_facies_ids=barrier_labels,
                  reference_component_summary=summary, seconds=round(time.perf_counter()-started, 3), workers=workers)
    result_path = output / "result.json"
    if result_path.exists():
        old = _read(result_path)
        if old.get("manifest") != result["manifest"]:
            raise ValueError("Completed Rule B bundle differs on resume")
        return old
    _write_json(result_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--physics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    print(json.dumps(build_barrier_geometry(args.parent, args.physics, args.output, spec_path=args.spec,
                                          workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
