"""Reproduce selected saved meeting fields without correction or phase inference.

The explicit request binds historical prepared inputs, corrected movie fields
and fitted parameters. Only byte-identical original anchor photographs are
eligible: a reduced uint8 display photograph cannot reproduce its unrounded
float32 optical input exactly. Outputs go to a new caller-owned result folder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from darsia.fluidflower import concentration_joint as joint
from darsia.fluidflower.mass_materialization import _image, _mass_analysis
from darsia.fluidflower.tools.phase_mass_joint_optics import build_joint_reference


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reproduce(request):
    started = time.perf_counter()
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    source = Path(request["prepared_root"])
    manifest_path = Path(request["corrected_manifest"])
    manifest = _read(manifest_path)
    fit_path = Path(request["fit"])
    fit = _read(fit_path)
    meta = _read(source / "prepared.json")
    references_path = source / "references.npz"
    checks = {
        "curve_sha_matches_corrected_manifest": _sha(fit_path) == manifest["curve_sha256"],
        "references_sha_matches_corrected_manifest": _sha(references_path) == manifest["optical_reference_sha256"],
        "prepared_sha_matches_corrected_manifest": _sha(source / "prepared.json") == manifest["original_prepared_sha256"],
        "parameters_match_corrected_manifest": fit["parameters"] == manifest["parameters"],
    }
    if not all(checks.values()):
        raise ValueError(f"Historical scientific input binding differs: {checks}")
    with np.load(references_path) as archive:
        references = {name: archive[name] for name in archive.files}
    reference = build_joint_reference(references["baseline"], references["clearwater"],
        references["facies"], domain_mask=references["pore_volume"] > 0)
    stores = {name: np.load(source / f"{name}.npy", mmap_mode="r")
              for name in ("rgb", "aq_volume", "gas_volume")}
    selected = list(request["movie_indices"])
    if not selected or len(set(selected)) != len(selected) or len(selected) > 8:
        raise ValueError("Choose one to eight unique bounded reproduction indices")
    rows = []
    bindings = {str(path): _sha(path) for path in (
        manifest_path, fit_path, source / "prepared.json", references_path,
        Path(joint.__file__), Path(__file__))}
    for index in selected:
        frame = manifest["frames"][index]
        receipt = _read(frame["record_path"])
        frozen = frame.get("frozen_frame_index")
        if frozen is None or not receipt.get("color_byte_identical") or not receipt.get("phase_reused"):
            raise ValueError(f"Frame {index} is not a verified identical historical anchor")
        expected_path = Path(frame["fields_path"])
        if receipt.get("field_sha256") and receipt["field_sha256"] != _sha(expected_path):
            raise ValueError(f"Saved movie field {index} changed")
        properties = source / f"properties_{frozen:03d}.npz"
        with np.load(properties) as archive:
            solubility, density = archive["solubility"], archive["density"]
        rgb, av, gv = (np.asarray(stores[name][frozen]) for name in ("rgb", "aq_volume", "gas_volume"))
        arrays, result = joint.evaluate_joint_fields(rgb, reference, av, gv, solubility,
            density, fit["parameters"], elapsed_h=frame["elapsed_h"], anchor=manifest["dilution_anchor"])
        with np.load(expected_path) as archive:
            expected_c = archive["concentration_g_l"]
            expected_aq, expected_gas = archive["aqueous_fraction"], archive["gas_fraction"]
        pv = references["pore_volume"]
        np.testing.assert_array_equal(arrays["concentration_g_l"], expected_c)
        for volume, fraction in ((av, expected_aq), (gv, expected_gas)):
            actual = np.divide(volume, pv, out=np.zeros_like(volume), where=pv > 0)
            np.testing.assert_array_equal(actual, fraction)
        for name in ("aqueous_mass_g", "gas_mass_g", "total_mass_g", "dilution_factor"):
            np.testing.assert_allclose(result[name], receipt[name], rtol=1e-12, atol=1e-12)
        # Check independent native DarSIA mass accounting from mixed-cell
        # concentration and gas saturation, using the saved physical state.
        g = meta["grid"]
        p = fit["parameters"]
        c_fraction = np.divide(arrays["concentration_g_l"], solubility,
            out=np.zeros_like(solubility), where=solubility > 0)
        se = np.divide(p["gas_saturation"] * gv.astype(float), pv,
            out=np.zeros_like(pv, dtype=float), where=pv > 0)
        liquid_volume = pv * (1 - se)
        ce = np.divide(c_fraction.astype(float) * av + (1 - p["gas_saturation"]) * gv,
            liquid_volume, out=np.zeros_like(pv, dtype=float), where=liquid_volume > 0)
        analyzer = _mass_analysis(pv.shape, g["width_m"], g["height_m"],
            receipt["pressure_bar"], receipt["temperature_c"],
            receipt["pressure_gradient_bar_per_m"], receipt["temperature_gradient_c_per_m"])
        masses = analyzer.mass_analysis(_image(ce, g["width_m"], g["height_m"]),
            _image(se, g["width_m"], g["height_m"]))
        native_aq = float(np.sum(masses.mass_aq.img * pv) * 1000)
        native_gas = float(np.sum(masses.mass_g.img * pv) * 1000)
        np.testing.assert_allclose([native_aq, native_gas],
            [result["aqueous_mass_g"], result["gas_mass_g"]], rtol=3e-6, atol=1e-9)
        record = dict(movie_index=index, original_frame_index=frozen,
            source_name=frame["source_name"], **result,
            concentration_bitwise_identical=True, phase_fractions_bitwise_identical=True,
            maximum_concentration_difference_g_l=0.,
            native_darsia_aqueous_g=native_aq, native_darsia_gas_g=native_gas,
            native_darsia_mass_parity=True,
            expected_field=str(expected_path), expected_field_sha256=_sha(expected_path))
        rows.append(record)
        for path in (properties, Path(frame["record_path"])):
            bindings[str(path)] = _sha(path)
        np.savez_compressed(output / f"frame_{index:05d}.npz",
            concentration_g_l=arrays["concentration_g_l"],
            record_json=np.asarray(json.dumps(record)))
    report = dict(format="joint-meeting-reproduction-v1", status="complete",
        created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
        checks=checks, rows=rows, input_bindings=bindings,
        phase_inference_or_training=False, color_correction_rerun=False,
        historical_gas_saturation_reference_only=fit["parameters"]["gas_saturation"],
        production_gas_bounds=list(joint.DEFAULT_GAS_BOUNDS),
        conclusion="Selected corrected movie anchors exactly reproduced; native DarSIA mass parity passed",
        scope="Only selected byte-identical historical anchors, not all inserted movie frames or current phase models")
    (output / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# RUN18 meeting-method reproduction", "", report["conclusion"], "",
        "| Source | Time [h] | Dissolved [g] | Gas [g] | Total [g] | Concentration |",
        "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['source_name']} | {row['elapsed_h']:.4f} | {row['aqueous_mass_g']:.6f} | {row['gas_mass_g']:.6f} | {row['total_mass_g']:.6f} | bitwise identical |")
    lines += ["", "Historical gas occupancy is retained only to reproduce the reference. New calibration uses explicit current gas bounds.",
        "No phase model, phase field, scientific input, or user Studio session was changed."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args()
    result = reproduce(_read(arguments.request))
    print(json.dumps({"status": result["status"], "frames": len(result["rows"]),
                      "seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
