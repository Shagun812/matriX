"""Evidence-first processing for user-supplied TiO2 measurements.

This module is deliberately separate from simulator Cases, checks, and
likelihood tables. It does not infer H1-H4 without calibrated real-data
likelihoods.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import least_squares


SPEC_PATH = Path(__file__).with_name("real_data_spec.yaml")
SPEC = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CompositionMeasurement:
    elements: dict[str, tuple[float, float, str]]


@dataclass(frozen=True)
class XRDMeasurement:
    two_theta: np.ndarray
    intensity: np.ndarray


@dataclass(frozen=True)
class OpticalMeasurement:
    x: np.ndarray
    signal: np.ndarray


@dataclass(frozen=True)
class RealRepeat:
    composition: CompositionMeasurement | None
    xrd: XRDMeasurement | None
    optical: OpticalMeasurement | None
    paths: dict[str, str]


@dataclass(frozen=True)
class RealSample:
    """User-provided measurements and provenance; contains no simulated state."""

    sample_id: str
    material: str
    composition: CompositionMeasurement
    xrd: XRDMeasurement
    optical: OpticalMeasurement
    metadata: dict[str, Any]
    repeat: RealRepeat | None
    provenance: dict[str, Any]


def _read_rows(path: str | Path, required: set[str]) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV header is required")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing CSV columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: CSV must contain at least one data row")
    return rows


def _finite_number(value: str, path: str | Path, column: str, row: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: row {row} has invalid {column}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path}: row {row} has non-finite {column}")
    return result


def _integral(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5))


def load_composition(path: str | Path) -> CompositionMeasurement:
    rows = _read_rows(path, {"element", "value", "uncertainty", "unit"})
    elements = {}
    for line, row in enumerate(rows, start=2):
        element = row["element"].strip()
        unit = row["unit"].strip()
        if not element or element in elements:
            raise ValueError(f"{path}: row {line} has a blank or duplicate element")
        if unit not in {"mol", "atom", "atomic_amount", "atomic_fraction"}:
            raise ValueError(
                f"{path}: row {line} unit must express atomic amounts "
                "(mol, atom, atomic_amount, or atomic_fraction)"
            )
        value = _finite_number(row["value"], path, "value", line)
        uncertainty = _finite_number(row["uncertainty"], path, "uncertainty", line)
        if value <= 0 or uncertainty < 0:
            raise ValueError(f"{path}: row {line} requires value > 0 and uncertainty >= 0")
        if unit == "atomic_fraction" and value > 1:
            raise ValueError(f"{path}: row {line} atomic_fraction cannot exceed 1")
        elements[element] = (value, uncertainty, unit)
    if "Ti" not in elements or "O" not in elements:
        raise ValueError(f"{path}: both Ti and O measurements are required")
    if elements["Ti"][2] != elements["O"][2]:
        raise ValueError(f"{path}: Ti and O must use the same atomic amount unit")
    return CompositionMeasurement(elements)


def load_xrd(path: str | Path) -> XRDMeasurement:
    rows = _read_rows(path, {"two_theta", "intensity"})
    two_theta = np.array([
        _finite_number(row["two_theta"], path, "two_theta", i)
        for i, row in enumerate(rows, start=2)
    ])
    intensity = np.array([
        _finite_number(row["intensity"], path, "intensity", i)
        for i, row in enumerate(rows, start=2)
    ])
    if len(two_theta) < 3 or np.any(np.diff(two_theta) <= 0):
        raise ValueError(f"{path}: at least three strictly increasing two_theta points required")
    if np.any(two_theta <= 0) or np.any(intensity < 0):
        raise ValueError(f"{path}: two_theta must be positive and intensity nonnegative")
    if not np.any(intensity > 0):
        raise ValueError(f"{path}: intensity must contain a positive value")
    return XRDMeasurement(two_theta, intensity)


def load_optical(path: str | Path) -> OpticalMeasurement:
    rows = _read_rows(path, {"x", "signal"})
    x = np.array([
        _finite_number(row["x"], path, "x", i) for i, row in enumerate(rows, start=2)
    ])
    signal = np.array([
        _finite_number(row["signal"], path, "signal", i)
        for i, row in enumerate(rows, start=2)
    ])
    if len(x) < 5 or np.any(np.diff(x) <= 0):
        raise ValueError(f"{path}: at least five strictly increasing x points required")
    if np.any(x <= 0) or np.any(signal < 0) or not np.any(signal > 0):
        raise ValueError(f"{path}: x must be positive and signal nonnegative with a positive value")
    return OpticalMeasurement(x, signal)


def _load_metadata(path: str | Path) -> dict[str, Any]:
    metadata = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata YAML must contain a mapping")
    required = {"sample_id", "material", "composition", "xrd", "optical"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"{path}: missing metadata: {', '.join(sorted(missing))}")
    if metadata["material"] != "tio2":
        raise ValueError("only material: tio2 (TiO2 anatase/rutile) is supported")
    if not str(metadata["sample_id"]).strip():
        raise ValueError("metadata sample_id must not be blank")
    composition = metadata["composition"]
    xrd = metadata["xrd"]
    optical = metadata["optical"]
    if (
        not isinstance(composition, dict)
        or composition.get("basis") != "atomic_ratio"
        or composition.get("uncertainty_type") not in {"standard_uncertainty", "standard_error"}
    ):
        raise ValueError(
            "metadata composition requires basis: atomic_ratio and "
            "uncertainty_type: standard_uncertainty or standard_error"
        )
    if not isinstance(xrd, dict):
        raise ValueError("metadata xrd must be a mapping")
    required_xrd = {"two_theta_unit", "radiation", "wavelength_angstrom", "geometry"}
    if required_xrd - set(xrd):
        raise ValueError(f"metadata xrd requires {', '.join(sorted(required_xrd))}")
    if xrd["two_theta_unit"] != "degree":
        raise ValueError("xrd.two_theta_unit must be degree")
    spec_xrd = SPEC["references"]["xrd"]
    if xrd["radiation"] != spec_xrd["radiation"]["name"]:
        raise ValueError("xrd.radiation must match the configured Cu Kalpha1 reference")
    if not math.isclose(float(xrd["wavelength_angstrom"]), spec_xrd["radiation"]["wavelength_angstrom"], rel_tol=0, abs_tol=1e-7):
        raise ValueError("xrd wavelength does not match the configured Cu Kalpha1 wavelength")
    if xrd["geometry"] != spec_xrd["geometry"]:
        raise ValueError(f"xrd.geometry must be {spec_xrd['geometry']}")
    if not isinstance(optical, dict):
        raise ValueError("metadata optical must be a mapping")
    required_optical = {
        "x_unit", "measurement_type", "signal_type",
        "tauc_fit_window_ev", "alternate_baseline_window_ev",
    }
    if required_optical - set(optical):
        raise ValueError(
            f"metadata optical requires {', '.join(sorted(required_optical))}"
        )
    if optical["x_unit"] not in {"eV", "nm"}:
        raise ValueError("optical.x_unit must be eV or nm")
    if optical["signal_type"] != "absorbance":
        raise ValueError("only optical.signal_type: absorbance is currently supported")
    if not optical["measurement_type"]:
        raise ValueError("optical.measurement_type must identify the instrument measurement")
    for name in ("tauc_fit_window_ev", "alternate_baseline_window_ev"):
        window = optical[name]
        if (
            not isinstance(window, list) or len(window) != 2
            or not all(math.isfinite(float(item)) for item in window)
            or float(window[0]) >= float(window[1])
        ):
            raise ValueError(f"optical.{name} must be [min_eV, max_eV]")
    if float(optical["alternate_baseline_window_ev"][1]) >= float(optical["tauc_fit_window_ev"][0]):
        raise ValueError("alternate_baseline_window_ev must lie below tauc_fit_window_ev")
    return metadata


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_real_sample(
    *,
    composition_path: str | Path,
    xrd_path: str | Path,
    optical_path: str | Path,
    metadata_path: str | Path,
    repeat_composition_path: str | Path | None = None,
    repeat_xrd_path: str | Path | None = None,
    repeat_optical_path: str | Path | None = None,
) -> RealSample:
    metadata = _load_metadata(metadata_path)
    paths = {
        "composition": str(composition_path),
        "xrd": str(xrd_path),
        "optical": str(optical_path),
        "metadata": str(metadata_path),
    }
    provenance = {
        "inputs": {key: {"path": value, "sha256": _sha256(value)} for key, value in paths.items()},
        "spec_path": str(SPEC_PATH),
        "spec_version": SPEC["spec_version"],
        "reference_provenance": metadata.get("xrd_reference"),
    }
    repeat_paths = {
        "composition": repeat_composition_path,
        "xrd": repeat_xrd_path,
        "optical": repeat_optical_path,
    }
    repeat = None
    if any(path is not None for path in repeat_paths.values()):
        loaded = {}
        for key, path in repeat_paths.items():
            loaded[key] = (
                load_composition(path) if key == "composition" and path else
                load_xrd(path) if key == "xrd" and path else
                load_optical(path) if key == "optical" and path else None
            )
        repeat = RealRepeat(
            composition=loaded["composition"],
            xrd=loaded["xrd"],
            optical=loaded["optical"],
            paths={key: str(path) for key, path in repeat_paths.items() if path},
        )
        provenance["repeat_inputs"] = {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in repeat_paths.items() if path
        }
    return RealSample(
        sample_id=str(metadata["sample_id"]),
        material=metadata["material"],
        composition=load_composition(composition_path),
        xrd=load_xrd(xrd_path),
        optical=load_optical(optical_path),
        metadata=metadata,
        repeat=repeat,
        provenance=provenance,
    )


def check_c1(sample: RealSample) -> dict[str, Any]:
    ti, ti_u, ti_unit = sample.composition.elements["Ti"]
    oxygen, oxygen_u, oxygen_unit = sample.composition.elements["O"]
    if ti_unit != oxygen_unit:
        raise ValueError("Ti and O units must match")
    ratio = oxygen / ti
    uncertainty = abs(ratio) * math.sqrt((oxygen_u / oxygen) ** 2 + (ti_u / ti) ** 2)
    difference = ratio - float(SPEC["composition"]["nominal_ratio"]["value"])
    return {
        "status": "AVAILABLE",
        "measured_O_to_Ti": ratio,
        "nominal_O_to_Ti": 2.0,
        "difference": difference,
        "standard_uncertainty": uncertainty,
        "uncertainty_type": sample.metadata["composition"]["uncertainty_type"],
        "standardized_difference": difference / uncertainty if uncertainty else None,
        "uncertainty_assumption": "independent supplied standard uncertainties; covariance not provided",
        "tolerance": None,
    }


def _load_reference_pattern(path: str | Path, provenance: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not provenance or not all(provenance.get(key) for key in ("citation", "source", "version")):
        raise ValueError("xrd reference requires provenance: citation, source, and version")
    rows = _read_rows(path, {"two_theta", "anatase_intensity", "rutile_intensity"})
    x = np.array([_finite_number(r["two_theta"], path, "two_theta", i) for i, r in enumerate(rows, 2)])
    anatase = np.array([_finite_number(r["anatase_intensity"], path, "anatase_intensity", i) for i, r in enumerate(rows, 2)])
    rutile = np.array([_finite_number(r["rutile_intensity"], path, "rutile_intensity", i) for i, r in enumerate(rows, 2)])
    if len(x) < 5 or np.any(np.diff(x) <= 0) or np.any(anatase < 0) or np.any(rutile < 0):
        raise ValueError("reference pattern requires increasing two_theta and nonnegative phase intensities")
    anatase_area = _integral(anatase, x)
    rutile_area = _integral(rutile, x)
    if anatase_area <= 0 or rutile_area <= 0:
        raise ValueError("reference pattern must contain both phase profiles")
    return x, anatase / anatase_area, rutile / rutile_area


def _fit_xrd(sample: RealSample, reference_path: str | Path | None) -> dict[str, Any]:
    if reference_path is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "validated, provenance-recorded anatase/rutile reference profiles were not supplied",
            "sample_displacement_deg": None,
            "shift_uncertainty_deg": None,
            "fit_quality": None,
            "relative_crystalline_fraction": None,
        }
    ref_x, anatase, rutile = _load_reference_pattern(
        reference_path, sample.metadata.get("xrd_reference")
    )
    x, observed = sample.xrd.two_theta, sample.xrd.intensity
    observed = observed / np.trapezoid(observed, x)
    shift_range = sample.metadata["xrd"].get("shift_search_range_deg")
    if (
        not isinstance(shift_range, list) or len(shift_range) != 2
        or float(shift_range[0]) >= float(shift_range[1])
    ):
        raise ValueError("metadata xrd.shift_search_range_deg must provide [min_deg, max_deg]")

    def residual(parameters):
        a, r, shift, background = parameters
        modeled = (
            a * np.interp(x - shift, ref_x, anatase, left=0.0, right=0.0)
            + r * np.interp(x - shift, ref_x, rutile, left=0.0, right=0.0)
            + background
        )
        return modeled - observed

    fit = least_squares(
        residual,
        x0=[0.5, 0.5, 0.0, 0.0],
        bounds=([0.0, 0.0, float(shift_range[0]), -np.inf],
                [np.inf, np.inf, float(shift_range[1]), np.inf]),
    )
    a, r, shift, _ = fit.x
    values = residual(fit.x)
    dof = len(values) - len(fit.x)
    fit_rank = int(np.linalg.matrix_rank(fit.jac))
    uncertainty = None
    if dof > 0 and fit_rank == len(fit.x):
        covariance = np.linalg.inv(fit.jac.T @ fit.jac) * float(values @ values / dof)
        uncertainty = float(math.sqrt(max(covariance[2, 2], 0.0)))
    total = a + r
    observed_centered = observed - float(np.mean(observed))
    ss_total = float(observed_centered @ observed_centered)
    fit_quality = {
        "rmse_normalized_intensity": float(math.sqrt(np.mean(values ** 2))),
        "r_squared": 1.0 - float(values @ values) / ss_total if ss_total else None,
        "degrees_of_freedom": dof,
        "jacobian_rank": fit_rank,
        "shift_uncertainty_method": (
            "local least-squares covariance from residuals; excludes instrument/systematic uncertainty"
            if uncertainty is not None else "unavailable: fit covariance is rank deficient"
        ),
    }
    return {
        "status": "AVAILABLE" if fit_rank == len(fit.x) else "UNAVAILABLE",
        "reason": None if fit_rank == len(fit.x) else "reference profile fit is not identifiable",
        "sample_displacement_deg": float(shift) if fit_rank == len(fit.x) else None,
        "shift_uncertainty_deg": uncertainty,
        "fit_quality": fit_quality,
        "relative_crystalline_fraction": {
            "anatase": float(a / total) if total and fit_rank == len(fit.x) else None,
            "rutile": float(r / total) if total and fit_rank == len(fit.x) else None,
            "interpretation": "relative crystalline fraction; not absolute wt%",
        },
        "reference": {
            "path": str(reference_path),
            "sha256": _sha256(reference_path),
            **sample.metadata["xrd_reference"],
        },
        "shift_interpretation": "global profile alignment; not by itself evidence of lattice or composition change",
    }


def _tauc_fit(
    energy: np.ndarray,
    absorbance: np.ndarray,
    exponent: float,
    window: list[float],
) -> dict[str, Any]:
    mask = (energy >= float(window[0])) & (energy <= float(window[1])) & (absorbance > 0)
    x = energy[mask]
    y = (absorbance[mask] * x) ** exponent
    if len(x) < 3:
        return {
            "band_gap_ev": None,
            "rmse": None,
            "r_squared": None,
            "n_points": int(len(x)),
            "status": "UNAVAILABLE: fewer than three positive points in fit window",
        }
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = y - predicted
    rmse = float(math.sqrt(np.mean(residual ** 2)))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    gap = -float(intercept / slope) if slope > 0 else None
    valid_gap = gap is not None and 0.0 < gap < float(x.max())
    return {
        "band_gap_ev": gap if valid_gap else None,
        "unconstrained_intercept_ev": gap,
        "rmse": rmse,
        "r_squared": 1.0 - float(residual @ residual) / ss_total if ss_total else None,
        "n_points": int(len(x)),
        "fitting_window_ev": [float(window[0]), float(window[1])],
        "exponent": exponent,
        "status": "AVAILABLE" if valid_gap else "UNAVAILABLE: nonpositive intercept or intercept beyond measured energy range",
    }


def _fit_optical(sample: RealSample) -> dict[str, Any]:
    cfg = sample.metadata["optical"]
    energy = sample.optical.x.copy()
    if cfg["x_unit"] == "nm":
        energy = 1239.841984 / energy
    signal = sample.optical.signal.copy()
    order = np.argsort(energy)
    energy, signal = energy[order], signal[order]
    if np.any(np.diff(energy) <= 0):
        raise ValueError("optical x values must map to unique photon energies")

    baseline_window = cfg["alternate_baseline_window_ev"]
    baseline_mask = (
        (energy >= float(baseline_window[0]))
        & (energy <= float(baseline_window[1]))
    )
    if int(baseline_mask.sum()) < 2:
        baseline_slope = baseline_intercept = None
        corrected = None
    else:
        baseline_slope, baseline_intercept = np.polyfit(
            energy[baseline_mask], signal[baseline_mask], 1
        )
        corrected = signal - (baseline_slope * energy + baseline_intercept)
        corrected = np.clip(corrected, 0.0, None)

    models = {}
    for phase, model in SPEC["checks"]["C4"]["models"].items():
        exponent = float(model["exponent"])
        primary = _tauc_fit(energy, signal, exponent, cfg["tauc_fit_window_ev"])
        alternate = (
            _tauc_fit(energy, corrected, exponent, cfg["tauc_fit_window_ev"])
            if corrected is not None else {
                "band_gap_ev": None,
                "rmse": None,
                "r_squared": None,
                "status": "UNAVAILABLE: alternate-baseline window has fewer than two points",
            }
        )
        primary_gap = primary["band_gap_ev"]
        alternate_gap = alternate["band_gap_ev"]
        models[phase] = {
            "transition": model["transition"],
            "exponent": exponent,
            "primary": primary,
            "alternate_baseline": alternate,
            "delta_band_gap_ev": (
                alternate_gap - primary_gap
                if alternate_gap is not None and primary_gap is not None else None
            ),
        }
    return {
        "status": "AVAILABLE",
        "measurement_type": cfg["measurement_type"],
        "signal_type": cfg["signal_type"],
        "x_unit": cfg["x_unit"],
        "energy_conversion": "energy_eV = 1239.841984 / wavelength_nm" if cfg["x_unit"] == "nm" else "x is photon energy in eV",
        "alternate_method": "linear baseline fitted over caller-supplied alternate_baseline_window_ev, clipped at zero before Tauc transform",
        "defect_tail_model": "not specified; no physical defect-tail model is fitted",
        "alternate_baseline_window_ev": baseline_window,
        "alternate_baseline_fit": (
            {"slope": float(baseline_slope), "intercept": float(baseline_intercept)}
            if baseline_slope is not None else None
        ),
        "models": models,
    }


def _check_c5(sample: RealSample, reference_path: str | Path | None) -> dict[str, Any]:
    if sample.repeat is None:
        return {"status": "UNAVAILABLE", "reason": "independent repeat measurement not supplied"}
    results = {}
    if sample.repeat.composition:
        repeated = replace(sample, composition=sample.repeat.composition)
        c1 = check_c1(repeated)
        c1_original = check_c1(sample)
        results["composition_O_to_Ti_delta"] = (
            c1["measured_O_to_Ti"] - c1_original["measured_O_to_Ti"]
        )
    if sample.repeat.xrd and reference_path:
        repeated = replace(sample, xrd=sample.repeat.xrd)
        first, second = _fit_xrd(sample, reference_path), _fit_xrd(repeated, reference_path)
        results["xrd_shift_delta_deg"] = (
            second["sample_displacement_deg"] - first["sample_displacement_deg"]
            if second["sample_displacement_deg"] is not None
            and first["sample_displacement_deg"] is not None else None
        )
    if sample.repeat.optical:
        repeated = replace(sample, optical=sample.repeat.optical)
        first, second = _fit_optical(sample), _fit_optical(repeated)
        results["optical_delta_band_gap_ev"] = {
            phase: (
                second["models"][phase]["primary"]["band_gap_ev"]
                - first["models"][phase]["primary"]["band_gap_ev"]
                if second["models"][phase]["primary"]["band_gap_ev"] is not None
                and first["models"][phase]["primary"]["band_gap_ev"] is not None
                else None
            )
            for phase in first["models"]
        }
    if not results:
        return {
            "status": "UNAVAILABLE",
            "reason": "repeat was supplied, but no supplied repeat measurement could be compared",
        }
    return {
        "status": "AVAILABLE",
        "measurements_compared": sorted(results),
        "deltas": results,
        "interpretation": "differences between supplied independent measurements; no repeat was simulated",
    }


class RealDataCheckRunner:
    """Run evidence checks on one RealSample, never a simulator Case."""

    def __init__(self, sample: RealSample, xrd_reference: str | Path | None = None):
        self.sample = sample
        self.xrd_reference = xrd_reference

    def run(self, check_id: str) -> dict[str, Any]:
        if check_id == "C1":
            return check_c1(self.sample)
        if check_id == "C2":
            return _fit_xrd(self.sample, self.xrd_reference)
        if check_id == "C3":
            result = _fit_xrd(self.sample, self.xrd_reference)
            fraction = result.get("relative_crystalline_fraction")
            if fraction is None:
                return {"status": "UNAVAILABLE", "reason": result["reason"]}
            return {
                "status": result["status"],
                **fraction,
                "fit_residual": result["fit_quality"]["rmse_normalized_intensity"],
                "sample_displacement_deg": result["sample_displacement_deg"],
            }
        if check_id == "C4":
            return _fit_optical(self.sample)
        if check_id == "C5":
            return _check_c5(self.sample, self.xrd_reference)
        raise ValueError(f"unknown real-data check: {check_id}")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def diagnose(
    *,
    composition_path: str | Path,
    xrd_path: str | Path,
    optical_path: str | Path,
    metadata_path: str | Path,
    out_dir: str | Path,
    xrd_reference: str | Path | None = None,
    repeat_composition_path: str | Path | None = None,
    repeat_xrd_path: str | Path | None = None,
    repeat_optical_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    sample = load_real_sample(
        composition_path=composition_path,
        xrd_path=xrd_path,
        optical_path=optical_path,
        metadata_path=metadata_path,
        repeat_composition_path=repeat_composition_path,
        repeat_xrd_path=repeat_xrd_path,
        repeat_optical_path=repeat_optical_path,
    )
    if xrd_reference:
        sample.provenance["xrd_reference_pattern"] = {
            "path": str(xrd_reference),
            "sha256": _sha256(xrd_reference),
            **(sample.metadata.get("xrd_reference") or {}),
        }
    runner = RealDataCheckRunner(sample, xrd_reference)
    checks = {check: runner.run(check) for check in ("C1", "C2", "C3", "C4", "C5")}
    diagnosis = {
        "status": SPEC["diagnosis"]["default"],
        "hypothesis": None,
        "confidence": None,
        "posterior": None,
        "reason": "real-data H1-H4 calibration is unavailable; report evidence without assigning a fault",
    }
    model_metadata = None
    if model_path is not None:
        from .real_calibration import load_real_model, predict_real

        model = load_real_model(model_path)
        prediction = predict_real(model, checks)
        diagnosis = {
            "status": prediction["diagnosis"],
            "hypothesis": (
                prediction["diagnosis"]
                if prediction["diagnosis"] in {"H1", "H2", "H3", "H4"} else None
            ),
            "confidence": prediction["confidence"],
            "posterior": prediction["posterior"],
            "reason": prediction["reason"],
            "model_domain": model["training_domain"],
            "model_dataset_id": model["dataset_id"],
        }
        model_metadata = {"path": str(model_path), "sha256": _sha256(model_path)}
    report = {
        "spec_version": SPEC["spec_version"],
        "sample": {
            "sample_id": sample.sample_id,
            "material": sample.material,
            "metadata": sample.metadata,
        },
        "measurements": {
            "composition": {
                element: {"value": value, "uncertainty": uncertainty, "unit": unit}
                for element, (value, uncertainty, unit) in sample.composition.elements.items()
            },
            "xrd": {
                "points": len(sample.xrd.two_theta),
                "two_theta_range_deg": [
                    float(sample.xrd.two_theta.min()), float(sample.xrd.two_theta.max())
                ],
            },
            "optical": {"points": len(sample.optical.x)},
            "repeat_supplied": sample.repeat is not None,
        },
        "evidence": checks,
        "diagnosis": diagnosis,
        "calibration_model": model_metadata,
        "provenance": sample.provenance,
        "limitations": [
            "No real-data-calibrated likelihood/prior is available; H1-H4 diagnosis is not made.",
            "Synthetic simulator likelihood tables and priors are not loaded or reused.",
            "C2 and C3 require a user-supplied reference profile with provenance; no reference profiles are bundled.",
            "C1 reports deviation and propagated supplied uncertainty without an invented tolerance.",
            "C3 fractions are relative crystalline fractions, not absolute weight fractions.",
            "C4 is an evidence-level Tauc fit; model selection and real-data validation are not established.",
            "C2 uncertainty is residual-based fit covariance and excludes instrumental/systematic uncertainty.",
        ],
    }
    report = _jsonable(report)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "evidence.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(_render_user_report(report), encoding="utf-8")
    return report


def _render_user_report(report: dict[str, Any]) -> str:
    lines = [
        "# TiO2 measurement evidence report",
        "",
        "> Evidence-first output. This report does not assign an H1-H4 fault diagnosis.",
        "",
        f"- Sample: `{report['sample']['sample_id']}`",
        f"- Material: `{report['sample']['material']}` (TiO2 anatase/rutile scope)",
        f"- Spec version: `{report['spec_version']}`",
        f"- Diagnosis: **{report['diagnosis']['status']}**",
        "",
        "## Checks",
        "",
    ]
    for check, result in report["evidence"].items():
        lines.append(f"### {check}: {result.get('status', 'reported')}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(result, indent=2))
        lines.extend(["```", ""])
    lines.extend([
        "## Limitations",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
    ])
    return "\n".join(lines)
