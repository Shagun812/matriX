"""Seeded phenomenological simulator for the MatriX benchmark.

LOCKED DESIGN DECISIONS
-----------------------
D1. The initial evidence is a coarse screen. Screen measurements use a degraded
    view of the same sample: low-resolution/higher-noise XRD, one higher-noise
    composition measurement, and one standard-baseline optical fit. C1-C4 use
    refined analyses and therefore must not simply reproduce screen values.

D2. C5 is a fresh higher-resolution repeat. H1/H2 intrinsic effects persist.
    H3 is re-mounted and its offset is redrawn near zero. H4 is re-processed
    and its distortion is redrawn at reduced magnitude.

D3. The oracle outcome is three binary screen flags plus five 3-bin check
    scores.

D4. The implementation is intentionally compact and deterministic. All random
    draws come from an explicit numpy Generator supplied by the caller.

This is a benchmark simulator, not a validated digital twin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schemas import AgentVisible, Case, HiddenInfo, HYPOTHESES
from .sim_constants import (
    ANATASE_PEAKS,
    COMPOSITION_TOL,
    H1_RUTILE_GAP_EV,
    H2_COMPOSITION_SHIFT_PER_MAG,
    H2_OPTICAL_SHIFT_PER_MAG,
    H2_XRD_SHIFT_PER_MAG,
    H3_OFFSET_PER_MAG,
    H3_REPEAT_OFFSET_SCALE,
    H4_OPTICAL_SHIFT_PER_MAG,
    H4_REPEAT_SCALE,
    MAGNITUDE_BANDS,
    NOISE_LEVELS,
    OPTICAL_GAP_TOL,
    REFERENCE_COMPOSITION,
    REFERENCE_GAP_EV,
    REFERENCE_MAIN_PEAK_2THETA,
    REFERENCE_PHASE,
    REPEAT_NOISE_FACTOR,
    RUTILE_PEAKS,
    SIMULATOR_VERSION,
    STRUCTURE_RMS_TOL,
    STRUCTURE_UNMATCHED_TOL,
)


@dataclass(frozen=True)
class SimConfig:
    """Explicit simulator configuration used by generation."""

    k_screen: float = 3.0
    composition_tol: float = COMPOSITION_TOL
    structure_rms_tol: float = STRUCTURE_RMS_TOL
    structure_unmatched_tol: float = STRUCTURE_UNMATCHED_TOL
    optical_gap_tol: float = OPTICAL_GAP_TOL
    simulator_version: str = SIMULATOR_VERSION


@dataclass
class Evidence:
    """All simulated evidence, including hidden/refined measurements."""

    # Screen measurements
    screen_composition: float
    screen_rms_2theta: float
    screen_unmatched_frac: float
    screen_gap_ev: float

    # Refined measurements
    composition_replicates: np.ndarray
    refined_peak_positions: np.ndarray
    refined_peak_weights: np.ndarray
    refined_gap_ev: float

    # Literature/reference metadata
    lit_ref_phase: str
    lit_composition: float
    lit_gap_ev: float
    lit_main_peak_2theta: float

    # Hidden simulator state useful to checks/oracle.
    true_fault: str
    magnitude: float
    noise_level: str
    seed: int


def _sample_magnitude(rng: np.random.Generator, band: str) -> float:
    lo, hi = MAGNITUDE_BANDS[band]
    return float(rng.uniform(lo, hi))


def _sample_noise(noise_level: str) -> float:
    return float(NOISE_LEVELS[noise_level])


def _fault_state(
    fault: str,
    magnitude: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Return phenomenological fault perturbations."""

    state = {
        "secondary_fraction": 0.0,
        "composition_offset": 0.0,
        "xrd_offset": 0.0,
        "optical_shift": 0.0,
    }

    if fault == "H1":
        state["secondary_fraction"] = magnitude

    elif fault == "H2":
        state["composition_offset"] = H2_COMPOSITION_SHIFT_PER_MAG * magnitude
        state["xrd_offset"] = H2_XRD_SHIFT_PER_MAG * magnitude
        state["optical_shift"] = H2_OPTICAL_SHIFT_PER_MAG * magnitude

    elif fault == "H3":
        # Constant instrumental offset across the pattern.
        state["xrd_offset"] = H3_OFFSET_PER_MAG * magnitude

    elif fault == "H4":
        state["optical_shift"] = H4_OPTICAL_SHIFT_PER_MAG * magnitude

    else:
        raise ValueError(f"unknown fault: {fault}")

    return state


def _simulate_xrd(
    fault: str,
    magnitude: float,
    sigma: float,
    rng: np.random.Generator,
    *,
    screen: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic peak positions and weights."""

    state = _fault_state(fault, magnitude, rng)
    noise = sigma * (3.0 if screen else 1.0)

    anatase = np.asarray(ANATASE_PEAKS, dtype=float)
    positions = anatase + state["xrd_offset"]
    weights = np.exp(-0.5 * ((positions - REFERENCE_MAIN_PEAK_2THETA) / 18.0) ** 2)

    if fault == "H1":
        rutile = np.asarray(RUTILE_PEAKS, dtype=float)
        n = min(len(rutile), len(positions))
        extra = np.full(n, magnitude)
        positions = np.concatenate([positions, rutile[:n]])
        weights = np.concatenate([weights, extra])

    positions = positions + rng.normal(0.0, noise, size=len(positions))
    weights = np.maximum(0.0, weights + rng.normal(0.0, noise * 0.2, size=len(weights)))

    return positions, weights


def _simulate_composition(
    fault: str,
    magnitude: float,
    sigma: float,
    rng: np.random.Generator,
    *,
    screen: bool,
) -> np.ndarray:
    """Generate one or more composition measurements."""

    state = _fault_state(fault, magnitude, rng)
    noise = sigma * (3.0 if screen else 1.0)
    n = 1 if screen else 3

    center = REFERENCE_COMPOSITION + state["composition_offset"]
    return center + rng.normal(0.0, noise, size=n)


def _simulate_optical(
    fault: str,
    magnitude: float,
    sigma: float,
    rng: np.random.Generator,
    *,
    screen: bool,
) -> float:
    """Generate a phenomenological Tauc-like fitted gap."""

    state = _fault_state(fault, magnitude, rng)
    noise = sigma * (3.0 if screen else 1.0)

    gap = REFERENCE_GAP_EV

    if fault == "H1":
        gap += magnitude * (H1_RUTILE_GAP_EV - REFERENCE_GAP_EV)

    gap += state["optical_shift"]

    # Screen uses a standard baseline; refined analysis uses a separate
    # baseline and therefore gets an independent realization.
    if screen:
        baseline_bias = 0.0
    else:
        baseline_bias = rng.normal(0.0, sigma * 0.25)

    return float(gap + baseline_bias + rng.normal(0.0, noise))


def _structure_screen(
    positions: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return coarse structure conflict magnitudes."""

    ref = np.asarray(ANATASE_PEAKS, dtype=float)

    # For each reference peak find its nearest observed peak.
    distances = np.min(np.abs(ref[:, None] - positions[None, :]), axis=1)
    rms = float(np.sqrt(np.mean(distances**2)))

    unmatched = float(np.mean(distances > 0.30))
    return rms, unmatched


def _conflict_score(
    t1: float,
    t2: float,
    t3: float,
    cfg: SimConfig,
) -> float:
    """Continuous pre-threshold screen conflict score, capped at 3."""

    vals = [
        min(abs(t1) / cfg.composition_tol, 3.0),
        min(abs(t2) / cfg.structure_rms_tol, 3.0),
        min(abs(t3) / cfg.optical_gap_tol, 3.0),
    ]
    return float(np.mean(vals))


def simulate_evidence(
    *,
    fault: str,
    magnitude_band: str,
    noise_level: str,
    seed: int,
    config: SimConfig | None = None,
) -> Evidence:
    """Simulate one accepted-or-not case without applying the screen filter."""

    cfg = config or SimConfig()
    rng = np.random.default_rng(seed)
    magnitude = _sample_magnitude(rng, magnitude_band)
    sigma = _sample_noise(noise_level)

    screen_xrd, _ = _simulate_xrd(
        fault, magnitude, sigma, rng, screen=True
    )
    screen_comp = _simulate_composition(
        fault, magnitude, sigma, rng, screen=True
    )[0]
    screen_gap = _simulate_optical(
        fault, magnitude, sigma, rng, screen=True
    )
    screen_rms, screen_unmatched = _structure_screen(
        screen_xrd, rng=rng
    )

    refined_xrd, refined_weights = _simulate_xrd(
        fault, magnitude, sigma, rng, screen=False
    )
    refined_comp = _simulate_composition(
        fault, magnitude, sigma, rng, screen=False
    )
    refined_gap = _simulate_optical(
        fault, magnitude, sigma, rng, screen=False
    )

    return Evidence(
        screen_composition=float(screen_comp),
        screen_rms_2theta=float(screen_rms),
        screen_unmatched_frac=float(screen_unmatched),
        screen_gap_ev=float(screen_gap),
        composition_replicates=refined_comp,
        refined_peak_positions=refined_xrd,
        refined_peak_weights=refined_weights,
        refined_gap_ev=refined_gap,
        lit_ref_phase=REFERENCE_PHASE,
        lit_composition=REFERENCE_COMPOSITION,
        lit_gap_ev=REFERENCE_GAP_EV,
        lit_main_peak_2theta=REFERENCE_MAIN_PEAK_2THETA,
        true_fault=fault,
        magnitude=float(magnitude),
        noise_level=noise_level,
        seed=seed,
    )


def make_case(
    *,
    case_id: str,
    fault: str,
    magnitude_band: str,
    noise_level: str,
    seed: int,
    config: SimConfig | None = None,
) -> Case | None:
    """Generate a case, returning None when the coarse screen is non-conflicting."""

    cfg = config or SimConfig()

    # We need the magnitude for HiddenInfo, so reproduce the first RNG draw
    # before running the full evidence simulation.
    probe_rng = np.random.default_rng(seed)
    magnitude = _sample_magnitude(probe_rng, magnitude_band)

    evidence = simulate_evidence(
        fault=fault,
        magnitude_band=magnitude_band,
        noise_level=noise_level,
        seed=seed,
        config=cfg,
    )

    t1 = abs(evidence.screen_composition - evidence.lit_composition)
    t2 = min(
        evidence.screen_rms_2theta / cfg.structure_rms_tol,
        evidence.screen_unmatched_frac / cfg.structure_unmatched_tol,
    )
    t3 = abs(evidence.screen_gap_ev - evidence.lit_gap_ev)

    t1_fail = t1 > cfg.composition_tol
    t2_fail = (
        evidence.screen_rms_2theta > cfg.structure_rms_tol
        or evidence.screen_unmatched_frac > cfg.structure_unmatched_tol
    )
    t3_fail = t3 > cfg.optical_gap_tol

    if not (t1_fail or t2_fail or t3_fail):
        return None

    visible = AgentVisible(
        numeric={
            "t1_mag": float(t1),
            "t2_mag": float(t2),
            "t3_mag": float(t3),
            "t1_fail": float(t1_fail),
            "t2_fail": float(t2_fail),
            "t3_fail": float(t3_fail),
            "screen_composition_dev": float(t1),
            "screen_rms_2theta": float(evidence.screen_rms_2theta),
            "screen_unmatched_frac": float(evidence.screen_unmatched_frac),
            "screen_gap_ev": float(evidence.screen_gap_ev),
            "lit_gap_ev": float(evidence.lit_gap_ev),
            "main_peak_2theta": float(
                evidence.refined_peak_positions[0]
            ),
            "lit_main_peak_2theta": float(evidence.lit_main_peak_2theta),
            "lit_composition": float(evidence.lit_composition),
        },
        categorical={"lit_ref_phase": evidence.lit_ref_phase},
    )

    hidden = HiddenInfo(
        fault=fault,
        magnitude=float(magnitude),
        magnitude_band=magnitude_band,
        noise_level=noise_level,
        resolvable=False,
        seed=seed,
        simulator_version=cfg.simulator_version,
        screen_conflict_score=_conflict_score(
            t1, t2, t3, cfg
        ),
    )

    return Case(
        case_id=case_id,
        agent_visible=visible,
        hidden=hidden,
        evidence=evidence.__dict__,
        simulator_version=cfg.simulator_version,
        seed=seed,
        permitted_checks=["C1", "C2", "C3", "C4", "C5"],
        label_generation_rule="oracle-v1",
    )


def generate_cases(
    n: int,
    *,
    seed: int = 0,
    faults: tuple[str, ...] = ("H1", "H2", "H3", "H4"),
    magnitude_bands: tuple[str, ...] = tuple(MAGNITUDE_BANDS),
    noise_levels: tuple[str, ...] = tuple(NOISE_LEVELS),
    config: SimConfig | None = None,
) -> list[Case]:
    """Generate n accepted single-fault cases deterministically."""

    if n < 0:
        raise ValueError("n must be non-negative")

    rng = np.random.default_rng(seed)
    cases: list[Case] = []
    attempts = 0

    while len(cases) < n:
        attempts += 1
        if attempts > max(1000, n * 100):
            raise RuntimeError("unable to generate enough conflicting cases")

        fault = faults[int(rng.integers(len(faults)))]
        band = magnitude_bands[int(rng.integers(len(magnitude_bands)))]
        noise = noise_levels[int(rng.integers(len(noise_levels)))]
        case_seed = int(rng.integers(0, 2**32 - 1))

        case = make_case(
            case_id=f"case-{seed}-{len(cases):04d}",
            fault=fault,
            magnitude_band=band,
            noise_level=noise,
            seed=case_seed,
            config=config,
        )

        if case is not None:
            cases.append(case)

    return cases


def generate_visible(
    n: int,
    seed: int,
    *,
    stream: str = "dev_likelihood",
) -> tuple[list[AgentVisible], list[str]]:
    """Generate accepted agent-visible training cases from the dev-likelihood cells."""

    if n < 0:
        raise ValueError("n must be non-negative")
    if stream != "dev_likelihood":
        raise ValueError(f"unsupported training stream: {stream}")

    magnitude_bands = ("B1", "B2", "B4", "B5")
    noise_levels = ("sigma1", "sigma2")
    rng = np.random.default_rng(seed)
    visibles: list[AgentVisible] = []
    labels: list[str] = []
    attempts = 0

    while len(visibles) < n:
        attempts += 1
        if attempts > max(1000, n * 100):
            raise RuntimeError("unable to generate enough conflicting training cases")

        fault = HYPOTHESES[int(rng.integers(len(HYPOTHESES)))]
        band = magnitude_bands[int(rng.integers(len(magnitude_bands)))]
        noise = noise_levels[int(rng.integers(len(noise_levels)))]
        case_seed = int(rng.integers(0, 2**32 - 1))
        case = make_case(
            case_id=f"prior-{seed}-{len(visibles):05d}",
            fault=fault,
            magnitude_band=band,
            noise_level=noise,
            seed=case_seed,
        )
        if case is not None:
            visibles.append(case.agent_visible)
            labels.append(fault)

    return visibles, labels


def repeat_evidence(
    case: Case,
    *,
    repeat_seed: int,
    config: SimConfig | None = None,
) -> dict[str, Any]:
    """Generate the D2 higher-resolution repeat for C5."""

    cfg = config or SimConfig()

    fault = case.hidden.fault
    magnitude = float(case.hidden.magnitude)
    noise_level = case.hidden.noise_level
    sigma = _sample_noise(noise_level) * REPEAT_NOISE_FACTOR
    rng = np.random.default_rng(repeat_seed)

    # H1/H2 persist; H3/H4 are partly or fully redrawn.
    if fault == "H3":
        xrd_fault = "H3"
        effective_mag = float(rng.normal(0.0, magnitude * H3_REPEAT_OFFSET_SCALE))
    elif fault == "H4":
        xrd_fault = "H4"
        effective_mag = magnitude * H4_REPEAT_SCALE
    else:
        xrd_fault = fault
        effective_mag = magnitude

    # Use direct measurements rather than resampling a magnitude band.
    if fault == "H3":
        xrd_offset = effective_mag * H3_OFFSET_PER_MAG
    elif fault == "H2":
        xrd_offset = effective_mag * H2_XRD_SHIFT_PER_MAG
    else:
        xrd_offset = 0.0

    positions = np.asarray(ANATASE_PEAKS, dtype=float) + xrd_offset
    positions = positions + rng.normal(0.0, sigma, size=len(positions))

    composition_center = (
        REFERENCE_COMPOSITION + effective_mag if fault == "H2" else REFERENCE_COMPOSITION
    )
    composition = composition_center + rng.normal(0.0, sigma)

    if fault == "H1":
        gap = REFERENCE_GAP_EV + effective_mag * (
            H1_RUTILE_GAP_EV - REFERENCE_GAP_EV
        )
    elif fault == "H4":
        gap = REFERENCE_GAP_EV + effective_mag * H4_OPTICAL_SHIFT_PER_MAG
    elif fault == "H2":
        gap = REFERENCE_GAP_EV + effective_mag * H2_OPTICAL_SHIFT_PER_MAG
    else:
        gap = REFERENCE_GAP_EV

    gap += rng.normal(0.0, sigma)

    rms, unmatched = _structure_screen(positions, rng=rng)

    t1 = abs(composition - REFERENCE_COMPOSITION)
    t2 = min(rms / cfg.structure_rms_tol, unmatched / cfg.structure_unmatched_tol)
    t3 = abs(gap - REFERENCE_GAP_EV)

    return {
        "composition": float(composition),
        "rms_2theta": float(rms),
        "unmatched_frac": float(unmatched),
        "gap_ev": float(gap),
        "conflict_score": _conflict_score(t1, t2, t3, cfg),
        "fault": fault,
        "effective_magnitude": float(effective_mag),
        "repeat_seed": repeat_seed,
    }
