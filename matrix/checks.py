"""Deterministic C1-C5 diagnostic checks for the MatriX benchmark."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import nnls

from .schemas import Case, CheckResult
from .sim import repeat_evidence
from .sim_constants import (
    ANATASE_PEAKS,
    COMPOSITION_TOL,
    MAX_PHASE_FRACTION,
    MIN_PHASE_FRACTION,
    OPTICAL_GAP_TOL,
    STRUCTURE_RMS_TOL,
)


def _require(case: Case, key: str):
    if key not in case.evidence:
        raise KeyError(f"case {case.case_id}: missing evidence field {key!r}")
    return case.evidence[key]


def _composition(case: Case) -> float:
    reps = np.asarray(_require(case, "composition_replicates"), dtype=float)

    if reps.size == 0:
        raise ValueError(f"case {case.case_id}: empty composition replicates")

    return float(np.mean(reps))


def _composition_deviation(case: Case) -> float:
    measured = _composition(case)
    nominal = float(_require(case, "lit_composition"))

    return abs(measured - nominal) / max(abs(nominal), 1e-12)


def check_c1(case: Case) -> CheckResult:
    """Composition consistency.

    Primary score is normalized absolute composition deviation.
    """

    deviation = _composition_deviation(case)
    measured = _composition(case)
    nominal = float(_require(case, "lit_composition"))

    return CheckResult(
        check_id="C1",
        primary_score=float(deviation),
        aux={
            "tolerance": float(COMPOSITION_TOL),
            "tolerance_flag": float(deviation > COMPOSITION_TOL),
            "measured_composition": measured,
            "nominal_composition": nominal,
        },
    )


def check_c2(case: Case) -> CheckResult:
    """Fit a constant XRD 2theta offset against anatase reference peaks.

    Primary score is the signed fitted offset.
    """

    observed = np.asarray(
        _require(case, "refined_peak_positions"),
        dtype=float,
    )
    reference = np.asarray(ANATASE_PEAKS, dtype=float)

    if observed.size < reference.size:
        raise ValueError(
            f"case {case.case_id}: expected at least "
            f"{reference.size} refined peaks, got {observed.size}"
        )

    # H1 can append rutile reflections.  The first six positions correspond
    # to the six anatase reference reflections, so fit the common offset only
    # to those expected anatase peaks.
    residuals = observed[: reference.size] - reference

    # Least-squares solution for y_i = reference_i + offset.
    offset = float(np.mean(residuals))
    rms = float(np.sqrt(np.mean((residuals - offset) ** 2)))

    return CheckResult(
        check_id="C2",
        primary_score=offset,
        aux={
            "rms_residual": rms,
            "abs_offset": abs(offset),
            "tolerance": float(STRUCTURE_RMS_TOL),
            "tolerance_flag": float(rms > STRUCTURE_RMS_TOL),
        },
    )


def check_c3(case: Case) -> CheckResult:
    """Estimate anatase/rutile phase fractions using NNLS.

    The simulator stores six anatase reflections followed by rutile
    reflections for H1 cases.  The corresponding reference peak-weight
    templates are represented by binary phase-membership vectors.
    """

    observed = np.asarray(
        _require(case, "refined_peak_weights"),
        dtype=float,
    )

    if observed.size < len(ANATASE_PEAKS):
        raise ValueError(
            f"case {case.case_id}: expected at least "
            f"{len(ANATASE_PEAKS)} peak weights, got {observed.size}"
        )

    anatase = np.zeros_like(observed)
    rutile = np.zeros_like(observed)

    n_anatase = min(len(ANATASE_PEAKS), observed.size)
    anatase[:n_anatase] = 1.0

    if observed.size > n_anatase:
        rutile[n_anatase:] = 1.0

    design = np.column_stack([anatase, rutile])

    coeffs, nnls_residual_norm = nnls(design, observed)
    total = float(coeffs.sum())

    if total > 0.0:
        fractions = coeffs / total
    else:
        fractions = np.array([1.0, 0.0], dtype=float)

    fractions = np.clip(
        fractions,
        MIN_PHASE_FRACTION,
        MAX_PHASE_FRACTION,
    )

    # Re-normalize after clipping for numerical consistency.
    fraction_sum = float(fractions.sum())
    if fraction_sum > 0.0:
        fractions = fractions / fraction_sum

    residual = float(
        np.linalg.norm(design @ coeffs - observed)
        / max(np.sqrt(observed.size), 1.0)
    )

    return CheckResult(
        check_id="C3",
        primary_score=float(fractions[1]),
        aux={
            "anatase_fraction": float(fractions[0]),
            "rutile_fraction": float(fractions[1]),
            "residual": residual,
            "nnls_residual_norm": float(nnls_residual_norm),
        },
    )


def check_c4(case: Case) -> CheckResult:
    """Alternate-baseline optical gap shift.

    Primary score is signed refined-gap minus literature-gap.
    """

    refined_gap = float(_require(case, "refined_gap_ev"))
    reference_gap = float(_require(case, "lit_gap_ev"))

    delta = refined_gap - reference_gap

    return CheckResult(
        check_id="C4",
        primary_score=float(delta),
        aux={
            "abs_delta_gap_ev": abs(delta),
            "tolerance": float(OPTICAL_GAP_TOL),
            "tolerance_flag": float(abs(delta) > OPTICAL_GAP_TOL),
            "reference_gap_ev": reference_gap,
            "refined_gap_ev": refined_gap,
        },
    )


def _screen_conflict(case: Case) -> float:
    """Continuous T1-T3 screen conflict score.

    Each continuous pre-threshold deviation is normalized by its frozen
    tolerance and capped at 3 before averaging.
    """

    composition = abs(
        float(_require(case, "screen_composition"))
        - float(_require(case, "lit_composition"))
    )

    rms = float(_require(case, "screen_rms_2theta"))

    gap = abs(
        float(_require(case, "screen_gap_ev"))
        - float(_require(case, "lit_gap_ev"))
    )

    components = [
        min(composition / max(COMPOSITION_TOL, 1e-12), 3.0),
        min(rms / max(STRUCTURE_RMS_TOL, 1e-12), 3.0),
        min(gap / max(OPTICAL_GAP_TOL, 1e-12), 3.0),
    ]

    return float(np.mean(components))


def _repeat_seed(case: Case) -> int:
    """Derive a deterministic independent seed from the case seed."""

    # Keep the repeat stream independent from the original simulator stream
    # while remaining completely reproducible for a case.
    seed_sequence = np.random.SeedSequence(
        [
            int(case.hidden.seed),
            0xC5,
        ]
    )

    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def check_c5(case: Case) -> CheckResult:
    """Independent higher-resolution repeat.

    Primary score is:

        repeat_conflict - original_conflict

    Negative values mean the repeat reduced the observed conflict.
    """

    original_conflict = _screen_conflict(case)
    repeat_seed = _repeat_seed(case)

    repeat = repeat_evidence(
        case,
        repeat_seed=repeat_seed,
    )

    repeat_conflict = float(repeat["conflict_score"])
    delta = repeat_conflict - original_conflict

    return CheckResult(
        check_id="C5",
        primary_score=float(delta),
        aux={
            "original_conflict": float(original_conflict),
            "repeat_conflict": repeat_conflict,
            "repeat_seed": float(repeat_seed),
            "repeat_composition": float(repeat["composition"]),
            "repeat_rms_2theta": float(repeat["rms_2theta"]),
            "repeat_unmatched_frac": float(repeat["unmatched_frac"]),
            "repeat_gap_ev": float(repeat["gap_ev"]),
            "repeat_effective_magnitude": float(
                repeat["effective_magnitude"]
            ),
        },
    )


_CHECKS: dict[str, Callable[[Case], CheckResult]] = {
    "C1": check_c1,
    "C2": check_c2,
    "C3": check_c3,
    "C4": check_c4,
    "C5": check_c5,
}


def run_check(case: Case, check_id: str) -> CheckResult:
    try:
        fn = _CHECKS[check_id]
    except KeyError as exc:
        raise ValueError(f"unknown check_id: {check_id}") from exc

    if check_id not in case.permitted_checks:
        raise ValueError(
            f"case {case.case_id}: check {check_id} is not permitted"
        )

    return fn(case)


def runner(
    cases: list[Case] | dict[str, Case],
    case_id: str,
    check_id: str,
) -> CheckResult:
    """Compatibility runner for the experiment harness."""

    if isinstance(cases, dict):
        case = cases[case_id]
    else:
        case = next(c for c in cases if c.case_id == case_id)

    return run_check(case, check_id)


__all__ = [
    "check_c1",
    "check_c2",
    "check_c3",
    "check_c4",
    "check_c5",
    "run_check",
    "runner",
]
