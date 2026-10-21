"""Oracle likelihood fitting and identifiability for the MatriX benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .checks import run_check
from .planner import LikelihoodTable
from .schemas import CHECKS, DEFAULT_COSTS, HYPOTHESES, N_BINS, Case

SCREEN_FLAGS = ("t1_fail", "t2_fail", "t3_fail")
JOINT_SCREEN_BINS = 2


class JointLikelihoodTable:
    """Joint oracle table over 3 binary screen flags and 5 ternary check bins."""

    def __init__(
        self,
        table,
        edges: Mapping[str, Sequence[float]],
        hyps: Sequence[str] = HYPOTHESES,
        checks: Sequence[str] = CHECKS,
        screen_flags: Sequence[str] = SCREEN_FLAGS,
    ):
        t = np.asarray(table, dtype=float)
        expected = (
            len(hyps), 2, 2, 2,
            N_BINS, N_BINS, N_BINS, N_BINS, N_BINS,
        )
        if t.shape != expected:
            raise ValueError(f"table shape {t.shape} != {expected}")
        if (t < 0).any():
            raise ValueError("negative probabilities")
        sums = t.reshape(len(hyps), -1).sum(axis=1)
        if (sums <= 0).any():
            raise ValueError("a hypothesis has zero total mass")
        self.hyps = tuple(hyps)
        self.checks = tuple(checks)
        self.screen_flags = tuple(screen_flags)
        self.table = t / sums.reshape((-1,) + (1,) * 8)
        self.edges = {c: np.asarray(edges[c], dtype=float) for c in self.checks}
        for c in self.checks:
            e = self.edges[c]
            if e.shape != (N_BINS - 1,) or e[0] > e[1]:
                raise ValueError(f"bad edges for {c}: {e}")

    @property
    def shape(self):
        return self.table.shape

    @property
    def n_observable_cells(self):
        return 2 ** len(self.screen_flags) * N_BINS ** len(self.checks)

    def bin(self, check: str, score: float) -> int:
        return int(np.searchsorted(self.edges[check], score, side="right"))

    def screen_bins(self, case: Case) -> tuple[int, int, int]:
        numeric = case.agent_visible.numeric
        return tuple(int(bool(numeric[name])) for name in self.screen_flags)

    def _idx(self, screen, observed):
        if len(screen) != len(self.screen_flags):
            raise ValueError("expected three screen flags")
        return (
            slice(None),
            *tuple(int(v) for v in screen),
            *tuple(observed[c] if c in observed else slice(None) for c in self.checks),
        )

    def lik(self, screen, observed=None) -> np.ndarray:
        observed = observed or {}
        sub = self.table[self._idx(screen, observed)]
        return sub.reshape(len(self.hyps), -1).sum(axis=1)

    def posterior(self, screen, observed=None) -> np.ndarray:
        prior = np.full(len(self.hyps), 1.0 / len(self.hyps))
        likelihood = self.lik(screen, observed)
        weights = prior * likelihood
        total = float(weights.sum())
        return prior if total <= 0 else weights / total

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            table=self.table,
            hyps=np.asarray(self.hyps),
            checks=np.asarray(self.checks),
            screen_flags=np.asarray(self.screen_flags),
            **{f"edges_{check}": edge for check, edge in self.edges.items()},
        )

    @classmethod
    def load(cls, path) -> "JointLikelihoodTable":
        data = np.load(path, allow_pickle=False)
        checks = [str(check) for check in data["checks"]]
        return cls(
            data["table"],
            {check: data[f"edges_{check}"] for check in checks},
            [str(hypothesis) for hypothesis in data["hyps"]],
            checks,
            [str(flag) for flag in data["screen_flags"]],
        )

from .schemas import CHECKS, DEFAULT_COSTS, HYPOTHESES, N_BINS, Case
from .sim import (
    _sample_noise,
    _simulate_composition,
    _simulate_optical,
    _simulate_xrd,
    _structure_screen,
)
from .sim_constants import (
    ANATASE_PEAKS,
    MAGNITUDE_BANDS,
    NOISE_LEVELS,
    ORACLE_BINS,
    ORACLE_DRAWS,
    TAU_ID,
)


@dataclass(frozen=True)
class OracleConfig:
    """Configuration for oracle Monte Carlo and identifiability."""

    draws_per_hypothesis_noise: int = 50_000
    case_draws: int = ORACLE_DRAWS
    bins: int = ORACLE_BINS
    tau_id: float = TAU_ID
    smoothing: float = 1.0
    seed: int = 0
    magnitude_bands: tuple[str, ...] = tuple(MAGNITUDE_BANDS)
    noise_levels: tuple[str, ...] = tuple(NOISE_LEVELS)

    def __post_init__(self) -> None:
        if self.bins != N_BINS:
            raise ValueError(
                f"planner contract requires {N_BINS} bins, got {self.bins}"
            )
        if self.draws_per_hypothesis_noise <= 0:
            raise ValueError("draws_per_hypothesis_noise must be positive")
        if self.case_draws <= 0:
            raise ValueError("case_draws must be positive")
        if self.smoothing <= 0:
            raise ValueError("smoothing must be positive")
        if not 0 < self.tau_id <= 1:
            raise ValueError("tau_id must be in (0, 1]")


@dataclass(frozen=True)
class SubsetResult:
    checks: tuple[str, ...]
    accuracy: float
    cost: float


@dataclass(frozen=True)
class OracleResult:
    table: LikelihoodTable
    subset_results: tuple[SubsetResult, ...]
    best_accuracy: float
    cheapest_identifiable: tuple[str, ...] | None
    joint_table: JointLikelihoodTable | None = None
    joint_subset_results: tuple[SubsetResult, ...] = ()
    joint_best_accuracy: float | None = None
    joint_cheapest_identifiable: tuple[str, ...] | None = None


def oracle_seed(seed: int, *parts: int) -> int:
    """Derive an independent deterministic uint32 seed."""
    ss = np.random.SeedSequence(
        [int(seed), *[int(part) for part in parts]]
    )
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _make_oracle_case(
    *,
    fault: str,
    magnitude: float,
    noise_level: str,
    seed: int,
) -> Case:
    """Generate one unfiltered oracle realization at a fixed magnitude."""

    rng = np.random.default_rng(seed)
    sigma = _sample_noise(noise_level)

    # Keep the same measurement ordering as simulate_evidence(), but fix the
    # magnitude rather than drawing a new one.
    screen_xrd, _ = _simulate_xrd(
        fault,
        magnitude,
        sigma,
        rng,
        screen=True,
    )
    screen_comp = _simulate_composition(
        fault,
        magnitude,
        sigma,
        rng,
        screen=True,
    )[0]
    screen_gap = _simulate_optical(
        fault,
        magnitude,
        sigma,
        rng,
        screen=True,
    )
    screen_rms, screen_unmatched = _structure_screen(
        screen_xrd,
        rng=rng,
    )

    refined_xrd, refined_weights = _simulate_xrd(
        fault,
        magnitude,
        sigma,
        rng,
        screen=False,
    )
    refined_comp = _simulate_composition(
        fault,
        magnitude,
        sigma,
        rng,
        screen=False,
    )
    refined_gap = _simulate_optical(
        fault,
        magnitude,
        sigma,
        rng,
        screen=False,
    )

    # Import constants here to keep the oracle's simulator construction
    # consistent with sim.py without duplicating their values.
    from .sim_constants import (
        REFERENCE_COMPOSITION,
        REFERENCE_GAP_EV,
        REFERENCE_MAIN_PEAK_2THETA,
        REFERENCE_PHASE,
        SIMULATOR_VERSION,
        COMPOSITION_TOL,
        STRUCTURE_RMS_TOL,
        STRUCTURE_UNMATCHED_TOL,
        OPTICAL_GAP_TOL,
    )

    evidence = {
        "screen_composition": float(screen_comp),
        "screen_rms_2theta": float(screen_rms),
        "screen_unmatched_frac": float(screen_unmatched),
        "screen_gap_ev": float(screen_gap),
        "composition_replicates": refined_comp,
        "refined_peak_positions": refined_xrd,
        "refined_peak_weights": refined_weights,
        "refined_gap_ev": float(refined_gap),
        "lit_ref_phase": REFERENCE_PHASE,
        "lit_composition": REFERENCE_COMPOSITION,
        "lit_gap_ev": REFERENCE_GAP_EV,
        "lit_main_peak_2theta": REFERENCE_MAIN_PEAK_2THETA,
        "true_fault": fault,
        "magnitude": float(magnitude),
        "noise_level": noise_level,
        "seed": int(seed),
    }

    # Checks only require evidence plus hidden metadata.
    t1 = abs(screen_comp - REFERENCE_COMPOSITION)
    t2 = min(
        screen_rms / STRUCTURE_RMS_TOL,
        screen_unmatched / STRUCTURE_UNMATCHED_TOL,
    )
    t3 = abs(screen_gap - REFERENCE_GAP_EV)

    return Case(
        case_id="oracle",
        agent_visible={
            "numeric": {
                "t1_mag": float(t1),
                "t2_mag": float(t2),
                "t3_mag": float(t3),
                "t1_fail": float(t1 > COMPOSITION_TOL),
                "t2_fail": float(
                    screen_rms > STRUCTURE_RMS_TOL
                    or screen_unmatched > STRUCTURE_UNMATCHED_TOL
                ),
                "t3_fail": float(t3 > OPTICAL_GAP_TOL),
            },
            "categorical": {"lit_ref_phase": REFERENCE_PHASE},
        },
        hidden={
            "fault": fault,
            "magnitude": float(magnitude),
            "magnitude_band": "oracle",
            "noise_level": noise_level,
            "resolvable": False,
            "identifiability": None,
            "seed": int(seed),
        },
        evidence=evidence,
        simulator_version=SIMULATOR_VERSION,
        seed=int(seed),
        permitted_checks=list(CHECKS),
        label_generation_rule="oracle-v1",
    )


def _oracle_score_vector(
    *,
    fault: str,
    magnitude: float,
    noise_level: str,
    seed: int,
) -> np.ndarray:
    """Return C1-C5 primary scores for one fresh oracle realization."""
    case = _make_oracle_case(
        fault=fault,
        magnitude=magnitude,
        noise_level=noise_level,
        seed=seed,
    )
    return np.asarray(
        [run_check(case, check).primary_score for check in CHECKS],
        dtype=float,
    )


def _sample_oracle_magnitude(
    rng: np.random.Generator,
    bands: Sequence[str],
) -> float:
    band = bands[int(rng.integers(len(bands)))]
    lo, hi = MAGNITUDE_BANDS[band]
    return float(rng.uniform(lo, hi))


def _quantile_edges(scores: np.ndarray) -> dict[str, np.ndarray]:
    """Pooled 1/3 and 2/3 score quantiles."""
    edges: dict[str, np.ndarray] = {}

    for j, check in enumerate(CHECKS):
        q = np.quantile(
            scores[:, j],
            [1.0 / 3.0, 2.0 / 3.0],
        )
        edges[check] = np.asarray(q, dtype=float)

    return edges


def _generate_oracle_samples(cfg: OracleConfig):
    """Generate the independent MC stream once for both oracle tables."""
    total_draws = len(HYPOTHESES) * len(cfg.noise_levels) * cfg.draws_per_hypothesis_noise
    scores = np.empty((total_draws, len(CHECKS)), dtype=float)
    owners = np.empty(total_draws, dtype=int)
    screen_flags = np.empty((total_draws, len(SCREEN_FLAGS)), dtype=np.int8)
    row = 0

    for h_idx, fault in enumerate(HYPOTHESES):
        for noise_idx, noise in enumerate(cfg.noise_levels):
            for draw_idx in range(cfg.draws_per_hypothesis_noise):
                seed = oracle_seed(cfg.seed, 1, h_idx, noise_idx, draw_idx)
                magnitude_rng = np.random.default_rng(
                    oracle_seed(cfg.seed, 11, h_idx, noise_idx, draw_idx)
                )
                magnitude = _sample_oracle_magnitude(magnitude_rng, cfg.magnitude_bands)
                case = _make_oracle_case(
                    fault=fault, magnitude=magnitude, noise_level=noise, seed=seed
                )
                scores[row] = [
                    run_check(case, check).primary_score for check in CHECKS
                ]
                screen_flags[row] = [
                    int(bool(case.agent_visible.numeric[name]))
                    for name in SCREEN_FLAGS
                ]
                owners[row] = h_idx
                row += 1
    return scores, owners, screen_flags


def _build_likelihood_from_samples(scores, owners, edges, cfg):
    counts = np.zeros(
        (len(HYPOTHESES),) + (cfg.bins,) * len(CHECKS), dtype=float
    )
    for row in range(len(scores)):
        bins = tuple(
            int(np.searchsorted(edges[check], scores[row, j], side="right"))
            for j, check in enumerate(CHECKS)
        )
        counts[(owners[row],) + bins] += 1.0
    counts += cfg.smoothing
    return LikelihoodTable(counts, edges, HYPOTHESES, CHECKS)


def _build_joint_likelihood_from_samples(scores, owners, screen_flags, edges, cfg):
    """Build all 8 * 3^5 = 1,944 observable screen/check cells."""
    counts = np.zeros(
        (len(HYPOTHESES), 2, 2, 2) + (cfg.bins,) * len(CHECKS),
        dtype=float,
    )
    for row in range(len(scores)):
        check_bins = tuple(
            int(np.searchsorted(edges[check], scores[row, j], side="right"))
            for j, check in enumerate(CHECKS)
        )
        screen = tuple(int(v) for v in screen_flags[row])
        counts[(owners[row],) + screen + check_bins] += 1.0
    counts += cfg.smoothing
    return JointLikelihoodTable(counts, edges, HYPOTHESES, CHECKS)


def build_oracle_tables(config: OracleConfig | None = None):
    """Build planner and D3 joint tables from one independent MC stream."""
    cfg = config or OracleConfig()
    scores, owners, screen_flags = _generate_oracle_samples(cfg)
    edges = _quantile_edges(scores)
    return (
        _build_likelihood_from_samples(scores, owners, edges, cfg),
        _build_joint_likelihood_from_samples(
            scores, owners, screen_flags, edges, cfg
        ),
    )


def build_likelihood_table(config: OracleConfig | None = None) -> LikelihoodTable:
    return build_oracle_tables(config)[0]


def build_joint_likelihood_table(
    config: OracleConfig | None = None,
) -> JointLikelihoodTable:
    return build_oracle_tables(config)[1]


def enumerate_subsets() -> tuple[tuple[str, ...], ...]:
    """Enumerate all 32 fixed subsets of the five checks."""
    subsets: list[tuple[str, ...]] = []

    for mask in range(1 << len(CHECKS)):
        subsets.append(
            tuple(
                CHECKS[i]
                for i in range(len(CHECKS))
                if mask & (1 << i)
            )
        )

    return tuple(subsets)


def _subset_accuracy(
    table: LikelihoodTable,
    checks: Sequence[str],
) -> float:
    """Bayes-optimal accuracy under a uniform four-hypothesis prior."""

    checks = tuple(checks)

    if not checks:
        return 1.0 / len(HYPOTHESES)

    observed_axes = {
        1 + table.checks.index(check)
        for check in checks
    }

    reduce_axes = tuple(
        axis
        for axis in range(1, table.table.ndim)
        if axis not in observed_axes
    )

    joint = table.table

    if reduce_axes:
        joint = joint.sum(axis=reduce_axes)

    flat = joint.reshape(len(HYPOTHESES), -1)

    # Uniform prior: sum_obs max_h P(h)P(obs | h).
    return float(
        np.max(flat, axis=0).sum() / len(HYPOTHESES)
    )


def evaluate_subsets(
    table: LikelihoodTable,
    *,
    costs: Mapping[str, float] | None = None,
) -> tuple[SubsetResult, ...]:
    """Evaluate the accuracy and reference cost of all 32 subsets."""

    costs = dict(costs or DEFAULT_COSTS)
    results: list[SubsetResult] = []

    for checks in enumerate_subsets():
        results.append(
            SubsetResult(
                checks=checks,
                accuracy=_subset_accuracy(table, checks),
                cost=float(sum(costs[c] for c in checks)),
            )
        )

    return tuple(
        sorted(
            results,
            key=lambda result: (
                result.cost,
                -result.accuracy,
                result.checks,
            ),
        )
    )


def choose_cheapest_identifiable_subset(
    subset_results: Sequence[SubsetResult],
    *,
    tau_id: float = TAU_ID,
) -> tuple[str, ...] | None:
    """Return the cheapest fixed subset reaching the identifiability target."""

    eligible = [
        result
        for result in subset_results
        if result.accuracy >= tau_id
    ]

    if not eligible:
        return None

    return min(
        eligible,
        key=lambda result: (
            result.cost,
            len(result.checks),
            result.checks,
        ),
    ).checks


def _joint_subset_accuracy(table: JointLikelihoodTable, checks: Sequence[str]) -> float:
    checks = tuple(checks)
    observed_axes = {4 + table.checks.index(check) for check in checks}
    reduce_axes = tuple(
        axis for axis in range(4, table.table.ndim) if axis not in observed_axes
    )
    joint = table.table.sum(axis=reduce_axes) if reduce_axes else table.table
    flat = joint.reshape(len(HYPOTHESES), -1)
    return float(np.max(flat, axis=0).sum() / len(HYPOTHESES))


def evaluate_joint_subsets(
    table: JointLikelihoodTable,
    *,
    costs: Mapping[str, float] | None = None,
) -> tuple[SubsetResult, ...]:
    costs = dict(costs or DEFAULT_COSTS)
    results = [
        SubsetResult(
            checks=checks,
            accuracy=_joint_subset_accuracy(table, checks),
            cost=float(sum(costs[c] for c in checks)),
        )
        for checks in enumerate_subsets()
    ]
    return tuple(sorted(
        results, key=lambda r: (r.cost, -r.accuracy, r.checks)
    ))


def assess_case_identifiability_joint(
    case: Case,
    table: JointLikelihoodTable,
    *,
    checks: Sequence[str] | None = None,
    draws: int = ORACLE_DRAWS,
    seed: int = 0,
    tau_id: float = TAU_ID,
) -> tuple[bool, float]:
    """Fresh-draw identifiability using observed screen flags plus checks."""
    selected = tuple(checks or CHECKS)
    screen = table.screen_bins(case)
    maxima = []
    for draw_idx in range(draws):
        scores = _oracle_score_vector(
            fault=case.hidden.fault,
            magnitude=float(case.hidden.magnitude),
            noise_level=case.hidden.noise_level,
            seed=oracle_seed(seed, 2, draw_idx),
        )
        observed = {
            check: table.bin(check, float(scores[CHECKS.index(check)]))
            for check in selected
        }
        maxima.append(float(np.max(table.posterior(screen, observed))))
    score = float(np.mean(maxima))
    return score >= tau_id, score


def _posterior(
    table: LikelihoodTable,
    observed: Mapping[str, int],
) -> np.ndarray:
    """Uniform-prior posterior for one binned observation."""

    prior = np.full(
        len(HYPOTHESES),
        1.0 / len(HYPOTHESES),
    )

    likelihood = table.lik(observed)
    weights = prior * likelihood
    total = float(weights.sum())

    if total <= 0:
        return prior

    return weights / total


def assess_case_identifiability(
    case: Case,
    table: LikelihoodTable,
    *,
    checks: Sequence[str] | None = None,
    draws: int = ORACLE_DRAWS,
    seed: int = 0,
    tau_id: float = TAU_ID,
) -> tuple[bool, float]:
    """Evaluate a case using fresh oracle realizations.

    The case's true magnitude and noise level are held fixed.  Only measurement
    noise is redrawn, which matches the intended fresh-draw identifiability
    evaluation.
    """

    selected = tuple(checks or CHECKS)

    if not selected:
        return False, 1.0 / len(HYPOTHESES)

    magnitude = float(case.hidden.magnitude)
    fault = case.hidden.fault
    noise = case.hidden.noise_level

    maxima: list[float] = []

    for draw_idx in range(draws):
        draw_seed = oracle_seed(
            seed,
            2,
            draw_idx,
        )

        scores = _oracle_score_vector(
            fault=fault,
            magnitude=magnitude,
            noise_level=noise,
            seed=draw_seed,
        )

        observed = {
            check: table.bin(
                check,
                float(scores[CHECKS.index(check)]),
            )
            for check in selected
        }

        posterior = _posterior(table, observed)
        maxima.append(float(np.max(posterior)))

    score = float(np.mean(maxima))

    return score >= tau_id, score


def annotate_cases(
    cases: Sequence[Case],
    table: LikelihoodTable,
    *,
    config: OracleConfig | None = None,
    checks: Sequence[str] | None = None,
) -> list[Case]:
    """Annotate copies of cases with oracle resolvability and identifiability."""

    cfg = config or OracleConfig()
    selected = tuple(checks or CHECKS)
    output: list[Case] = []

    subset_results = evaluate_subsets(table)
    best_accuracy = max(
        result.accuracy
        for result in subset_results
    )

    globally_identifiable = best_accuracy >= cfg.tau_id

    for index, case in enumerate(cases):
        seed = oracle_seed(
            cfg.seed,
            3,
            index,
        )

        if globally_identifiable:
            resolvable, identifiability = assess_case_identifiability(
                case,
                table,
                checks=selected,
                draws=cfg.case_draws,
                seed=seed,
                tau_id=cfg.tau_id,
            )
        else:
            resolvable = False
            identifiability = best_accuracy

        data = case.model_dump()
        data["hidden"]["resolvable"] = bool(resolvable)
        data["hidden"]["identifiability"] = float(identifiability)

        output.append(Case.model_validate(data))

    return output

def fit_oracle(
    *,
    config: OracleConfig | None = None,
    costs: Mapping[str, float] | None = None,
) -> OracleResult:
    """Fit both the original oracle and the D3 joint screen+check oracle."""
    cfg = config or OracleConfig()
    table, joint_table = build_oracle_tables(cfg)

    subsets = evaluate_subsets(table, costs=costs)
    best_accuracy = max(r.accuracy for r in subsets)
    cheapest = choose_cheapest_identifiable_subset(subsets, tau_id=cfg.tau_id)

    joint_subsets = evaluate_joint_subsets(joint_table, costs=costs)
    joint_best_accuracy = max(r.accuracy for r in joint_subsets)
    joint_cheapest = choose_cheapest_identifiable_subset(
        joint_subsets, tau_id=cfg.tau_id
    )

    return OracleResult(
        table=table,
        subset_results=subsets,
        best_accuracy=best_accuracy,
        cheapest_identifiable=cheapest,
        joint_table=joint_table,
        joint_subset_results=joint_subsets,
        joint_best_accuracy=joint_best_accuracy,
        joint_cheapest_identifiable=joint_cheapest,
    )
