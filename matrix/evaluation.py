"""Fair evaluation utilities for the MatriX benchmark.

This module consumes frozen cases/traces and reports the metrics described in
README.md.  It deliberately does not generate the oracle likelihood table:
that responsibility stays in matrix.oracle.

The core evaluation path is deterministic given:
  * a case JSON file,
  * one trace JSON per case/arm,
  * a frozen Planner/LikelihoodTable,
  * a frozen tau_stop grid.

The CLI is intentionally small; richer experiment orchestration can be layered
on top without changing the metric functions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .planner import LikelihoodTable, Planner, decide
from .schemas import (
    CHECKS,
    DEFAULT_COSTS,
    HYPOTHESES,
    Case,
    Decision,
    Status,
    StopReason,
    Trace,
)


DEFAULT_TAU_GRID = (0.60, 0.70, 0.80, 0.90, 0.95)
DEFAULT_BOOTSTRAP = 5000
DEFAULT_PERMUTATIONS = 10_000


@dataclass(frozen=True)
class CaseMetric:
    case_id: str
    arm: str
    tau_stop: float
    status: str
    hypothesis: str | None
    correct: bool
    eligible_resolvable: bool
    unresolved_positive: bool
    false_diagnosis: bool
    premature_commitment: bool
    cost: float
    n_checks: int
    stop_reason: str
    complete_trace: bool


@dataclass(frozen=True)
class Summary:
    arm: str
    tau_stop: float
    n_cases: int
    n_resolvable: int
    resolvable_accuracy: float
    abstention_precision: float
    abstention_recall: float
    abstention_f1: float
    false_diagnosis_rate: float
    premature_commitment_rate: float
    mean_cost: float
    mean_checks: float
    budget_exhaustion_rate: float
    trace_completeness: float


@dataclass(frozen=True)
class CurvePoint:
    arm: str
    tau_stop: float
    accuracy: float
    mean_cost: float


@dataclass(frozen=True)
class AUCResult:
    arm_a: str
    arm_b: str
    auc_a: float | None
    auc_b: float | None
    delta_auc: float | None
    common_cost_lo: float | None
    common_cost_hi: float | None
    degenerate: bool


@dataclass(frozen=True)
class ComparisonResult:
    arm_a: str
    arm_b: str
    delta_auc: float | None
    bootstrap_ci_low: float | None
    bootstrap_ci_high: float | None
    effect_size: float | None
    p_raw: float | None
    p_holm: float | None


def load_cases(path: str | Path) -> list[Case]:
    return [
        Case.model_validate(obj)
        for obj in json.loads(Path(path).read_text())
    ]


def load_traces(path: str | Path, arm: str) -> dict[str, Trace]:
    directory = Path(path) / arm
    if not directory.exists():
        return {}
    return {
        p.stem: Trace.model_validate_json(p.read_text())
        for p in sorted(directory.glob("*.json"))
    }


def save_json(data: object, path: str | Path) -> None:
    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"not JSON serializable: {type(obj).__name__}")

    Path(path).write_text(json.dumps(data, indent=2, default=default))


def _decision(
    trace: Trace,
    tau_stop: float,
    planner: Planner,
) -> Decision:
    """Apply a frozen tau_stop to an already-recorded trajectory."""
    return decide(trace, tau_stop)


def _case_fault(case: Case) -> str | None:
    return getattr(case.hidden, "fault", None)


def _case_resolvable(case: Case) -> bool:
    return bool(getattr(case.hidden, "resolvable", False))


def _budget_from_trace(
    trace: Trace,
    costs: Mapping[str, float],
) -> float | None:
    """Recover an explicit budget only when traces recorded one in meta."""
    value = trace.meta.get("budget") if trace.meta else None
    return None if value is None else float(value)


def _legal_remaining(
    trace: Trace,
    costs: Mapping[str, float],
    budget: float | None,
    at_step: int,
) -> list[str]:
    """Checks that were still legal immediately after a decision step."""
    used = {
        step.action
        for step in trace.steps[: at_step + 1]
        if step.action in costs
    }
    cumulative = float(trace.steps[at_step].cumulative_cost)
    remaining = [c for c in CHECKS if c not in used]

    if budget is None:
        return remaining

    return [
        c for c in remaining
        if cumulative + float(costs[c]) <= budget + 1e-12
    ]


def _is_premature(
    trace: Trace,
    decision: Decision,
    costs: Mapping[str, float],
    budget: float | None,
) -> bool:
    if decision.status != Status.CONFIRMED or decision.hypothesis is None:
        return False
    if decision.step is None:
        return False

    if budget is None:
        # If the trace ended before all checks, another check could have been
        # run.  For an exhausted trajectory there is no premature commitment.
        return len({
            s.action for s in trace.steps if s.action in costs
        }) < len(CHECKS)

    return bool(_legal_remaining(trace, costs, budget, decision.step))


def case_metric(
    case: Case,
    trace: Trace,
    tau_stop: float,
    planner: Planner,
    costs: Mapping[str, float] | None = None,
    budget: float | None = None,
) -> CaseMetric:
    costs = dict(costs or DEFAULT_COSTS)
    decision = _decision(trace, tau_stop, planner)

    fault = _case_fault(case)
    correct = (
        decision.status == Status.CONFIRMED
        and decision.hypothesis is not None
        and fault is not None
        and decision.hypothesis == fault
    )

    unresolved = decision.status == Status.UNRESOLVED
    false_diagnosis = (
        decision.status == Status.CONFIRMED
        and not correct
    )

    # A complete trace is structurally replayable: step indices are contiguous
    # from zero and every recorded action is a legal check name.  Stopping early
    # is allowed and does not make a trace incomplete.
    complete = (
        bool(trace.steps)
        and trace.steps[0].step == 0
        and trace.steps[0].action is None
        and trace.steps[0].result is None
        and trace.steps[0].n_checks == 0
        and all(
            step.step == i
            and step.n_checks == i
            and step.action in CHECKS
            and step.result is not None
            and step.result.check_id == step.action
            for i, step in enumerate(trace.steps[1:], start=1)
        )
        and len({
            step.action for step in trace.steps[1:]
        }) == len(trace.steps) - 1
    )
    premature = _is_premature(trace, decision, costs, budget)

    step_index = decision.step
    if step_index is None:
        step_index = len(trace.steps) - 1

    step = trace.steps[step_index]
    return CaseMetric(
        case_id=case.case_id,
        arm=trace.arm,
        tau_stop=float(tau_stop),
        status=decision.status.value,
        hypothesis=decision.hypothesis,
        correct=bool(correct),
        eligible_resolvable=_case_resolvable(case),
        unresolved_positive=unresolved,
        false_diagnosis=bool(false_diagnosis),
        premature_commitment=bool(premature),
        cost=float(step.cumulative_cost),
        n_checks=int(step.n_checks),
        stop_reason=trace.stop_reason.value,
        complete_trace=complete,
    )


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def summarize(metrics: Sequence[CaseMetric], arm: str, tau_stop: float) -> Summary:
    metrics = list(metrics)
    resolvable = [m for m in metrics if m.eligible_resolvable]

    # Positive class for abstention precision/recall/F1 is an unresolved case.
    tp = sum(m.unresolved_positive and m.eligible_resolvable for m in metrics)
    fp = sum(m.unresolved_positive and not m.eligible_resolvable for m in metrics)
    fn = sum((not m.unresolved_positive) and m.eligible_resolvable for m in metrics)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (
        _safe_div(2 * precision * recall, precision + recall)
        if np.isfinite(precision) and np.isfinite(recall)
        else float("nan")
    )

    return Summary(
        arm=arm,
        tau_stop=float(tau_stop),
        n_cases=len(metrics),
        n_resolvable=len(resolvable),
        resolvable_accuracy=_safe_div(
            sum(m.correct for m in resolvable), len(resolvable)
        ),
        abstention_precision=precision,
        abstention_recall=recall,
        abstention_f1=f1,
        false_diagnosis_rate=_safe_div(
            sum(m.false_diagnosis for m in metrics), len(metrics)
        ),
        premature_commitment_rate=_safe_div(
            sum(m.premature_commitment for m in metrics), len(metrics)
        ),
        mean_cost=float(np.mean([m.cost for m in metrics])) if metrics else float("nan"),
        mean_checks=float(np.mean([m.n_checks for m in metrics])) if metrics else float("nan"),
        budget_exhaustion_rate=_safe_div(
            sum(m.status == Status.BUDGET_EXCEEDED.value for m in metrics),
            len(metrics),
        ),
        trace_completeness=_safe_div(
            sum(m.complete_trace for m in metrics), len(metrics)
        ),
    )


def evaluate_arm(
    cases: Sequence[Case],
    traces: Mapping[str, Trace],
    tau_stop: float,
    planner: Planner,
    *,
    costs: Mapping[str, float] | None = None,
    budget: float | None = None,
) -> tuple[list[CaseMetric], Summary]:
    metrics: list[CaseMetric] = []
    for case in cases:
        trace = traces.get(case.case_id)
        if trace is None:
            raise ValueError(
                f"missing trace for case {case.case_id!r} in evaluation input"
            )
        metrics.append(
            case_metric(
                case,
                trace,
                tau_stop,
                planner,
                costs=costs,
                budget=budget,
            )
        )
    return metrics, summarize(metrics, traces[next(iter(traces))].arm if traces else "", tau_stop)


def accuracy_cost_curve(
    cases: Sequence[Case],
    traces: Mapping[str, Trace],
    planner: Planner,
    *,
    tau_grid: Sequence[float] = DEFAULT_TAU_GRID,
    costs: Mapping[str, float] | None = None,
    budget: float | None = None,
) -> tuple[list[CurvePoint], dict[float, list[CaseMetric]]]:
    points: list[CurvePoint] = []
    by_tau: dict[float, list[CaseMetric]] = {}

    arm = next(iter(traces.values())).arm if traces else ""
    for tau in tau_grid:
        metrics, summary = evaluate_arm(
            cases, traces, tau, planner, costs=costs, budget=budget
        )
        by_tau[float(tau)] = metrics
        points.append(
            CurvePoint(
                arm=arm,
                tau_stop=float(tau),
                accuracy=summary.resolvable_accuracy,
                mean_cost=summary.mean_cost,
            )
        )
    return points, by_tau


def _interp_area(
    x: np.ndarray,
    y: np.ndarray,
    lo: float,
    hi: float,
) -> float:
    """Integrate a piecewise-linear curve only inside [lo, hi]."""
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]

    keep = (x >= lo) & (x <= hi)
    xx = x[keep]
    yy = y[keep]

    if xx.size == 0:
        xx = np.array([lo, hi], dtype=float)
        yy = np.array([
            np.interp(lo, x, y),
            np.interp(hi, x, y),
        ])
    else:
        if xx[0] > lo:
            xx = np.insert(xx, 0, lo)
            yy = np.insert(yy, 0, np.interp(lo, x, y))
        if xx[-1] < hi:
            xx = np.append(xx, hi)
            yy = np.append(yy, np.interp(hi, x, y))

    if xx.size < 2 or hi <= lo:
        return float("nan")
    return float(np.trapezoid(yy, xx))


def curve_auc(
    curve_a: Sequence[CurvePoint],
    curve_b: Sequence[CurvePoint],
) -> AUCResult:
    """AUC over the common observed mean-cost interval; never extrapolate."""
    if not curve_a or not curve_b:
        raise ValueError("both curves are required")

    xa = np.array([p.mean_cost for p in curve_a], dtype=float)
    ya = np.array([p.accuracy for p in curve_a], dtype=float)
    xb = np.array([p.mean_cost for p in curve_b], dtype=float)
    yb = np.array([p.accuracy for p in curve_b], dtype=float)

    lo = max(float(np.min(xa)), float(np.min(xb)))
    hi = min(float(np.max(xa)), float(np.max(xb)))

    arm_a = curve_a[0].arm
    arm_b = curve_b[0].arm

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return AUCResult(
            arm_a, arm_b, None, None, None, lo, hi, True
        )

    auc_a = _interp_area(xa, ya, lo, hi)
    auc_b = _interp_area(xb, yb, lo, hi)

    return AUCResult(
        arm_a=arm_a,
        arm_b=arm_b,
        auc_a=auc_a,
        auc_b=auc_b,
        delta_auc=float(auc_a - auc_b),
        common_cost_lo=lo,
        common_cost_hi=hi,
        degenerate=False,
    )


def paired_auc_bootstrap(
    case_metrics_a: Mapping[float, Sequence[CaseMetric]],
    case_metrics_b: Mapping[float, Sequence[CaseMetric]],
    *,
    arm_a: str,
    arm_b: str,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for AUC difference.

    Cases are resampled jointly. For each resample the full tau curve is
    recomputed from per-case correctness/cost observations.
    """
    taus = sorted(set(case_metrics_a) & set(case_metrics_b))
    if not taus:
        raise ValueError("no common tau values")

    by_a = {
        tau: {m.case_id: m for m in case_metrics_a[tau]}
        for tau in taus
    }
    by_b = {
        tau: {m.case_id: m for m in case_metrics_b[tau]}
        for tau in taus
    }
    ids = sorted(set.intersection(*(set(by_a[t]) for t in taus),
                                  *(set(by_b[t]) for t in taus)))
    if len(ids) < 2:
        raise ValueError("paired bootstrap needs at least two common cases")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)

        ca: list[CurvePoint] = []
        cb: list[CurvePoint] = []
        for tau in taus:
            ma = [by_a[tau][cid] for cid in sampled]
            mb = [by_b[tau][cid] for cid in sampled]
            sa = summarize(ma, arm_a, tau)
            sb = summarize(mb, arm_b, tau)
            ca.append(CurvePoint(arm_a, tau, sa.resolvable_accuracy, sa.mean_cost))
            cb.append(CurvePoint(arm_b, tau, sb.resolvable_accuracy, sb.mean_cost))

        result = curve_auc(ca, cb)
        diffs[b] = np.nan if result.delta_auc is None else result.delta_auc

    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan"), float("nan"), float("nan")

    return (
        float(np.median(diffs)),
        float(np.quantile(diffs, 0.025)),
        float(np.quantile(diffs, 0.975)),
    )


def paired_permutation_pvalue(
    case_metrics_a: Mapping[float, Sequence[CaseMetric]],
    case_metrics_b: Mapping[float, Sequence[CaseMetric]],
    *,
    n_perm: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> float:
    """Two-sided paired permutation test for AUC difference.

    The same swap is applied to both arms at every tau for each case.
    """
    taus = sorted(set(case_metrics_a) & set(case_metrics_b))
    if not taus:
        raise ValueError("no common tau values")

    by_a = {tau: {m.case_id: m for m in case_metrics_a[tau]} for tau in taus}
    by_b = {tau: {m.case_id: m for m in case_metrics_b[tau]} for tau in taus}
    ids = sorted(set.intersection(*(set(by_a[t]) for t in taus),
                                  *(set(by_b[t]) for t in taus)))
    if not ids:
        raise ValueError("no paired cases")

    def observed_auc(swaps: np.ndarray | None) -> float:
        ca: list[CurvePoint] = []
        cb: list[CurvePoint] = []
        for tau in taus:
            ma: list[CaseMetric] = []
            mb: list[CaseMetric] = []
            for i, cid in enumerate(ids):
                swap = bool(swaps[i]) if swaps is not None else False
                if not swap:
                    ma.append(by_a[tau][cid])
                    mb.append(by_b[tau][cid])
                else:
                    ma.append(by_b[tau][cid])
                    mb.append(by_a[tau][cid])
            sa = summarize(ma, "A", tau)
            sb = summarize(mb, "B", tau)
            ca.append(CurvePoint("A", tau, sa.resolvable_accuracy, sa.mean_cost))
            cb.append(CurvePoint("B", tau, sb.resolvable_accuracy, sb.mean_cost))
        result = curve_auc(ca, cb)
        return float(result.delta_auc) if result.delta_auc is not None else float("nan")

    observed = observed_auc(None)
    if not np.isfinite(observed):
        return float("nan")

    rng = np.random.default_rng(seed)
    extreme = 0
    valid = 0
    for _ in range(n_perm):
        swaps = rng.integers(0, 2, size=len(ids), dtype=np.int8)
        value = observed_auc(swaps)
        if not np.isfinite(value):
            continue
        valid += 1
        if abs(value) >= abs(observed) - 1e-15:
            extreme += 1

    if not valid:
        return float("nan")
    return float((extreme + 1) / (valid + 1))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjustment for a named family of p-values."""
    items = sorted(
        ((name, float(p)) for name, p in p_values.items() if np.isfinite(p)),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(items)

    for rank, (name, p) in enumerate(items):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[name] = running

    for name in p_values:
        if name not in adjusted:
            adjusted[name] = float("nan")
    return adjusted


def paired_effect_size(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> float:
    """Standardized mean paired difference."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    if d.size < 2:
        return float("nan")
    sd = float(np.std(d, ddof=1))
    return float(np.mean(d) / sd) if sd > 0 else float("inf")


def mcnemar_exact(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
) -> float:
    """Exact two-sided McNemar p-value for paired binary correctness."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired inputs must have equal length")

    discordant_a = int(np.sum(a & ~b))
    discordant_b = int(np.sum(~a & b))
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0

    # Exact binomial tail, implemented without requiring scipy.
    k = min(discordant_a, discordant_b)
    tail = sum(
        math.comb(n, i) * (0.5 ** n)
        for i in range(k + 1)
    )
    return float(min(1.0, 2.0 * tail))


def operating_point(
    dev_summaries: Sequence[Summary],
    *,
    target_accuracy: float = 0.90,
) -> float:
    """Choose the cheapest tau reaching the requested dev accuracy."""
    eligible = [
        s for s in dev_summaries
        if np.isfinite(s.resolvable_accuracy)
        and s.resolvable_accuracy >= target_accuracy
    ]
    if not dev_summaries:
        raise ValueError("dev_summaries must not be empty")
    if not eligible:
        # Frozen fallback: highest tau in the supplied grid.
        return float(max(s.tau_stop for s in dev_summaries))
    return float(min(eligible, key=lambda s: (s.mean_cost, s.tau_stop)).tau_stop)


def plot_accuracy_cost(
    curves: Mapping[str, Sequence[CurvePoint]],
    path: str | Path,
) -> None:
    """Save an accuracy-vs-cost figure without imposing plot styling."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for arm, curve in curves.items():
        x = [p.mean_cost for p in curve]
        y = [p.accuracy for p in curve]
        ax.plot(x, y, marker="o", label=arm)
    ax.set_xlabel("Mean normalized cost")
    ax.set_ylabel("Resolvable accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MatriX traces.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--tau", nargs="*", type=float, default=list(DEFAULT_TAU_GRID))
    parser.add_argument("--out", default="artifacts/evaluation.json")
    args = parser.parse_args()

    cases = load_cases(args.cases)

    # A likelihood table is required for tau-truncation decisions. This CLI
    # expects a serialized planner table to be supplied by the caller in code;
    # it intentionally refuses to invent one from evaluation data.
    raise SystemExit(
        "CLI orchestration is intentionally not wired to construct a frozen "
        "LikelihoodTable. Use evaluate_arm()/accuracy_cost_curve() with the "
        "frozen planner from the experiment configuration."
    )


if __name__ == "__main__":
    _cli()
