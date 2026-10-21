"""Dev-time analysis utilities for Person A (tuning, calibration, sanity checks).

Reads hidden labels from DEV cases only, never from inside an agent. Person B's evaluation.py is the
authority for every number that gets reported; cross-check these against it.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

from .planner import decide, top
from .schemas import CHECKS, Case, Decision, Status, Trace

TAU_GRID = [0.6, 0.7, 0.8, 0.9, 0.95]


def _remaining_legal(trace: Trace, d: Decision) -> bool:
    cfg = trace.meta.get("config", {})
    costs, budget = cfg.get("costs", {}), cfg.get("budget")
    run = {s.action for s in trace.steps[: d.step + 1] if s.action}
    for c in CHECKS:
        if c not in run and (budget is None or d.cost + costs.get(c, 0) <= budget + 1e-9):
            return True
    return False


def score(traces: Mapping[str, Trace], cases: Mapping[str, Case], tau: float) -> dict:
    n = n_res = correct = false_dx = premature = tp = fp = fn = 0
    cost = checks = 0.0
    for cid, tr in traces.items():
        h = cases[cid].hidden
        d = decide(tr, tau)
        n += 1; cost += d.cost; checks += d.n_checks
        ok = d.status == Status.CONFIRMED and d.hypothesis == h.fault
        wrong = d.status == Status.CONFIRMED and not ok
        if h.resolvable:
            n_res += 1; correct += ok
        if d.status == Status.CONFIRMED and (d.hypothesis != h.fault or not h.resolvable):
            false_dx += 1
        if wrong and _remaining_legal(tr, d):
            premature += 1
        pred_unres, true_unres = d.status == Status.UNRESOLVED, not h.resolvable
        tp += pred_unres and true_unres; fp += pred_unres and not true_unres; fn += (not pred_unres) and true_unres
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if tp and p + r else (0.0 if tp + fp + fn else float("nan"))
    return {"tau": tau, "n": n, "resolvable_acc": correct / n_res if n_res else float("nan"),
            "false_dx_rate": false_dx / n if n else float("nan"), "premature": premature,
            "abstain_p": p, "abstain_r": r, "abstain_f1": f1,
            "mean_cost": cost / n if n else float("nan"), "mean_checks": checks / n if n else float("nan")}


def curve(traces, cases, taus: Sequence[float] = TAU_GRID) -> list[dict]:
    return [score(traces, cases, t) for t in taus]


def auc(points: Sequence[dict], lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    """Trapezoid area of resolvable accuracy over mean cost; no extrapolation beyond observed range."""
    pts = sorted((p["mean_cost"], p["resolvable_acc"]) for p in points if not math.isnan(p["resolvable_acc"]))
    if len(pts) < 2:
        return float("nan")
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    lo = xs.min() if lo is None else max(lo, xs.min())
    hi = xs.max() if hi is None else min(hi, xs.max())
    if hi <= lo:
        return float("nan")
    grid = np.unique(np.concatenate([[lo, hi], xs[(xs > lo) & (xs < hi)]]))
    vals = np.interp(grid, xs, ys)
    return float(np.sum((grid[1:] - grid[:-1]) * (vals[1:] + vals[:-1]) / 2))


def choose_operating_point(points: Sequence[dict], target: float = 0.9) -> tuple[float, bool]:
    """Cheapest tau reaching >= target resolvable accuracy (README 8.2). Returns (tau, target_met).
    If none reaches it, returns the most accurate tau and False: say so in the write-up."""
    ok = [p for p in points if p["resolvable_acc"] >= target]
    if ok:
        return min(ok, key=lambda p: (p["mean_cost"], p["tau"]))["tau"], True
    return max(points, key=lambda p: (p["resolvable_acc"], -p["mean_cost"]))["tau"], False


def reliability(traces, cases, n_bins: int = 10, all_steps: bool = False) -> dict:
    """Reliability of the top posterior (README 7 'Known approximation'). Uses each trajectory's final
    step (or every step). overconfident=True when mean confidence exceeds accuracy by > 0.10."""
    conf, hit = [], []
    for cid, tr in traces.items():
        steps = tr.steps if all_steps else tr.steps[-1:]
        for s in steps:
            h, p = top(s.posterior)
            conf.append(p); hit.append(h == cases[cid].hidden.fault)
    conf, hit = np.array(conf), np.array(hit, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bins.append({"conf": float(conf[m].mean()), "acc": float(hit[m].mean()), "n": int(m.sum())})
            ece += m.mean() * abs(conf[m].mean() - hit[m].mean())
    gap = float(conf.mean() - hit.mean()) if len(conf) else float("nan")
    return {"bins": bins, "ece": float(ece), "gap": gap, "overconfident": bool(gap > 0.10), "n": int(len(conf))}


def plot_reliability(results: Mapping[str, dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, r in results.items():
        ax.plot([b["conf"] for b in r["bins"]], [b["acc"] for b in r["bins"]], "o-",
                label=f"{name} (ECE {r['ece']:.2f})")
    ax.set_xlabel("top posterior"); ax.set_ylabel("fraction correct"); ax.legend(); ax.set_title("Dev reliability")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)