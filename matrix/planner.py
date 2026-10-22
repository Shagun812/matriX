"""Deterministic Bayesian planner (README Section 7, "Planner model").

Likelihoods are a JOINT binned table over the five check scores (no independence assumption),
with unobserved checks marginalized out. Screen evidence lives in the prior, not here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .schemas import (CHECKS, DEFAULT_COSTS, HYPOTHESES, N_BINS, Decision, Status, StopReason, Trace)


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    nz = p[p > 1e-12]
    return float(-(nz * np.log2(nz)).sum())


class LikelihoodTable:
    """table[h, b1, b2, b3, b4, b5] = P(binned scores | h), fitted on the dev-likelihood stream.

    Bin convention (Person B must use the SAME one when building the table):
    two ascending edges per check; bin 0 if score < e0, 1 if e0 <= score < e1, 2 if score >= e1.
    """

    def __init__(self, table, edges: Mapping[str, Sequence[float]],
                 hyps: Sequence[str] = HYPOTHESES, checks: Sequence[str] = CHECKS):
        t = np.asarray(table, dtype=float)
        expected = (len(hyps),) + (N_BINS,) * len(checks)
        if t.shape != expected:
            raise ValueError(f"table shape {t.shape} != {expected}")
        if (t < 0).any():
            raise ValueError("negative probabilities")
        sums = t.reshape(len(hyps), -1).sum(axis=1)
        if (sums <= 0).any():
            raise ValueError("a hypothesis has zero total mass")
        self.hyps, self.checks = tuple(hyps), tuple(checks)
        self.table = t / sums.reshape((-1,) + (1,) * len(checks))
        self.edges = {}
        for c in checks:
            e = np.asarray(edges[c], float)
            if e.shape != (N_BINS - 1,) or e[0] > e[1]:
                raise ValueError(f"bad edges for {c}: {e}")
            self.edges[c] = e

    def bin(self, check: str, score: float) -> int:
        return int(np.searchsorted(self.edges[check], score, side="right"))

    def _idx(self, observed: Mapping[str, int]):
        return (slice(None),) + tuple(observed[c] if c in observed else slice(None) for c in self.checks)

    def lik(self, observed: Mapping[str, int]) -> np.ndarray:
        """P(observed bins | h), unobserved checks marginalized. Shape (H,)."""
        sub = self.table[self._idx(observed)]
        return sub.reshape(len(self.hyps), -1).sum(axis=1)

    def joint_with(self, observed: Mapping[str, int], check: str) -> np.ndarray:
        """P(observed bins, bin of `check` | h). Shape (H, N_BINS)."""
        if check in observed:
            raise ValueError(f"{check} already observed")
        sub = self.table[self._idx(observed)]
        free = [c for c in self.checks if c not in observed]
        pos = free.index(check) + 1
        axes = tuple(a for a in range(1, sub.ndim) if a != pos)
        return sub.sum(axis=axes) if axes else sub

    def save(self, path) -> None:
        np.savez(path, table=self.table, hyps=np.array(self.hyps), checks=np.array(self.checks),
                 training_domain=np.array("synthetic"), synthetic_data_used=np.array(True),
                 **{f"edges_{c}": e for c, e in self.edges.items()})

    @classmethod
    def load(cls, path) -> "LikelihoodTable":
        z = np.load(Path(path), allow_pickle=False)
        if "training_domain" in z and str(z["training_domain"].item()) != "synthetic":
            raise ValueError("synthetic likelihood table has an unexpected training domain")
        if "synthetic_data_used" in z and not bool(z["synthetic_data_used"].item()):
            raise ValueError("synthetic likelihood table is missing its synthetic-data marker")
        checks = [str(c) for c in z["checks"]]
        return cls(z["table"], {c: z[f"edges_{c}"] for c in checks}, [str(h) for h in z["hyps"]], checks)


class Planner:
    def __init__(self, table: LikelihoodTable, costs: Optional[Mapping[str, float]] = None,
                 eps_gain: float = 0.01):
        self.table = table
        self.costs = dict(costs or DEFAULT_COSTS)
        self.eps_gain = eps_gain

    def posterior(self, prior: np.ndarray, observed: Mapping[str, int]) -> np.ndarray:
        w = np.asarray(prior, float) * self.table.lik(observed)
        s = w.sum()
        return w / s if s > 0 else np.asarray(prior, float)

    def expected_gain(self, prior: np.ndarray, observed: Mapping[str, int], check: str) -> float:
        """Expected entropy reduction (bits) from running `check` next."""
        post = self.posterior(prior, observed)
        lik = self.table.lik(observed)
        cond = self.table.joint_with(observed, check) / lik[:, None]  # P(bin | h, observed)
        pb = post @ cond
        gain = entropy(post)
        for b in range(N_BINS):
            if pb[b] > 1e-12:
                gain -= pb[b] * entropy(post * cond[:, b] / pb[b])
        return max(gain, 0.0)

    def gains(self, prior, observed, candidates: Sequence[str]) -> dict[str, float]:
        """Expected information gain PER COST for each candidate."""
        return {c: self.expected_gain(prior, observed, c) / self.costs[c] for c in candidates}

    def pick(self, gains: Mapping[str, float]) -> str:
        """Highest gain/cost; ties -> cheaper -> lower id (deterministic)."""
        return min(gains, key=lambda c: (-round(gains[c], 12), self.costs[c], c))


def top(posterior: Mapping[str, float]) -> tuple[str, float]:
    h = max(HYPOTHESES, key=lambda k: (posterior[k], -HYPOTHESES.index(k)))
    return h, posterior[h]


def decide(trace: Trace, tau_stop: float) -> Decision:
    """Trajectory truncation (README 7/8.2): CONFIRMED at the first step whose top posterior >= tau.

    Step 0 (prior only, cost 0) counts as a decision point. B0 only decides at its final step.
    If nothing reaches tau: BUDGET_EXCEEDED if the budget ended the run, else UNRESOLVED.
    """
    candidates = [trace.steps[-1]] if trace.final_only else trace.steps
    for s in candidates:
        h, p = top(s.posterior)
        if p >= tau_stop:
            return Decision(status=Status.CONFIRMED, hypothesis=h, cost=s.cumulative_cost,
                            n_checks=s.n_checks, step=s.step)
    last = trace.steps[-1]
    status = Status.BUDGET_EXCEEDED if trace.stop_reason == StopReason.BUDGET else Status.UNRESOLVED
    return Decision(status=status, hypothesis=None, cost=last.cumulative_cost,
                    n_checks=last.n_checks, step=last.step)