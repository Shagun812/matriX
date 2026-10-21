"""Planner-side kill-gate checks (README 6.4). Person B owns the generator-side checks and the final
gate; these cover the items that depend on the planner and the likelihood table."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .agents import AgentConfig, B2CheapestFirst, B3GreedyIG
from .analysis import score
from .planner import LikelihoodTable, Planner
from .schemas import CHECKS, HYPOTHESES, Case


def c5_separation(table: LikelihoodTable, min_tv: float = 0.2) -> dict:
    """C5's marginal over bins must separate {H3,H4} (artefacts) from {H1,H2} (intrinsic)."""
    ax = tuple(range(1, 5))  # sum out C1..C4 -> (H, bins of C5)
    m = table.table.sum(axis=ax)
    art, intr = m[[2, 3]].mean(0), m[[0, 1]].mean(0)
    tv = float(0.5 * np.abs(art - intr).sum())
    return {"tv_artefact_vs_intrinsic": tv, "pass": tv >= min_tv}


def selection_coverage(table, cases: Sequence[Case], runner_for: Callable, cfg: AgentConfig | None = None) -> dict:
    """Every check must be picked by the planner on at least some dev cases (else redundant/flat)."""
    cfg = cfg or AgentConfig(eps_gain=0.0)
    ag = B3GreedyIG(Planner(table, cfg.costs, cfg.eps_gain), cfg)
    picked = {c: 0 for c in CHECKS}
    for case in cases:
        tr = ag.run(case.case_id, case.agent_visible, runner_for(case))
        for s in tr.steps:
            if s.action:
                picked[s.action] += 1
    return {"picked": picked, "pass": all(v > 0 for v in picked.values())}


def cost_changes_policy(table, cases, runner_for, cheap_c5: float = 0.5, frac_min: float = 0.05) -> dict:
    """Costs should change the preferred check sequence on some cases (C5 at cost 10 vs 0.5)."""
    base = AgentConfig(eps_gain=0.0)
    alt = AgentConfig(eps_gain=0.0, costs={**base.costs, "C5": cheap_c5})
    a, b = (B3GreedyIG(Planner(table, c.costs, 0.0), c) for c in (base, alt))
    diff = 0
    for case in cases:
        sa = [s.action for s in a.run(case.case_id, case.agent_visible, runner_for(case)).steps]
        sb = [s.action for s in b.run(case.case_id, case.agent_visible, runner_for(case)).steps]
        diff += sa != sb
    f = diff / max(len(cases), 1)
    return {"frac_changed": f, "pass": f >= frac_min}


def baselines_not_trivial(table, cases, runner_for, tau: float = 0.9, max_acc: float = 0.99) -> dict:
    """Cheapest-first must not solve every resolvable case (benchmark too easy otherwise)."""
    cfg = AgentConfig(eps_gain=0.0)
    ag = B2CheapestFirst(Planner(table, cfg.costs, 0.0), cfg)
    traces = {c.case_id: ag.run(c.case_id, c.agent_visible, runner_for(c)) for c in cases}
    acc = score(traces, {c.case_id: c for c in cases}, tau)["resolvable_acc"]
    return {"b2_resolvable_acc": acc, "pass": not (acc >= max_acc)}


def run_all(table, cases, runner_for) -> dict:
    r = {"c5_separation": c5_separation(table), "selection_coverage": selection_coverage(table, cases, runner_for),
         "cost_changes_policy": cost_changes_policy(table, cases, runner_for),
         "baselines_not_trivial": baselines_not_trivial(table, cases, runner_for)}
    r["all_pass"] = all(v["pass"] for v in r.values())
    return r