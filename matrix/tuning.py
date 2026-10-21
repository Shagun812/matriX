"""Choose eps_gain and the operating tau ON DEV, then freeze (README Section 7/8.2).

Rule (fixed here, before looking at test): operating tau = cheapest tau reaching >= `target` resolvable
accuracy for B3 with no early stopping; eps_gain = LARGEST value on the grid whose dev accuracy at that
tau is within `tol` of the eps_gain=0 run (stops earlier without hurting accuracy).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from .agents import AgentConfig, B3GreedyIG, B3bLearnedPrior
from .analysis import TAU_GRID, choose_operating_point, curve, score
from .planner import LikelihoodTable, Planner
from .priors import PriorModel
from .schemas import Case


def _run(table, cfg, cases, runner_for, prior_model):
    agents = {"B3": B3GreedyIG(Planner(table, cfg.costs, cfg.eps_gain), cfg)}
    if prior_model is not None:
        agents["B3b"] = B3bLearnedPrior(Planner(table, cfg.costs, cfg.eps_gain), prior_model, cfg)
    return {name: {c.case_id: ag.run(c.case_id, c.agent_visible, runner_for(c)) for c in cases}
            for name, ag in agents.items()}


def tune(table: LikelihoodTable, cases: Sequence[Case], runner_for: Callable, prior_model: Optional[PriorModel] = None,
         base: Optional[AgentConfig] = None, grid: Sequence[float] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1),
         target: float = 0.9, tol: float = 0.01, taus: Sequence[float] = TAU_GRID) -> dict:
    base = base or AgentConfig()
    by_id = {c.case_id: c for c in cases}
    ref_cfg = AgentConfig(**{**base.__dict__, "eps_gain": 0.0})
    ref = _run(table, ref_cfg, cases, runner_for, prior_model)
    op_tau, met = choose_operating_point(curve(ref["B3"], by_id, taus), target)
    rows, ref_acc = [], {k: score(v, by_id, op_tau)["resolvable_acc"] for k, v in ref.items()}
    for eps in grid:
        cfg = AgentConfig(**{**base.__dict__, "eps_gain": eps})
        res = _run(table, cfg, cases, runner_for, prior_model)
        sc = {k: score(v, by_id, op_tau) for k, v in res.items()}
        ok = all(sc[k]["resolvable_acc"] >= ref_acc[k] - tol for k in sc)
        rows.append({"eps_gain": eps, "ok": ok, **{f"{k}_acc": sc[k]["resolvable_acc"] for k in sc},
                     **{f"{k}_cost": sc[k]["mean_cost"] for k in sc}})
    chosen = max((r["eps_gain"] for r in rows if r["ok"]), default=0.0)
    return {"operating_tau": op_tau, "target_met": met, "eps_gain": chosen, "rows": rows, "ref_acc": ref_acc}