"""Arms B0-B5 (README Section 7). One interface: agent.run(case_id, visible, run_check) -> Trace.

`run_check(check_id) -> CheckResult` is supplied by checks.py bound to one case, so
agents can never touch hidden fields. Cost accounting uses AgentConfig.costs (so C5-cost
sensitivity needs no re-simulation).
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

from .features import summarize_evidence
from .llm import CachedLLM, parse_distribution
from .planner import Planner
from .priors import PriorModel
from .prompts import B4_SYSTEM, PRIOR_SYSTEM, b4_user, prior_user
from .schemas import (CHECKS, DEFAULT_COSTS, HYPOTHESES, AgentVisible, CheckResult, Step, StopReason, Trace)

RunCheck = Callable[[str], CheckResult]


@dataclass
class AgentConfig:
    costs: dict = field(default_factory=lambda: dict(DEFAULT_COSTS))
    budget: Optional[float] = None   # None = unlimited. README 8.4 sensitivity: 5 / full.
    eps_gain: float = 0.01           # stop when every legal check's gain/cost is below this (freeze on dev)
    prior_eps: float = 0.05          # uniform mixture on LLM (and B3b) priors
    k_samples: int = 3               # LLM prior samples to average
    seed: int = 0


def _d(p: np.ndarray) -> dict[str, float]:
    return {h: float(x) for h, x in zip(HYPOTHESES, p)}


def _mix(p: np.ndarray, eps: float) -> np.ndarray:
    return (1 - eps) * p + eps / len(HYPOTHESES)


def _rng(case_id: str, arm: str, seed: int) -> np.random.Generator:
    h = hashlib.sha256(f"{case_id}|{arm}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


class Agent(ABC):
    name: str

    def __init__(self, cfg: Optional[AgentConfig] = None):
        self.cfg = cfg or AgentConfig()

    def _legal(self, done, spent, ignore_budget=False) -> tuple[list[str], list[str]]:
        remaining = [c for c in CHECKS if c not in done]
        if ignore_budget or self.cfg.budget is None:
            return remaining, remaining
        return remaining, [c for c in remaining if spent + self.cfg.costs[c] <= self.cfg.budget + 1e-9]

    @abstractmethod
    def run(self, case_id: str, visible: AgentVisible, run_check: RunCheck) -> Trace: ...


class PlannerAgent(Agent):
    """Shared loop for B0-B3b and B5. Subclasses differ in prior and check selection only."""

    ignore_budget = False
    use_gain_stop = True
    final_only = False

    def __init__(self, planner: Planner, cfg: Optional[AgentConfig] = None):
        super().__init__(cfg)
        self.planner = planner
        self.planner.costs = dict(self.cfg.costs)
        self.planner.eps_gain = self.cfg.eps_gain

    def make_prior(self, case_id: str, visible: AgentVisible) -> tuple[np.ndarray, dict]:
        return np.full(len(HYPOTHESES), 1 / len(HYPOTHESES)), {}

    def choose(self, legal: list[str], gains: dict[str, float], rng) -> str:
        return self.planner.pick(gains)

    def run(self, case_id, visible, run_check) -> Trace:
        pl, cfg = self.planner, self.cfg
        prior, meta = self.make_prior(case_id, visible)
        rng = _rng(case_id, self.name, cfg.seed)
        observed: dict[str, int] = {}
        spent = 0.0
        steps = [Step(step=0, posterior=_d(pl.posterior(prior, observed)), cumulative_cost=0.0, n_checks=0)]
        while True:
            remaining, legal = self._legal(observed, spent, self.ignore_budget)
            if not remaining:
                stop = StopReason.ALL_USED; break
            if not legal:
                stop = StopReason.BUDGET; break
            gains = pl.gains(prior, observed, legal)
            if self.use_gain_stop and max(gains.values()) < cfg.eps_gain:
                stop = StopReason.LOW_GAIN; break
            c = self.choose(legal, gains, rng)
            res = run_check(c)
            observed[c] = pl.table.bin(c, res.primary_score)
            spent += cfg.costs[c]
            steps.append(Step(step=len(steps), action=c, result=res, observed_bin=observed[c],
                              posterior=_d(pl.posterior(prior, observed)), cumulative_cost=spent,
                              n_checks=len(observed), candidate_gain_per_cost=gains))
        return Trace(case_id=case_id, arm=self.name, prior=_d(prior), steps=steps, stop_reason=stop,
                     final_only=self.final_only, meta={**meta, "config": asdict(cfg)})


class B0FullCheck(PlannerAgent):
    """Run every check (cheapest first, budget ignored), MAP under a uniform prior."""
    name = "B0"
    ignore_budget = True
    use_gain_stop = False
    final_only = True

    def choose(self, legal, gains, rng):
        return min(legal, key=lambda c: (self.cfg.costs[c], c))


class B1Random(PlannerAgent):
    name = "B1"

    def choose(self, legal, gains, rng):
        return str(rng.choice(sorted(legal)))


class B2CheapestFirst(PlannerAgent):
    name = "B2"

    def choose(self, legal, gains, rng):
        return min(legal, key=lambda c: (self.cfg.costs[c], c))


class B3GreedyIG(PlannerAgent):
    """Greedy information gain per cost, uniform prior."""
    name = "B3"


class B3bLearnedPrior(PlannerAgent):
    """Same planner as B3/B5; prior from logistic regression on the shared feature encoder."""
    name = "B3b"

    def __init__(self, planner: Planner, prior_model: PriorModel, cfg: Optional[AgentConfig] = None):
        super().__init__(planner, cfg)
        self.prior_model = prior_model

    def make_prior(self, case_id, visible):
        raw = self.prior_model.predict(visible)
        return _mix(raw, self.cfg.prior_eps), {"raw_prior": _d(raw)}


class B5Hybrid(PlannerAgent):
    """LLM prior (k samples averaged, eps-mixed) + the same deterministic planner."""
    name = "B5"

    def __init__(self, planner: Planner, llm: CachedLLM, cfg: Optional[AgentConfig] = None):
        super().__init__(planner, cfg)
        self.llm = llm

    def make_prior(self, case_id, visible):
        user = prior_user(summarize_evidence(visible))
        samples, metas = [], []
        for i in range(self.cfg.k_samples):
            p, m = self.llm.get_json(PRIOR_SYSTEM, user, lambda d: parse_distribution(d["distribution"]), sample_idx=i)
            metas.append(m)
            if p is not None:
                samples.append(p)
        fallback = len(samples) == 0
        if fallback:  # documented fallback: neutral prior
            avg = np.full(len(HYPOTHESES), 1 / len(HYPOTHESES))
        else:
            avg = np.mean(samples, axis=0)
        disp = np.std(samples, axis=0).tolist() if len(samples) > 1 else None
        meta = {"llm_prior": {"samples": [_d(s) for s in samples], "valid": len(samples), "requested": self.cfg.k_samples,
                              "dispersion_std": disp, "fallback_neutral": fallback, "call_meta": metas}}
        return _mix(avg, self.cfg.prior_eps), meta


class B4LLMOnly(Agent):
    """LLM picks checks and self-reports a distribution after every step. No likelihood tables.

    Fallbacks (logged): invalid output after one retry -> keep previous distribution (uniform at
    step 0) and run the cheapest legal check so the trajectory continues.
    """
    name = "B4"

    def __init__(self, llm: CachedLLM, cfg: Optional[AgentConfig] = None):
        super().__init__(cfg)
        self.llm = llm

    def _ask(self, evidence, history, legal, spent, prev):
        cfg = self.cfg
        left = None if cfg.budget is None else cfg.budget - spent
        user = b4_user(evidence, cfg.costs, history, legal, left)

        def validate(d):
            nxt = str(d["next_check"]).upper()
            if nxt != "STOP" and nxt not in CHECKS:
                raise ValueError(f"bad next_check {nxt}")
            return parse_distribution(d["distribution"]), nxt

        out, meta = self.llm.get_json(B4_SYSTEM, user, validate, sample_idx=len(history))
        if out is None:
            nxt = min(legal, key=lambda c: (cfg.costs[c], c)) if legal else "STOP"
            return prev, nxt, {"fallback": True, **meta}
        dist, nxt = out
        if nxt != "STOP" and nxt not in legal:  # illegal choice (already run / over budget)
            nxt = min(legal, key=lambda c: (cfg.costs[c], c)) if legal else "STOP"
            return dist, nxt, {"fallback": True, "reason": "illegal_next_check", **meta}
        return dist, nxt, {"fallback": False, **meta}

    def run(self, case_id, visible, run_check) -> Trace:
        cfg = self.cfg
        evidence = summarize_evidence(visible)
        uniform = np.full(len(HYPOTHESES), 1 / len(HYPOTHESES))
        history: list[tuple[str, float]] = []
        done: dict[str, int] = {}
        spent, calls = 0.0, []
        _, legal = self._legal(done, spent)
        dist, nxt, m = self._ask(evidence, history, legal, spent, uniform)
        calls.append(m)
        prior = dist
        steps = [Step(step=0, posterior=_d(dist), cumulative_cost=0.0, n_checks=0)]
        while True:
            remaining, legal = self._legal(done, spent)
            if not remaining:
                stop = StopReason.ALL_USED; break
            if not legal:
                stop = StopReason.BUDGET; break
            if nxt == "STOP":
                stop = StopReason.LLM_STOP; break
            res = run_check(nxt)
            done[nxt] = 1
            spent += cfg.costs[nxt]
            history.append((nxt, res.primary_score))
            _, legal = self._legal(done, spent)
            dist, nxt_new, m = self._ask(evidence, history, legal, spent, dist)
            calls.append(m)
            steps.append(Step(step=len(steps), action=nxt, result=res, posterior=_d(dist),
                              cumulative_cost=spent, n_checks=len(done)))
            nxt = nxt_new
        fallbacks = sum(1 for c in calls if c.get("fallback"))
        return Trace(case_id=case_id, arm=self.name, prior=_d(prior), steps=steps, stop_reason=stop,
                     meta={"fallback_events": fallbacks, "llm_calls": calls, "config": asdict(cfg),
                           "note": "posteriors are self-reported and uncalibrated"})