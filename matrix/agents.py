"""Agent arms B0-B5.

All agents receive only AgentVisible plus a callable that executes a permitted
check. Hidden case metadata never enters an agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import hashlib
import numpy as np

from .features import scan_leak, summarize_evidence
from .llm import CachedLLM, parse_distribution
from .planner import Planner
from .priors import PriorModel
from .prompts import B4_SYSTEM, PRIOR_SYSTEM, b4_user, prior_user
from .schemas import (
    CHECKS,
    HYPOTHESES,
    AgentVisible,
    CheckResult,
    Step,
    StopReason,
    Trace,
)


@dataclass(frozen=True)
class AgentConfig:
    costs: dict[str, float] | None = None
    budget: float | None = None
    eps_gain: float = 0.01
    prior_eps: float = 0.05
    k_samples: int = 3
    seed: int = 0

    def __post_init__(self):
        if self.costs is None:
            object.__setattr__(
                self,
                "costs",
                {"C1": 0.5, "C2": 1.0, "C3": 2.0, "C4": 2.0, "C5": 10.0},
            )


Runner = Callable[[str], CheckResult]


def _uniform_prior() -> np.ndarray:
    return np.full(len(HYPOTHESES), 1.0 / len(HYPOTHESES))


def _posterior_dict(p: np.ndarray) -> dict[str, float]:
    return {h: float(p[i]) for i, h in enumerate(HYPOTHESES)}


def _score_bin(score: float) -> int:
    """Fallback three-bin encoding for B4 traces when no Planner is available.

    B4 does not receive the likelihood table, so its trace records a
    deterministic tercile-style score bin. Bayesian arms use the frozen
    Planner table directly.
    """
    x = float(score)
    if not np.isfinite(x):
        return 0
    if x < 1.0 / 3.0:
        return 0
    if x < 2.0 / 3.0:
        return 1
    return 2


def _legal_checks(
    observed: dict[str, int],
    costs: dict[str, float],
    budget: float | None,
    cumulative_cost: float,
) -> list[str]:
    remaining = [c for c in CHECKS if c not in observed]
    if budget is None:
        return remaining
    return [c for c in remaining if cumulative_cost + costs[c] <= budget + 1e-12]


def _append_step(
    steps: list[Step],
    check_id: str | None,
    result: CheckResult | None,
    posterior: np.ndarray,
    cumulative_cost: float,
    gains: dict[str, float] | None = None,
    observed_bin: int | None = None,
) -> None:
    gains = {} if gains is None else dict(gains)

    steps.append(
        Step(
            step=len(steps),
            action=check_id,
            result=result,
            observed_bin=observed_bin,
            posterior=_posterior_dict(posterior),
            cumulative_cost=float(cumulative_cost),
            n_checks=len(steps),
            candidate_gain_per_cost=gains,
        )
    )


class Agent(ABC):
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.costs = dict(cfg.costs or {})

    @abstractmethod
    def run(self, case_id: str, visible: AgentVisible, run_check: Runner) -> Trace:
        raise NotImplementedError

    def _trace(
        self,
        case_id: str,
        arm: str,
        prior: np.ndarray,
        steps: list[Step],
        stop_reason: StopReason,
        *,
        final_only: bool = False,
        meta: dict | None = None,
    ) -> Trace:
        return Trace(
            case_id=case_id,
            arm=arm,
            prior=_posterior_dict(prior),
            steps=steps,
            stop_reason=stop_reason,
            final_only=final_only,
            meta=meta or {},
        )


class _BayesianAgent(Agent):
    def __init__(self, planner: Planner, cfg: AgentConfig):
        super().__init__(cfg)
        self.planner = planner

    def _observe(
        self,
        result: CheckResult,
        observed: dict[str, int],
    ) -> None:
        observed[result.check_id] = self.planner.table.bin(
            result.check_id, result.primary_score
        )

    def _run_planned(
        self,
        case_id: str,
        visible: AgentVisible,
        run_check: Runner,
        prior: np.ndarray,
        arm: str,
        picker: Callable[[np.ndarray, dict[str, int], list[str]], tuple[str, dict[str, float]]],
    ) -> Trace:
        observed: dict[str, int] = {}
        steps: list[Step] = []
        cumulative = 0.0

        _append_step(steps, None, None, prior, cumulative)

        while True:
            legal = _legal_checks(observed, self.costs, self.cfg.budget, cumulative)
            if not legal:
                return self._trace(
                    case_id, arm, prior, steps, StopReason.BUDGET,
                    meta={"observed": observed},
                )

            check, gains = picker(prior, observed, legal)

            if self.cfg.budget is not None and cumulative + self.costs[check] > self.cfg.budget + 1e-12:
                return self._trace(
                    case_id, arm, prior, steps, StopReason.BUDGET,
                    meta={"observed": observed},
                )

            result = run_check(check)
            cumulative += self.costs[check]
            self._observe(result, observed)

            posterior = self.planner.posterior(prior, observed)

            _append_step(
                steps,
                check,
                result,
                posterior,
                cumulative,
                gains,
                observed_bin=observed[result.check_id],
            )

            # Stop once the planner's expected gain is below threshold,
            # but always retain the just-observed result.
            remaining = _legal_checks(
                observed, self.costs, self.cfg.budget, cumulative
            )
            if not remaining:
                reason = (
                    StopReason.BUDGET
                    if self.cfg.budget is not None
                    else StopReason.ALL_USED
                )
                return self._trace(
                    case_id, arm, prior, steps, reason,
                    meta={"observed": observed},
                )

            gains = self.planner.gains(prior, observed, remaining)
            if max(gains.values(), default=0.0) < self.cfg.eps_gain:
                return self._trace(
                    case_id, arm, prior, steps, StopReason.LOW_GAIN,
                    meta={"observed": observed},
                )


class B0FullCheck(_BayesianAgent):
    """B0: no refined checks; decide from the screen-derived prior only."""

    def __init__(self, planner: Planner, cfg: AgentConfig):
        super().__init__(planner, cfg)

    def run(self, case_id: str, visible: AgentVisible, run_check: Runner) -> Trace:
        prior = np.full(len(HYPOTHESES), 0.25)
        steps: list[Step] = []
        _append_step(steps, None, None, prior, 0.0)

        return self._trace(
            case_id,
            "B0",
            prior,
            steps,
            StopReason.LOW_GAIN,
            final_only=True,
        )


class B1Random(_BayesianAgent):
    """B1: random legal check selection."""

    def __init__(self, planner: Planner, cfg: AgentConfig):
        super().__init__(planner, cfg)

    def run(self, case_id: str, visible: AgentVisible, run_check: Runner) -> Trace:
        case_seed = int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:4], 'little')
        seed = np.random.SeedSequence([self.cfg.seed, case_seed])
        rng = np.random.default_rng(seed)

        def pick(prior, observed, legal):
            gains = self.planner.gains(prior, observed, legal)
            check = str(rng.choice(legal))
            return check, gains

        return self._run_planned(
            case_id, visible, run_check, _uniform_prior(), "B1", pick
        )


class B2CheapestFirst(_BayesianAgent):
    """B2: always run the cheapest remaining legal check."""

    def __init__(self, planner: Planner, cfg: AgentConfig):
        super().__init__(planner, cfg)

    def run(self, case_id: str, visible: AgentVisible, run_check: Runner) -> Trace:
        def pick(prior, observed, legal):
            gains = self.planner.gains(prior, observed, legal)
            check = min(legal, key=lambda c: (self.costs[c], c))
            return check, gains

        return self._run_planned(
            case_id, visible, run_check, _uniform_prior(), "B2", pick
        )


class B3GreedyIG(_BayesianAgent):
    """B3: greedy expected information gain per cost."""

    def __init__(self, planner: Planner, cfg: AgentConfig):
        super().__init__(planner, cfg)

    def run(self, case_id: str, visible: AgentVisible, run_check: Runner) -> Trace:
        def pick(prior, observed, legal):
            gains = self.planner.gains(prior, observed, legal)
            check = self.planner.pick(gains)
            return check, gains

        return self._run_planned(
            case_id, visible, run_check, _uniform_prior(), "B3", pick
        )


class B3bLearnedPrior(_BayesianAgent):
    """B3b: learned screen prior, followed by B3 greedy IG/cost."""

    def __init__(self, planner: Planner, prior_model: PriorModel, cfg: AgentConfig):
        super().__init__(planner, cfg)
        self.prior_model = prior_model

    def run(self, case_id: str, visible: AgentVisible, run_check: Runner) -> Trace:
        prior = np.asarray(self.prior_model.predict(visible), dtype=float)
        prior = prior + self.cfg.prior_eps
        prior = prior / prior.sum()

        def pick(prior_now, observed, legal):
            gains = self.planner.gains(prior_now, observed, legal)
            check = self.planner.pick(gains)
            return check, gains

        return self._run_planned(
            case_id, visible, run_check, prior, "B3b", pick
        )


class B4LLMOnly(Agent):
    """B4: LLM supplies posterior and chooses each next check."""

    def __init__(self, llm: CachedLLM, cfg: AgentConfig):
        super().__init__(cfg)
        self.llm = llm

    def _call(
        self,
        evidence: str,
        history: list[tuple[str, float]],
        legal: list[str],
        budget_left: float | None,
        sample_idx: int,
    ):
        user = b4_user(
            evidence,
            self.costs,
            history,
            legal,
            budget_left,
        )

        def validate(d):
            p = parse_distribution(d["distribution"])
            nxt = d.get("next_check", "STOP")
            if nxt != "STOP" and nxt not in CHECKS:
                raise ValueError(f"illegal next_check {nxt}")
            if nxt != "STOP" and nxt not in legal:
                raise ValueError(f"next_check {nxt} is not legal")
            return p, nxt

        parsed, meta = self.llm.get_json(
            B4_SYSTEM,
            user,
            validate,
            sample_idx=sample_idx,
            retries=1,
        )
        return parsed, meta

    def run(self, case_id: str, visible: AgentVisible, run_check: Runner) -> Trace:
        evidence = summarize_evidence(visible)
        leak = scan_leak(evidence)
        if leak:
            raise ValueError(f"LLM evidence leak: {leak}")

        prior = _uniform_prior()
        steps: list[Step] = []
        history: list[tuple[str, float]] = []
        observed: dict[str, int] = {}
        cumulative = 0.0

        _append_step(steps, None, None, prior, cumulative)

        for sample_idx in range(self.cfg.k_samples):
            legal = _legal_checks(
                observed,
                self.costs,
                self.cfg.budget,
                cumulative,
            )

            budget_left = (
                None if self.cfg.budget is None else self.cfg.budget - cumulative
            )

            parsed, meta = self._call(
                evidence,
                history,
                legal,
                budget_left,
                sample_idx,
            )

            if parsed is None:
                # Invalid/API output: stop rather than inventing an action.
                return self._trace(
                    case_id,
                    "B4",
                    prior,
                    steps,
                    StopReason.LLM_STOP,
                    meta={"llm": meta, "history": history},
                )

            posterior, next_check = parsed
            prior = posterior

            if next_check == "STOP" or not legal:
                return self._trace(
                    case_id,
                    "B4",
                    prior,
                    steps,
                    StopReason.LLM_STOP,
                    meta={"history": history, "llm": meta},
                )

            result = run_check(next_check)
            cumulative += self.costs[next_check]
            observed[next_check] = _score_bin(result.primary_score)
            history.append((next_check, result.primary_score))

            _append_step(
                steps,
                next_check,
                result,
                posterior,
                cumulative,
            )

            if len(observed) == len(CHECKS):
                return self._trace(
                    case_id,
                    "B4",
                    prior,
                    steps,
                    StopReason.ALL_USED,
                    meta={"history": history},
                )

        return self._trace(
            case_id,
            "B4",
            prior,
            steps,
            StopReason.LLM_STOP,
            meta={"history": history},
        )


class B5Hybrid(_BayesianAgent):
    """B5: LLM-derived prior, Bayesian planner selects refined checks."""

    def __init__(self, planner: Planner, llm: CachedLLM, cfg: AgentConfig):
        super().__init__(planner, cfg)
        self.llm = llm

    def _llm_prior(self, visible: AgentVisible) -> tuple[np.ndarray | None, dict]:
        evidence = summarize_evidence(visible)
        leak = scan_leak(evidence)
        if leak:
            raise ValueError(f"LLM evidence leak: {leak}")

        def validate(d):
            return parse_distribution(d["distribution"])

        samples = []
        metas = []

        for i in range(self.cfg.k_samples):
            parsed, meta = self.llm.get_json(
                PRIOR_SYSTEM,
                prior_user(evidence),
                validate,
                sample_idx=i,
                retries=1,
            )
            metas.append(meta)
            if parsed is not None:
                samples.append(parsed)

        if not samples:
            return None, {"samples": 0, "calls": metas}

        p = np.mean(np.stack(samples), axis=0)
        p = p + self.cfg.prior_eps
        p = p / p.sum()
        return p, {"samples": len(samples), "calls": metas}

    def run(self, case_id: str, visible: AgentVisible, run_check: Runner) -> Trace:
        prior, meta = self._llm_prior(visible)

        if prior is None:
            fallback = _uniform_prior()
            steps: list[Step] = []
            _append_step(steps, None, None, fallback, 0.0)
            return self._trace(
                case_id,
                "B5",
                fallback,
                steps,
                StopReason.LLM_STOP,
                meta={"llm": meta},
            )

        def pick(prior_now, observed, legal):
            gains = self.planner.gains(prior_now, observed, legal)
            check = self.planner.pick(gains)
            return check, gains

        trace = self._run_planned(
            case_id, visible, run_check, prior, "B5", pick
        )
        trace.meta["llm"] = meta
        return trace