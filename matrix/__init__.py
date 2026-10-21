"""Synthetic stand-ins so the agents side runs end-to-end BEFORE Person B delivers sim/oracle.

Used by tests and for early demo wiring only. Nothing here is a result or a claim about the real
benchmark; replace with `likelihoods.py` output and `checks.py` runners when they exist.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .llm import LLMClient
from .planner import LikelihoodTable
from .schemas import CHECKS, HYPOTHESES, AgentVisible, Case, CheckResult, HiddenInfo

# P(bin | hypothesis) per check, bins low/mid/high. Independent across checks (mock only).
_M = {
    "C1": {"H1": (.8, .15, .05), "H2": (.05, .15, .8), "H3": (.8, .15, .05), "H4": (.8, .15, .05)},
    "C2": {"H1": (.8, .15, .05), "H2": (.15, .7, .15), "H3": (.05, .15, .8), "H4": (.8, .15, .05)},
    "C3": {"H1": (.05, .15, .8), "H2": (.8, .15, .05), "H3": (.8, .15, .05), "H4": (.7, .2, .1)},
    "C4": {"H1": (.2, .6, .2), "H2": (.8, .15, .05), "H3": (.8, .15, .05), "H4": (.05, .15, .8)},
    "C5": {"H1": (.1, .3, .6), "H2": (.1, .3, .6), "H3": (.8, .15, .05), "H4": (.7, .2, .1)},
}
EDGES = {c: [1 / 3, 2 / 3] for c in CHECKS}
_CENTERS = (1 / 6, 0.5, 5 / 6)


def mock_table() -> LikelihoodTable:
    t = np.zeros((4,) + (3,) * 5)
    for hi, h in enumerate(HYPOTHESES):
        joint = np.ones((3,) * 5)
        for ci, c in enumerate(CHECKS):
            shape = [1] * 5
            shape[ci] = 3
            joint = joint * np.array(_M[c][h]).reshape(shape)
        t[hi] = joint
    return LikelihoodTable(t, EDGES)


def mock_visible(rng, h: str) -> AgentVisible:
    i = HYPOTHESES.index(h)
    base = np.array([[1.2, .3, .3, .8], [.3, 1.0, .8, .3], [.2, .8, 1.2, .2], [1.0, .2, .2, 1.2]])[i]
    f = base + rng.normal(0, .5, 4)
    num = {"t1_mag": float(abs(f[1])), "t2_mag": float(abs(f[2])), "t3_mag": float(abs(f[0])),
           "t1_fail": float(f[1] > .7), "t2_fail": float(f[2] > .7), "t3_fail": float(f[0] > .7),
           "screen_gap_ev": float(3.2 + .1 * f[3]), "lit_gap_ev": 3.2}
    return AgentVisible(numeric=num, categorical={"lit_ref_phase": "anatase"})


def mock_case(rng, h: str, table: LikelihoodTable | None = None):
    """Returns (visible, run_check, true_bins). Bins are drawn jointly from the mock table."""
    table = table or mock_table()
    flat = rng.choice(3 ** 5, p=table.table[HYPOTHESES.index(h)].reshape(-1))
    bins = dict(zip(CHECKS, np.unravel_index(flat, (3,) * 5)))

    def run_check(c: str) -> CheckResult:
        return CheckResult(check_id=c, primary_score=_CENTERS[int(bins[c])] + float(rng.uniform(-.05, .05)))

    return mock_visible(rng, h), run_check, bins


class FakeLLM(LLMClient):
    """Deterministic stand-in: returns the mock-table-agnostic mildly peaked distribution it is told to."""
    model = "fake"
    params: dict = {}

    def __init__(self, responder=None):
        self.responder = responder or (lambda system, user: json.dumps(
            {"distribution": {"H1": .25, "H2": .25, "H3": .25, "H4": .25}, "next_check": "C1"}))
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.responder(system, user)


# ---------- Case-level mocks (full pipeline without Person B) ----------
def mock_identifiability(table: LikelihoodTable | None = None) -> dict[str, float]:
    """Exact I(h) under the mock table (uniform prior, all five checks)."""
    table = table or mock_table()
    flat = table.table.reshape(4, -1)
    post = flat / flat.sum(axis=0, keepdims=True)
    return {h: float((flat[i] * post[i]).sum()) for i, h in enumerate(HYPOTHESES)}


def mock_cases(n: int, seed: int = 0, tau_id: float = 0.8) -> list[Case]:
    table = mock_table()
    ident = mock_identifiability(table)
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        h = HYPOTHESES[i % 4]
        flat = rng.choice(3 ** 5, p=table.table[HYPOTHESES.index(h)].reshape(-1))
        bins = {c: int(b) for c, b in zip(CHECKS, np.unravel_index(flat, (3,) * 5))}
        cases.append(Case(case_id=f"mock-{seed}-{i:04d}", agent_visible=mock_visible(rng, h),
                          hidden=HiddenInfo(fault=h, magnitude=0.5, magnitude_band="B2", noise_level="s1",
                                            resolvable=ident[h] >= tau_id, identifiability=ident[h]),
                          mock_bins=bins))
    return cases


def case_runner(case: Case):
    """Deterministic per (case, check): identical results across arms, as the fair comparison needs."""
    bins = case.model_extra["mock_bins"]

    def run_check(c: str) -> CheckResult:
        u = int.from_bytes(hashlib.sha256(f"{case.case_id}|{c}".encode()).digest()[:4], "big") / 2 ** 32
        return CheckResult(check_id=c, primary_score=_CENTERS[bins[c]] + (u - .5) * .1)

    return run_check


def training_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    hs = [HYPOTHESES[i % 4] for i in range(n)]
    return [mock_visible(rng, h) for h in hs], hs


def llm_responder(system: str, user: str) -> str:
    """Mock LLM: uniform prior; as B4, runs the cheapest legal check then stops."""
    uni = {h: .25 for h in HYPOTHESES}
    if "Checks you may still run:" not in user:
        return json.dumps({"distribution": uni})
    line = user.split("Checks you may still run:")[1].split("\n")[0].strip()
    legal = [c.strip() for c in line.split(",") if c.strip().startswith("C")]
    return json.dumps({"distribution": uni, "next_check": legal[0] if legal else "STOP"})