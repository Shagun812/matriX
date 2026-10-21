"""Prompts for B4/B5. Bump PROMPT_VERSION on ANY change (it is part of the cache key)."""
from __future__ import annotations

PROMPT_VERSION = "v0"

HYPOTHESIS_TEXT = (
    "H1: secondary or wrong phase present (reflections from the other TiO2 polymorph)\n"
    "H2: composition discrepancy in the sample\n"
    "H3: XRD instrumental artefact (constant 2-theta offset)\n"
    "H4: optical-processing artefact (baseline or defect-tail distortion of the fitted gap)"
)

# Neutral descriptions only: no statement of which check separates which hypothesis.
CHECK_CATALOGUE = {
    "C1": "Composition consistency (replicate-averaged). Returns normalized absolute composition deviation.",
    "C2": "XRD zero-shift fit (full resolution). Returns signed fitted 2-theta offset.",
    "C3": "Two-phase fit (NNLS). Returns fitted secondary-phase fraction.",
    "C4": "Alternate optical baseline. Returns signed change in fitted optical gap.",
    "C5": "Higher-resolution repeat measurement. Returns change in overall conflict score vs the original.",
}

PRIOR_SYSTEM = (
    "You assess simulated materials-characterization evidence for anatase/rutile TiO2. "
    "Screening tests disagree. Give a probability distribution over the hypotheses below. "
    "Reply with ONLY a JSON object: {\"distribution\": {\"H1\": p, \"H2\": p, \"H3\": p, \"H4\": p}} "
    "with probabilities summing to 1.\n\nHypotheses:\n" + HYPOTHESIS_TEXT
)


def prior_user(evidence: str) -> str:
    return f"Screening evidence:\n{evidence}\n\nReturn the JSON distribution."


B4_SYSTEM = (
    "You investigate why simulated materials-characterization evidence disagrees, choosing refined "
    "checks one at a time. Each check has a cost; total budget may be limited. After every step report "
    "your probability distribution over the hypotheses, then choose the next check or STOP. "
    "Reply with ONLY a JSON object: {\"distribution\": {\"H1\": p, \"H2\": p, \"H3\": p, \"H4\": p}, "
    "\"next_check\": \"C1\"|\"C2\"|\"C3\"|\"C4\"|\"C5\"|\"STOP\"}. Probabilities sum to 1. "
    "You cannot invent checks or run code.\n\nHypotheses:\n" + HYPOTHESIS_TEXT
)


def b4_user(evidence: str, costs: dict, history: list[tuple[str, float]], legal: list[str],
            budget_left: float | None) -> str:
    cat = "\n".join(f"{c} (cost {costs[c]}): {d}" for c, d in CHECK_CATALOGUE.items())
    done = "\n".join(f"{c}: primary_score = {s:.5g}" for c, s in history) or "(none yet)"
    budget = "unlimited" if budget_left is None else f"{budget_left:g}"
    return (f"Screening evidence:\n{evidence}\n\nCheck catalogue:\n{cat}\n\nResults so far:\n{done}\n\n"
            f"Checks you may still run: {', '.join(legal) if legal else '(none)'}\nBudget left: {budget}\n"
            "Return the JSON object.")