"""Shared agent-visible feature encoder (README Section 7, step 2).

B3b (logistic regression) and B5 (LLM text summary) must see EXACTLY the same information.
FEATURE_SPEC is the single place that defines it. Person B: make `sim.py` emit these keys in
`AgentVisible`, or extend the spec together with Person A (a spec-coverage test will catch drift).
"""
from __future__ import annotations

import re

import numpy as np

from .schemas import AgentVisible

# PROVISIONAL names: confirm against sim.py in hour 1.
NUMERIC_KEYS = [
    "t1_mag", "t2_mag", "t3_mag",            # continuous screen conflict magnitudes (coarse)
    "t1_fail", "t2_fail", "t3_fail",         # 0/1 flags
    "screen_composition_dev",                # normalized composition deviation (screen)
    "screen_rms_2theta", "screen_unmatched_frac",
    "screen_gap_ev", "lit_gap_ev",           # fitted vs literature gap
    "main_peak_2theta", "lit_main_peak_2theta",
    "lit_composition",
]
CATEGORICAL_VOCAB = {"lit_ref_phase": ["anatase", "rutile"]}

# Missing-value rule (fixed, documented): numeric -> 0.0 plus `<k>__missing`=1; categorical -> all-zero one-hot plus `<k>__missing`=1.
FORBIDDEN_TOKENS = [
    "H1", "H2", "H3", "H4", "fault", "inject", "hidden", "magnitude_band", "seed", "resolvable",
    "identifiab", "ground truth", "label",
]


def feature_names() -> list[str]:
    names = []
    for k in NUMERIC_KEYS:
        names += [k, f"{k}__missing"]
    for k, vocab in CATEGORICAL_VOCAB.items():
        names += [f"{k}={v}" for v in vocab] + [f"{k}__missing"]
    return names


def encode(visible: AgentVisible) -> np.ndarray:
    out: list[float] = []
    for k in NUMERIC_KEYS:
        v = visible.numeric.get(k)
        out += [0.0, 1.0] if v is None else [float(v), 0.0]
    for k, vocab in CATEGORICAL_VOCAB.items():
        v = visible.categorical.get(k)
        out += [1.0 if v == x else 0.0 for x in vocab] + [1.0 if v is None or v not in vocab else 0.0]
    return np.asarray(out, float)


def unknown_keys(visible: AgentVisible) -> list[str]:
    """Keys the case carries that the spec ignores (spec-coverage test)."""
    return sorted((set(visible.numeric) - set(NUMERIC_KEYS)) | (set(visible.categorical) - set(CATEGORICAL_VOCAB)))


def summarize_evidence(visible: AgentVisible) -> str:
    """Plain-text evidence summary for the LLM. Same fields as `encode`, nothing else."""
    lines = []
    for k in NUMERIC_KEYS:
        if k in visible.numeric:
            lines.append(f"{k}: {visible.numeric[k]:.4g}")
    for k in CATEGORICAL_VOCAB:
        if k in visible.categorical:
            lines.append(f"{k}: {visible.categorical[k]}")
    return "\n".join(lines)


def scan_leak(text: str) -> list[str]:
    """Find forbidden terms as words/variants, avoiding arbitrary substring matches."""
    hits = []
    for tok in FORBIDDEN_TOKENS:
        if tok == "identifiab":
            pattern = r"(?<!\w)" + re.escape(tok) + r"\w*"
        elif tok in {"fault", "inject", "hidden", "resolvable", "label"}:
            pattern = r"(?<!\w)" + re.escape(tok) + r"\w*"
        else:
            pattern = r"(?<!\w)" + re.escape(tok) + r"(?!\w)"
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(tok)
    return hits
