"""Measure first-attempt schema validity of the LLM prompts on dev (README gate: >= 95%).

  python -m materialfalsify.llm_check --cases cases/dev.json --model <model> [--k 3]
Calls use the same cache keys as B5, so this also warms the cache for the real run.
"""
from __future__ import annotations

import argparse
import sys

from .features import scan_leak, summarize_evidence
from .llm import CachedLLM, parse_distribution
from .prompts import PRIOR_SYSTEM, PROMPT_VERSION, prior_user
from .schemas import AgentVisible


def prior_validity(llm: CachedLLM, visibles: list[AgentVisible], k: int = 3) -> dict:
    total = valid = 0
    for v in visibles:
        ev = summarize_evidence(v)
        assert not scan_leak(ev), f"leak in evidence summary: {scan_leak(ev)}"
        for i in range(k):
            out, _ = llm.get_json(PRIOR_SYSTEM, prior_user(ev), lambda d: parse_distribution(d["distribution"]),
                                  sample_idx=i, retries=0)
            total += 1; valid += out is not None
    rate = valid / total if total else float("nan")
    return {"calls": total, "valid": valid, "rate": rate, "pass": rate >= 0.95, "stats": dict(llm.stats)}


def main(argv=None) -> int:
    from .llm import GeminiClient
    from .run_arms import load_cases

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=3); ap.add_argument("--cache-dir", default="cache/llm")
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args(argv)
    llm = CachedLLM(GeminiClient(a.model, a.temperature), a.cache_dir, PROMPT_VERSION)
    r = prior_validity(llm, [c.agent_visible for c in load_cases(a.cases)], a.k)
    print(r)
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())