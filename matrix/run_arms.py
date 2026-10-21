"""Run arms on a split and save one replayable JSON trace per (arm, case).

  python -m materialfalsify.run_arms --split dev --arms B0 B1 B2 B3 B3b --cases cases/dev.json \
      --table artifacts/table.npz --prior-model artifacts/prior.pkl --out traces/dev
  python -m materialfalsify.run_arms --split dev --arms B4 B5 --llm-model <model> --cache-dir cache/llm ...
  python -m materialfalsify.run_arms --split test --arms all --frozen frozen.json ...   # frozen enforced
  python -m materialfalsify.run_arms --mock --split dev --arms all --out /tmp/t          # no Person B needed

Resumable: existing traces are skipped (use --force to redo), so an API failure never costs a full rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional
import numpy as np

from . import mock
from .agents import (Agent, AgentConfig, B0FullCheck, B1Random, B2CheapestFirst, B3GreedyIG, B3bLearnedPrior,
                     B4LLMOnly, B5Hybrid)
from .config import load_frozen, verify
from .llm import GeminiClient, CachedLLM
from .planner import LikelihoodTable, Planner
from .priors import PriorModel
from .prompts import PROMPT_VERSION
from .schemas import Case, Trace

ALL_ARMS = ["B0", "B1", "B2", "B3", "B3b", "B4", "B5"]


def load_cases(path) -> list[Case]:
    return [Case.model_validate(c) for c in json.loads(Path(path).read_text())]


def save_cases(cases: list[Case], path: str | Path) -> None:
    def json_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    Path(path).write_text(
        json.dumps([c.model_dump() for c in cases], default=json_default)
    )

def trace_path(out: Path, arm: str, case_id: str) -> Path:
    return out / arm / f"{case_id}.json"


def load_traces(out, arm: str) -> dict[str, Trace]:
    d = Path(out) / arm
    return {p.stem: Trace.model_validate_json(p.read_text()) for p in sorted(d.glob("*.json"))}


def build_agents(arms: list[str], table: LikelihoodTable, cfg: AgentConfig,
                 prior_model: Optional[PriorModel] = None, llm: Optional[CachedLLM] = None) -> dict[str, Agent]:
    def pl() -> Planner:
        return Planner(table, cfg.costs, cfg.eps_gain)

    out: dict[str, Agent] = {}
    for a in arms:
        if a == "B0": out[a] = B0FullCheck(pl(), cfg)
        elif a == "B1": out[a] = B1Random(pl(), cfg)
        elif a == "B2": out[a] = B2CheapestFirst(pl(), cfg)
        elif a == "B3": out[a] = B3GreedyIG(pl(), cfg)
        elif a == "B3b":
            if prior_model is None: raise ValueError("B3b needs --prior-model")
            out[a] = B3bLearnedPrior(pl(), prior_model, cfg)
        elif a in ("B4", "B5"):
            if llm is None: raise ValueError(f"{a} needs --llm-model (or --mock)")
            out[a] = B4LLMOnly(llm, cfg) if a == "B4" else B5Hybrid(pl(), llm, cfg)
        else:
            raise ValueError(f"unknown arm {a}")
    return out


def run_split(agents: dict[str, Agent], cases: list[Case], runner_for: Callable[[Case], Callable],
              out, force: bool = False) -> dict:
    out = Path(out)
    counts = {a: {"run": 0, "skipped": 0} for a in agents}
    for arm, agent in agents.items():
        for case in cases:
            p = trace_path(out, arm, case.case_id)
            if p.exists() and not force:
                counts[arm]["skipped"] += 1
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            tr = agent.run(case.case_id, case.agent_visible, runner_for(case))  # agents never see case.hidden
            p.write_text(tr.model_dump_json())
            counts[arm]["run"] += 1
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--arms", nargs="+", default=["all"])
    ap.add_argument("--cases"); ap.add_argument("--out", required=True)
    ap.add_argument("--table"); ap.add_argument("--prior-model")
    ap.add_argument("--frozen"); ap.add_argument("--mock", action="store_true")
    ap.add_argument("--llm-model"); ap.add_argument("--llm-temperature", type=float, default=1.0)
    ap.add_argument("--cache-dir", default="cache/llm")
    ap.add_argument("--eps-gain", type=float, default=0.01); ap.add_argument("--budget", type=float)
    ap.add_argument("--prior-eps", type=float, default=0.05); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-mock", type=int, default=40)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    arms = ALL_ARMS if a.arms == ["all"] else a.arms

    fc = load_frozen(a.frozen) if a.frozen else None
    if a.split == "test" and fc is None:
        ap.error("the test split requires --frozen (freeze thresholds on dev first)")
    cfg = fc.agent_config() if fc else AgentConfig(eps_gain=a.eps_gain, budget=a.budget, prior_eps=a.prior_eps,
                                                   k_samples=a.k, seed=a.seed)
    if a.mock:
        table, cases, runner_for = mock.mock_table(), mock.mock_cases(a.n_mock, a.seed), mock.case_runner
        pm = PriorModel().fit(*mock.training_data(800, 1)) if "B3b" in arms else None
        llm = CachedLLM(mock.FakeLLM(mock.llm_responder), a.cache_dir, PROMPT_VERSION) if {"B4", "B5"} & set(arms) else None
    else:
        if not (a.cases and a.table):
            ap.error("--cases and --table are required unless --mock")
        from .checks import runner as real_runner  # Person B's module
        table, cases = LikelihoodTable.load(a.table), load_cases(a.cases)
        runner_for = lambda case: lambda check_id: real_runner(cases, case.case_id, check_id)
        pm = PriorModel.load(a.prior_model) if a.prior_model else None
        llm = None
        if {"B4", "B5"} & set(arms):
            if not a.llm_model:
                ap.error("B4/B5 need --llm-model")
            llm = CachedLLM(GeminiClient(a.llm_model, a.llm_temperature), a.cache_dir, PROMPT_VERSION)
        if fc:
            verify(fc, cfg, a.table, a.prior_model, PROMPT_VERSION, a.llm_model,
                   llm.client.params if llm else None)
    agents = build_agents(arms, table, cfg, pm, llm)
    counts = run_split(agents, cases, runner_for, a.out, a.force)
    manifest = {"split": a.split, "arms": arms, "n_cases": len(cases), "counts": counts, "prompt_version": PROMPT_VERSION,
                "llm_stats": llm.stats if llm else None, "frozen": bool(fc), "mock": a.mock}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())