"""Frozen configuration (README Section 7, "Fair comparison"): thresholds chosen on dev, then frozen.

`freeze()` writes the file ONCE (refuses to overwrite). `run_arms` refuses to run the test split unless
a frozen file exists and the table, prior model, prompt version and agent settings still match it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .agents import AgentConfig


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenConfig(BaseModel):
    eps_gain: float
    prior_eps: float
    k_samples: int
    budget: Optional[float]
    costs: dict
    tau_grid: list[float]
    operating_tau: float
    seed: int = 0
    llm_model: Optional[str] = None
    llm_params: dict = Field(default_factory=dict)
    prompt_version: str
    table_sha256: str
    prior_model_sha256: Optional[str] = None
    created: str
    notes: str = ""

    def agent_config(self) -> AgentConfig:
        return AgentConfig(costs=dict(self.costs), budget=self.budget, eps_gain=self.eps_gain,
                           prior_eps=self.prior_eps, k_samples=self.k_samples, seed=self.seed)


def freeze(path, cfg: AgentConfig, table_path, prior_path, prompt_version: str, tau_grid, operating_tau: float,
           llm_model: Optional[str] = None, llm_params: Optional[dict] = None, notes: str = "",
           overwrite: bool = False) -> FrozenConfig:
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} already exists; a frozen config must not change after the test run starts")
    fc = FrozenConfig(eps_gain=cfg.eps_gain, prior_eps=cfg.prior_eps, k_samples=cfg.k_samples, budget=cfg.budget,
                      costs=cfg.costs, tau_grid=list(tau_grid), operating_tau=operating_tau, seed=cfg.seed,
                      llm_model=llm_model, llm_params=llm_params or {}, prompt_version=prompt_version,
                      table_sha256=sha256_file(table_path),
                      prior_model_sha256=sha256_file(prior_path) if prior_path else None,
                      created=dt.datetime.now(dt.timezone.utc).isoformat(), notes=notes)
    serialized = fc.model_dump_json(indent=2)

    if overwrite:
        p.write_text(serialized, encoding="utf-8")
    else:
        # Exclusive creation prevents concurrent runs from overwriting a freeze.
        with p.open("x", encoding="utf-8") as f:
            f.write(serialized)

    return fc


def load_frozen(path) -> FrozenConfig:
    return FrozenConfig.model_validate_json(Path(path).read_text())


def verify(fc: FrozenConfig, cfg: AgentConfig, table_path, prior_path, prompt_version: str,
           llm_model: Optional[str] = None, llm_params: Optional[dict] = None) -> None:
    """Raise ValueError listing every mismatch between what is about to run and what was frozen."""
    bad = []
    if fc.agent_config() != cfg:
        bad.append("agent config differs from frozen")
    if table_path is None or sha256_file(table_path) != fc.table_sha256:
        bad.append("likelihood table differs from frozen")
    if fc.prior_model_sha256 and (prior_path is None or sha256_file(prior_path) != fc.prior_model_sha256):
        bad.append("B3b prior model differs from frozen")
    if prompt_version != fc.prompt_version:
        bad.append(f"prompt version {prompt_version} != frozen {fc.prompt_version}")
    if llm_model is not None and llm_model != fc.llm_model:
        bad.append(f"llm model {llm_model} != frozen {fc.llm_model}")
    if llm_params is not None and fc.llm_params and llm_params != fc.llm_params:
        bad.append("llm params differ from frozen")
    if bad:
        raise ValueError("; ".join(bad))