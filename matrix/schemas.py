"""Shared contracts between the agents side (Person A) and the simulator/eval side (Person B).

Drafted by Person A, reviewed by Person B in hour 1. Changing a field here means telling the
other person. Agents only ever see `AgentVisible`; `HiddenInfo` is for evaluation only.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

HYPOTHESES = ("H1", "H2", "H3", "H4")
CHECKS = ("C1", "C2", "C3", "C4", "C5")
# Normalized decision costs (README Section 5). Overridable via AgentConfig for sensitivity runs.
DEFAULT_COSTS = {"C1": 0.5, "C2": 1.0, "C3": 2.0, "C4": 2.0, "C5": 10.0}
N_BINS = 3  # per check score (README 6.2, D3)


class Status(str, Enum):
    CONFIRMED = "CONFIRMED"
    UNRESOLVED = "UNRESOLVED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


class StopReason(str, Enum):
    ALL_USED = "ALL_USED"
    BUDGET = "BUDGET"
    LOW_GAIN = "LOW_GAIN"
    LLM_STOP = "LLM_STOP"


class AgentVisible(BaseModel):
    """Everything an agent may see about a case before running any check.

    Flat on purpose: `features.FEATURE_SPEC` lists the keys. Flags are stored as 0/1 numerics
    (e.g. `t1_fail`). No fault names, generator fields, or family-encoding IDs allowed.
    """

    numeric: dict[str, float]
    categorical: dict[str, str] = Field(default_factory=dict)


class HiddenInfo(BaseModel):
    """Never passed to an agent. Used by evaluation only."""

    model_config = ConfigDict(extra="allow")
    fault: str  # one of HYPOTHESES
    magnitude: float
    magnitude_band: str
    noise_level: str
    resolvable: bool
    identifiability: Optional[float] = None


class Case(BaseModel):
    model_config = ConfigDict(extra="allow")
    case_id: str  # opaque; must not encode the fault family
    agent_visible: AgentVisible
    hidden: HiddenInfo


class CheckResult(BaseModel):
    """What a check runner returns. `primary_score` is the scalar that gets binned."""

    check_id: str
    primary_score: float
    aux: dict[str, float] = Field(default_factory=dict)


class Step(BaseModel):
    step: int
    action: Optional[str] = None  # check run to reach this step (None at step 0)
    result: Optional[CheckResult] = None
    observed_bin: Optional[int] = None
    posterior: dict[str, float]  # planner posterior (B1-B3b, B5) or self-reported (B4)
    cumulative_cost: float
    n_checks: int
    candidate_gain_per_cost: dict[str, float] = Field(default_factory=dict)  # for the demo


class Trace(BaseModel):
    """Replayable record of one trajectory. Serialize with `.model_dump_json()`."""

    case_id: str
    arm: str
    prior: dict[str, float]
    steps: list[Step]
    stop_reason: StopReason
    final_only: bool = False  # B0: only the last step is a decision point
    meta: dict = Field(default_factory=dict)  # LLM samples, dispersion, fallbacks, config


class Decision(BaseModel):
    status: Status
    hypothesis: Optional[str] = None
    cost: float
    n_checks: int
    step: int