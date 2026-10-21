import numpy as np
import pytest

from matrix.agents import (
    AgentConfig,
    B0FullCheck,
    B1Random,
    B2CheapestFirst,
    B3GreedyIG,
)
from matrix.mock import mock_table
from matrix.planner import Planner
from matrix.schemas import AgentVisible, CHECKS, CheckResult, StopReason


@pytest.fixture
def table():
    return mock_table()


@pytest.fixture
def visible():
    return AgentVisible(numeric={}, categorical={})


def make_runner(table, calls):
    def run_check(check_id):
        calls.append(check_id)
        return CheckResult(
            check_id=check_id,
            primary_score=0.5,
            aux={},
        )
    return run_check


def make_agent(agent_cls, table, cfg):
    return agent_cls(Planner(table), cfg)


def test_b0_runs_every_check_and_ignores_budget(table, visible):
    calls = []
    cfg = AgentConfig(
        costs={c: 1.0 for c in CHECKS},
        budget=0.0,
    )
    agent = make_agent(B0FullCheck, table, cfg)

    trace = agent.run("case-b0", visible, make_runner(table, calls))

    assert set(calls) == set(CHECKS)
    assert len(calls) == len(CHECKS)
    assert trace.stop_reason == StopReason.ALL_USED
    assert trace.steps[-1].n_checks == len(CHECKS)
    assert trace.final_only is True


def test_b2_uses_cheapest_check_first(table, visible):
    calls = []
    costs = {c: 10.0 for c in CHECKS}
    costs["C3"] = 0.25
    cfg = AgentConfig(costs=costs, budget=0.25, eps_gain=0.0)
    agent = make_agent(B2CheapestFirst, table, cfg)

    trace = agent.run("case-b2", visible, make_runner(table, calls))

    assert calls[0] == "C3"
    assert trace.steps[1].cumulative_cost == pytest.approx(0.25)
    assert trace.stop_reason == StopReason.BUDGET


def test_budget_prevents_unaffordable_checks(table, visible):
    calls = []
    cfg = AgentConfig(
        costs={c: 1.0 for c in CHECKS},
        budget=0.5,
        eps_gain=0.0,
    )
    agent = make_agent(B3GreedyIG, table, cfg)

    trace = agent.run("case-budget", visible, make_runner(table, calls))

    assert calls == []
    assert trace.stop_reason == StopReason.BUDGET
    assert trace.steps[-1].cumulative_cost == 0.0


def test_random_agent_is_reproducible_for_same_case(table, visible):
    cfg = AgentConfig(
        costs={c: 1.0 for c in CHECKS},
        budget=None,
        eps_gain=0.0,
        seed=17,
    )

    calls1, calls2 = [], []
    a1 = make_agent(B1Random, table, cfg)
    a2 = make_agent(B1Random, table, cfg)

    t1 = a1.run("same-case", visible, make_runner(table, calls1))
    t2 = a2.run("same-case", visible, make_runner(table, calls2))

    assert calls1 == calls2
    assert [s.action for s in t1.steps] == [s.action for s in t2.steps]


def test_trace_starts_with_zero_cost_prior_step(table, visible):
    calls = []
    cfg = AgentConfig(
        costs={c: 1.0 for c in CHECKS},
        budget=1.0,
        eps_gain=0.0,
    )
    agent = make_agent(B3GreedyIG, table, cfg)

    trace = agent.run("case-step0", visible, make_runner(table, calls))
    first = trace.steps[0]

    assert first.step == 0
    assert first.action is None
    assert first.result is None
    assert first.n_checks == 0
    assert first.cumulative_cost == 0.0
    assert np.isclose(sum(first.posterior.values()), 1.0)
