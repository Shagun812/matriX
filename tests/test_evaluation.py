from matrix.agents import AgentConfig, B0FullCheck
from matrix.evaluation import case_metric, summarize
from matrix.mock import mock_table
from matrix.planner import Planner
from matrix.schemas import AgentVisible, Case, HiddenInfo


def test_b0_prior_only_trace_is_complete():
    case = Case(
        case_id="case-b0",
        agent_visible=AgentVisible(numeric={}),
        hidden=HiddenInfo(
            fault="H1",
            magnitude=0.5,
            magnitude_band="B2",
            noise_level="s1",
            resolvable=True,
        ),
    )
    agent = B0FullCheck(Planner(mock_table()), AgentConfig())
    trace = agent.run(
        case.case_id,
        case.agent_visible,
        lambda check_id: None,
    )

    metric = case_metric(case, trace, 0.6, agent.planner)
    summary = summarize([metric], trace.arm, 0.6)

    assert metric.complete_trace
    assert summary.trace_completeness == 1.0
    assert summary.budget_exhaustion_rate == 0.0
