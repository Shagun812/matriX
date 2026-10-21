import numpy as np
import pytest

from matrix.mock import mock_table
from matrix.planner import LikelihoodTable, Planner, entropy


@pytest.fixture
def table():
    return mock_table()


@pytest.fixture
def planner(table):
    return Planner(table)


def test_mock_likelihood_table_is_normalized(table):
    assert table.table.shape == (4, 3, 3, 3, 3, 3)

    totals = table.table.reshape(4, -1).sum(axis=1)
    np.testing.assert_allclose(totals, np.ones(4))


def test_likelihoods_are_nonnegative(table):
    assert np.all(np.isfinite(table.table))
    assert np.all(table.table >= 0)


def test_entropy_uniform_distribution():
    p = np.full(4, 0.25)
    assert entropy(p) == pytest.approx(2.0)


def test_entropy_certain_distribution():
    p = np.array([1.0, 0.0, 0.0, 0.0])
    assert entropy(p) == pytest.approx(0.0)


def test_posterior_is_normalized(planner):
    prior = np.full(4, 0.25)
    posterior = planner.posterior(prior, {})

    assert np.all(np.isfinite(posterior))
    assert np.all(posterior >= 0)
    assert posterior.sum() == pytest.approx(1.0)


def test_expected_gain_is_finite_and_nonnegative(planner):
    prior = np.full(4, 0.25)

    for check in planner.table.checks:
        gain = planner.expected_gain(prior, {}, check)
        assert np.isfinite(gain)
        assert gain >= 0


def test_gain_per_cost_is_finite(planner):
    prior = np.full(4, 0.25)
    gains = planner.gains(prior, {}, planner.table.checks)

    assert set(gains) == set(planner.table.checks)
    assert all(np.isfinite(value) and value >= 0
               for value in gains.values())


def test_pick_uses_deterministic_tie_breaking(planner):
    costs = planner.costs

    # Equal gain: lower cost wins; if cost also ties, lower check ID wins.
    checks = list(planner.table.checks)
    equal_costs = {c: 1.0 for c in checks}
    planner.costs = equal_costs

    winner = planner.pick({c: 0.5 for c in checks})
    assert winner == min(checks)

    planner.costs = costs
