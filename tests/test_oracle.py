import numpy as np

from matrix.oracle import JointLikelihoodTable


def test_joint_likelihood_table_round_trip(tmp_path):
    table = JointLikelihoodTable(
        np.ones((4, 2, 2, 2, 3, 3, 3, 3, 3)),
        {f"C{i}": (0.25, 0.75) for i in range(1, 6)},
    )
    path = tmp_path / "joint.npz"

    table.save(path)
    loaded = JointLikelihoodTable.load(path)

    assert loaded.hyps == table.hyps
    assert loaded.checks == table.checks
    assert loaded.screen_flags == table.screen_flags
    assert np.allclose(loaded.table, table.table)
    assert all(np.array_equal(loaded.edges[c], table.edges[c]) for c in table.checks)
