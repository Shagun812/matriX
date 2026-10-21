from matrix.schemas import HYPOTHESES
from matrix.sim import generate_visible


def test_generate_visible_uses_deterministic_dev_likelihood_stream():
    visibles, labels = generate_visible(12, 19)
    repeated_visibles, repeated_labels = generate_visible(12, 19)

    assert labels == repeated_labels
    assert labels and set(labels) <= set(HYPOTHESES)
    assert [v.model_dump() for v in visibles] == [
        v.model_dump() for v in repeated_visibles
    ]
