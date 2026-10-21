import numpy as np

from matrix.features import (
    encode,
    feature_names,
    scan_leak,
    summarize_evidence,
    unknown_keys,
)
from matrix.schemas import AgentVisible


def test_encoding_has_expected_dimension():
    visible = AgentVisible(numeric={}, categorical={})
    encoded = encode(visible)

    assert encoded.shape == (len(feature_names()),)
    assert np.isfinite(encoded).all()


def test_missing_numeric_values_have_missing_indicators():
    from matrix.features import NUMERIC_KEYS

    visible = AgentVisible(numeric={}, categorical={})
    encoded = encode(visible)

    for i, _ in enumerate(NUMERIC_KEYS):
        value_idx = 2 * i
        missing_idx = value_idx + 1

        assert encoded[value_idx] == 0.0
        assert encoded[missing_idx] == 1.0


def test_unknown_keys_are_reported():
    visible = AgentVisible(
        numeric={"unexpected_numeric": 1.0},
        categorical={"unexpected_category": "x"},
    )

    assert unknown_keys(visible) == [
        "unexpected_category",
        "unexpected_numeric",
    ]


def test_evidence_summary_only_contains_specified_fields():
    visible = AgentVisible(
        numeric={"t1_mag": 0.25, "secret_label": 1.0},
        categorical={"lit_ref_phase": "anatase"},
    )

    summary = summarize_evidence(visible)

    assert "t1_mag:" in summary
    assert "lit_ref_phase: anatase" in summary
    assert "secret_label" not in summary


def test_leak_scanner_detects_explicit_forbidden_terms():
    assert scan_leak("The ground truth is H2") 
    assert scan_leak("resolvable=True")


def test_leak_scanner_accepts_ordinary_evidence():
    text = "t1_mag: 0.25\nlit_ref_phase: anatase"
    assert scan_leak(text) == []


def test_leak_scanner_does_not_flag_fault_as_substring_of_default():
    assert scan_leak("default evidence summary") == []


def test_leak_scanner_detects_forbidden_word_variants():
    assert "fault" in scan_leak("faulty measurement")
    assert "identifiab" in scan_leak("identifiability assessment")
    assert "label" in scan_leak("labeling metadata")
