import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from matrix.real_data import (
    RealDataCheckRunner,
    check_c1,
    diagnose,
    load_composition,
    load_optical,
    load_real_sample,
    load_xrd,
)


FIXTURES = Path(__file__).parent / "fixtures" / "real_data"


def _sample(**overrides):
    args = {
        "composition_path": FIXTURES / "composition.csv",
        "xrd_path": FIXTURES / "xrd.csv",
        "optical_path": FIXTURES / "optical.csv",
        "metadata_path": FIXTURES / "metadata.yaml",
    }
    args.update(overrides)
    return load_real_sample(**args)


def test_csv_parsers_load_composition_xrd_and_optical():
    assert load_composition(FIXTURES / "composition.csv").elements["O"][0] == 0.667
    assert len(load_xrd(FIXTURES / "xrd.csv").two_theta) == 11
    assert len(load_optical(FIXTURES / "optical.csv").x) == 11


def test_missing_required_metadata_is_an_error(tmp_path):
    metadata = yaml.safe_load((FIXTURES / "metadata.yaml").read_text())
    del metadata["xrd"]["geometry"]
    path = tmp_path / "metadata.yaml"
    path.write_text(yaml.safe_dump(metadata))
    with pytest.raises(ValueError, match="geometry"):
        _sample(metadata_path=path)


def test_uncertainty_type_must_be_explicit(tmp_path):
    metadata = yaml.safe_load((FIXTURES / "metadata.yaml").read_text())
    del metadata["composition"]["uncertainty_type"]
    path = tmp_path / "metadata.yaml"
    path.write_text(yaml.safe_dump(metadata))
    with pytest.raises(ValueError, match="uncertainty_type"):
        _sample(metadata_path=path)


def test_c1_uses_atomic_ratio_and_propagates_uncertainty():
    result = check_c1(_sample())
    assert result["measured_O_to_Ti"] == pytest.approx(0.667 / 0.333)
    assert result["standard_uncertainty"] > 0
    assert result["tolerance"] is None


def test_c2_fits_global_shift_and_c3_reports_relative_fractions():
    runner = RealDataCheckRunner(_sample(), FIXTURES / "xrd_reference.csv")
    c2 = runner.run("C2")
    c3 = runner.run("C3")
    assert c2["status"] == "AVAILABLE"
    assert c2["sample_displacement_deg"] == pytest.approx(0, abs=0.05)
    assert c2["fit_quality"]["rmse_normalized_intensity"] < 0.02
    assert c3["status"] == "AVAILABLE"
    assert c3["anatase"] + c3["rutile"] == pytest.approx(1)
    assert "not absolute wt%" in c3["interpretation"]


def test_xrd_checks_are_unavailable_without_reference_profiles():
    runner = RealDataCheckRunner(_sample())
    assert runner.run("C2")["status"] == "UNAVAILABLE"
    assert runner.run("C3")["status"] == "UNAVAILABLE"


def test_xrd_fit_recovers_shift_and_relative_phase_mix(tmp_path):
    reference = np.genfromtxt(
        FIXTURES / "xrd_reference.csv", delimiter=",", names=True
    )
    grid = reference["two_theta"]
    anatase = reference["anatase_intensity"]
    rutile = reference["rutile_intensity"]
    anatase = anatase / np.sum(np.diff(grid) * (anatase[:-1] + anatase[1:]) * 0.5)
    rutile = rutile / np.sum(np.diff(grid) * (rutile[:-1] + rutile[1:]) * 0.5)
    shift = 0.2
    intensity = (
        0.7 * np.interp(grid - shift, grid, anatase, left=0.0, right=0.0)
        + 0.3 * np.interp(grid - shift, grid, rutile, left=0.0, right=0.0)
    )
    shifted_path = tmp_path / "shifted.csv"
    shifted_path.write_text(
        "two_theta,intensity\n"
        + "".join(f"{x},{y}\n" for x, y in zip(grid, intensity))
    )
    runner = RealDataCheckRunner(
        _sample(xrd_path=shifted_path), FIXTURES / "xrd_reference.csv"
    )
    c2, c3 = runner.run("C2"), runner.run("C3")
    assert c2["sample_displacement_deg"] == pytest.approx(shift, abs=0.05)
    assert c3["anatase"] == pytest.approx(0.7, abs=0.03)
    assert c3["rutile"] == pytest.approx(0.3, abs=0.03)


def test_c4_runs_primary_and_alternate_tauc_analyses():
    result = RealDataCheckRunner(_sample()).run("C4")
    assert result["status"] == "AVAILABLE"
    for phase in ("anatase", "rutile"):
        model = result["models"][phase]
        assert model["primary"]["n_points"] >= 3
        assert "band_gap_ev" in model["alternate_baseline"]
        assert model["primary"]["exponent"] > 0
    assert result["models"]["anatase"]["primary"]["band_gap_ev"] == pytest.approx(2.0)
    assert "not specified" in result["defect_tail_model"]


def test_c5_is_unavailable_without_a_supplied_repeat():
    assert RealDataCheckRunner(_sample()).run("C5") == {
        "status": "UNAVAILABLE",
        "reason": "independent repeat measurement not supplied",
    }


def test_c5_uses_only_supplied_repeat_measurements():
    sample = _sample(repeat_composition_path=FIXTURES / "repeat_composition.csv")
    result = RealDataCheckRunner(sample).run("C5")
    assert result["status"] == "AVAILABLE"
    assert result["measurements_compared"] == ["composition_O_to_Ti_delta"]
    assert result["deltas"]["composition_O_to_Ti_delta"] != 0


def test_real_data_runner_does_not_load_synthetic_planner():
    import matrix.real_data as real_data

    assert not hasattr(real_data, "LikelihoodTable")
    assert not hasattr(real_data, "PriorModel")
    assert "likelihood_table_50k.npz" not in Path(real_data.__file__).read_text()


def test_cli_writes_evidence_report_and_unresolved_diagnosis(tmp_path):
    output = tmp_path / "result"
    result = diagnose(
        composition_path=FIXTURES / "composition.csv",
        xrd_path=FIXTURES / "xrd.csv",
        optical_path=FIXTURES / "optical.csv",
        metadata_path=FIXTURES / "metadata.yaml",
        xrd_reference=FIXTURES / "xrd_reference.csv",
        out_dir=output,
    )
    assert result["diagnosis"]["status"] == "UNRESOLVED"
    assert (output / "diagnosis.json").is_file()
    assert (output / "evidence.json").is_file()
    assert (output / "report.md").is_file()
    saved = json.loads((output / "diagnosis.json").read_text())
    assert saved["diagnosis"]["hypothesis"] is None


def test_cli_entry_point_dispatches_diagnosis(tmp_path):
    from matrix.real_cli import main

    result = main([
        "--composition", str(FIXTURES / "composition.csv"),
        "--xrd", str(FIXTURES / "xrd.csv"),
        "--optical", str(FIXTURES / "optical.csv"),
        "--metadata", str(FIXTURES / "metadata.yaml"),
        "--out", str(tmp_path / "cli"),
    ])
    assert result == 0
