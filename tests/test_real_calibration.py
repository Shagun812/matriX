import json
from pathlib import Path

import pytest
import yaml

from matrix.real_calibration import (
    calibrate,
    load_real_model,
    predict_real,
    validate_heldout,
)
from matrix.real_dataset import read_dataset, validate_dataset
from matrix.real_data import diagnose


FIXTURES = Path(__file__).parent / "fixtures" / "real_data"


def make_dataset(root: Path, dataset_id: str, labels, *, reuse=None):
    samples = []
    salt = (sum(ord(character) for character in dataset_id) % 50) / 10000
    for index, label in enumerate(labels):
        sample_id = f"{dataset_id}-sample-{index}"
        folder = root / f"sample-{index}"
        folder.mkdir(parents=True, exist_ok=True)
        if reuse is not None and index == 0:
            composition_source, xrd_source, optical_source, metadata_source = reuse
            composition_path = composition_source
            xrd_path = xrd_source
            optical_path = optical_source
            metadata_path = metadata_source
        else:
            composition_path = folder / "composition.csv"
            xrd_path = folder / "xrd.csv"
            optical_path = folder / "optical.csv"
            ti = 0.333 + salt + index * 0.0001
            oxygen = 1.0 - ti
            composition_path.write_text(
                f"element,value,uncertainty,unit\nTi,{ti:.6f},0.002,atomic_fraction\n"
                f"O,{oxygen:.6f},0.003,atomic_fraction\n"
            )
            factor = 1 + salt + index * 0.011
            xrd_lines = (FIXTURES / "xrd.csv").read_text().splitlines()
            xrd_path.write_text(
                xrd_lines[0] + "\n" + "\n".join(
                    f"{line.split(',')[0]},{float(line.split(',')[1]) * factor:.8f}"
                    for line in xrd_lines[1:]
                ) + "\n"
            )
            optical_lines = (FIXTURES / "optical.csv").read_text().splitlines()
            optical_path.write_text(
                optical_lines[0] + "\n" + "\n".join(
                    f"{line.split(',')[0]},{float(line.split(',')[1]) * (1 + salt + index * 0.013):.10f}"
                    for line in optical_lines[1:]
                ) + "\n"
            )
        metadata_path = folder / "metadata.yaml"
        metadata = yaml.safe_load((FIXTURES / "metadata.yaml").read_text())
        metadata["sample_id"] = sample_id
        metadata_path.write_text(yaml.safe_dump(metadata))

        samples.append({
            "sample_id": sample_id,
            "material": "tio2",
            "ground_truth": label,
            "ground_truth_confidence": None if label == "UNKNOWN" else 0.9,
            "composition": str(composition_path.relative_to(root)) if composition_path.is_relative_to(root) else str(composition_path),
            "xrd": str(xrd_path.relative_to(root)) if xrd_path.is_relative_to(root) else str(xrd_path),
            "optical": str(optical_path.relative_to(root)) if optical_path.is_relative_to(root) else str(optical_path),
            "metadata": str(metadata_path.relative_to(root)) if metadata_path.is_relative_to(root) else str(metadata_path),
            "xrd_reference": str((FIXTURES / "xrd_reference.csv").resolve()),
            "provenance": {
                "source": "clearly synthetic software test fixture; not experimental",
                "measurement_ids": {
                    "composition": f"{sample_id}-c",
                    "xrd": f"{sample_id}-x",
                    "optical": f"{sample_id}-o",
                },
            },
        })
    manifest = {"schema_version": 1, "dataset_id": dataset_id, "samples": samples}
    path = root / f"{dataset_id}.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path


@pytest.fixture
def train_dataset(tmp_path):
    return make_dataset(
        tmp_path / "train",
        "synthetic-software-test-train",
        ["H1", "H2", "H3", "H4", "UNKNOWN", "UNRESOLVED"],
    )


def test_dataset_parser_and_validator_allow_unknown_and_unresolved(train_dataset):
    dataset = read_dataset(train_dataset)
    result = validate_dataset(train_dataset)
    assert len(dataset.samples) == 6
    assert result["valid"]
    assert result["label_counts"]["UNKNOWN"] == 1
    assert result["label_counts"]["UNRESOLVED"] == 1


def test_dataset_validator_detects_duplicate_sample_ids(tmp_path, train_dataset):
    doc = yaml.safe_load(train_dataset.read_text())
    doc["samples"][1]["sample_id"] = doc["samples"][0]["sample_id"]
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(yaml.safe_dump(doc))
    result = validate_dataset(duplicate)
    assert not result["valid"]
    assert any("duplicate sample_id" in issue["message"] for issue in result["errors"])


def test_dataset_validator_detects_duplicate_raw_file_hashes(tmp_path):
    path = make_dataset(tmp_path / "leak", "duplicate-raw", ["H1", "H2"])
    doc = yaml.safe_load(path.read_text())
    doc["samples"][1]["composition"] = doc["samples"][0]["composition"]
    doc["samples"][1]["provenance"]["measurement_ids"]["composition"] = "unique-composition-id"
    path.write_text(yaml.safe_dump(doc))
    result = validate_dataset(path)
    assert not result["valid"]
    assert any("duplicate raw-file leakage" in issue["message"] for issue in result["errors"])


def test_dataset_validator_rejects_invalid_ground_truth(tmp_path):
    path = make_dataset(tmp_path / "invalid", "invalid-label", ["H1"])
    doc = yaml.safe_load(path.read_text())
    doc["samples"][0]["ground_truth"] = "FAULT"
    path.write_text(yaml.safe_dump(doc))
    assert not validate_dataset(path)["valid"]


def test_calibration_serializes_domain_prior_and_likelihoods(tmp_path, train_dataset):
    model = calibrate(
        train_dataset, tmp_path / "calibrated", decision_threshold=0.7
    )
    loaded = load_real_model(tmp_path / "calibrated" / "real_model.json")
    assert loaded["training_domain"] == "real"
    assert loaded["synthetic_data_used"] is False
    assert loaded["class_counts"] == {"H1": 1, "H2": 1, "H3": 1, "H4": 1}
    assert loaded["excluded_label_counts"] == {"UNRESOLVED": 1, "UNKNOWN": 1}
    assert loaded["prior"]["type"] == "uniform"
    assert set(loaded["likelihoods"]) == set(loaded["feature_definitions"]["features"])
    assert model["dataset_hash"] == loaded["dataset_hash"]
    assert (tmp_path / "calibrated" / "evidence.csv").is_file()


def test_calibration_supports_empirical_and_explicit_priors(tmp_path, train_dataset):
    empirical = calibrate(
        train_dataset, tmp_path / "empirical", prior_mode="empirical",
        decision_threshold=0.7,
    )
    assert empirical["prior"]["type"] == "empirical_training_set"
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps({"H1": 4, "H2": 3, "H3": 2, "H4": 1}))
    explicit = calibrate(
        train_dataset, tmp_path / "explicit", prior_mode="explicit",
        explicit_prior_path=prior_path, decision_threshold=0.7,
    )
    assert explicit["prior"]["type"] == "explicit_user_supplied"
    assert explicit["prior"]["probabilities"]["H1"] == pytest.approx(0.4)


def test_calibration_requires_all_four_labeled_classes(tmp_path):
    path = make_dataset(tmp_path / "classes", "missing-h4", ["H1", "H2", "H3", "UNKNOWN"])
    with pytest.raises(ValueError, match="missing H4"):
        calibrate(path, tmp_path / "model", decision_threshold=0.7)


def test_model_rejects_synthetic_domain(tmp_path, train_dataset):
    model = calibrate(train_dataset, tmp_path / "cal", decision_threshold=0.7)
    path = tmp_path / "synthetic.json"
    model["training_domain"] = "synthetic"
    model["synthetic_data_used"] = True
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="real-domain"):
        load_real_model(path)


def test_calibration_likelihoods_reproduce_for_same_dataset(tmp_path, train_dataset):
    first = calibrate(train_dataset, tmp_path / "first", decision_threshold=0.7)
    second = calibrate(train_dataset, tmp_path / "second", decision_threshold=0.7)
    assert first["likelihoods"] == second["likelihoods"]
    assert first["bin_definitions"] == second["bin_definitions"]


def test_heldout_validation_reports_metrics_and_excludes_unknown(tmp_path, train_dataset):
    model_path = tmp_path / "cal" / "real_model.json"
    calibrate(train_dataset, model_path.parent, decision_threshold=0.7)
    heldout = make_dataset(
        tmp_path / "heldout",
        "synthetic-software-test-heldout",
        ["H1", "H2", "H3", "H4", "UNKNOWN", "UNRESOLVED"],
    )
    result = validate_heldout(model_path, heldout, tmp_path / "validation")
    metrics = result["metrics"]
    assert metrics["eligible_labeled_sample_count"] == 5
    assert metrics["unknown_unlabeled_excluded"] == 1
    assert set(metrics["per_class_recall"]) == {"H1", "H2", "H3", "H4"}
    assert metrics["overall_accuracy_95_ci"] is not None
    assert metrics["per_check_usage_available_sample_count"]["C5"] == 0
    assert (tmp_path / "validation" / "validation.json").is_file()


def test_heldout_validation_fails_on_sample_id_leakage(tmp_path, train_dataset):
    model_path = tmp_path / "cal" / "real_model.json"
    calibrate(train_dataset, model_path.parent, decision_threshold=0.7)
    heldout = make_dataset(tmp_path / "heldout", "different-dataset", ["H1"])
    doc = yaml.safe_load(heldout.read_text())
    training = read_dataset(train_dataset)
    doc["samples"][0]["sample_id"] = training.samples[0].sample_id
    held_metadata = Path(heldout.parent / doc["samples"][0]["metadata"])
    metadata = yaml.safe_load(held_metadata.read_text())
    metadata["sample_id"] = training.samples[0].sample_id
    held_metadata.write_text(yaml.safe_dump(metadata))
    heldout.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="leakage by sample_id"):
        validate_heldout(model_path, heldout, tmp_path / "validation")


def test_heldout_validation_fails_on_measurement_id_leakage(tmp_path, train_dataset):
    model_path = tmp_path / "cal" / "real_model.json"
    calibrate(train_dataset, model_path.parent, decision_threshold=0.7)
    heldout = make_dataset(tmp_path / "heldout", "different-dataset", ["H1"])
    doc = yaml.safe_load(heldout.read_text())
    training = read_dataset(train_dataset)
    doc["samples"][0]["provenance"]["measurement_ids"]["xrd"] = (
        training.samples[0].measurement_ids["xrd"]
    )
    heldout.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="leakage by measurement ID"):
        validate_heldout(model_path, heldout, tmp_path / "validation")


def test_heldout_validation_fails_on_raw_hash_leakage(tmp_path, train_dataset):
    model_path = tmp_path / "cal" / "real_model.json"
    calibrate(train_dataset, model_path.parent, decision_threshold=0.7)
    train = read_dataset(train_dataset)
    heldout = make_dataset(
        tmp_path / "heldout", "different-dataset", ["H1"],
        reuse=(
            train.samples[0].resolved_paths["composition"],
            train.samples[0].resolved_paths["xrd"],
            train.samples[0].resolved_paths["optical"],
            None,
        ),
    )
    doc = yaml.safe_load(heldout.read_text())
    doc["samples"][0]["provenance"]["measurement_ids"] = {
        "composition": "new-c", "xrd": "new-x", "optical": "new-o"
    }
    heldout.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="leakage by raw-file hash"):
        validate_heldout(model_path, heldout, tmp_path / "validation")


def test_diagnosis_uses_model_when_supplied_and_unresolved_without_it(tmp_path, train_dataset):
    model_path = tmp_path / "cal" / "real_model.json"
    calibrate(train_dataset, model_path.parent, decision_threshold=0.99)
    fixture = FIXTURES
    common = {
        "composition_path": fixture / "composition.csv",
        "xrd_path": fixture / "xrd.csv",
        "optical_path": fixture / "optical.csv",
        "metadata_path": fixture / "metadata.yaml",
        "xrd_reference": fixture / "xrd_reference.csv",
    }
    without = diagnose(**common, out_dir=tmp_path / "without")
    with_model = diagnose(**common, out_dir=tmp_path / "with", model_path=model_path)
    assert without["diagnosis"]["status"] == "UNRESOLVED"
    assert with_model["diagnosis"]["model_domain"] == "real"
    assert with_model["diagnosis"]["posterior"] is not None


def test_predict_prior_and_missing_c5_are_explicit(tmp_path, train_dataset):
    model = calibrate(train_dataset, tmp_path / "cal", decision_threshold=0.7)
    assert model["prior"]["type"] == "uniform"
    assert model["feature_definitions"]["availability"].startswith("missing measurements")
    assert "C5.composition_O_to_Ti_delta" in model["feature_definitions"]["features"]


def test_real_calibration_cli_workflow(tmp_path, train_dataset):
    from matrix.__main__ import main

    heldout = make_dataset(
        tmp_path / "heldout",
        "synthetic-software-test-cli-heldout",
        ["H1", "H2", "H3", "H4"],
    )
    validated = tmp_path / "dataset-validation"
    assert main(["real-dataset-validate", "--dataset", str(train_dataset), "--out", str(validated)]) == 0

    calibrated = tmp_path / "calibration"
    assert main([
        "real-calibrate",
        "--dataset", str(train_dataset),
        "--out", str(calibrated),
        "--decision-threshold", "0.8",
    ]) == 0

    validation = tmp_path / "heldout-validation"
    assert main([
        "real-validate",
        "--model", str(calibrated / "real_model.json"),
        "--dataset", str(heldout),
        "--out", str(validation),
    ]) == 0
    assert (validated / "validation.json").is_file()
    assert (calibrated / "real_model.json").is_file()
    assert (validation / "validation.json").is_file()
