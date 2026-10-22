"""Calibration and held-out validation for explicitly real-domain evidence."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .real_dataset import (
    CHECK_FEATURES,
    EVIDENCE_FEATURES,
    REAL_LABELS,
    SUPERVISED_LABELS,
    DatasetValidationError,
    dataset_provenance,
    extract_dataset,
    file_sha256,
    load_dataset_samples,
    read_dataset,
    utc_timestamp,
    validate_dataset,
    write_evidence_csv,
)
from .real_data_split import check_train_validation_leakage

MODEL_SCHEMA_VERSION = 1
MODEL_FORMAT = "matrix-real-likelihood-model"


def _validated_dataset(path: str | Path):
    validation = validate_dataset(path)
    if not validation["valid"]:
        raise DatasetValidationError(validation["errors"])
    return read_dataset(path)


def _prior(
    rows: list[dict[str, Any]],
    mode: str,
    explicit_path: str | Path | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    counts = Counter(
        row["ground_truth"] for row in rows
        if row["ground_truth"] in SUPERVISED_LABELS
    )
    if mode == "uniform":
        values = {label: 0.25 for label in SUPERVISED_LABELS}
        config = {"type": "uniform", "probabilities": values}
    elif mode == "empirical":
        n = sum(counts.values())
        values = {label: counts[label] / n for label in SUPERVISED_LABELS}
        config = {
            "type": "empirical_training_set",
            "probabilities": values,
            "warning": "class prevalence is the calibration-set prevalence, not assumed real-world prevalence",
        }
    elif mode == "explicit":
        if explicit_path is None:
            raise ValueError("--explicit-prior is required when --prior explicit")
        values = json.loads(Path(explicit_path).read_text(encoding="utf-8"))
        if set(values) != set(SUPERVISED_LABELS):
            raise ValueError(f"explicit prior must define exactly {', '.join(SUPERVISED_LABELS)}")
        values = {label: float(values[label]) for label in SUPERVISED_LABELS}
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("explicit prior values must be finite and nonnegative")
        total = sum(values.values())
        if total <= 0:
            raise ValueError("explicit prior must have positive total mass")
        values = {label: value / total for label, value in values.items()}
        config = {
            "type": "explicit_user_supplied",
            "source_path": str(explicit_path),
            "source_sha256": file_sha256(explicit_path),
            "probabilities": values,
        }
    else:
        raise ValueError("prior must be uniform, empirical, or explicit")
    return values, config


def _category(value: float | None, edges: list[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "MISSING"
    return f"bin_{int(np.digitize(float(value), edges, right=False))}"


def _calibrate_feature_model(
    rows: list[dict[str, Any]],
    alpha: float,
    bins: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labeled = [row for row in rows if row["ground_truth"] in SUPERVISED_LABELS]
    if not labeled:
        raise ValueError("no labeled H1-H4 samples available for calibration")
    edges_by_feature = {}
    likelihoods = {}
    categories_by_feature = {}
    for feature in EVIDENCE_FEATURES:
        finite_values = [
            float(row["features"][feature])
            for row in labeled
            if row["features"][feature] is not None
            and math.isfinite(float(row["features"][feature]))
        ]
        edges = (
            np.quantile(finite_values, np.linspace(0, 1, bins + 1)[1:-1]).tolist()
            if len(finite_values) >= 2 else []
        )
        edges = sorted({float(edge) for edge in edges})
        categories = {"MISSING", *(f"bin_{i}" for i in range(len(edges) + 1))}
        counts = {label: Counter() for label in SUPERVISED_LABELS}
        for row in labeled:
            counts[row["ground_truth"]][
                _category(row["features"][feature], edges)
            ] += 1
        total_categories = len(categories)
        likelihoods[feature] = {}
        for label in SUPERVISED_LABELS:
            denominator = sum(counts[label].values()) + alpha * total_categories
            likelihoods[feature][label] = {
                category: (counts[label][category] + alpha) / denominator
                for category in sorted(categories)
            }
        edges_by_feature[feature] = edges
        categories_by_feature[feature] = sorted(categories)

    return (
        {
            "bin_definitions": {
                feature: {
                    "method": "training-set empirical quantile thresholds",
                    "edges": edges_by_feature[feature],
                    "categories": categories_by_feature[feature],
                }
                for feature in EVIDENCE_FEATURES
            },
            "likelihoods": likelihoods,
        },
        {"edges": edges_by_feature},
    )


def calibrate(
    dataset_path: str | Path,
    out_dir: str | Path,
    *,
    prior_mode: str = "uniform",
    explicit_prior_path: str | Path | None = None,
    smoothing: float = 1.0,
    bins: int = 3,
    decision_threshold: float = 0.8,
) -> dict[str, Any]:
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing must be positive and finite")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if not 0 < decision_threshold <= 1:
        raise ValueError("decision threshold must be in (0, 1]")
    dataset = _validated_dataset(dataset_path)
    samples = load_dataset_samples(dataset)
    evidence_rows = extract_dataset(dataset)
    counts = Counter(
        row["ground_truth"] for row in evidence_rows
        if row["ground_truth"] in SUPERVISED_LABELS
    )
    missing_classes = [label for label in SUPERVISED_LABELS if counts[label] == 0]
    if missing_classes:
        raise ValueError(
            "calibration requires at least one labeled sample per H1-H4 class; "
            f"missing {', '.join(missing_classes)}"
        )
    prior_values, prior_config = _prior(evidence_rows, prior_mode, explicit_prior_path)
    model_features, bin_data = _calibrate_feature_model(evidence_rows, smoothing, bins)
    sample_provenance = dataset_provenance(dataset)
    # Per-sample provenance enables exact held-out overlap checks.
    sample_provenance["samples"] = {
        sample["sample_id"]: {
            "raw_file_hashes": list(sample["raw_hashes"].values()),
            "measurement_ids": list(sample["measurement_ids"].values()),
        }
        for sample in samples
    }
    model = {
        "format": MODEL_FORMAT,
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_id": f"real-{dataset.dataset_id}-{dataset.dataset_hash[:12]}",
        "training_domain": "real",
        "synthetic_data_used": False,
        "dataset_id": dataset.dataset_id,
        "dataset_hash": dataset.dataset_hash,
        "training_sample_count": sum(counts.values()),
        "dataset_sample_count": len(evidence_rows),
        "class_counts": {label: counts[label] for label in SUPERVISED_LABELS},
        "excluded_label_counts": {
            label: sum(row["ground_truth"] == label for row in evidence_rows)
            for label in ("UNRESOLVED", "UNKNOWN")
        },
        "feature_definitions": {
            "features": list(EVIDENCE_FEATURES),
            "by_check": CHECK_FEATURES,
            "availability": "missing measurements/check results are encoded as MISSING categories, never numeric zero",
        },
        "bin_definitions": model_features["bin_definitions"],
        "smoothing": {"method": "additive categorical Laplace smoothing", "alpha": smoothing},
        "likelihoods": model_features["likelihoods"],
        "prior": prior_config,
        "decision_threshold": decision_threshold,
        "training_provenance": sample_provenance,
        "calibration_timestamp_utc": utc_timestamp(),
        "calibration_version": "matrix-real-calibration-1",
    }
    output_path = Path(out_dir)
    if output_path.suffix.lower() == ".json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model_path = output_path
        evidence_path = output_path.with_name(f"{output_path.stem}_evidence.csv")
        summary_path = output_path.with_name(f"{output_path.stem}_calibration.json")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        model_path = output_path / "real_model.json"
        evidence_path = output_path / "evidence.csv"
        summary_path = output_path / "calibration.json"
    model["artifact_path"] = str(model_path)
    model_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    write_evidence_csv(evidence_rows, evidence_path)
    calibration_summary = {
        "model_id": model["model_id"],
        "training_domain": "real",
        "synthetic_data_used": False,
        "dataset_id": dataset.dataset_id,
        "dataset_hash": dataset.dataset_hash,
        "sample_count": model["dataset_sample_count"],
        "class_counts": model["class_counts"],
        "excluded_label_counts": model["excluded_label_counts"],
        "prior": model["prior"],
    }
    summary_path.write_text(
        json.dumps(calibration_summary, indent=2) + "\n", encoding="utf-8"
    )
    return model


def load_real_model(path: str | Path) -> dict[str, Any]:
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("format") != MODEL_FORMAT or model.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported or malformed real calibration model")
    if model.get("training_domain") != "real" or model.get("synthetic_data_used") is not False:
        raise ValueError("model rejected: real diagnosis accepts only real-domain models with synthetic_data_used=false")
    if set(model.get("likelihoods", {})) != set(EVIDENCE_FEATURES):
        raise ValueError("model rejected: feature definitions do not match this real-data implementation")
    if set(model.get("prior", {}).get("probabilities", {})) != set(SUPERVISED_LABELS):
        raise ValueError("model rejected: invalid prior configuration")
    for feature in EVIDENCE_FEATURES:
        if set(model["likelihoods"][feature]) != set(SUPERVISED_LABELS):
            raise ValueError(f"model rejected: invalid likelihood classes for {feature}")
    return model


def predict_real(model: dict[str, Any], checks: dict[str, Any]) -> dict[str, Any]:
    if model.get("training_domain") != "real" or model.get("synthetic_data_used") is not False:
        raise ValueError("real prediction rejected a non-real calibration model")
    from .real_dataset import _flatten

    features = _flatten(checks)
    log_probs = {
        label: (
            math.log(model["prior"]["probabilities"][label])
            if model["prior"]["probabilities"][label] > 0 else -math.inf
        )
        for label in SUPERVISED_LABELS
    }
    for feature in EVIDENCE_FEATURES:
        edge_info = model["bin_definitions"][feature]
        category = _category(features[feature], edge_info["edges"])
        for label in SUPERVISED_LABELS:
            if log_probs[label] == -math.inf:
                continue
            probability = model["likelihoods"][feature][label].get(category)
            if probability is None:
                raise ValueError(f"model missing likelihood category {category} for {feature}")
            log_probs[label] += math.log(max(float(probability), 1e-300))
    max_log = max(log_probs.values())
    probabilities = {label: math.exp(value - max_log) for label, value in log_probs.items()}
    normalizer = sum(probabilities.values())
    probabilities = {label: value / normalizer for label, value in probabilities.items()}
    best = max(SUPERVISED_LABELS, key=lambda label: probabilities[label])
    confidence = probabilities[best]
    diagnosis = best if confidence >= float(model["decision_threshold"]) else "UNRESOLVED"
    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "posterior": probabilities,
        "reason": (
            "real calibration posterior met the configured decision threshold"
            if diagnosis != "UNRESOLVED"
            else "real calibration posterior did not meet the configured decision threshold"
        ),
    }


def _check_leakage(model: dict[str, Any], dataset) -> None:
    training = model.get("training_provenance", {})
    check_train_validation_leakage(
        training_sample_ids=training.get("sample_ids", []),
        training_raw_hashes=training.get("raw_file_hashes", []),
        training_measurement_ids=training.get("measurement_ids", []),
        validation=dataset,
    )


def _wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _validation_metrics(rows, model, predictions):
    labeled = [
        (row, pred) for row, pred in zip(rows, predictions)
        if row["ground_truth"] in SUPERVISED_LABELS
    ]
    unresolved_truth = [
        (row, pred) for row, pred in zip(rows, predictions)
        if row["ground_truth"] == "UNRESOLVED"
    ]
    resolvable = labeled
    labels = [*SUPERVISED_LABELS, "UNRESOLVED"]
    matrix = {true: {pred: 0 for pred in labels} for true in labels}
    for row, pred in (*labeled, *unresolved_truth):
        matrix[row["ground_truth"]][pred["diagnosis"]] += 1
    correct = sum(row["ground_truth"] == pred["diagnosis"] for row, pred in labeled)
    resolvable_correct = sum(row["ground_truth"] == pred["diagnosis"] for row, pred in resolvable)
    recalls = {}
    recall_ci = {}
    for label in SUPERVISED_LABELS:
        members = [(row, pred) for row, pred in resolvable if row["ground_truth"] == label]
        hits = sum(pred["diagnosis"] == label for _, pred in members)
        recalls[label] = hits / len(members) if members else None
        recall_ci[label] = _wilson(hits, len(members))
    confirmations = [(row, pred) for row, pred in labeled if pred["diagnosis"] in SUPERVISED_LABELS]
    wrong_confirmations = sum(row["ground_truth"] != pred["diagnosis"] for row, pred in confirmations)
    wrong_all = sum(
        row["ground_truth"] != pred["diagnosis"] and pred["diagnosis"] in SUPERVISED_LABELS
        for row, pred in labeled
    )
    unresolved = sum(pred["diagnosis"] == "UNRESOLVED" for _, pred in labeled)

    probabilities = []
    targets = []
    for row, pred in resolvable:
        probabilities.append([pred["posterior"][label] for label in SUPERVISED_LABELS])
        targets.append([1.0 if row["ground_truth"] == label else 0.0 for label in SUPERVISED_LABELS])
    if probabilities:
        probs, truth = np.asarray(probabilities), np.asarray(targets)
        brier = float(np.mean(np.sum((probs - truth) ** 2, axis=1)))
        confidence = probs.max(axis=1)
        correct_pred = (probs.argmax(axis=1) == truth.argmax(axis=1)).astype(float)
        ece = 0.0
        for left in np.linspace(0, 1, 11)[:-1]:
            mask = (confidence >= left) & (confidence < left + 0.1)
            if mask.any():
                ece += float(mask.mean() * abs(correct_pred[mask].mean() - confidence[mask].mean()))
    else:
        brier = ece = None

    check_usage = {
        check: sum(row["availability"][check] for row in rows)
        for check in ("C1", "C2", "C3", "C4", "C5")
    }
    return {
        "eligible_labeled_sample_count": len(labeled),
        "unknown_unlabeled_excluded": sum(row["ground_truth"] == "UNKNOWN" for row in rows),
        "unresolved_ground_truth_excluded": len(unresolved_truth),
        "overall_accuracy": correct / len(labeled) if labeled else None,
        "overall_accuracy_95_ci": _wilson(correct, len(labeled)),
        "accuracy_on_resolvable": resolvable_correct / len(resolvable) if resolvable else None,
        "accuracy_on_resolvable_95_ci": _wilson(resolvable_correct, len(resolvable)),
        "resolvable_sample_count": len(resolvable),
        "per_class_recall": recalls,
        "per_class_recall_95_ci": recall_ci,
        "confusion_matrix": matrix,
        "wrong_confirmation_count": wrong_confirmations,
        "wrong_confirmation_rate_among_confirmations": (
            wrong_confirmations / len(confirmations) if confirmations else None
        ),
        "wrong_confirmation_rate_among_eligible": wrong_all / len(labeled) if labeled else None,
        "unresolved_count": unresolved,
        "unresolved_rate": unresolved / len(labeled) if labeled else None,
        "calibration": {
            "multiclass_brier_score_on_H1_H4_labeled": brier,
            "expected_calibration_error_10_bins_on_H1_H4_labeled": ece,
        },
        "per_check_usage_available_sample_count": check_usage,
        "decision_threshold": model["decision_threshold"],
    }


def validate_heldout(
    model_path: str | Path,
    dataset_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    model = load_real_model(model_path)
    dataset = _validated_dataset(dataset_path)
    _check_leakage(model, dataset)
    rows = extract_dataset(dataset)
    predictions = [predict_real(model, row["evidence"]) for row in rows]
    metrics = _validation_metrics(rows, model, predictions)
    result = {
        "training_domain": "real",
        "synthetic_data_used": False,
        "model_id": model["model_id"],
        "calibration_dataset_id": model["dataset_id"],
        "heldout_dataset_id": dataset.dataset_id,
        "heldout_dataset_hash": dataset.dataset_hash,
        "metrics": metrics,
        "sample_results": [
            {
                "sample_id": row["sample_id"],
                "ground_truth": row["ground_truth"],
                "prediction": pred["diagnosis"],
                "confidence": pred["confidence"],
                "posterior": pred["posterior"],
            }
            for row, pred in zip(rows, predictions)
        ],
    }
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_evidence_csv(rows, output / "evidence.csv")
    (output / "report.md").write_text(_render_validation_report(result), encoding="utf-8")
    return result


def _render_validation_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    accuracy = metrics["overall_accuracy"]
    lines = [
        "# Real-data held-out validation",
        "",
        f"- Calibration dataset: `{result['calibration_dataset_id']}`",
        f"- Held-out dataset: `{result['heldout_dataset_id']}`",
        f"- Eligible labeled samples: {metrics['eligible_labeled_sample_count']}",
        f"- Overall accuracy: {'not reported (no eligible labeled samples)' if accuracy is None else f'{accuracy:.3f}'}",
        f"- Resolvable accuracy: {metrics['accuracy_on_resolvable']}",
        f"- Wrong-confirmation rate: {metrics['wrong_confirmation_rate_among_confirmations']}",
        f"- Unresolved rate: {metrics['unresolved_rate']}",
        "",
        "## Confusion matrix",
        "",
        "| True \\ Predicted | " + " | ".join(metrics["confusion_matrix"]) + " |",
        "| --- | " + " | ".join("---:" for _ in metrics["confusion_matrix"]) + " |",
    ]
    for truth, row in metrics["confusion_matrix"].items():
        lines.append(f"| {truth} | " + " | ".join(str(row[label]) for label in row) + " |")
    lines.extend([
        "",
        "These metrics apply only to this held-out dataset and this real-calibrated model.",
        "They do not establish external or population-level validity.",
        "",
    ])
    return "\n".join(lines)
