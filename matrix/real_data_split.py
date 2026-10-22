"""Leakage checks and deterministic sample-level splits for real datasets."""

from __future__ import annotations

import random
from typing import Iterable

from .real_dataset import RealDataset


def deterministic_sample_split(
    dataset: RealDataset,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return calibration and validation sample IDs, keeping each sample intact.

    This helper is for development/testing only. Scientific evaluation should
    normally use separately supplied calibration and held-out datasets.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    sample_ids = sorted(sample.sample_id for sample in dataset.samples)
    if len(sample_ids) < 2:
        raise ValueError("at least two samples are required for a split")
    shuffled = sample_ids.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = min(
        len(shuffled) - 1,
        max(1, round(len(shuffled) * validation_fraction)),
    )
    validation_ids = set(shuffled[:validation_count])
    calibration = tuple(sample_id for sample_id in sample_ids if sample_id not in validation_ids)
    validation = tuple(sample_id for sample_id in sample_ids if sample_id in validation_ids)
    return calibration, validation


def check_train_validation_leakage(
    *,
    training_sample_ids: Iterable[str],
    training_raw_hashes: Iterable[str],
    training_measurement_ids: Iterable[str],
    validation: RealDataset,
) -> None:
    sample_overlap = set(training_sample_ids) & {
        sample.sample_id for sample in validation.samples
    }
    hash_overlap = set(training_raw_hashes) & set(validation.raw_hashes)
    measurement_overlap = set(training_measurement_ids) & set(validation.measurement_ids)
    issues = []
    if sample_overlap:
        issues.append(f"sample_id: {', '.join(sorted(sample_overlap))}")
    if hash_overlap:
        issues.append(f"raw-file hash: {len(hash_overlap)} overlapping file(s)")
    if measurement_overlap:
        issues.append(f"measurement ID: {', '.join(sorted(measurement_overlap))}")
    if issues:
        raise ValueError("train/validation leakage by " + "; ".join(issues))
