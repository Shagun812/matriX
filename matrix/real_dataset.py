"""Dataset schema, provenance, leakage validation, and evidence extraction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .real_data import RealDataCheckRunner, load_real_sample


REAL_LABELS = {"H1", "H2", "H3", "H4", "UNRESOLVED", "UNKNOWN"}
SUPERVISED_LABELS = ("H1", "H2", "H3", "H4")
EVIDENCE_FEATURES = (
    "C1.ratio_difference",
    "C1.standardized_difference",
    "C2.sample_displacement_deg",
    "C2.shift_uncertainty_deg",
    "C2.fit_rmse",
    "C3.anatase_fraction",
    "C3.rutile_fraction",
    "C3.fit_residual",
    "C4.anatase.primary_Eg",
    "C4.anatase.alternate_delta_Eg",
    "C4.anatase.primary_rmse",
    "C4.rutile.primary_Eg",
    "C4.rutile.alternate_delta_Eg",
    "C4.rutile.primary_rmse",
    "C5.composition_O_to_Ti_delta",
    "C5.xrd_shift_delta_deg",
    "C5.optical_anatase_delta_Eg",
    "C5.optical_rutile_delta_Eg",
)
CHECK_FEATURES = {
    "C1": EVIDENCE_FEATURES[0:2],
    "C2": EVIDENCE_FEATURES[2:5],
    "C3": EVIDENCE_FEATURES[5:8],
    "C4": EVIDENCE_FEATURES[8:14],
    "C5": EVIDENCE_FEATURES[14:18],
}


class DatasetValidationError(ValueError):
    def __init__(self, issues: list[dict[str, str]]):
        self.issues = issues
        super().__init__(
            "dataset validation failed:\n"
            + "\n".join(
                f"- {issue.get('sample_id', 'dataset')}: {issue['message']}"
                for issue in issues
            )
        )


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    ground_truth: str
    ground_truth_confidence: float | None
    paths: dict[str, str]
    metadata_path: str
    provenance: dict[str, Any]
    measurement_ids: dict[str, str]
    resolved_paths: dict[str, Path]


@dataclass(frozen=True)
class RealDataset:
    path: Path
    dataset_id: str
    samples: tuple[DatasetSample, ...]
    manifest: dict[str, Any]
    dataset_hash: str
    raw_hashes: frozenset[str]
    measurement_ids: frozenset[str]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sample_paths(root: Path, raw: dict[str, Any]) -> tuple[dict[str, str], str, dict[str, Path]]:
    measurement_fields = ("composition", "xrd", "optical")
    missing = [name for name in (*measurement_fields, "metadata") if not raw.get(name)]
    if missing:
        raise ValueError(f"missing required measurements or metadata: {', '.join(missing)}")
    paths = {name: str(raw[name]) for name in measurement_fields}
    paths["metadata"] = str(raw["metadata"])
    resolved = {name: _resolve(root, value, name) for name, value in paths.items()}

    repeat = raw.get("repeat")
    if repeat is not None:
        if not isinstance(repeat, dict):
            raise ValueError("repeat must be a mapping of measurement paths")
        for key in measurement_fields:
            if repeat.get(key):
                name = f"repeat_{key}"
                paths[name] = str(repeat[key])
                resolved[name] = _resolve(root, str(repeat[key]), name)
    if raw.get("xrd_reference"):
        paths["xrd_reference"] = str(raw["xrd_reference"])
        resolved["xrd_reference"] = _resolve(root, str(raw["xrd_reference"]), "xrd_reference")
    return paths, str(raw["metadata"]), resolved


def read_dataset(path: str | Path) -> RealDataset:
    path = Path(path).resolve()
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: dataset must be a YAML mapping")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError(f"{path}: dataset_id must be a non-empty string")
    rows = manifest.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: samples must be a non-empty list")

    samples = []
    raw_hashes: set[str] = set()
    measurement_ids: set[str] = set()
    issues = []
    seen_ids: set[str] = set()
    seen_file_hashes: dict[str, str] = {}
    seen_measurement_ids: dict[str, str] = {}
    for index, raw in enumerate(rows):
        sample_id = raw.get("sample_id", f"row-{index}") if isinstance(raw, dict) else f"row-{index}"
        try:
            if not isinstance(raw, dict):
                raise ValueError("sample entry must be a mapping")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError("sample_id must be a non-empty string")
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id {sample_id!r}")
            seen_ids.add(sample_id)

            label = raw.get("ground_truth")
            label = "UNKNOWN" if label is None else str(label).upper()
            if label not in REAL_LABELS:
                raise ValueError(f"invalid ground_truth {label!r}")
            if "ground_truth_confidence" not in raw:
                raise ValueError("ground_truth_confidence field is required (use null if unknown)")
            confidence = raw.get("ground_truth_confidence")
            if confidence is not None:
                confidence = float(confidence)
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("ground_truth_confidence must be between 0 and 1")
            if not isinstance(raw.get("provenance"), dict):
                raise ValueError("provenance mapping is required")
            provenance = raw["provenance"]
            if not provenance.get("source") or not provenance.get("measurement_ids"):
                raise ValueError("provenance requires source and measurement_ids")
            paths, metadata_path, resolved = _sample_paths(path.parent, raw)
            ids = provenance["measurement_ids"]
            if not isinstance(ids, dict) or not ids:
                raise ValueError("provenance.measurement_ids must map measurement types to IDs")
            parsed_ids = {}
            for measurement, identifier in ids.items():
                if not isinstance(identifier, str) or not identifier.strip():
                    raise ValueError(f"measurement ID for {measurement} must be non-empty")
                if identifier in seen_measurement_ids:
                    raise ValueError(
                        f"duplicate measurement ID {identifier!r} also used by "
                        f"{seen_measurement_ids[identifier]}"
                    )
                seen_measurement_ids[identifier] = sample_id
                measurement_ids.add(identifier)
                parsed_ids[str(measurement)] = identifier

            for name, raw_path in resolved.items():
                if not raw_path.is_file():
                    raise ValueError(f"{name} file does not exist: {raw_path}")
                if name in {"metadata", "xrd_reference"}:
                    continue
                digest = file_sha256(raw_path)
                previous = seen_file_hashes.get(digest)
                if previous:
                    raise ValueError(
                        f"duplicate raw-file leakage: {name} file matches {previous}"
                    )
                seen_file_hashes[digest] = f"{sample_id}:{name}"
                raw_hashes.add(digest)
            if raw.get("material") != "tio2":
                raise ValueError("material must be tio2")
            samples.append(
                DatasetSample(
                    sample_id=sample_id,
                    ground_truth=label,
                    ground_truth_confidence=confidence,
                    paths=paths,
                    metadata_path=metadata_path,
                    provenance=provenance,
                    measurement_ids=parsed_ids,
                    resolved_paths=resolved,
                )
            )
        except (TypeError, ValueError, OSError) as exc:
            issues.append({"sample_id": str(sample_id), "message": str(exc)})
    if issues:
        raise DatasetValidationError(issues)
    canonical = {
        "dataset_id": dataset_id,
        "manifest_sha256": file_sha256(path),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "ground_truth": sample.ground_truth,
                "raw_hashes": {
                    key: file_sha256(value)
                    for key, value in sample.resolved_paths.items()
                    if key not in {"metadata", "xrd_reference"}
                },
                "metadata_hash": file_sha256(sample.resolved_paths["metadata"]),
                "reference_hash": (
                    file_sha256(sample.resolved_paths["xrd_reference"])
                    if "xrd_reference" in sample.resolved_paths else None
                ),
                "measurement_ids": sample.measurement_ids,
            }
            for sample in samples
        ],
    }
    dataset_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return RealDataset(
        path=path,
        dataset_id=dataset_id,
        samples=tuple(samples),
        manifest=manifest,
        dataset_hash=dataset_hash,
        raw_hashes=frozenset(raw_hashes),
        measurement_ids=frozenset(measurement_ids),
    )


def validate_dataset(path: str | Path) -> dict[str, Any]:
    try:
        dataset = read_dataset(path)
        issues = []
        for entry in dataset.samples:
            try:
                sample = load_real_sample(
                    composition_path=entry.resolved_paths["composition"],
                    xrd_path=entry.resolved_paths["xrd"],
                    optical_path=entry.resolved_paths["optical"],
                    metadata_path=entry.resolved_paths["metadata"],
                    repeat_composition_path=entry.resolved_paths.get("repeat_composition"),
                    repeat_xrd_path=entry.resolved_paths.get("repeat_xrd"),
                    repeat_optical_path=entry.resolved_paths.get("repeat_optical"),
                )
                if sample.sample_id != entry.sample_id:
                    raise ValueError(
                        f"metadata sample_id {sample.sample_id!r} does not match dataset sample_id {entry.sample_id!r}"
                    )
                for check_id in ("C1", "C2", "C3", "C4", "C5"):
                    RealDataCheckRunner(
                        sample, entry.resolved_paths.get("xrd_reference")
                    ).run(check_id)
            except (ValueError, OSError, KeyError) as exc:
                issues.append({"sample_id": entry.sample_id, "message": str(exc)})
        return {
            "valid": not issues,
            "dataset_id": dataset.dataset_id,
            "dataset_hash": dataset.dataset_hash,
            "sample_count": len(dataset.samples),
            "label_counts": _label_counts(dataset),
            "errors": issues,
        }
    except (DatasetValidationError, OSError, ValueError, yaml.YAMLError) as exc:
        errors = exc.issues if isinstance(exc, DatasetValidationError) else [
            {"sample_id": "dataset", "message": str(exc)}
        ]
        return {"valid": False, "errors": errors}


def _label_counts(dataset: RealDataset) -> dict[str, int]:
    return {
        label: sum(sample.ground_truth == label for sample in dataset.samples)
        for label in ("H1", "H2", "H3", "H4", "UNRESOLVED", "UNKNOWN")
    }


def _flatten(evidence: dict[str, Any]) -> dict[str, float | None]:
    c1, c2, c3, c4, c5 = (evidence[name] for name in ("C1", "C2", "C3", "C4", "C5"))
    c4_models = c4.get("models", {})
    c4_anatase = c4_models.get("anatase", {})
    c4_rutile = c4_models.get("rutile", {})
    c5_delta = c5.get("deltas", {})
    optical_delta = c5_delta.get("optical_delta_band_gap_ev", {})
    xrd = c2.get("sample_displacement_deg")
    c3_fractions = c3.get("relative_crystalline_fraction") or {}
    # The C3 runner reports fractions at the top level.
    anatase = c3.get("anatase", c3_fractions.get("anatase"))
    rutile = c3.get("rutile", c3_fractions.get("rutile"))
    return {
        EVIDENCE_FEATURES[0]: c1.get("difference") if c1.get("status") == "AVAILABLE" else None,
        EVIDENCE_FEATURES[1]: c1.get("standardized_difference") if c1.get("status") == "AVAILABLE" else None,
        EVIDENCE_FEATURES[2]: xrd,
        EVIDENCE_FEATURES[3]: c2.get("shift_uncertainty_deg"),
        EVIDENCE_FEATURES[4]: (
            c2.get("fit_quality", {}).get("rmse_normalized_intensity")
            if c2.get("status") == "AVAILABLE" else None
        ),
        EVIDENCE_FEATURES[5]: anatase,
        EVIDENCE_FEATURES[6]: rutile,
        EVIDENCE_FEATURES[7]: (
            c3.get("fit_quality", {}).get("rmse_normalized_intensity")
            if c3.get("status") == "AVAILABLE" else None
        ),
        EVIDENCE_FEATURES[8]: c4_anatase.get("primary", {}).get("band_gap_ev"),
        EVIDENCE_FEATURES[9]: c4_anatase.get("delta_band_gap_ev"),
        EVIDENCE_FEATURES[10]: c4_anatase.get("primary", {}).get("rmse"),
        EVIDENCE_FEATURES[11]: c4_rutile.get("primary", {}).get("band_gap_ev"),
        EVIDENCE_FEATURES[12]: c4_rutile.get("delta_band_gap_ev"),
        EVIDENCE_FEATURES[13]: c4_rutile.get("primary", {}).get("rmse"),
        EVIDENCE_FEATURES[14]: c5_delta.get("composition_O_to_Ti_delta"),
        EVIDENCE_FEATURES[15]: c5_delta.get("xrd_shift_delta_deg"),
        EVIDENCE_FEATURES[16]: optical_delta.get("anatase"),
        EVIDENCE_FEATURES[17]: optical_delta.get("rutile"),
    }


def extract_sample(entry: DatasetSample) -> dict[str, Any]:
    sample = load_real_sample(
        composition_path=entry.resolved_paths["composition"],
        xrd_path=entry.resolved_paths["xrd"],
        optical_path=entry.resolved_paths["optical"],
        metadata_path=entry.resolved_paths["metadata"],
        repeat_composition_path=entry.resolved_paths.get("repeat_composition"),
        repeat_xrd_path=entry.resolved_paths.get("repeat_xrd"),
        repeat_optical_path=entry.resolved_paths.get("repeat_optical"),
    )
    if sample.sample_id != entry.sample_id:
        raise ValueError(
            f"metadata sample_id {sample.sample_id!r} does not match dataset sample_id {entry.sample_id!r}"
        )
    runner = RealDataCheckRunner(sample, entry.resolved_paths.get("xrd_reference"))
    evidence = {check: runner.run(check) for check in ("C1", "C2", "C3", "C4", "C5")}
    return {
        "sample_id": entry.sample_id,
        "ground_truth": entry.ground_truth,
        "ground_truth_confidence": entry.ground_truth_confidence,
        "features": _flatten(evidence),
        "availability": {
            check: evidence[check].get("status") == "AVAILABLE"
            for check in ("C1", "C2", "C3", "C4", "C5")
        },
        "uncertainties": {
            "C1.O_to_Ti_standard_uncertainty": evidence["C1"].get("standard_uncertainty"),
            "C2.shift_uncertainty_deg": evidence["C2"].get("shift_uncertainty_deg"),
        },
        "evidence": evidence,
    }


def extract_dataset(dataset: RealDataset) -> list[dict[str, Any]]:
    rows = []
    for entry in dataset.samples:
        try:
            rows.append(extract_sample(entry))
        except (ValueError, OSError, KeyError) as exc:
            raise DatasetValidationError(
                [{"sample_id": entry.sample_id, "message": str(exc)}]
            ) from exc
    return rows


def write_evidence_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    fields = ["sample_id", "ground_truth", "ground_truth_confidence"]
    fields += [f"{check}_available" for check in ("C1", "C2", "C3", "C4", "C5")]
    fields += list(EVIDENCE_FEATURES)
    fields += ["C1.O_to_Ti_standard_uncertainty"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sample_id": row["sample_id"],
                "ground_truth": row["ground_truth"],
                "ground_truth_confidence": row["ground_truth_confidence"],
                **{f"{check}_available": row["availability"][check] for check in ("C1", "C2", "C3", "C4", "C5")},
                **row["features"],
                "C1.O_to_Ti_standard_uncertainty": (
                    row["evidence"]["C1"].get("standard_uncertainty")
                ),
            })


def dataset_provenance(dataset: RealDataset) -> dict[str, Any]:
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_hash": dataset.dataset_hash,
        "sample_ids": sorted(sample.sample_id for sample in dataset.samples),
        "raw_file_hashes": sorted(dataset.raw_hashes),
        "measurement_ids": sorted(dataset.measurement_ids),
    }


def load_dataset_samples(dataset: RealDataset) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample.sample_id,
            "ground_truth": sample.ground_truth,
            "ground_truth_confidence": sample.ground_truth_confidence,
            "provenance": sample.provenance,
            "measurement_ids": sample.measurement_ids,
            "raw_hashes": {
                name: file_sha256(path)
                for name, path in sample.resolved_paths.items()
                if name not in {"metadata", "xrd_reference"}
            },
        }
        for sample in dataset.samples
    ]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
