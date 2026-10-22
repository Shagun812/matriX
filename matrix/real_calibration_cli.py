"""CLI commands for real dataset validation, calibration, and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .real_calibration import calibrate, validate_heldout
from .real_dataset import validate_dataset


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        args = ["--help"]
    command, *command_args = args
    parser = argparse.ArgumentParser(prog=f"python -m matrix {command}")
    if command in {"real-dataset-validate", "real-validate-dataset"}:
        parser.add_argument("--dataset", "--manifest", dest="dataset", required=True)
        parser.add_argument("--out", help="optional output directory for validation.json")
        parsed = parser.parse_args(command_args)
        result = validate_dataset(parsed.dataset)
        print(json.dumps(result, indent=2))
        if parsed.out:
            output = Path(parsed.out)
            output.mkdir(parents=True, exist_ok=True)
            (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
        return 0 if result["valid"] else 1
    if command == "real-calibrate":
        parser.add_argument("--dataset", "--manifest", dest="dataset", required=True)
        parser.add_argument("--out", required=True)
        parser.add_argument("--prior", choices=["uniform", "empirical", "explicit"], default="uniform")
        parser.add_argument("--explicit-prior", help="JSON file with H1,H2,H3,H4 probabilities")
        parser.add_argument("--smoothing", type=float, default=1.0)
        parser.add_argument("--bins", type=int, default=3)
        parser.add_argument("--decision-threshold", type=float, required=True)
        parsed = parser.parse_args(command_args)
        try:
            validation = validate_dataset(parsed.dataset)
            if validation.get("synthetic_fixture"):
                raise ValueError("synthetic software-test fixtures cannot be used for calibration")
            result = calibrate(
                parsed.dataset,
                parsed.out,
                prior_mode=parsed.prior,
                explicit_prior_path=parsed.explicit_prior,
                smoothing=parsed.smoothing,
                bins=parsed.bins,
                decision_threshold=parsed.decision_threshold,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(f"Real-data calibration model written to: {result['artifact_path']}")
        print(f"Dataset: {result['dataset_id']} ({result['training_sample_count']} labeled H1-H4 samples)")
        print(f"Prior: {result['prior']['type']}")
        return 0
    if command in {"real-validate", "real-evaluate"}:
        parser.add_argument("--model", required=True)
        parser.add_argument("--dataset", "--manifest", dest="dataset", required=True)
        parser.add_argument("--out", required=True)
        parsed = parser.parse_args(command_args)
        try:
            validation = validate_dataset(parsed.dataset)
            if validation.get("synthetic_fixture"):
                raise ValueError("synthetic software-test fixtures cannot be used for held-out validation")
            result = validate_heldout(parsed.model, parsed.dataset, parsed.out)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        metrics = result["metrics"]
        accuracy = metrics["overall_accuracy"]
        print(f"Real-data validation written to: {parsed.out}/validation.json")
        print(
            "Overall accuracy: "
            + ("not reported (no eligible labeled samples)" if accuracy is None else f"{accuracy:.3f}")
        )
        return 0
    parser.error(
        "command must be real-validate-dataset, real-dataset-validate, "
        "real-calibrate, real-evaluate, or real-validate"
    )
    return 2
