"""Command-line entry point for real-measurement workflows."""

from __future__ import annotations

import argparse

from .real_data import diagnose


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m matrix diagnose",
        description="Create an evidence-first TiO2 report from supplied measurements.",
    )
    parser.add_argument("--composition", required=True, help="CSV: element,value,uncertainty,unit")
    parser.add_argument("--xrd", required=True, help="CSV: two_theta,intensity")
    parser.add_argument("--optical", required=True, help="CSV: x,signal")
    parser.add_argument("--metadata", required=True, help="YAML specifying sample and measurement metadata")
    parser.add_argument(
        "--xrd-reference",
        help="optional validated reference profile CSV: two_theta,anatase_intensity,rutile_intensity",
    )
    parser.add_argument("--repeat-composition", help="optional independent repeat composition CSV")
    parser.add_argument("--repeat-xrd", help="optional independent repeat XRD CSV")
    parser.add_argument("--repeat-optical", help="optional independent repeat optical CSV")
    parser.add_argument("--model", help="optional real-data-calibrated model JSON")
    parser.add_argument(
        "--out",
        default="results/matrix-real-demo",
        help="output directory (default: results/matrix-real-demo)",
    )
    args = parser.parse_args(argv)
    try:
        result = diagnose(
            composition_path=args.composition,
            xrd_path=args.xrd,
            optical_path=args.optical,
            metadata_path=args.metadata,
            out_dir=args.out,
            xrd_reference=args.xrd_reference,
            repeat_composition_path=args.repeat_composition,
            repeat_xrd_path=args.repeat_xrd,
            repeat_optical_path=args.repeat_optical,
            model_path=args.model,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("matriX TiO2 evidence report")
    print(f"Sample: {result['sample']['sample_id']}")
    diagnosis = result["diagnosis"]
    qualifier = (
        "real-calibrated model"
        if diagnosis.get("model_domain") == "real"
        else "no real-calibrated model"
    )
    print(f"Diagnosis: {diagnosis['status']} ({qualifier})")
    for check, evidence in result["evidence"].items():
        print(f"{check}: {evidence.get('status', 'reported')}")
    print(f"Report written to: {args.out}")
    return 0
