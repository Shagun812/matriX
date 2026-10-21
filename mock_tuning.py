import json
import sys
from pathlib import Path

from matrix.mock import mock_table, mock_cases, case_runner
from matrix.priors import PriorModel
from matrix.tuning import tune


def main():
    print("=" * 60)
    print("MaterialFalsify — Mock Development Tuning")
    print("=" * 60)

    prior_path = Path("artifacts/mock_prior.pkl")
    if not prior_path.exists():
        raise FileNotFoundError(
            f"{prior_path} not found. Train the mock prior first."
        )

    table = mock_table()
    cases = mock_cases(n=200, seed=42)
    prior = PriorModel.load(prior_path)

    print(f"Cases: {len(cases)}")
    print(f"Mock prior: {prior_path}")
    print("Tuning eps_gain and operating tau on mock data...\n")

    report = tune(
        table=table,
        cases=cases,
        runner_for=case_runner,
        prior_model=prior,
    )

    print("Operating tau:", report["operating_tau"])
    print("Target met:", report["target_met"])
    print("Selected eps_gain:", report["eps_gain"])
    print("Reference accuracy:", report["ref_acc"])
    print("\nGrid results:")
    for row in report["rows"]:
        print(json.dumps(row, sort_keys=True))

    out = Path("mock_run/tuning_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nReport saved to: {out}")
    print("WARNING: These are mock-only tuning parameters.")
    print("Do not use them to freeze the real benchmark configuration.")


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
