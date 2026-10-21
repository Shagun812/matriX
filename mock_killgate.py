import json
import sys
from pathlib import Path

from matrix.mock import mock_table, mock_cases, case_runner
from matrix.killgate import run_all


def main():
    print("=" * 60)
    print("MaterialFalsify — Planner Kill Gate (MOCK)")
    print("=" * 60)

    table = mock_table()
    cases = mock_cases(n=200, seed=42)
    print(f"Mock development cases: {len(cases)}")
    print("Likelihood table: mock table")
    print("Check runner: deterministic mock runner")
    print("\nRunning four planner-side checks...\n")

    report = run_all(table, cases, case_runner)

    for name, result in report.items():
        if name == "all_pass":
            continue
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"[{status}] {name}")
        print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("OVERALL:", "PASS" if report["all_pass"] else "FAIL")
    print("=" * 60)

    out = Path("mock_run/killgate_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report saved to: {out}")

    if not report["all_pass"]:
        print(
            "\nAt least one mock kill-gate check failed. "
            "Inspect its result before proceeding."
        )
        return 1

    print(
        "\nAll planner-side mock checks passed. "
        "This is not validation on the real simulator."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(2)
