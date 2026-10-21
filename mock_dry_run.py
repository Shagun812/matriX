#!/usr/bin/env python3
"""Smoke-test the MaterialFalsify agent pipeline using mock data."""

import importlib
import sys
import traceback

MODULES = (
    "matrix.schemas",
    "matrix.planner",
    "matrix.agents",
    "matrix.features",
    "matrix.priors",
    "matrix.run_arms",
    "matrix.tuning",
    "matrix.killgate",
)

def main():
    print("=" * 60)
    print("MaterialFalsify — mock dry run")
    print("=" * 60)

    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"[OK] {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    if failed:
        print("\nDry run could not complete. Missing modules or dependencies:")
        for name, exc in failed:
            print(f"  - {name}: {exc}")
        print(
            "\nThis smoke test checks imports only; it does not yet execute "
            "agents against mock cases."
        )
        sys.exit(1)

    print("\nAll required modules imported successfully.")
    print(
        "Import smoke test passed. A full mock rehearsal requires the "
        "project's mock fixtures and runner API."
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
