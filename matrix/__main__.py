"""Top-level matriX command dispatcher."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"diagnose", "real-diagnose"}:
        from .real_cli import main as diagnose_main

        return diagnose_main(args[1:])
    calibration_commands = {
        "real-dataset-validate",
        "real-validate-dataset",
        "real-calibrate",
        "real-validate",
        "real-evaluate",
    }
    if args and args[0] in calibration_commands:
        from .real_calibration_cli import main as calibration_main

        return calibration_main(args)
    parser = argparse.ArgumentParser(prog="python -m matrix")
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "diagnose",
            "real-diagnose",
            "real-dataset-validate",
            "real-validate-dataset",
            "real-calibrate",
            "real-validate",
            "real-evaluate",
        ],
    )
    parser.print_help()
    return 0 if not args or args in (["--help"], ["-h"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
