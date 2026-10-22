"""Generate a concise report from saved MatriX cases and traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from statistics import median
from dataclasses import asdict
from pathlib import Path

from .config import load_frozen
from .evaluation import (
    accuracy_cost_curve,
    curve_auc,
    evaluate_arm,
    load_cases,
    load_traces,
)
from .planner import LikelihoodTable, Planner
from .schemas import CHECKS


ARMS = ("B0", "B1", "B2", "B3", "B3b", "B4", "B5")


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _format(value, digits: int = 3) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _render_markdown(report: dict) -> str:
    lines = [
        "# Development / Evaluation Report",
        "",
        "Generated from simulated development cases and saved agent traces. "
        "This is not a report from real user material measurements.",
        "",
        f"- Cases: {report['n_cases']}",
        f"- Decision threshold: {report['tau_stop']}",
        f"- Cases file: `{report['cases_path']}`",
        f"- Traces directory: `{report['traces_dir']}`",
        "",
        "| Arm | Cases | Resolvable accuracy | Wrong confirmations | Unresolved | Mean cost | Median cost | Mean checks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = report["arms"].get(arm)
        if row is None:
            lines.append(f"| {arm} | — | unavailable | — | — | — | — | — |")
            continue
        lines.append(
            f"| {arm} | {row['n_cases']} | "
            f"{_format(None if row['accuracy'] is None else row['accuracy'] * 100, 1)}% "
            f"| {row['wrong_confirmations']} | {row['unresolved']} "
            f"| {_format(row['mean_cost'])} | {_format(row['median_cost'])} "
            f"| {_format(row['mean_checks'])} |"
        )

    lines.extend(["", "## Check usage", "", "| Arm | " + " | ".join(CHECKS) + " |",
                  "| --- | " + " | ".join("---:" for _ in CHECKS) + " |"])
    for arm, row in report["arms"].items():
        lines.append(
            f"| {arm} | " + " | ".join(str(row["check_usage"][check]) for check in CHECKS) + " |"
        )

    lines.extend(["", "## Stopping reasons", "",
                  "| Arm | " + " | ".join(report["stop_reasons"]) + " |",
                  "| --- | " + " | ".join("---:" for _ in report["stop_reasons"]) + " |"])
    for arm, row in report["arms"].items():
        lines.append(
            f"| {arm} | " + " | ".join(
                str(row["stop_reasons"].get(reason, 0)) for reason in report["stop_reasons"]
            ) + " |"
        )

    lines.extend(["", "## Accuracy-cost curves", "",
                  "| Arm | Threshold | Resolvable accuracy | Mean cost |",
                  "| --- | ---: | ---: | ---: |"])
    for arm, points in report["curves"].items():
        for point in points:
            accuracy = point["accuracy"]
            lines.append(
                f"| {arm} | {_format(point['tau_stop'])} "
                f"| {_format(None if accuracy is None else accuracy * 100, 1)}% "
                f"| {_format(point['mean_cost'])} |"
            )

    lines.extend(["", "## Accuracy-cost comparison", "",
                  "Descriptive AUC over each pair's common observed cost interval; "
                  "no new inferential statistical tests were run.",
                  "", "| Comparison | Delta AUC | Common cost interval |",
                  "| --- | ---: | ---: |"])
    for comparison in report["accuracy_cost_comparisons"]:
        lo, hi = comparison["common_cost_interval"]
        interval = (
            f"{_format(lo)}–{_format(hi)}"
            if not comparison["degenerate"] and lo is not None and hi is not None
            else "no overlapping nonzero cost interval"
        )
        lines.append(
            f"| {comparison['arm']} vs {comparison['baseline']} "
            f"| {_format(comparison['delta_auc'])} | {interval} |"
        )
    missing = [arm for arm in ARMS if arm not in report["arms"]]
    if missing:
        lines.extend([
            "",
            "## Unavailable arms",
            "",
            "No complete set of matching traces was found for: " + ", ".join(missing) + ".",
        ])
    return "\n".join(lines) + "\n"


def generate_report(
    cases_path: str | Path,
    traces_dir: str | Path,
    table_path: str | Path,
    frozen_path: str | Path,
    out_dir: str | Path,
    tau: float | None = None,
) -> dict:
    cases_path = Path(cases_path)
    traces_dir = Path(traces_dir)
    table_path = Path(table_path)
    frozen_path = Path(frozen_path)
    out_dir = Path(out_dir)

    cases = load_cases(cases_path)
    frozen = load_frozen(frozen_path)
    threshold = frozen.operating_tau if tau is None else tau
    tau_grid = frozen.tau_grid or [threshold]
    planner = Planner(
        LikelihoodTable.load(table_path),
        costs=frozen.costs,
        eps_gain=frozen.eps_gain,
    )

    summaries = {}
    curves = {}
    for arm in ARMS:
        arm_dir = traces_dir / arm
        if not arm_dir.is_dir():
            continue
        traces = load_traces(traces_dir, arm)
        case_ids = {case.case_id for case in cases}
        missing_ids = case_ids - traces.keys()
        if missing_ids:
            continue

        metrics, summary = evaluate_arm(
            cases, traces, threshold, planner, costs=frozen.costs, budget=frozen.budget
        )
        cost_values = [metric.cost for metric in metrics]
        check_usage = {check: 0 for check in CHECKS}
        stop_reasons = {}
        for metric in metrics:
            trace = traces[metric.case_id]
            stop_reasons[trace.stop_reason.value] = (
                stop_reasons.get(trace.stop_reason.value, 0) + 1
            )
            for step in trace.steps[1 : metric.n_checks + 1]:
                if step.action in check_usage:
                    check_usage[step.action] += 1
        summaries[arm] = {
            "n_cases": summary.n_cases,
            "n_resolvable": summary.n_resolvable,
            "accuracy": summary.resolvable_accuracy,
            "wrong_confirmations": sum(metric.false_diagnosis for metric in metrics),
            "unresolved": sum(metric.unresolved_positive for metric in metrics),
            "mean_cost": summary.mean_cost,
            "median_cost": float(median(cost_values)),
            "mean_checks": summary.mean_checks,
            "check_usage": check_usage,
            "stop_reasons": stop_reasons,
        }
        curve, _ = accuracy_cost_curve(
            cases,
            traces,
            planner,
            tau_grid=tau_grid,
            costs=frozen.costs,
            budget=frozen.budget,
        )
        curves[arm] = curve

    if not summaries:
        raise ValueError(
            f"no arm has complete traces matching all {len(cases)} cases "
            f"from {cases_path} under {traces_dir}"
        )

    comparisons = []
    baseline = "B0" if "B0" in curves else next(iter(curves))
    for arm, curve in curves.items():
        if arm == baseline:
            continue
        result = curve_auc(curves[baseline], curve)
        comparisons.append({
            "baseline": baseline,
            "arm": arm,
            "auc_baseline": result.auc_a,
            "auc_arm": result.auc_b,
            "delta_auc": result.delta_auc,
            "common_cost_interval": (
                [result.common_cost_lo, result.common_cost_hi]
                if not result.degenerate else [None, None]
            ),
            "degenerate": result.degenerate,
        })

    report = _json_safe({
        "report_type": "Development / Evaluation Report",
        "n_cases": len(cases),
        "tau_stop": threshold,
        "cases_path": str(cases_path),
        "traces_dir": str(traces_dir),
        "table_path": str(table_path),
        "frozen_path": str(frozen_path),
        "arms": summaries,
        "stop_reasons": sorted({
            reason
            for row in summaries.values()
            for reason in row["stop_reasons"]
        }),
        "curves": {arm: [asdict(point) for point in curve] for arm, curve in curves.items()},
        "accuracy_cost_comparisons": comparisons,
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "report.md").write_text(_render_markdown(report))
    with (out_dir / "arm_summary.csv").open("w", newline="") as file:
        fields = [
            "arm", "n_cases", "n_resolvable", "accuracy", "wrong_confirmations",
            "unresolved", "mean_cost", "median_cost", "mean_checks",
            *(f"{check}_usage" for check in CHECKS),
            *(f"stop_{reason.lower()}" for reason in report["stop_reasons"]),
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for arm, row in summaries.items():
            writer.writerow({
                "arm": arm,
                **{key: row[key] for key in fields if key in row},
                **{f"{check}_usage": row["check_usage"][check] for check in CHECKS},
                **{
                    f"stop_{reason.lower()}": row["stop_reasons"].get(reason, 0)
                    for reason in report["stop_reasons"]
                },
            })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="report existing development data and write artifacts/dev_report",
    )
    parser.add_argument("--cases", default="artifacts/dev_cases.json")
    parser.add_argument("--traces-dir", default="artifacts/dev_runs")
    parser.add_argument("--table", default="artifacts/likelihood_table_50k.npz")
    parser.add_argument("--frozen", default="artifacts/frozen_dev.json")
    parser.add_argument("--tau", type=float)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    out_dir = args.out or "artifacts/dev_report"
    try:
        report = generate_report(
            args.cases, args.traces_dir, args.table, args.frozen, out_dir, args.tau
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print("matriX final report")
    print(f"Cases: {report['n_cases']}")
    for arm in ARMS:
        row = report["arms"].get(arm)
        if row is None:
            print(f"{arm:<4} unavailable")
        else:
            print(
                f"{arm:<4} accuracy="
                f"{_format(None if row['accuracy'] is None else row['accuracy'] * 100, 1)}% "
                f"wrong={row['wrong_confirmations']} unresolved={row['unresolved']} "
                f"mean_cost={_format(row['mean_cost'])}"
            )
    print(f"Report written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
