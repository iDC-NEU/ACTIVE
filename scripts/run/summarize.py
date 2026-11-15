#!/usr/bin/env python3
"""Summarize ACTIVE logs without third-party dependencies."""
import argparse
import csv
import json
from pathlib import Path
import re

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
SEARCH_L = re.compile(r"SEARCH_L\s*:\s*(\d+)")
SEARCH_STATE = re.compile(r"Start Search\(\s*(\d+)\s+iter\)")
LATENCY = re.compile(r"search time:\s*(" + NUMBER + r")")
ACCURACY = re.compile(r"(\d+)\s+NN accuracy:\s*(" + NUMBER + r")")
ROUND = re.compile(r"=+\s*ROUND\s+(\d+)\s*=+")
UPDATE = re.compile(r"\b(insert|delete)\s+(\d+)\s+entries\s+TOTAL\s+(?:INSERT|DELETE)\s+TIME:\s*("
                    + NUMBER + r")s\s+.*?\bOPS\s+(" + NUMBER + r")", re.I)


def parse_logs(sources, expected_rounds=10):
    result = {"search": [], "delete": {}, "insert": {}, "warnings": []}
    states = {"delete": {}, "insert": {}}
    updates = {"delete": {}, "insert": {}}
    warnings = result["warnings"]
    for text, hint in sources:
        mode, state, trace_round, search_l, latency = hint, None, None, None, None
        for line in text.splitlines():
            if "=== Static search" in line:
                mode = "search"
            elif "=== Delete-only" in line:
                mode = "delete"
            elif "=== Insert-only" in line:
                mode = "insert"
            elif "=== Build " in line:
                mode = None
            if match := ROUND.search(line):
                trace_round = int(match[1])
            if match := SEARCH_STATE.search(line):
                state = int(match[1])
            if match := SEARCH_L.search(line):
                search_l, latency = int(match[1]), None
            if match := LATENCY.search(line):
                latency = float(match[1]) * 1000
            if match := ACCURACY.search(line):
                k, recall = int(match[1]), float(match[2])
                row = {"L": search_l, "K": k, "recall": recall, "latency_ms": latency}
                if mode == "search":
                    result["search"].append(row)
                elif mode in states and state is not None:
                    if state in states[mode]:
                        warnings.append(f"Duplicate {mode} search state {state}; using latest value")
                    states[mode][state] = row
                else:
                    warnings.append("Search result could not be assigned to a stage/state")
            if match := UPDATE.search(line):
                operation, count, seconds, ops = match[1].lower(), int(match[2]), float(match[3]), float(match[4])
                if trace_round is None:
                    warnings.append(f"{operation} OPS has no trace round")
                    continue
                updates[operation][trace_round] = {
                    "round": trace_round + 1, "trace_round": trace_round,
                    "count": count, "update_seconds": seconds, "ops_per_second": ops,
                }
    if not result["search"]:
        warnings.append("No static search results found")
    for mode in ("delete", "insert"):
        rows = []
        for r in range(expected_rounds):
            update = updates[mode].get(r)
            before, after = states[mode].get(r, {}), states[mode].get(r + 1, {})
            row = {"round": r + 1, "trace_round": r, "count": None,
                   "update_seconds": None, "ops_per_second": None}
            if update:
                row.update(update)
            else:
                warnings.append(f"Missing {mode} update metrics for trace round {r:02d}")
            row.update({"recall_before": before.get("recall"), "recall_after": after.get("recall"),
                        "latency_after_ms": after.get("latency_ms"),
                        "L": after.get("L"), "K": after.get("K")})
            if not after:
                warnings.append(f"Missing {mode} post-update search state {r + 1}")
            rows.append(row)
        count = sum(row["count"] or 0 for row in rows)
        seconds = sum(row["update_seconds"] or 0 for row in rows)
        result[mode] = {"initial_state": states[mode].get(0), "rounds": rows,
                        "aggregate_ops_per_second": count / seconds if seconds else None}
        if not states[mode].get(0):
            warnings.append(f"Missing {mode} initial search state")
    return result


def format_summary(result):
    def fmt(value, places=6):
        return "N/A" if value is None else f"{value:.{places}f}"
    lines = ["ACTIVE run summary", f"Run exit code: {result.get('run_exit_code', 'unknown')}",
             "Recall is a fraction in [0,1]; latency is milliseconds per query.",
             "Update rounds are 1-based; recall_after is measured AFTER that update.",
             "OPS is the program-reported throughput of the timed update section.", "", "Static search", "L\tK\tRecall\tLatency_ms"]
    for row in result["search"]:
        lines.append(f"{row['L']}\t{row['K']}\t{fmt(row['recall'])}\t{fmt(row['latency_ms'])}")
    for mode in ("delete", "insert"):
        block = result[mode]
        initial = block["initial_state"] or {}
        lines += ["", mode.capitalize(), f"Initial recall: {fmt(initial.get('recall'))}",
                  "Round\tOPS/s\tRecall_before\tRecall_after\tLatency_after_ms"]
        for row in block["rounds"]:
            lines.append(f"{row['round']}\t{fmt(row['ops_per_second'], 3)}\t"
                         f"{fmt(row['recall_before'])}\t{fmt(row['recall_after'])}\t{fmt(row['latency_after_ms'])}")
        lines.append(f"Aggregate throughput (total operations / total update seconds): {fmt(block['aggregate_ops_per_second'], 3)} OPS/s")
    if result["warnings"]:
        lines += ["", "Warnings (missing data is N/A, never zero):", *result["warnings"]]
    return "\n".join(lines) + "\n"


def write_outputs(result, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "summary.txt").write_text(format_summary(result), encoding="utf-8")
    for name, rows, fields in (
        ("search", result["search"], ["L", "K", "recall", "latency_ms"]),
        *((mode, result[mode]["rounds"], ["round", "trace_round", "count", "update_seconds",
           "ops_per_second", "recall_before", "recall_after", "latency_after_ms", "L", "K"])
          for mode in ("delete", "insert")),
    ):
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, help="Log directory or one combined run log")
    parser.add_argument("--output", type=Path, help="Default: log directory (or input file parent)")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--run-exit-code", type=int)
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    sources = []
    if args.logs.is_file():
        sources.append((args.logs.read_text(encoding="utf-8", errors="replace"), None))
        default_output = args.logs.parent
    elif args.logs.is_dir():
        for mode in ("search", "delete", "insert"):
            path = args.logs / f"{mode}.log"
            if path.exists():
                sources.append((path.read_text(encoding="utf-8", errors="replace"), mode))
        if not sources and (args.logs / "run_all.log").exists():
            sources.append(((args.logs / "run_all.log").read_text(encoding="utf-8", errors="replace"), None))
        default_output = args.logs
    else:
        parser.error(f"Input does not exist: {args.logs}")
    if not sources:
        # Also useful after compilation/preflight failure: emit an explicit incomplete report.
        sources = [("", None)]
    result = parse_logs(sources, args.rounds)
    result["run_exit_code"] = args.run_exit_code
    write_outputs(result, args.output or default_output)
    print(format_summary(result), end="")
    print(f"Logs and summary: {(args.output or default_output).resolve()}")


if __name__ == "__main__":
    main()
