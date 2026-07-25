#!/usr/bin/env python3

"""Print a compact HTCondor snapshot for bilby_pipe analysis runs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


DZ_RE = re.compile(r"dlogz:\s*([^\s>\]]+)\s*>\s*([^\s\]]+)")
LABEL_RE = re.compile(r"(?:^|\s)--label\s+(\S+)")
FATAL_RE = re.compile(
    r"(fatal|error|exception|killed|segmentation fault|traceback)", re.IGNORECASE
)
STATUS = {
    1: "IDLE",
    2: "RUNNING",
    3: "REMOVED",
    4: "DONE",
    5: "HELD",
    6: "RUNNING",
    7: "SUSPENDED",
}
SYMBOL = {
    "RUNNING": "●",
    "IDLE": "○",
    "WAITING": "·",
    "HELD": "◆",
    "FAILED": "✖",
    "DONE": "✓",
    "REMOVED": "✖",
    "SUSPENDED": "◆",
}
COLOUR = {
    "RUNNING": "\033[32m",
    "IDLE": "\033[36m",
    "WAITING": "\033[90m",
    "HELD": "\033[33m",
    "FAILED": "\033[31m",
    "DONE": "\033[34m",
    "REMOVED": "\033[31m",
    "SUSPENDED": "\033[33m",
}
RESET = "\033[0m"


def condor_json(command: list[str]) -> list[dict]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(message or f"{command[0]} failed")
    return json.loads(result.stdout or "[]")


def analysis_label(ad: dict) -> str | None:
    arguments = ad.get("Args") or ""
    if "bilby_pipe_analysis" not in arguments:
        return None
    match = LABEL_RE.search(arguments)
    return match.group(1) if match else None


def tail_text(path: Path, size: int = 2_000_000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode(errors="replace")
    except OSError:
        return ""


def latest_dz(run_dir: Path, label: str) -> str:
    matches = []
    for suffix in ("out", "err"):
        path = run_dir / "log_data_analysis" / f"{label}.{suffix}"
        found = DZ_RE.findall(tail_text(path))
        if found:
            matches.append((path.stat().st_mtime, *found[-1]))
    if not matches:
        return "starting; no dZ yet"
    _, value, target = max(matches)
    return f"dZ {value}  (target {target})"


def fatal_error(run_dir: Path, label: str, ad: dict) -> str:
    error_path = ad.get("Err")
    path = Path(error_path) if error_path else None
    if path and not path.is_absolute():
        path = Path(ad.get("Iwd") or run_dir.parent) / path
    if not path or not path.is_file():
        path = run_dir / "log_data_analysis" / f"{label}.err"

    lines = tail_text(path).replace("\r", "\n").splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line and FATAL_RE.search(line) and "warning" not in line.lower():
            return line[:180]

    if ad.get("ExitBySignal"):
        return f"terminated by signal {ad.get('ExitSignal', 'unknown')}"
    return f"nonzero exit code {ad.get('ExitCode', 'unknown')}"


def latest_ads(ads: list[dict], root: Path) -> dict[str, dict]:
    selected = {}
    for ad in ads:
        if Path(ad.get("Iwd") or "").resolve() != root:
            continue
        label = analysis_label(ad)
        if not label:
            continue
        key = (ad.get("ClusterId", 0), ad.get("ProcId", 0))
        previous = selected.get(label)
        previous_key = (
            previous.get("ClusterId", 0),
            previous.get("ProcId", 0),
        ) if previous else (-1, -1)
        if key > previous_key:
            selected[label] = ad
    return selected


def active_roots(ads: list[dict]) -> list[Path]:
    roots = {
        Path(ad["Iwd"]).resolve()
        for ad in ads
        if ad.get("Iwd")
    }
    return sorted(
        root
        for root in roots
        if any(root.glob("*/submit/*_analysis_*_par*.submit"))
    )


def paint(state: str, text: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{COLOUR[state]}{text}{RESET}"


def print_snapshot(
    root: Path,
    queued: list[dict],
    history: list[dict],
    colour: bool,
) -> None:
    submit_files = sorted(root.glob("*/submit/*_analysis_*_par*.submit"))
    queue_by_label = latest_ads(queued, root)
    history_by_label = latest_ads(history, root)
    rows = []
    for submit_file in submit_files:
        label = submit_file.stem
        run_dir = submit_file.parents[1]
        job = label.rsplit("_", 1)[-1]
        ad = queue_by_label.get(label)
        if ad:
            state = STATUS.get(ad.get("JobStatus"), "UNKNOWN")
            if state == "RUNNING":
                detail = latest_dz(run_dir, label)
            elif state in {"HELD", "SUSPENDED"}:
                detail = ad.get("HoldReason") or "no hold reason reported"
            elif state == "REMOVED":
                detail = ad.get("RemoveReason") or "removed from queue"
            else:
                detail = "waiting for a slot"
        else:
            ad = history_by_label.get(label)
            failed = ad and (
                ad.get("ExitBySignal") or ad.get("ExitCode") not in (None, 0)
            )
            if failed:
                state = "FAILED"
                detail = fatal_error(run_dir, label, ad)
            elif ad:
                state = "DONE"
                detail = "completed successfully"
            else:
                state = "WAITING"
                detail = "waiting for DAG dependencies"
        rows.append((run_dir.name, job, state, detail))

    counts = Counter(row[2] for row in rows)
    summary_order = ("RUNNING", "IDLE", "HELD", "FAILED", "DONE", "WAITING")
    summary = "  ".join(
        f"{name.lower()} {counts[name]}" for name in summary_order if counts[name]
    )
    print(f"\nBilby run snapshot  ·  {root}")
    print(f"{len(rows)} analysis jobs  ·  {summary}\n")

    name_width = max(len(row[0]) for row in rows)
    current_run = None
    for run_name, job, state, detail in rows:
        if current_run is not None and run_name != current_run:
            print()
        current_run = run_name
        marker = paint(state, f"{SYMBOL[state]} {state:<7}", colour)
        print(f"{run_name:<{name_width}}  {job:<4}  {marker}  {detail}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show the state and latest dZ of bilby_pipe analysis jobs."
    )
    parser.add_argument(
        "run_directory",
        nargs="?",
        type=Path,
        help="specific run root to inspect (default: discover active roots from Condor)",
    )
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    attributes = (
        "ClusterId,ProcId,JobStatus,HoldReason,ExitCode,ExitBySignal,"
        "ExitSignal,RemoveReason,Iwd,Args,Out,Err"
    )
    try:
        queued = condor_json(
            ["condor_q", "-json", "-attributes", attributes]
        )
        history = condor_json(
            [
                "condor_history",
                "-limit",
                "10000",
                "-json",
                "-attributes",
                attributes,
            ]
        )
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"monitor_runs.py: unable to query HTCondor: {error}", file=sys.stderr)
        return 1

    if args.run_directory:
        roots = [args.run_directory.expanduser().resolve()]
        if not any(roots[0].glob("*/submit/*_analysis_*_par*.submit")):
            parser.error(f"no analysis submit files found below {roots[0]}")
    else:
        roots = active_roots(queued)
        if not roots:
            print("No active bilby_pipe run directories found in your Condor queue.")
            return 0

    colour = sys.stdout.isatty() and not args.no_color
    for root in roots:
        print_snapshot(root, queued, history, colour)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
