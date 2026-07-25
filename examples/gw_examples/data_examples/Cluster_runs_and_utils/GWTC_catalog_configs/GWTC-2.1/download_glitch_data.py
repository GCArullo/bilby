#!/usr/bin/env python3

"""Download the public glitch-subtracted frames required by seven templates."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


RECORD_ID = 6477075
OUTPUT_DIR = Path(__file__).resolve().parent / "glitch_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event",
        action="append",
        help="Download only frames containing these GWTC-2.1 event IDs",
    )
    return parser.parse_args()


def gps_from_event(event: str) -> float:
    from prepare_gwtc21 import event_gps

    return event_gps(event)


def frame_interval(filename: str) -> tuple[int, int]:
    stem = filename.removesuffix(".gwf")
    start_text, duration_text = stem.rsplit("-", maxsplit=2)[-2:]
    start = int(start_text)
    return start, start + int(duration_text)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file_pointer:
        for block in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    with urllib.request.urlopen(
        f"https://zenodo.org/api/records/{RECORD_ID}"
    ) as response:
        record = json.load(response)
    files = [item for item in record["files"] if item["key"].endswith(".gwf")]
    if args.event:
        event_times = [gps_from_event(event) for event in args.event]
        files = [
            item
            for item in files
            if any(
                frame_interval(item["key"])[0] <= gps <= frame_interval(item["key"])[1]
                for gps in event_times
            )
        ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(files, start=1):
        output = OUTPUT_DIR / item["key"]
        expected = item["checksum"].removeprefix("md5:")
        if (
            output.is_file()
            and output.stat().st_size == item["size"]
            and md5(output) == expected
        ):
            print(f"[{index}/{len(files)}] verified {output.name}")
            continue
        print(f"[{index}/{len(files)}] downloading {output.name}")
        temporary = output.with_suffix(".gwf.part")
        with (
            urllib.request.urlopen(item["links"]["self"]) as response,
            temporary.open("wb") as file_pointer,
        ):
            while block := response.read(1024 * 1024):
                file_pointer.write(block)
        if temporary.stat().st_size != item["size"] or md5(temporary) != expected:
            raise RuntimeError(f"Checksum failed for {temporary}")
        temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
