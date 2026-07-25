#!/usr/bin/env python3

"""Extract GWTC-4 XPHM-SpinTaylor settings as nocosmo run templates."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parent
COMMON_PATH = ROOT.parent / "GWTC-2.1" / "prepare_gwtc21.py"
SPEC = importlib.util.spec_from_file_location("prepare_igwn_catalog", COMMON_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {COMMON_PATH}")
common = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = common
SPEC.loader.exec_module(common)

RECORD_IDS = (17602505,)
EVENT_PATTERN = re.compile(
    r"-(GW\d{6}_\d{6})-combined_PEDataRelease\.hdf5$"
)
SELECTED_RUN = "C00:IMRPhenomXPHM-SpinTaylor"
FALLBACK_RUNS = {
    "GW230529_181500": "C00:IMRPhenomXPHM:LowSpin",
    "GW240925_005809": "C01:IMRPhenomXPHM-SpinTaylor",
}
CATALOG_NAME = "GWTC-4"
DOCUMENTATION_URL = "https://gwosc.org/GWTC-4.0/"


def fetch_record() -> list[dict[str, object]]:
    events = []
    for record_id in RECORD_IDS:
        with urllib.request.urlopen(
            f"https://zenodo.org/api/records/{record_id}"
        ) as response:
            record = json.load(response)
        for item in record["files"]:
            match = EVENT_PATTERN.search(item["key"])
            if not match:
                continue
            events.append(
                {
                    "record_id": record_id,
                    "record_event": match.group(1),
                    "filename": item["key"],
                    "checksum": item["checksum"],
                    "size": item["size"],
                    "url": (
                        f"https://zenodo.org/records/{record_id}/files/"
                        f"{item['key']}?download=1"
                    ),
                }
            )
    return sorted(events, key=lambda item: item["record_event"])


def select_run(h5_file: h5py.File, record_event: str) -> str:
    selected = FALLBACK_RUNS.get(record_event, SELECTED_RUN)
    if selected not in h5_file:
        raise RuntimeError(f"{selected} is absent for {record_event}")
    return selected


base_make_prior = common.make_prior


def make_prior(run: h5py.Group, record_event: str) -> tuple[str, bool]:
    text, reconstructed = base_make_prior(run, record_event)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("luminosity_distance = "):
            continue
        minimum = re.search(r"minimum=([0-9.eE+-]+)", line)
        maximum = re.search(r"maximum=([0-9.eE+-]+)", line)
        if minimum is None or maximum is None:
            raise RuntimeError(
                f"Unable to read luminosity-distance bounds for {record_event}"
            )
        lines[index] = (
            "luminosity_distance = PowerLaw("
            f"alpha=2, minimum={minimum.group(1)}, maximum={maximum.group(1)}, "
            "name='luminosity_distance', latex_label='$d_L$', "
            "unit='Mpc', boundary=None)"
        )
        return "\n".join(lines) + "\n", reconstructed
    raise RuntimeError(f"No luminosity-distance prior found for {record_event}")


base_prepare_event = common.prepare_event


def prepare_event(item: dict[str, object], local_h5_dir: Path | None):
    for attempt in range(1, 6):
        try:
            return base_prepare_event(item, local_h5_dir)
        except Exception as error:
            rate_limited = getattr(error, "status", None) == 429 or "429" in str(error)
            if not rate_limited or attempt == 5:
                raise
            print(
                f"Zenodo rate limit reached; retrying in 60 seconds "
                f"({attempt}/5)",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(60)


def make_manifest(events: list[dict[str, object]]) -> None:
    manifest = {
        "record_ids": list(RECORD_IDS),
        "record_urls": [
            f"https://zenodo.org/records/{record_id}" for record_id in RECORD_IDS
        ],
        "events": events,
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def make_html(_events: list[dict[str, object]]) -> None:
    return None


def configure() -> None:
    common.RECORD_ID = RECORD_IDS[0]
    common.REMOTE_BLOCK_SIZE = 1024 * 1024
    common.SOURCE_CONFIG_SCOPE = "selected"
    common.ROOT = ROOT
    common.EVENT_PATTERN = EVENT_PATTERN
    common.GWTC1_PREFIXES = ()
    common.GLITCH_FRAMES = {}
    common.LOW_FREQUENCY_MITIGATION = {}
    common.TRIGGER_TIME_OVERRIDES = {}
    common.SOURCES = {
        "release": f"https://zenodo.org/records/{RECORD_IDS[0]}",
        "documentation": DOCUMENTATION_URL,
    }
    common.fetch_record = fetch_record
    common.select_run = select_run
    common.make_prior = make_prior
    common.prepare_event = prepare_event
    common.make_manifest = make_manifest
    common.make_html = make_html
    catalog_number = CATALOG_NAME.removeprefix("GWTC-")
    common.__file__ = str(ROOT / f"prepare_gwtc{catalog_number}.py")


configure()


if __name__ == "__main__":
    raise SystemExit(common.main())
