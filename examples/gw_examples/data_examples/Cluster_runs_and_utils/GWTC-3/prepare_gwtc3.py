#!/usr/bin/env python3

"""Extract GWTC-3 nocosmo configs and make bilby_pipe run templates.

This reuses the GWTC-2.1 extraction code while selecting the corresponding
GWTC-3 v2 release, event names, waveform runs, and de-glitched frames.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import time
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

RECORD_ID = 8177023
EVENT_PATTERN = re.compile(
    r"v2-(GW\d{6}_\d{6})_PEDataRelease_mixed_nocosmo\.h5$"
)
LOW_SPIN_EVENTS = {"GW191219_163120", "GW200115_042309"}
GLITCH_DATA = {
    "GW191105_143521": {
        "V1": (
            "V-V1Online_T1700406_v4-1256998000-2000.gwf",
            "Hrec_hoft_16384Hz_T1700406_v4",
        ),
    },
    "GW191109_010717": {
        "H1": (
            "H-H1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1257296641-3327.gwf",
            "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01_T1700406_v4",
        ),
        "L1": (
            "L-L1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1257295872-4096.gwf",
            "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01_T1700406_v4",
        ),
    },
    "GW191113_071753": {
        "H1": (
            "H-H1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1257664512-4096.gwf",
            "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01_T1700406_v4",
        ),
    },
    "GW191127_050227": {
        "H1": (
            "H-H1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1258864640-4096.gwf",
            "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01_T1700406_v4",
        ),
    },
    "GW191219_163120": {
        "H1": (
            "H-H1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1260806144-4096.gwf",
            "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01_T1700406_v4",
        ),
        "L1": (
            "L-L1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1260806144-4096.gwf",
            "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01_T1700406_v4",
        ),
    },
    "GW200105_162426": {
        "L1": (
            "L-L1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1262276608-1440.gwf",
            "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01_T1700406_v4",
        ),
    },
    "GW200115_042309": {
        "L1": (
            "L-L1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4-1263095808-4096.gwf",
            "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01_T1700406_v4",
        ),
    },
    "GW200129_065458": {
        "L1": (
            "L-L1_HOFT_CLEAN_SUB60HZ_C01_P1800169_v4-1264314068-4096.gwf",
            "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01_P1800169_v4",
        ),
    },
}


def select_run(h5_file: h5py.File, record_event: str) -> str:
    suffix = ":LowSpin" if record_event in LOW_SPIN_EVENTS else ""
    selected = f"C01:IMRPhenomXPHM{suffix}"
    if selected not in h5_file:
        raise RuntimeError(f"{selected} is absent for {record_event}")
    return selected


base_make_template = common.make_template


def make_template(
    event: str,
    record_event: str,
    run_name: str,
    settings: dict[str, object],
    psd_detectors: list[str],
    calibration_detectors: list[str],
) -> str:
    text = base_make_template(
        event,
        record_event,
        run_name,
        settings,
        psd_detectors,
        calibration_detectors,
    )
    special_data = GLITCH_DATA.get(record_event)
    if not special_data:
        return text
    channels = {detector: "GWOSC" for detector in settings["detectors"]}
    data = {}
    for detector, (filename, channel) in special_data.items():
        channels[detector] = channel
        data[detector] = f"__WORKING_DIRECTORY__/glitch_data/{filename}"
    text = common.replace_setting(text, "channel-dict", common.mapping_text(channels))
    return common.replace_setting(text, "data-dict", common.mapping_text(data))


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


def make_html(_events: list[dict[str, object]]) -> None:
    return None


common.RECORD_ID = RECORD_ID
common.ROOT = ROOT
common.EVENT_PATTERN = EVENT_PATTERN
common.GWTC1_PREFIXES = ()
common.GLITCH_FRAMES = {}
common.LOW_FREQUENCY_MITIGATION = {}
common.TRIGGER_TIME_OVERRIDES = {}
common.SOURCES = {
    "release": "https://zenodo.org/records/8177023",
    "paper": "https://dcc.ligo.org/LIGO-P2000318/public",
    "glitches": "https://zenodo.org/records/5546680",
    "documentation": "https://gwosc.org/GWTC-3/",
}
common.select_run = select_run
common.make_template = make_template
common.prepare_event = prepare_event
common.make_html = make_html
common.__file__ = __file__


if __name__ == "__main__":
    raise SystemExit(common.main())
