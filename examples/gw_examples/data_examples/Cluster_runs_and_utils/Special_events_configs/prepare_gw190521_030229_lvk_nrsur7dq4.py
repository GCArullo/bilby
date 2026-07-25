#!/usr/bin/env python3

"""Prepare the released LVK GW190521 NRSur7dq4 PSDs and calibration files."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import h5py
import numpy as np


URL = "https://dcc.ligo.org/public/0168/P2000158/004/GW190521_posterior_samples.h5"
EXPECTED_MD5 = "8af9bce0b55b5ebed7853dbfaa69a2d5"
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "data" / "GW190521_030229_LVK_NRSur7dq4"
ARCHIVE = OUTPUT_DIR / "GW190521_posterior_samples.h5"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file_pointer:
        for block in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download() -> None:
    if ARCHIVE.is_file() and md5(ARCHIVE) == EXPECTED_MD5:
        print(f"Verified {ARCHIVE}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ARCHIVE.with_suffix(".h5.part")
    print(f"Downloading {URL}")
    with (
        urllib.request.urlopen(URL) as response,
        temporary.open("wb") as file_pointer,
    ):
        while block := response.read(1024 * 1024):
            file_pointer.write(block)
    if md5(temporary) != EXPECTED_MD5:
        raise RuntimeError(f"Checksum failed for {temporary}")
    temporary.replace(ARCHIVE)


def extract() -> None:
    with h5py.File(ARCHIVE, "r") as h5_file:
        run = h5_file["NRSur7dq4"]
        psd_dir = OUTPUT_DIR / "psds"
        calibration_dir = OUTPUT_DIR / "calibration"
        psd_dir.mkdir(parents=True, exist_ok=True)
        calibration_dir.mkdir(parents=True, exist_ok=True)
        for detector in ("H1", "L1", "V1"):
            np.savetxt(
                psd_dir / f"{detector}.dat",
                np.asarray(run[f"psds/{detector}"], dtype=float),
                fmt="%.12e",
                header="f psd(f)",
            )
            np.savetxt(
                calibration_dir / f"{detector}.txt",
                np.asarray(run[f"calibration_envelope/{detector}"], dtype=float),
                fmt="%.12e",
                header=(
                    "f amp_median phase_median amp_lower phase_lower "
                    "amp_upper phase_upper"
                ),
            )
    print(f"Prepared NRSur7dq4 products under {OUTPUT_DIR}")


def main() -> int:
    download()
    extract()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
