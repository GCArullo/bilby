#!/usr/bin/env python3

"""Extract GW150914 calibration and PSD products from the IGWN C01 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR.parent
    / "GW150914"
    / "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_nocosmo.h5"
)
DEFAULT_RUN_GROUP = "C01:IMRPhenomXPHM"
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR.parent
    / "GW150914"
    / "Data"
    / "GW150914_C01_IMRPhenomXPHM"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the IGWN HDF5 file.",
    )
    parser.add_argument(
        "--run-group",
        default=DEFAULT_RUN_GROUP,
        help="Run group inside the HDF5 file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where calibration and PSD files are written.",
    )
    return parser.parse_args()


def extract_products(
    input_h5: Path,
    run_group: str,
    output_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    if not input_h5.is_file():
        raise FileNotFoundError(f"Missing input file: {input_h5}")

    calibration_dir = output_dir / "calibration"
    psd_dir = output_dir / "psds"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    psd_dir.mkdir(parents=True, exist_ok=True)

    calibration_paths: dict[str, str] = {}
    psd_paths: dict[str, str] = {}

    with h5py.File(input_h5, "r") as input_file:
        if run_group not in input_file:
            raise KeyError(f"Run group '{run_group}' not found in {input_h5}")
        run = input_file[run_group]

        if "calibration_envelope" not in run:
            raise KeyError(f"'{run_group}/calibration_envelope' is missing")
        if "psds" not in run:
            raise KeyError(f"'{run_group}/psds' is missing")

        calibration_group = run["calibration_envelope"]
        psd_group = run["psds"]
        detectors = sorted(set(calibration_group.keys()) & set(psd_group.keys()))
        if not detectors:
            raise RuntimeError(
                "No overlapping detectors in calibration_envelope and psds"
            )

        for detector in detectors:
            calibration_path = (calibration_dir / f"{detector}.txt").resolve()
            psd_path = (psd_dir / f"{detector}.dat").resolve()

            np.savetxt(
                calibration_path,
                np.asarray(calibration_group[detector][:], dtype=float),
                header=(
                    "f amp_median phase_median "
                    "amp_lower amp_upper phase_lower phase_upper"
                ),
            )
            np.savetxt(
                psd_path,
                np.asarray(psd_group[detector][:], dtype=float),
                header="f psd(f)",
            )

            calibration_paths[detector] = str(calibration_path)
            psd_paths[detector] = str(psd_path)

    return calibration_paths, psd_paths


def main() -> int:
    args = parse_args()
    calibration_paths, psd_paths = extract_products(
        input_h5=args.input_h5.expanduser().resolve(),
        run_group=args.run_group,
        output_dir=args.output_dir.expanduser().resolve(),
    )

    print("Extracted calibration files:")
    for detector, path in calibration_paths.items():
        print(f"  {detector}: {path}")
    print("Extracted PSD files:")
    for detector, path in psd_paths.items():
        print(f"  {detector}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
