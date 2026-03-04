#!/usr/bin/env python3

"""Generate and optionally submit Student-t bilby_pipe runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


LABEL_PREFIX = "GW231123_t_Student"
OUTDIR_BASE = "/home/gregorio.carullo/GW231123/t_Student/Runs"
WEBDIR_BASE = "/home/gregorio.carullo/public_html/GW231123/t_Student/Runs"
DEFAULT_DETECTORS = ("H1", "L1")

def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc

    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_argument_parser(script_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate bilby_pipe ini/prior files for Student-t runs and optionally "
            "submit them."
        )
    )
    parser.add_argument(
        "num_frequency_bands",
        type=positive_int,
        help=(
            "Positive integer. In single mode this is the exact band count. "
            "In range mode this is the maximum band count."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--single",
        action="store_true",
        help="Generate and submit only the requested band count (default).",
    )
    mode.add_argument(
        "--range",
        dest="range_mode",
        action="store_true",
        help="Generate and submit runs for every band count from 1 to N.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate files but do not call bilby_pipe.",
    )
    parser.add_argument(
        "--detector-dependent-nu",
        action="store_true",
        help="Generate detector-specific Student-t nu parameters.",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=list(DEFAULT_DETECTORS),
        help=(
            "Detector names used when building detector-dependent nu priors. "
            "Defaults to H1 L1."
        ),
    )
    parser.add_argument(
        "--ini-dir",
        type=Path,
        default=script_dir / "ini_files",
        help="Directory where generated ini files are written.",
    )
    parser.add_argument(
        "--prior-dir",
        type=Path,
        default=script_dir / "Priors",
        help="Directory where generated prior files are written.",
    )
    return parser


def load_template(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing template: {path}")
    return path.read_text(encoding="utf-8")


def build_nu_priors(
    band_count: int,
    *,
    detector_dependent_nu: bool = False,
    detectors: list[str] | tuple[str, ...] = DEFAULT_DETECTORS,
) -> str:
    if not detector_dependent_nu:
        if band_count == 1:
            return "nu = Uniform(name='nu', minimum=2.1, maximum=1000)"
        return "\n".join(
            f"nu_{index} = Uniform(name='nu_{index}', minimum=2.1, maximum=1000)"
            for index in range(1, band_count + 1)
        )
    return "\n".join(
        (
            f"nu_{detector} = Uniform(name='nu_{detector}', minimum=2.1, maximum=1000)"
            if band_count == 1
            else f"nu_{detector}_{index} = Uniform(name='nu_{detector}_{index}', minimum=2.1, maximum=1000)"
        )
        for detector in detectors
        for index in range(1, band_count + 1)
    )


def render_prior(
    prior_template: str,
    band_count: int,
    *,
    detector_dependent_nu: bool = False,
    detectors: list[str] | tuple[str, ...] = DEFAULT_DETECTORS,
) -> str:
    return prior_template.replace(
        "__NU_PRIORS__",
        build_nu_priors(
            band_count,
            detector_dependent_nu=detector_dependent_nu,
            detectors=detectors,
        ),
    )


def render_ini(
    ini_template: str,
    *,
    label: str,
    outdir: str,
    webdir: str,
    prior_file: Path,
    band_count: int,
    detector_dependent_nu: bool,
) -> str:
    replacements = {
        "__LABEL__": label,
        "__OUTDIR__": outdir,
        "__WEBDIR__": webdir,
        "__PRIOR_FILE__": str(prior_file),
        "__NUM_FREQUENCY_BANDS__": str(band_count),
        "__DETECTOR_DEPENDENT_NU__": str(detector_dependent_nu),
    }
    rendered = ini_template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def prepare_run(
    *,
    band_count: int,
    ini_template: str,
    prior_template: str,
    ini_dir: Path,
    prior_dir: Path,
    detector_dependent_nu: bool,
    detectors: list[str] | tuple[str, ...],
) -> Path:
    mode_suffix = "_detector_dependent_nu" if detector_dependent_nu else ""
    label = f"{LABEL_PREFIX}{mode_suffix}_N{band_count}"
    run_outdir = f"{OUTDIR_BASE}/student{mode_suffix}_N{band_count}"
    run_webdir = f"{WEBDIR_BASE}/student{mode_suffix}_N{band_count}"
    prior_path = (prior_dir / f"GW231123{mode_suffix}_N{band_count}.prior").resolve()
    ini_path = (ini_dir / f"GW231123_t_student{mode_suffix}_N{band_count}.ini").resolve()

    prior_path.write_text(
        render_prior(
            prior_template,
            band_count,
            detector_dependent_nu=detector_dependent_nu,
            detectors=detectors,
        ),
        encoding="utf-8",
    )
    ini_path.write_text(
        render_ini(
            ini_template,
            label=label,
            outdir=run_outdir,
            webdir=run_webdir,
            prior_file=prior_path,
            band_count=band_count,
            detector_dependent_nu=detector_dependent_nu,
        ),
        encoding="utf-8",
    )

    print(f"Prepared N={band_count}:")
    print(f"  prior: {prior_path}")
    print(f"  ini:   {ini_path}")
    return ini_path


def submit_run(ini_path: Path) -> None:
    subprocess.run(["bilby_pipe", str(ini_path), "--submit"], check=True)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    args = build_argument_parser(script_dir).parse_args()

    ini_template_path = script_dir / "GW231123_t_student_template.ini"
    prior_template_path = script_dir / "GW231123_template.prior"

    ini_template = load_template(ini_template_path)
    prior_template = load_template(prior_template_path)

    ini_dir = args.ini_dir.expanduser().resolve()
    prior_dir = args.prior_dir.expanduser().resolve()
    ini_dir.mkdir(parents=True, exist_ok=True)
    prior_dir.mkdir(parents=True, exist_ok=True)

    if args.range_mode:
        band_counts = range(1, args.num_frequency_bands + 1)
    else:
        band_counts = [args.num_frequency_bands]

    for band_count in band_counts:
        ini_path = prepare_run(
            band_count=band_count,
            ini_template=ini_template,
            prior_template=prior_template,
            ini_dir=ini_dir,
            prior_dir=prior_dir,
            detector_dependent_nu=args.detector_dependent_nu,
            detectors=args.detectors,
        )
        if not args.dry_run:
            submit_run(ini_path)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
