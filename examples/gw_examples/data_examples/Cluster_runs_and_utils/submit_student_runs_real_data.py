#!/usr/bin/env python3

"""Generate and optionally submit Student-t bilby_pipe runs.

Optionally, Gaussian-likelihood runs can be generated and submitted alongside
the Student-t runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DETECTORS = ("H1", "L1")
DEFAULT_EVENT = "GW231123"


@dataclass(frozen=True)
class EventDefaults:
    label_prefix: str
    outdir_base: str
    webdir_base: str
    file_prefix: str
    ini_template: str
    prior_template: str
    detectors: tuple[str, ...]


EVENT_DEFAULTS: dict[str, EventDefaults] = {
    "GW231123": EventDefaults(
        label_prefix="GW231123_t_Student",
        outdir_base="/home/gregorio.carullo/GW231123/t_Student/Runs",
        webdir_base="/home/gregorio.carullo/public_html/GW231123/t_Student/Runs",
        file_prefix="GW231123",
        ini_template="Initialisation_file_templates/GW231123_t_student_template.ini",
        prior_template="Prior_templates/GW231123_template.prior",
        detectors=("H1", "L1"),
    ),
    "GW230814": EventDefaults(
        label_prefix="GW230814_t_Student_pSEOB",
        outdir_base="/home/gregorio.carullo/GW230814/t_Student_pSEOB/Runs",
        webdir_base="/home/gregorio.carullo/public_html/GW230814/t_Student_pSEOB/Runs",
        file_prefix="GW230814",
        ini_template="Initialisation_file_templates/GW230814_t_student_pSEOB_template.ini",
        prior_template="Prior_templates/GW230814_template.prior",
        detectors=("L1",),
    ),
}

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
            "submit them. Gaussian runs can also be added."
        )
    )
    parser.add_argument(
        "--event",
        choices=sorted(EVENT_DEFAULTS.keys()),
        default=DEFAULT_EVENT,
        help=(
            "Event defaults to use for template paths, output prefixes, and "
            "detectors. Default: GW231123."
        ),
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
        "--add-gaussian",
        action="store_true",
        help=(
            "Also generate and submit Gaussian-likelihood runs for each requested "
            "band count."
        ),
    )
    parser.add_argument(
        "--detector-dependent-nu",
        action="store_true",
        help="Generate detector-specific Student-t nu parameters.",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=None,
        help=(
            "Detector names used when building detector-dependent nu priors. "
            "Defaults to the selected event."
        ),
    )
    parser.add_argument(
        "--ini-template",
        type=Path,
        default=None,
        help=(
            "Optional path to the Student-t ini template. "
            "If omitted, the selected event default is used."
        ),
    )
    parser.add_argument(
        "--prior-template",
        type=Path,
        default=None,
        help=(
            "Optional path to the prior template containing __NU_PRIORS__. "
            "If omitted, the selected event default is used."
        ),
    )
    parser.add_argument(
        "--label-prefix",
        default=None,
        help="Optional run label prefix override.",
    )
    parser.add_argument(
        "--outdir-base",
        default=None,
        help="Optional base outdir override.",
    )
    parser.add_argument(
        "--webdir-base",
        default=None,
        help="Optional base webdir override.",
    )
    parser.add_argument(
        "--file-prefix",
        default=None,
        help="Optional filename prefix for generated ini/prior files.",
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


def resolve_path(path: Path | None, default_path: Path) -> Path:
    if path is None:
        return default_path.resolve()
    return path.expanduser().resolve()


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
    include_nu_priors: bool = True,
    detector_dependent_nu: bool = False,
    detectors: list[str] | tuple[str, ...] = DEFAULT_DETECTORS,
) -> str:
    if not include_nu_priors:
        return prior_template.replace("__NU_PRIORS__", "")
    return prior_template.replace(
        "__NU_PRIORS__",
        build_nu_priors(
            band_count,
            detector_dependent_nu=detector_dependent_nu,
            detectors=detectors,
        ),
    )


def replace_line(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    prefix = f"{key}="
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f"{key}={value}"
            return "\n".join(lines) + "\n"
    raise ValueError(f"Unable to find config key '{key}' in template")


def render_ini(
    ini_template: str,
    *,
    hypothesis: str,
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

    if hypothesis == "student":
        rendered = replace_line(
            rendered,
            "likelihood-type",
            "bilby.gw.likelihood.StudentTGravitationalWaveTransient",
        )
    elif hypothesis == "gaussian":
        rendered = replace_line(
            rendered,
            "likelihood-type",
            "bilby.gw.likelihood.GravitationalWaveTransient",
        )
        rendered = replace_line(rendered, "extra-likelihood-kwargs", "None")
    else:
        raise ValueError(f"Unknown hypothesis '{hypothesis}'")
    return rendered


def prepare_run(
    *,
    hypothesis: str,
    band_count: int,
    ini_template: str,
    prior_template: str,
    ini_dir: Path,
    prior_dir: Path,
    detector_dependent_nu: bool,
    detectors: list[str] | tuple[str, ...],
    label_prefix: str,
    outdir_base: str,
    webdir_base: str,
    file_prefix: str,
) -> Path:
    if hypothesis == "student":
        mode_suffix = "_detector_dependent_nu" if detector_dependent_nu else ""
        label = f"{label_prefix}{mode_suffix}_N{band_count}"
        run_outdir = f"{outdir_base}/student{mode_suffix}_N{band_count}"
        run_webdir = f"{webdir_base}/student{mode_suffix}_N{band_count}"
        prior_path = (
            prior_dir / f"{file_prefix}{mode_suffix}_N{band_count}.prior"
        ).resolve()
        ini_path = (
            ini_dir / f"{file_prefix}_t_student{mode_suffix}_N{band_count}.ini"
        ).resolve()
        include_nu_priors = True
        run_detector_dependent_nu = detector_dependent_nu
    elif hypothesis == "gaussian":
        label = f"{label_prefix}_gaussian_N{band_count}"
        run_outdir = f"{outdir_base}/gaussian_N{band_count}"
        run_webdir = f"{webdir_base}/gaussian_N{band_count}"
        prior_path = (prior_dir / f"{file_prefix}_gaussian_N{band_count}.prior").resolve()
        ini_path = (ini_dir / f"{file_prefix}_gaussian_N{band_count}.ini").resolve()
        include_nu_priors = False
        run_detector_dependent_nu = False
    else:
        raise ValueError(f"Unknown hypothesis '{hypothesis}'")

    prior_path.write_text(
        render_prior(
            prior_template,
            band_count,
            include_nu_priors=include_nu_priors,
            detector_dependent_nu=run_detector_dependent_nu,
            detectors=detectors,
        ),
        encoding="utf-8",
    )
    ini_path.write_text(
        render_ini(
            ini_template,
            hypothesis=hypothesis,
            label=label,
            outdir=run_outdir,
            webdir=run_webdir,
            prior_file=prior_path,
            band_count=band_count,
            detector_dependent_nu=run_detector_dependent_nu,
        ),
        encoding="utf-8",
    )

    print(f"Prepared {hypothesis} N={band_count}:")
    print(f"  prior: {prior_path}")
    print(f"  ini:   {ini_path}")
    return ini_path


def submit_run(ini_path: Path) -> None:
    subprocess.run(["bilby_pipe", str(ini_path), "--submit"], check=True)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    args = build_argument_parser(script_dir).parse_args()

    defaults = EVENT_DEFAULTS[args.event]
    ini_template_path = resolve_path(
        args.ini_template,
        script_dir / defaults.ini_template,
    )
    prior_template_path = resolve_path(
        args.prior_template,
        script_dir / defaults.prior_template,
    )
    label_prefix = args.label_prefix or defaults.label_prefix
    outdir_base = args.outdir_base or defaults.outdir_base
    webdir_base = args.webdir_base or defaults.webdir_base
    file_prefix = args.file_prefix or defaults.file_prefix
    detectors = tuple(args.detectors) if args.detectors else defaults.detectors

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
    hypotheses = ["student"]
    if args.add_gaussian:
        hypotheses.append("gaussian")

    for band_count in band_counts:
        for hypothesis in hypotheses:
            ini_path = prepare_run(
                hypothesis=hypothesis,
                band_count=band_count,
                ini_template=ini_template,
                prior_template=prior_template,
                ini_dir=ini_dir,
                prior_dir=prior_dir,
                detector_dependent_nu=args.detector_dependent_nu,
                detectors=detectors,
                label_prefix=label_prefix,
                outdir_base=outdir_base,
                webdir_base=webdir_base,
                file_prefix=file_prefix,
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
