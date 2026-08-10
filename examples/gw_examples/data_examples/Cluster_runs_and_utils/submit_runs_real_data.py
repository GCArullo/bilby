#!/usr/bin/env python3

"""Generate and optionally submit Student-t bilby_pipe runs.

Generate either Student-t or Gaussian-likelihood runs. Student-t runs may
also generate a single-band Gaussian companion run by default.
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from container_creation.submission_container_utils import (
    add_container_arguments,
    resolve_container_image,
)
from submission_sine_gaussian_utils import (
    add_sine_gaussian_arguments,
    apply_sine_gaussian_waveform_settings,
    build_sine_gaussian_prior_block,
    combine_prior_blocks,
    effective_nlive,
    positive_int,
    read_template_settings,
    require_supported_sine_gaussian_source_model,
    resolve_sine_gaussian_configurations,
    validate_submission_local_paths,
)


DEFAULT_DETECTORS = ("H1", "L1")
DEFAULT_EVENT = "GW231123"
SPIN_TAYLOR_SUFFIX = "_SpinTaylor"
SPIN_TAYLOR_PREC_VERSION = 320
DEFAULT_CONTAINER_IMAGES_FILE = (
    Path(__file__).resolve().parent / "container_creation" / "container_images.json"
)


def default_accounting_user() -> str:
    for home_path in (os.environ.get("HOME"), str(Path.home())):
        if not home_path:
            continue
        home_name = Path(home_path).name
        if home_name:
            return home_name
    return getpass.getuser()


DEFAULT_HOME_DIR = Path.home()
DEFAULT_ACCOUNTING_USER = default_accounting_user()
DEFAULT_NUM_FREQUENCY_BANDS = 1
DEFAULT_ENVIRONMENT_VARIABLES = {
    "HDF5_USE_FILE_LOCKING": False,
    "NUMBA_CACHE_DIR": "/tmp",
    "OMP_NUM_THREADS": 1,
    "OMP_PROC_BIND": False,
    "LAL_DATA_PATH": "/scratch/lalsimulation",
}
DEFAULT_PESUMMARY_ARGUMENTS = {
    "multi_process": 6,
    "disable_expert": True,
    "disable_interactive": True,
    "gw": True,
    "no_ligo_skymap": True,
    "redshift_method": "exact",
    "evolve_spins_forwards": True,
    "evolve_spins_backwards": True,
    "NRSur_fits": True,
    "calculate_multipole_snr": True,
    "ignore_parameters": ["recalib*"],
}


@dataclass(frozen=True)
class EventDefaults:
    label_prefix: str
    run_subdir: str
    file_prefix: str
    ini_template: str
    prior_template: str
    working_directory: str
    detectors: tuple[str, ...]


EVENT_DEFAULTS: dict[str, EventDefaults] = {
    "GW150914": EventDefaults(
        label_prefix="GW150914_IMRPhenomXPHM",
        run_subdir="GW150914/Runs",
        file_prefix="GW150914_IGWN_C01_IMRPhenomXPHM",
        ini_template=(
            "Special_events_configs/templates/"
            "GW150914_t_student_igwn_template.ini"
        ),
        prior_template=(
            "Special_events_configs/priors/GW150914_igwn_template.prior"
        ),
        working_directory="LVK_posteriors/GW150914",
        detectors=("H1", "L1"),
    ),
    "GW190521_030229_LVK_NRSur7dq4": EventDefaults(
        label_prefix="GW190521_030229_LVK_NRSur7dq4",
        run_subdir="GWTC_parametric_noise/Runs/GW190521_030229",
        file_prefix="GW190521_030229_LVK_NRSur7dq4",
        ini_template=(
            "Special_events_configs/templates/"
            "GW190521_030229_LVK_NRSur7dq4.ini"
        ),
        prior_template=(
            "Special_events_configs/priors/"
            "GW190521_030229_LVK_NRSur7dq4.prior"
        ),
        working_directory="Special_events_configs",
        detectors=("H1", "L1", "V1"),
    ),
    "GW231123": EventDefaults(
        label_prefix="GW231123_t_Student",
        run_subdir="GW231123/t_Student/Runs",
        file_prefix="GW231123",
        ini_template="Initialisation_file_templates/GW231123_t_student_template.ini",
        prior_template="Prior_templates/GW231123_template.prior",
        working_directory="LVK_posteriors/GW231123",
        detectors=("H1", "L1"),
    ),
    "GW230814": EventDefaults(
        label_prefix="GW230814_t_Student_pSEOB",
        run_subdir="GW230814/t_Student_pSEOB/Runs",
        file_prefix="GW230814",
        ini_template="Initialisation_file_templates/GW230814_t_student_pSEOB_template.ini",
        prior_template="Prior_templates/GW230814_template.prior",
        working_directory="LVK_posteriors/GW230814",
        detectors=("L1",),
    ),
}


def outdir_label(value: str) -> str:
    label = value.strip()
    if not label:
        raise argparse.ArgumentTypeError("outdir label must not be empty")
    if any(separator and separator in label for separator in (os.sep, os.altsep)):
        raise argparse.ArgumentTypeError("outdir label must not contain path separators")
    return label


def build_argument_parser(script_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate bilby_pipe ini/prior files for Student-t or Gaussian runs "
            "and optionally submit them."
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
        "--num-frequency-bands",
        type=positive_int,
        default=None,
        help=(
            "Positive integer. In single mode this is the exact band count. "
            "In range mode this is the maximum band count. "
            f"Defaults to {DEFAULT_NUM_FREQUENCY_BANDS}."
        ),
    )
    parser.add_argument(
        "--maxmcmc",
        type=positive_int,
        default=None,
        help=(
            "Optional Dynesty maxmcmc value written into sampler-kwargs. "
            "Defaults to the value in the selected ini template."
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
        "--condor-job-priority",
        type=int,
        default=None,
        help=(
            "Value written into condor-job-priority. Larger values are matched "
            "first among your own idle jobs. Defaults to the ini template value."
        ),
    )
    parser.add_argument(
        "--require-epnfs",
        action="store_true",
        help=(
            "Set queue=EPNFS in generated ini files so bilby_pipe emits "
            "requirements = ((TARGET.EPNFS =?= True)) in Condor submit files."
        ),
    )
    parser.add_argument(
        "--likelihood",
        choices=("student", "gaussian"),
        default="gaussian",
        help=(
            "Primary recovery likelihood to generate. Gaussian runs always use "
            f"the default single frequency band ({DEFAULT_NUM_FREQUENCY_BANDS})."
        ),
    )
    parser.add_argument(
        "--add-gaussian",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When --likelihood student is selected, also generate a single-band "
            "Gaussian companion run. Enabled by default for Student runs; pass "
            "--no-add-gaussian to disable it."
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
            "Detectors to analyse, also used when building detector-dependent nu "
            "priors. Defaults to the selected event. A single detector switches "
            "the run to reference-frame=sky, since one detector cannot localise."
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
        "--working-directory",
        type=Path,
        default=None,
        help=(
            "Base directory used to resolve event-specific relative paths in the "
            "template. Defaults to the selected event under LVK_posteriors."
        ),
    )
    parser.add_argument(
        "--home-dir",
        type=Path,
        default=DEFAULT_HOME_DIR,
        help=(
            "Base home directory containing public_html, used to build default "
            "output paths when --outdir-base/--webdir-base are not provided."
        ),
    )
    parser.add_argument(
        "--accounting-user",
        default=DEFAULT_ACCOUNTING_USER,
        help=(
            "Value written into accounting-user in the generated ini files. "
            f"Default: {DEFAULT_ACCOUNTING_USER}."
        ),
    )
    parser.add_argument(
        "--label-prefix",
        default=None,
        help="Optional run label prefix override.",
    )
    parser.add_argument(
        "--outdir-label",
        type=outdir_label,
        default=None,
        help=(
            "Optional suffix appended to the standard run outdir/webdir name. "
            "This does not change the bilby label or ini/prior filenames."
        ),
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
    parser.add_argument(
        "--waveform-approximant",
        default=None,
        help=(
            "Override the waveform approximant set in the ini template. "
            "When the override differs from the template default a suffix "
            "(e.g. _SEOBNRv5PHM) is appended to all labels and filenames. "
            "If omitted the template value is used unchanged."
        ),
    )
    add_container_arguments(
        parser,
        default_image_file=DEFAULT_CONTAINER_IMAGES_FILE,
    )
    add_sine_gaussian_arguments(parser)
    return parser


def hypothesis_list(args: argparse.Namespace) -> list[str]:
    explicit_num_frequency_bands = getattr(
        args,
        "num_frequency_bands_was_explicit",
        args.num_frequency_bands is not None,
    )
    resolved_num_frequency_bands = (
        DEFAULT_NUM_FREQUENCY_BANDS
        if args.num_frequency_bands is None
        else args.num_frequency_bands
    )
    if args.likelihood == "gaussian":
        if args.add_gaussian is True:
            raise ValueError(
                "--add-gaussian requires --likelihood student. "
                "Gaussian is already the selected primary likelihood."
            )
        if explicit_num_frequency_bands:
            raise ValueError(
                "--likelihood gaussian cannot be combined with --num-frequency-bands. "
                f"Gaussian runs always use the default value {DEFAULT_NUM_FREQUENCY_BANDS}."
            )
        return ["gaussian"]
    if args.add_gaussian is not False:
        return ["student", "gaussian"]
    return ["student"]


def build_run_requests(
    args: argparse.Namespace,
    *,
    band_counts,
) -> list[tuple[str, int]]:
    if args.likelihood == "gaussian":
        return [("gaussian", DEFAULT_NUM_FREQUENCY_BANDS)]

    requests = [("student", band_count) for band_count in band_counts]
    if args.add_gaussian is not False:
        requests.append(("gaussian", DEFAULT_NUM_FREQUENCY_BANDS))
    return requests


def resolve_path(path: Path | None, default_path: Path) -> Path:
    if path is None:
        return default_path.resolve()
    return path.expanduser().resolve()


def determine_submit_directory(outdir_base: str, webdir_base: str) -> Path:
    return Path(
        os.path.commonpath(
            [str(Path(outdir_base).resolve()), str(Path(webdir_base).resolve())]
        )
    )


def default_output_bases(home_dir: Path, run_subdir: str) -> tuple[str, str]:
    resolved_home = home_dir.expanduser()
    outdir_base = resolved_home / "public_html" / run_subdir
    webdir_base = outdir_base
    return str(outdir_base), str(webdir_base)


def build_run_directory_name(name: str, outdir_label: str | None) -> str:
    if outdir_label is None:
        return name
    return f"{name}_{outdir_label}"


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
    template_settings: dict[str, object],
    sine_gaussian_config,
) -> str:
    nu_prior_block = ""
    if include_nu_priors:
        nu_prior_block = build_nu_priors(
            band_count,
            detector_dependent_nu=detector_dependent_nu,
            detectors=detectors,
        )
    sine_gaussian_prior_block = build_sine_gaussian_prior_block(
        sine_gaussian_config,
        minimum_frequency=template_settings["minimum_frequency"],
        maximum_frequency=template_settings["maximum_frequency"],
    )
    return prior_template.replace(
        "__NU_PRIORS__",
        combine_prior_blocks(nu_prior_block, sine_gaussian_prior_block),
    )


def minimum_frequency_for_pesummary(minimum_frequency):
    if isinstance(minimum_frequency, dict):
        detector_frequencies = [
            value for key, value in minimum_frequency.items()
            if key != "waveform"
        ]
        if detector_frequencies:
            return min(detector_frequencies)
        return minimum_frequency["waveform"]
    return minimum_frequency


def maximum_frequency_for_pesummary(maximum_frequency):
    if isinstance(maximum_frequency, dict):
        return max(maximum_frequency.values())
    return maximum_frequency


def restrict_to_detectors(mapping, detectors):
    """Drop the entries of detectors that this run does not analyse."""
    if detectors is None or not isinstance(mapping, dict):
        return mapping
    names = {str(name) for name in detectors}
    return {key: value for key, value in mapping.items() if key in names}


def build_pesummary_arguments(
    template_settings: dict[str, object],
    *,
    page_label: str,
    detectors: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    arguments = dict(DEFAULT_PESUMMARY_ARGUMENTS)
    # pesummary names its outputs '<label>_<label>_<parameter>.html' and
    # '<label>_<result file name>'. Left to bilby_pipe the label is the full
    # merge-result basename, so those names run past the 255 byte file name
    # limit and the results page aborts with OSError errno 36. Name the page
    # after the run directory instead.
    arguments["labels"] = [page_label]
    f_low = minimum_frequency_for_pesummary(template_settings["minimum_frequency"])
    f_ref = template_settings["reference_frequency"]
    arguments.update(
        f_low=f_low,
        f_start=f_ref,
        f_ref=f_ref,
        f_final=maximum_frequency_for_pesummary(template_settings["maximum_frequency"]),
        approximant=[template_settings["waveform_approximant"]],
    )
    calibration = template_settings["spline_calibration_envelope_dict"]
    if calibration not in (None, "None"):
        arguments["calibration"] = restrict_to_detectors(calibration, detectors)
    psd_dict = template_settings.get("psd_dict")
    if psd_dict not in (None, "None"):
        arguments["psd"] = restrict_to_detectors(psd_dict, detectors)
    return arguments


def replace_line(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    prefix = f"{key}="
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f"{key}={value}"
            return "\n".join(lines) + "\n"
    raise ValueError(f"Unable to find config key '{key}' in template")


def replace_or_append_line(
    text: str,
    key: str,
    value: str,
) -> str:
    lines = text.splitlines()
    prefix = f"{key}="
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f"{key}={value}"
            return "\n".join(lines) + "\n"
    lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def resolve_spin_taylor_approximant(approximant: str) -> tuple[str, dict | None]:
    """Split `<approximant>_SpinTaylor` into its LAL name and precession version.

    LALSuite has no standalone SpinTaylor approximant: the SpinTaylor precession
    prescription is selected on IMRPhenomXPHM through PhenomXPrecVersion.
    """
    if not approximant.endswith(SPIN_TAYLOR_SUFFIX):
        return approximant, None
    return (
        approximant[: -len(SPIN_TAYLOR_SUFFIX)],
        {"PhenomXPrecVersion": SPIN_TAYLOR_PREC_VERSION},
    )


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
    working_directory: Path,
    accounting_user: str,
    container_image: str | None,
    require_epnfs: bool,
    maxmcmc: int | None,
    template_settings: dict[str, object],
    sine_gaussian_config,
    detectors: list[str] | tuple[str, ...] | None = None,
    condor_job_priority: int | None = None,
    waveform_arguments: dict | None = None,
) -> str:
    replacements = {
        "__LABEL__": label,
        "__OUTDIR__": outdir,
        "__WEBDIR__": webdir,
        "__PRIOR_FILE__": str(prior_file),
        "__NUM_FREQUENCY_BANDS__": str(band_count),
        "__DETECTOR_DEPENDENT_NU__": str(detector_dependent_nu),
        "__WORKING_DIRECTORY__": str(working_directory),
    }
    rendered = ini_template
    rendered = replace_line(rendered, "accounting-user", accounting_user)
    rendered = replace_or_append_line(
        rendered,
        "container",
        container_image or "None",
    )
    rendered = replace_or_append_line(rendered, "transfer-files", "True")
    rendered = replace_or_append_line(rendered, "osg", "True")
    rendered = replace_or_append_line(rendered, "desired-sites", "None")
    if require_epnfs:
        rendered = replace_line(rendered, "queue", "EPNFS")
    if condor_job_priority is not None:
        # Not every event template sets this key.
        rendered = replace_or_append_line(
            rendered,
            "condor-job-priority",
            str(condor_job_priority),
        )
    if detectors is not None:
        rendered = replace_line(
            rendered,
            "detectors",
            repr([str(name) for name in detectors]),
        )
        if len(detectors) == 1:
            # A single detector cannot triangulate, so sample the sky directly
            # instead of in the two-detector zenith/azimuth frame.
            rendered = replace_line(rendered, "reference-frame", "sky")
            rendered = replace_line(rendered, "time-reference", str(detectors[0]))
    rendered = replace_line(rendered, "create-summary", "True")
    rendered = replace_line(
        rendered,
        "environment-variables",
        repr(DEFAULT_ENVIRONMENT_VARIABLES),
    )
    rendered = replace_line(
        rendered,
        "summarypages-arguments",
        repr(
            build_pesummary_arguments(
                template_settings,
                page_label=Path(outdir).name,
                detectors=detectors,
            )
        ),
    )
    sampler_kwargs = dict(template_settings["sampler_kwargs"])
    sampler_kwargs["nlive"] = effective_nlive(
        int(sampler_kwargs["nlive"]),
        sine_gaussian_config,
    )
    if maxmcmc is not None:
        sampler_kwargs["maxmcmc"] = maxmcmc
    rendered = replace_line(rendered, "sampler-kwargs", repr(sampler_kwargs))

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
    
    rendered = replace_line(
        rendered,
        "waveform-approximant",
        template_settings["waveform_approximant"],
    )
    if waveform_arguments is not None:
        rendered = replace_line(
            rendered,
            "waveform-arguments-dict",
            repr(waveform_arguments),
        )
    rendered = replace_line(
        rendered,
        "minimum-frequency",
        repr(template_settings["minimum_frequency"]),
    )
    GW_SIGNAL_MODELS = {"SEOBNRv5PHM", "SEOBNRv5HM"}
    if template_settings["waveform_approximant"] in GW_SIGNAL_MODELS:
        # waveform generator now calls the correct waveform generation function depending on the flag.
        rendered = replace_line(
            rendered,
            "waveform-generator",
            "bilby.gw.waveform_generator.WaveformGenerator",
        )
        # Sine-Gaussian runs override this below via cbc_plus_sine_gaussians,
        # which auto-detects these approximants; without sine-Gaussians, route
        # the CBC baseline through gwsignal directly.
        rendered = replace_line(
            rendered,
            "frequency-domain-source-model",
            "bilby.gw.source.gwsignal_binary_black_hole",
        )
    rendered = apply_sine_gaussian_waveform_settings(
        rendered,
        sine_gaussian_config,
        replace_line=replace_line,
    )
    # Substituted last: lines written above (summarypages-arguments in
    # particular) carry template placeholders through from template_settings.
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def prepare_run(
    *,
    hypothesis: str,
    band_count: int,
    ini_template: str,
    prior_template: str,
    template_settings: dict[str, object],
    ini_dir: Path,
    prior_dir: Path,
    detector_dependent_nu: bool,
    detectors: list[str] | tuple[str, ...],
    label_prefix: str,
    outdir_base: str,
    webdir_base: str,
    outdir_label: str | None,
    file_prefix: str,
    working_directory: Path,
    accounting_user: str,
    container_image: str | None,
    require_epnfs: bool,
    condor_job_priority: int | None,
    maxmcmc: int | None,
    waveform_arguments: dict | None,
    sine_gaussian_config,
    approximant_suffix: str = "",
    detector_suffix: str = "",
) -> Path:
    waveform_suffix = (
        sine_gaussian_config.label_suffix + approximant_suffix + detector_suffix
    )
    if hypothesis == "student":
        run_band_count = band_count
        mode_suffix = "_detector_dependent_nu" if detector_dependent_nu else ""
        label = f"{label_prefix}{mode_suffix}_N{run_band_count}{waveform_suffix}"
        run_directory_name = build_run_directory_name(
            f"student{mode_suffix}_N{run_band_count}{waveform_suffix}",
            outdir_label,
        )
        run_outdir = f"{outdir_base}/{run_directory_name}"
        prior_path = (
            prior_dir
            / f"{file_prefix}{mode_suffix}_N{run_band_count}{waveform_suffix}.prior"
        ).resolve()
        ini_path = (
            ini_dir
            / f"{file_prefix}_t_student{mode_suffix}_N{run_band_count}{waveform_suffix}.ini"
        ).resolve()
        include_nu_priors = True
        run_detector_dependent_nu = detector_dependent_nu
    elif hypothesis == "gaussian":
        run_band_count = DEFAULT_NUM_FREQUENCY_BANDS
        label = f"{label_prefix}_gaussian{waveform_suffix}"
        run_directory_name = build_run_directory_name(
            f"gaussian{waveform_suffix}",
            outdir_label,
        )
        run_outdir = f"{outdir_base}/{run_directory_name}"
        prior_path = (prior_dir / f"{file_prefix}_gaussian{waveform_suffix}.prior").resolve()
        ini_path = (ini_dir / f"{file_prefix}_gaussian{waveform_suffix}.ini").resolve()
        include_nu_priors = False
        run_detector_dependent_nu = False
    else:
        raise ValueError(f"Unknown hypothesis '{hypothesis}'")

    run_webdir = f"{webdir_base}/{run_directory_name}/web"

    prior_path.write_text(
        render_prior(
            prior_template,
            run_band_count,
            include_nu_priors=include_nu_priors,
            detector_dependent_nu=run_detector_dependent_nu,
            detectors=detectors,
            template_settings=template_settings,
            sine_gaussian_config=sine_gaussian_config,
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
            band_count=run_band_count,
            detector_dependent_nu=run_detector_dependent_nu,
            detectors=detectors,
            working_directory=working_directory,
            accounting_user=accounting_user,
            container_image=container_image,
            require_epnfs=require_epnfs,
            condor_job_priority=condor_job_priority,
            maxmcmc=maxmcmc,
            waveform_arguments=waveform_arguments,
            template_settings=template_settings,
            sine_gaussian_config=sine_gaussian_config,
        ),
        encoding="utf-8",
    )

    band_fragment = f" N={run_band_count}" if hypothesis == "student" else ""
    print(
        f"Prepared {hypothesis}{band_fragment} "
        f"({sine_gaussian_config.description}):"
    )
    print(f"  prior: {prior_path}")
    print(f"  ini:   {ini_path}")
    return ini_path


def submit_run(ini_path: Path, *, submit_directory: Path) -> None:
    validate_submission_local_paths(
        ini_path.read_text(encoding="utf-8"),
        base_directory=submit_directory,
    )
    subprocess.run(
        ["bilby_pipe", str(ini_path), "--submit"],
        check=True,
        cwd=submit_directory,
    )


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    args = build_argument_parser(script_dir).parse_args()
    args.num_frequency_bands_was_explicit = args.num_frequency_bands is not None
    if args.num_frequency_bands is None:
        args.num_frequency_bands = DEFAULT_NUM_FREQUENCY_BANDS
    container_image = resolve_container_image(
        use_container=args.container,
        container_image=args.container_image,
        default_image_file=DEFAULT_CONTAINER_IMAGES_FILE,
    )

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
    default_outdir_base, _ = default_output_bases(
        args.home_dir,
        defaults.run_subdir,
    )
    outdir_base = args.outdir_base or default_outdir_base
    webdir_base = args.webdir_base or outdir_base
    file_prefix = args.file_prefix or defaults.file_prefix
    detectors = tuple(args.detectors) if args.detectors else defaults.detectors
    working_directory = resolve_path(
        args.working_directory,
        script_dir / defaults.working_directory,
    )
    submit_directory = determine_submit_directory(outdir_base, webdir_base)

    ini_template = load_template(ini_template_path)
    prior_template = load_template(prior_template_path)
    template_settings = read_template_settings(ini_template)

    waveform_arguments = None
    if args.waveform_approximant is not None:
        template_approximant = template_settings["waveform_approximant"]
        lal_approximant, waveform_arguments = resolve_spin_taylor_approximant(
            args.waveform_approximant,
        )
        min_freq = template_settings["minimum_frequency"]
        if isinstance(min_freq, dict):
            detector_freqs = [v for k, v in min_freq.items() if k != "waveform"]
            min_freq = dict(min_freq, waveform=min(detector_freqs) if detector_freqs else 20.0)
        template_settings = dict(
            template_settings,
            waveform_approximant=lal_approximant,
            minimum_frequency=min_freq,
        )
        approximant_suffix = (
            f"_{args.waveform_approximant}"
            if args.waveform_approximant != template_approximant
            else ""
        )
    else:
        approximant_suffix = ""

    detector_suffix = (
        "_" + "".join(detectors) + "only"
        if tuple(detectors) != defaults.detectors
        else ""
    )

    sine_gaussian_configs = resolve_sine_gaussian_configurations(
        num_sine_gaussians=args.num_sine_gaussians,
        range_mode=args.sine_gaussian_range,
        mode=args.sine_gaussian_mode,
        incoherent_detectors=args.incoherent_detectors,
        incoherent_counts_spec=args.incoherent_sg_counts,
        detectors=detectors,
    )
    for sine_gaussian_config in sine_gaussian_configs:
        require_supported_sine_gaussian_source_model(
            template_settings,
            sine_gaussian_config,
        )

    ini_dir = args.ini_dir.expanduser().resolve()
    prior_dir = args.prior_dir.expanduser().resolve()
    ini_dir.mkdir(parents=True, exist_ok=True)
    prior_dir.mkdir(parents=True, exist_ok=True)

    if args.range_mode:
        band_counts = range(1, args.num_frequency_bands + 1)
    else:
        band_counts = [args.num_frequency_bands]
    run_requests = build_run_requests(args, band_counts=band_counts)

    for sine_gaussian_config in sine_gaussian_configs:
        for hypothesis, band_count in run_requests:
            ini_path = prepare_run(
                hypothesis=hypothesis,
                band_count=band_count,
                ini_template=ini_template,
                prior_template=prior_template,
                template_settings=template_settings,
                ini_dir=ini_dir,
                prior_dir=prior_dir,
                detector_dependent_nu=args.detector_dependent_nu,
                detectors=detectors,
                label_prefix=label_prefix,
                outdir_base=outdir_base,
                webdir_base=webdir_base,
                outdir_label=args.outdir_label,
                file_prefix=file_prefix,
                working_directory=working_directory,
                accounting_user=args.accounting_user,
                container_image=container_image,
                require_epnfs=args.require_epnfs,
                condor_job_priority=args.condor_job_priority,
                maxmcmc=args.maxmcmc,
                waveform_arguments=waveform_arguments,
                sine_gaussian_config=sine_gaussian_config,
                approximant_suffix=approximant_suffix,
                detector_suffix=detector_suffix,
            )
            if not args.dry_run:
                submit_run(ini_path, submit_directory=submit_directory)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
