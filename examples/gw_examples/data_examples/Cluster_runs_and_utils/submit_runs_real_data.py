#!/usr/bin/env python3

"""Generate and optionally submit parametric-noise bilby_pipe runs.

Generate Student-t, Hyperbolic, Gaussian-parametric, or Gaussian-likelihood
runs. Parametric-noise runs may also generate a single-band Gaussian companion
run by default.
"""

from __future__ import annotations

import argparse
import getpass
import json
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
)


DEFAULT_DETECTORS = ("H1", "L1")
DEFAULT_EVENT = "GW231123"
DEFAULT_CONTAINER_IMAGES_FILE = (
    Path(__file__).resolve().parent / "container_creation" / "container_images.json"
)
CATALOG_CONFIGS_DIR = "GWTC_catalog_configs"
SPECIAL_EVENTS_CONFIGS_DIR = "Special_events_configs"


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
HEAVY_TAILED_LIKELIHOODS = ("student", "hyperbolic")
PARAMETRIC_NOISE_LIKELIHOODS = (*HEAVY_TAILED_LIKELIHOODS, "gaussian-parametric")
DEFAULT_HYPERBOLIC_ALPHA = 10.0
DEFAULT_HYPERBOLIC_DELTA = 1.0
DEFAULT_HYPERBOLIC_ALPHA_MIN = 1e-6
DEFAULT_HYPERBOLIC_ALPHA_MAX = 30.0
DEFAULT_HYPERBOLIC_DELTA_MIN = 1e-6
DEFAULT_HYPERBOLIC_DELTA_MAX = 30.0
DEFAULT_LOG_PSD_SCALE_MIN = -1.0
DEFAULT_LOG_PSD_SCALE_MAX = 1.0
DEFAULT_REQUEST_CPUS = 16
DEFAULT_REQUEST_MEMORY_GB = 24.0
WORKING_DIRECTORY_PLACEHOLDER = "__WORKING_DIRECTORY__"
NOISE_ONLY_DEFAULT_PRIOR = "bilby.core.prior.PriorDict"
NOISE_ONLY_SOURCE_MODEL = "bilby.gw.source.zero_waveform"
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
LOCAL_DATA_SETTINGS = (
    "data_dict",
    "psd_dict",
    "spline_calibration_envelope_dict",
)


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
            f"{SPECIAL_EVENTS_CONFIGS_DIR}/templates/"
            "GW150914_t_student_igwn_template.ini"
        ),
        prior_template=(
            f"{SPECIAL_EVENTS_CONFIGS_DIR}/priors/"
            "GW150914_igwn_template.prior"
        ),
        working_directory="LVK_posteriors/GW150914",
        detectors=("H1", "L1"),
    ),
    "GW231123": EventDefaults(
        label_prefix="GW231123",
        run_subdir="GW231123/Runs",
        file_prefix="GW231123",
        ini_template=(
            f"{SPECIAL_EVENTS_CONFIGS_DIR}/templates/"
            "GW231123_t_student_template.ini"
        ),
        prior_template=(
            f"{SPECIAL_EVENTS_CONFIGS_DIR}/priors/"
            "GW231123_template.prior"
        ),
        working_directory="LVK_posteriors/GW231123",
        detectors=("H1", "L1"),
    ),
    "GW230814": EventDefaults(
        label_prefix="GW230814_pSEOB",
        run_subdir="GW230814/Runs",
        file_prefix="GW230814",
        ini_template=(
            f"{SPECIAL_EVENTS_CONFIGS_DIR}/templates/"
            "GW230814_t_student_pSEOB_template.ini"
        ),
        prior_template=(
            f"{SPECIAL_EVENTS_CONFIGS_DIR}/priors/"
            "GW230814_template.prior"
        ),
        working_directory="LVK_posteriors/GW230814",
        detectors=("L1",),
    ),
}


def load_catalog_event_defaults() -> dict[str, EventDefaults]:
    defaults = {}
    for catalog in ("GWTC-2.1", "GWTC-3", "GWTC-4", "GWTC-5"):
        manifest_path = (
            Path(__file__).resolve().parent
            / CATALOG_CONFIGS_DIR
            / catalog
            / "manifest.json"
        )
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["events"]:
            event = item["event"]
            approximant = item["approximant"]
            prefix = f"{event}_IGWN_C01_{approximant}"
            defaults[event] = EventDefaults(
                label_prefix=f"{event}_{approximant}",
                run_subdir=f"GWTC_parametric_noise/Runs/{event}",
                file_prefix=prefix,
                ini_template=f"{CATALOG_CONFIGS_DIR}/{catalog}/{item['template']}",
                prior_template=f"{CATALOG_CONFIGS_DIR}/{catalog}/{item['prior']}",
                working_directory=f"{CATALOG_CONFIGS_DIR}/{catalog}",
                detectors=tuple(item["detectors"]),
            )
    return defaults


EVENT_DEFAULTS.update(load_catalog_event_defaults())

EVENT_DEFAULTS["GW200129_065458_Hannam"] = EventDefaults(
    label_prefix="GW200129_065458_Hannam_NRSur7dq4",
    run_subdir="GW200129_065458_Hannam/Runs",
    file_prefix="GW200129_065458_Hannam_NRSur7dq4",
    ini_template=(
        f"{SPECIAL_EVENTS_CONFIGS_DIR}/templates/"
        "GW200129_065458_Hannam_NRSur7dq4.ini"
    ),
    prior_template=(
        f"{SPECIAL_EVENTS_CONFIGS_DIR}/priors/"
        "GW200129_065458_Hannam_NRSur7dq4.prior"
    ),
    working_directory=f"{CATALOG_CONFIGS_DIR}/GWTC-3",
    detectors=("H1", "L1", "V1"),
)

EVENT_DEFAULTS["GW190521_030229_LVK_NRSur7dq4"] = EventDefaults(
    label_prefix="GW190521_030229_LVK_NRSur7dq4",
    run_subdir="GWTC_parametric_noise/Runs/GW190521_030229",
    file_prefix="GW190521_030229_LVK_NRSur7dq4",
    ini_template=(
        f"{SPECIAL_EVENTS_CONFIGS_DIR}/templates/"
        "GW190521_030229_LVK_NRSur7dq4.ini"
    ),
    prior_template=(
        f"{SPECIAL_EVENTS_CONFIGS_DIR}/priors/"
        "GW190521_030229_LVK_NRSur7dq4.prior"
    ),
    working_directory=SPECIAL_EVENTS_CONFIGS_DIR,
    detectors=("H1", "L1", "V1"),
)


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
            "Generate bilby_pipe ini/prior files for Student-t, Hyperbolic, "
            "Gaussian-parametric, or Gaussian runs and optionally submit them."
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
        "--require-epnfs",
        action="store_true",
        help=(
            "Set queue=EPNFS in generated ini files so bilby_pipe emits "
            "requirements = ((TARGET.EPNFS =?= True)) in Condor submit files."
        ),
    )
    parser.add_argument(
        "--disable-calibration",
        action="store_true",
        help="Disable calibration uncertainties in every generated run.",
    )
    parser.add_argument(
        "--likelihood",
        choices=(*PARAMETRIC_NOISE_LIKELIHOODS, "gaussian"),
        default="gaussian",
        help=(
            "Primary recovery likelihood to generate. Standard Gaussian runs "
            "always use the default single frequency band "
            f"({DEFAULT_NUM_FREQUENCY_BANDS})."
        ),
    )
    parser.add_argument(
        "--add-gaussian",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When a parametric-noise likelihood is selected, also generate a "
            "single-band standard-Gaussian companion run. Enabled by default; "
            "pass --no-add-gaussian to disable it."
        ),
    )
    parser.add_argument(
        "--noise-only-inference",
        action="store_true",
        help=(
            "Generate a parametric-noise run that infers only the noise "
            "parameter(s). The recovery waveform source model is replaced with "
            "an identically zero waveform and the generated prior keeps only "
            "the selected likelihood's noise parameter(s)."
        ),
    )
    parser.add_argument(
        "--detector-dependent-noise",
        "--detector-dependent-nu",
        dest="detector_dependent_noise",
        action="store_true",
        help=(
            "Generate detector-specific noise parameters for parametric-noise "
            "likelihoods."
        ),
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help=(
            "Use a joint network Student-t or Hyperbolic likelihood. By default, "
            "detector likelihoods are factorized, including when their noise "
            "parameters are shared."
        ),
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=None,
        help=(
            "Detector names used when building detector-dependent noise-parameter "
            "priors. Defaults to the selected event."
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
    validate_likelihood_arguments(args)
    explicit_num_frequency_bands = getattr(
        args,
        "num_frequency_bands_was_explicit",
        args.num_frequency_bands is not None,
    )
    if args.likelihood == "gaussian":
        if args.add_gaussian is True:
            raise ValueError(
                "--add-gaussian requires a parametric-noise likelihood. "
                "Gaussian is already the selected primary likelihood."
            )
        if explicit_num_frequency_bands:
            raise ValueError(
                "--likelihood gaussian cannot be combined with --num-frequency-bands. "
                f"Gaussian runs always use the default value {DEFAULT_NUM_FREQUENCY_BANDS}."
            )
        return ["gaussian"]
    if args.noise_only_inference:
        return [args.likelihood]
    if args.add_gaussian is not False:
        return [args.likelihood, "gaussian"]
    if args.likelihood in PARAMETRIC_NOISE_LIKELIHOODS:
        return [args.likelihood]
    raise ValueError(f"Unknown likelihood selection: {args.likelihood}")


def heavy_tailed_likelihood_selected(likelihood: str) -> bool:
    return likelihood in HEAVY_TAILED_LIKELIHOODS


def parametric_noise_likelihood_selected(likelihood: str) -> bool:
    return likelihood in PARAMETRIC_NOISE_LIKELIHOODS


def validate_noise_only_arguments(args: argparse.Namespace) -> None:
    if not getattr(args, "noise_only_inference", False):
        return
    if not parametric_noise_likelihood_selected(args.likelihood):
        raise ValueError(
            "--noise-only-inference requires a parametric-noise likelihood "
            "because standard Gaussian recovery has no sampled noise parameters."
        )
    if args.add_gaussian is True:
        raise ValueError(
            "--noise-only-inference cannot be combined with --add-gaussian "
            "because the Gaussian companion run has no sampled noise parameters."
        )


def validate_likelihood_arguments(args: argparse.Namespace) -> None:
    validate_noise_only_arguments(args)
    if not getattr(args, "joint", False):
        return
    if not heavy_tailed_likelihood_selected(args.likelihood):
        raise ValueError(
            "--joint requires --likelihood student or hyperbolic"
        )
    if args.detector_dependent_noise:
        raise ValueError(
            "--joint cannot be combined with --detector-dependent-noise"
        )


def build_run_requests(
    args: argparse.Namespace,
    *,
    band_counts,
) -> list[tuple[str, int]]:
    validate_likelihood_arguments(args)
    if args.likelihood == "gaussian":
        return [("gaussian", DEFAULT_NUM_FREQUENCY_BANDS)]

    requests = [(args.likelihood, band_count) for band_count in band_counts]
    if args.noise_only_inference:
        return requests
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


def noise_dependency_directory_token(detector_dependent_noise: bool) -> str:
    if detector_dependent_noise:
        return "detector_dependent_noise"
    return "detector_independent_noise"


def explicit_run_directory_stem(
    *,
    hypothesis: str,
    band_count: int,
    detector_dependent_noise: bool,
    waveform_suffix: str,
    joint: bool = False,
) -> str:
    joint_suffix = "_joint" if joint else ""
    return (
        f"{hypothesis}_"
        f"{noise_dependency_directory_token(detector_dependent_noise)}"
        f"{joint_suffix}_"
        f"N{band_count}{waveform_suffix}"
    )


def explicit_run_label(
    *,
    label_prefix: str,
    hypothesis: str,
    band_count: int,
    detector_dependent_noise: bool,
    waveform_suffix: str,
    joint: bool = False,
) -> str:
    run_directory_stem = explicit_run_directory_stem(
        hypothesis=hypothesis,
        band_count=band_count,
        detector_dependent_noise=detector_dependent_noise,
        waveform_suffix=waveform_suffix,
        joint=joint,
    )
    return f"{label_prefix}_{run_directory_stem}"


def load_template(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing template: {path}")
    return path.read_text(encoding="utf-8")


def build_nu_priors(
    band_count: int,
    *,
    detector_dependent_noise: bool = False,
    detectors: list[str] | tuple[str, ...] = DEFAULT_DETECTORS,
) -> str:
    if not detector_dependent_noise:
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


def build_hyperbolic_priors(
    band_count: int,
    *,
    detector_dependent_noise: bool = False,
    detectors: list[str] | tuple[str, ...] = DEFAULT_DETECTORS,
) -> str:
    lines = []
    if detector_dependent_noise:
        if band_count == 1:
            parameter_keys = [
                (f"alpha_{detector}", "alpha") for detector in detectors
            ] + [
                (f"delta_{detector}", "delta") for detector in detectors
            ]
        else:
            parameter_keys = [
                (f"alpha_{detector}_{index}", "alpha")
                for detector in detectors
                for index in range(1, band_count + 1)
            ] + [
                (f"delta_{detector}_{index}", "delta")
                for detector in detectors
                for index in range(1, band_count + 1)
            ]
    elif band_count == 1:
        parameter_keys = [("alpha", "alpha"), ("delta", "delta")]
    else:
        parameter_keys = [
            (f"alpha_{index}", "alpha")
            for index in range(1, band_count + 1)
        ] + [
            (f"delta_{index}", "delta")
            for index in range(1, band_count + 1)
        ]

    for key, parameter_name in parameter_keys:
        if parameter_name == "alpha":
            minimum = DEFAULT_HYPERBOLIC_ALPHA_MIN
            maximum = DEFAULT_HYPERBOLIC_ALPHA_MAX
        else:
            minimum = DEFAULT_HYPERBOLIC_DELTA_MIN
            maximum = DEFAULT_HYPERBOLIC_DELTA_MAX
        lines.append(
            f"{key} = Uniform(name='{key}', minimum={minimum}, maximum={maximum})"
        )
    return "\n".join(lines)


def build_log_psd_scale_priors(
    band_count: int,
    *,
    detector_dependent_noise: bool = False,
    detectors: list[str] | tuple[str, ...] = DEFAULT_DETECTORS,
) -> str:
    if detector_dependent_noise:
        if band_count == 1:
            parameter_keys = [
                f"log_psd_scale_{detector}" for detector in detectors
            ]
        else:
            parameter_keys = [
                f"log_psd_scale_{detector}_{index}"
                for detector in detectors
                for index in range(1, band_count + 1)
            ]
    elif band_count == 1:
        parameter_keys = ["log_psd_scale"]
    else:
        parameter_keys = [
            f"log_psd_scale_{index}" for index in range(1, band_count + 1)
        ]
    return "\n".join(
        f"{key} = Uniform(name='{key}', minimum={DEFAULT_LOG_PSD_SCALE_MIN}, "
        f"maximum={DEFAULT_LOG_PSD_SCALE_MAX})"
        for key in parameter_keys
    )


def render_prior(
    prior_template: str,
    band_count: int,
    *,
    hypothesis: str,
    detector_dependent_noise: bool = False,
    detectors: list[str] | tuple[str, ...] = DEFAULT_DETECTORS,
    template_settings: dict[str, object],
    sine_gaussian_config,
    noise_only_inference: bool = False,
) -> str:
    if noise_only_inference:
        return render_noise_only_prior(
            hypothesis=hypothesis,
            band_count=band_count,
            detector_dependent_noise=detector_dependent_noise,
            detectors=detectors,
            template_settings=template_settings,
        )

    if hypothesis == "student":
        noise_prior_block = build_nu_priors(
            band_count,
            detector_dependent_noise=detector_dependent_noise,
            detectors=detectors,
        )
    elif hypothesis == "hyperbolic":
        noise_prior_block = build_hyperbolic_priors(
            band_count,
            detector_dependent_noise=detector_dependent_noise,
            detectors=detectors,
        )
    elif hypothesis == "gaussian-parametric":
        noise_prior_block = build_log_psd_scale_priors(
            band_count,
            detector_dependent_noise=detector_dependent_noise,
            detectors=detectors,
        )
    elif hypothesis == "gaussian":
        noise_prior_block = ""
    else:
        raise ValueError(f"Unknown hypothesis '{hypothesis}'")
    sine_gaussian_prior_block = build_sine_gaussian_prior_block(
        sine_gaussian_config,
        minimum_frequency=template_settings["minimum_frequency"],
        maximum_frequency=template_settings["maximum_frequency"],
    )
    rendered = prior_template.replace(
        "__NU_PRIORS__",
        combine_prior_blocks(noise_prior_block, sine_gaussian_prior_block),
    )
    for class_name in (
        "UniformInComponentsChirpMass",
        "UniformInComponentsMassRatio",
    ):
        rendered = rendered.replace(
            f"= {class_name}(",
            f"= bilby.gw.prior.{class_name}(",
        )
    return rendered


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


def build_pesummary_arguments(template_settings: dict[str, object]) -> dict[str, object]:
    arguments = dict(DEFAULT_PESUMMARY_ARGUMENTS)
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
        arguments["calibration"] = calibration
    psd_dict = template_settings.get("psd_dict")
    if psd_dict not in (None, "None"):
        arguments["psd"] = psd_dict
    return arguments


def replace_template_setting_placeholder(
    value,
    *,
    placeholder: str,
    replacement: str,
):
    if isinstance(value, str):
        return value.replace(placeholder, replacement)
    if isinstance(value, dict):
        return {
            key: replace_template_setting_placeholder(
                item,
                placeholder=placeholder,
                replacement=replacement,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            replace_template_setting_placeholder(
                item,
                placeholder=placeholder,
                replacement=replacement,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            replace_template_setting_placeholder(
                item,
                placeholder=placeholder,
                replacement=replacement,
            )
            for item in value
        )
    return value


def resolve_template_settings(
    template_settings: dict[str, object],
    *,
    working_directory: Path,
) -> dict[str, object]:
    resolved_working_directory = str(working_directory)
    return {
        key: replace_template_setting_placeholder(
            value,
            placeholder=WORKING_DIRECTORY_PLACEHOLDER,
            replacement=resolved_working_directory,
        )
        for key, value in template_settings.items()
    }


def required_local_data_paths(
    template_settings: dict[str, object],
    *,
    working_directory: Path,
) -> list[Path]:
    resolved = resolve_template_settings(
        template_settings,
        working_directory=working_directory,
    )
    paths = []
    for setting in LOCAL_DATA_SETTINGS:
        values = resolved.get(setting)
        if values in (None, "None"):
            continue
        if isinstance(values, str):
            values = [values]
        elif isinstance(values, dict):
            values = values.values()
        else:
            raise ValueError(f"{setting} must resolve to a path or dictionary")
        for value in values:
            if not isinstance(value, str):
                raise ValueError(f"{setting} paths must be strings")
            if "://" not in value:
                paths.append(Path(value).expanduser())
    return sorted(set(paths))


def preflight_local_data(
    template_settings: dict[str, object],
    *,
    event: str,
    working_directory: Path,
) -> None:
    required_paths = required_local_data_paths(
        template_settings,
        working_directory=working_directory,
    )
    missing = [path for path in required_paths if not path.is_file()]
    if not missing:
        return

    downloader = working_directory / "download_glitch_data.py"
    glitch_directory = working_directory / "glitch_data"
    downloadable = [
        path for path in missing if path.parent == glitch_directory
    ]
    if downloadable and downloader.is_file():
        print(
            f"Missing {len(downloadable)} local data file(s) for {event}; "
            "running the catalog downloader."
        )
        subprocess.run(
            [sys.executable, str(downloader), "--event", event],
            check=True,
            cwd=working_directory,
        )

    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing required local data for {event}:\n{formatted}"
        )


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
    *,
    insert_after: str | None = None,
) -> str:
    lines = text.splitlines()
    prefix = f"{key}="
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f"{key}={value}"
            return "\n".join(lines) + "\n"
    if insert_after is not None:
        anchor = f"{insert_after}="
        for index, line in enumerate(lines):
            if line.startswith(anchor):
                lines.insert(index + 1, f"{key}={value}")
                return "\n".join(lines) + "\n"
    lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def disable_calibration_settings(text: str) -> str:
    rendered = replace_line(text, "calibration-model", "None")
    rendered = replace_line(rendered, "spline-calibration-envelope-dict", "None")
    rendered = replace_line(
        rendered,
        "spline-calibration-amplitude-uncertainty-dict",
        "None",
    )
    rendered = replace_line(
        rendered,
        "spline-calibration-phase-uncertainty-dict",
        "None",
    )
    rendered = replace_line(rendered, "calibration-marginalization", "False")
    rendered = replace_line(rendered, "calibration-lookup-table", "None")
    return rendered


def apply_noise_only_inference_settings(text: str) -> str:
    rendered = replace_line(text, "default-prior", NOISE_ONLY_DEFAULT_PRIOR)
    rendered = replace_line(rendered, "distance-marginalization", "False")
    rendered = replace_line(rendered, "phase-marginalization", "False")
    rendered = replace_line(rendered, "time-marginalization", "False")
    rendered = replace_line(rendered, "jitter-time", "False")
    rendered = replace_line(
        rendered,
        "frequency-domain-source-model",
        NOISE_ONLY_SOURCE_MODEL,
    )
    return disable_calibration_settings(rendered)


def format_prior_value(value: float) -> str:
    return repr(float(value))


def time_parameter_name(time_reference: str) -> str:
    normalized = str(time_reference).strip().lower()
    if normalized in {"geocent", "geocenter"}:
        return "geocent_time"
    return f"{time_reference}_time"


def render_noise_only_prior(
    *,
    hypothesis: str,
    band_count: int,
    detector_dependent_noise: bool,
    detectors: list[str] | tuple[str, ...],
    template_settings: dict[str, object],
) -> str:
    if hypothesis == "student":
        noise_prior_block = build_nu_priors(
            band_count,
            detector_dependent_noise=detector_dependent_noise,
            detectors=detectors,
        )
    elif hypothesis == "hyperbolic":
        noise_prior_block = build_hyperbolic_priors(
            band_count,
            detector_dependent_noise=detector_dependent_noise,
            detectors=detectors,
        )
    elif hypothesis == "gaussian-parametric":
        noise_prior_block = build_log_psd_scale_priors(
            band_count,
            detector_dependent_noise=detector_dependent_noise,
            detectors=detectors,
        )
    elif hypothesis == "gaussian":
        noise_prior_block = ""
    else:
        raise ValueError(f"Unknown hypothesis '{hypothesis}'")

    prior_blocks = [noise_prior_block]
    time_parameter = time_parameter_name(
        str(template_settings.get("time_reference", "geocent"))
    )
    prior_blocks.append(
        "{} = DeltaFunction(name='{}', peak={})".format(
            time_parameter,
            time_parameter,
            format_prior_value(template_settings["trigger_time"]),
        )
    )
    return "\n\n".join(block for block in prior_blocks if block) + "\n"


def render_ini(
    ini_template: str,
    *,
    hypothesis: str,
    label: str,
    outdir: str,
    webdir: str,
    prior_file: Path,
    band_count: int,
    detector_dependent_noise: bool,
    working_directory: Path,
    accounting_user: str,
    container_image: str | None,
    require_epnfs: bool,
    maxmcmc: int | None,
    template_settings: dict[str, object],
    sine_gaussian_config,
    noise_only_inference: bool = False,
    disable_calibration: bool = False,
    joint: bool = False,
) -> str:
    resolved_template_settings = resolve_template_settings(
        template_settings,
        working_directory=working_directory,
    )
    replacements = {
        "__LABEL__": label,
        "__OUTDIR__": outdir,
        "__WEBDIR__": webdir,
        "__PRIOR_FILE__": str(prior_file),
        "__NUM_FREQUENCY_BANDS__": str(band_count),
        "__DETECTOR_DEPENDENT_NOISE__": str(detector_dependent_noise),
        WORKING_DIRECTORY_PLACEHOLDER: str(working_directory),
    }
    rendered = ini_template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    rendered = replace_line(rendered, "accounting-user", accounting_user)
    rendered = replace_line(rendered, "container", container_image or "None")
    rendered = replace_or_append_line(
        rendered,
        "request-memory",
        str(DEFAULT_REQUEST_MEMORY_GB),
    )
    rendered = replace_or_append_line(
        rendered,
        "request-memory-generation",
        str(DEFAULT_REQUEST_MEMORY_GB),
        insert_after="request-memory",
    )
    rendered = replace_or_append_line(
        rendered,
        "request-cpus",
        str(DEFAULT_REQUEST_CPUS),
        insert_after="request-memory-generation",
    )
    rendered = replace_or_append_line(rendered, "transfer-files", "True")
    rendered = replace_or_append_line(rendered, "osg", "True")
    rendered = replace_or_append_line(rendered, "desired-sites", "None")
    if require_epnfs:
        rendered = replace_line(rendered, "queue", "EPNFS")
    rendered = replace_line(
        rendered,
        "environment-variables",
        repr(DEFAULT_ENVIRONMENT_VARIABLES),
    )
    if noise_only_inference:
        rendered = replace_line(rendered, "create-summary", "False")
        rendered = replace_line(rendered, "summarypages-arguments", "None")
    else:
        rendered = replace_line(rendered, "create-summary", "True")
        summary_settings = dict(resolved_template_settings)
        if disable_calibration:
            summary_settings["spline_calibration_envelope_dict"] = None
        rendered = replace_line(
            rendered,
            "summarypages-arguments",
            repr(build_pesummary_arguments(summary_settings)),
        )
    if noise_only_inference:
        rendered = apply_noise_only_inference_settings(rendered)
    elif disable_calibration:
        rendered = disable_calibration_settings(rendered)
    sampler_kwargs = dict(template_settings["sampler_kwargs"])
    sampler_kwargs["nlive"] = effective_nlive(
        int(sampler_kwargs["nlive"]),
        sine_gaussian_config,
    )
    if "npool" in sampler_kwargs:
        sampler_kwargs["npool"] = min(
            int(sampler_kwargs["npool"]),
            DEFAULT_REQUEST_CPUS,
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
        rendered = replace_line(
            rendered,
            "extra-likelihood-kwargs",
            (
                "{'nu': 8.0, 'infer_nu': True, "
                f"'num_frequency_bands': {band_count}, "
                f"'detector_dependent_noise': {detector_dependent_noise}, "
                f"'joint': {joint}"
                "}"
            ),
        )
    elif hypothesis == "hyperbolic":
        rendered = replace_line(
            rendered,
            "likelihood-type",
            "bilby.gw.likelihood.HyperbolicGravitationalWaveTransient",
        )
        rendered = replace_line(
            rendered,
            "extra-likelihood-kwargs",
            (
                "{'alpha': "
                f"{DEFAULT_HYPERBOLIC_ALPHA}, 'delta': {DEFAULT_HYPERBOLIC_DELTA}, "
                "'infer_alpha': True, 'infer_delta': True, "
                f"'num_frequency_bands': {band_count}, "
                f"'detector_dependent_noise': {detector_dependent_noise}, "
                f"'joint': {joint}"
                "}"
            ),
        )
    elif hypothesis == "gaussian-parametric":
        rendered = replace_line(
            rendered,
            "likelihood-type",
            "bilby.gw.likelihood.GaussianParametricGravitationalWaveTransient",
        )
        rendered = replace_line(
            rendered,
            "extra-likelihood-kwargs",
            (
                "{'log_psd_scale': 0.0, 'infer_log_psd_scale': True, "
                f"'num_psd_frequency_bands': {band_count}, "
                f"'detector_dependent_noise': {detector_dependent_noise}"
                "}"
            ),
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
    detector_dependent_noise: bool,
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
    maxmcmc: int | None,
    sine_gaussian_config,
    noise_only_inference: bool,
    disable_calibration: bool,
    approximant_suffix: str = "",
    joint: bool = False,
) -> Path:
    waveform_suffix = sine_gaussian_config.label_suffix + approximant_suffix
    if hypothesis == "student":
        run_band_count = band_count
        mode_suffix = (
            "_detector_dependent_noise" if detector_dependent_noise else ""
        )
        joint_suffix = "_joint" if joint else ""
        run_directory_stem = explicit_run_directory_stem(
            hypothesis=hypothesis,
            band_count=run_band_count,
            detector_dependent_noise=detector_dependent_noise,
            waveform_suffix=waveform_suffix,
            joint=joint,
        )
        label = explicit_run_label(
            label_prefix=label_prefix,
            hypothesis=hypothesis,
            band_count=run_band_count,
            detector_dependent_noise=detector_dependent_noise,
            waveform_suffix=waveform_suffix,
            joint=joint,
        )
        run_directory_name = build_run_directory_name(run_directory_stem, outdir_label)
        run_outdir = f"{outdir_base}/{run_directory_name}"
        prior_path = (
            prior_dir
            / f"{file_prefix}{mode_suffix}{joint_suffix}_N"
            f"{run_band_count}{waveform_suffix}.prior"
        ).resolve()
        ini_path = (
            ini_dir
            / f"{file_prefix}_t_student{mode_suffix}{joint_suffix}_N"
            f"{run_band_count}{waveform_suffix}.ini"
        ).resolve()
        run_detector_dependent_noise = detector_dependent_noise
    elif hypothesis == "hyperbolic":
        run_band_count = band_count
        mode_suffix = (
            "_detector_dependent_noise" if detector_dependent_noise else ""
        )
        joint_suffix = "_joint" if joint else ""
        run_directory_stem = explicit_run_directory_stem(
            hypothesis=hypothesis,
            band_count=run_band_count,
            detector_dependent_noise=detector_dependent_noise,
            waveform_suffix=waveform_suffix,
            joint=joint,
        )
        label = explicit_run_label(
            label_prefix=label_prefix,
            hypothesis=hypothesis,
            band_count=run_band_count,
            detector_dependent_noise=detector_dependent_noise,
            waveform_suffix=waveform_suffix,
            joint=joint,
        )
        run_directory_name = build_run_directory_name(run_directory_stem, outdir_label)
        run_outdir = f"{outdir_base}/{run_directory_name}"
        run_webdir = f"{webdir_base}/{run_directory_name}"
        prior_path = (
            prior_dir
            / f"{file_prefix}_hyperbolic{mode_suffix}{joint_suffix}_N"
            f"{run_band_count}{waveform_suffix}.prior"
        ).resolve()
        ini_path = (
            ini_dir
            / f"{file_prefix}_hyperbolic{mode_suffix}{joint_suffix}_N"
            f"{run_band_count}{waveform_suffix}.ini"
        ).resolve()
        run_detector_dependent_noise = detector_dependent_noise
    elif hypothesis == "gaussian-parametric":
        run_band_count = band_count
        mode_suffix = (
            "_detector_dependent_noise" if detector_dependent_noise else ""
        )
        run_directory_stem = explicit_run_directory_stem(
            hypothesis=hypothesis,
            band_count=run_band_count,
            detector_dependent_noise=detector_dependent_noise,
            waveform_suffix=waveform_suffix,
        )
        label = explicit_run_label(
            label_prefix=label_prefix,
            hypothesis=hypothesis,
            band_count=run_band_count,
            detector_dependent_noise=detector_dependent_noise,
            waveform_suffix=waveform_suffix,
        )
        run_directory_name = build_run_directory_name(run_directory_stem, outdir_label)
        run_outdir = f"{outdir_base}/{run_directory_name}"
        prior_path = (
            prior_dir
            / f"{file_prefix}_gaussian_parametric{mode_suffix}_N"
            f"{run_band_count}{waveform_suffix}.prior"
        ).resolve()
        ini_path = (
            ini_dir
            / f"{file_prefix}_gaussian_parametric{mode_suffix}_N"
            f"{run_band_count}{waveform_suffix}.ini"
        ).resolve()
        run_detector_dependent_noise = detector_dependent_noise
    elif hypothesis == "gaussian":
        run_band_count = DEFAULT_NUM_FREQUENCY_BANDS
        run_directory_stem = explicit_run_directory_stem(
            hypothesis=hypothesis,
            band_count=run_band_count,
            detector_dependent_noise=False,
            waveform_suffix=waveform_suffix,
        )
        label = explicit_run_label(
            label_prefix=label_prefix,
            hypothesis=hypothesis,
            band_count=run_band_count,
            detector_dependent_noise=False,
            waveform_suffix=waveform_suffix,
        )
        run_directory_name = build_run_directory_name(run_directory_stem, outdir_label)
        run_outdir = f"{outdir_base}/{run_directory_name}"
        prior_path = (prior_dir / f"{file_prefix}_gaussian{waveform_suffix}.prior").resolve()
        ini_path = (ini_dir / f"{file_prefix}_gaussian{waveform_suffix}.ini").resolve()
        run_detector_dependent_noise = False
    else:
        raise ValueError(f"Unknown hypothesis '{hypothesis}'")

    run_webdir = f"{webdir_base}/{run_directory_name}/web"

    prior_path.write_text(
        render_prior(
            prior_template,
            run_band_count,
            hypothesis=hypothesis,
            detector_dependent_noise=run_detector_dependent_noise,
            detectors=detectors,
            template_settings=template_settings,
            sine_gaussian_config=sine_gaussian_config,
            noise_only_inference=noise_only_inference,
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
            detector_dependent_noise=run_detector_dependent_noise,
            working_directory=working_directory,
            accounting_user=accounting_user,
            container_image=container_image,
            require_epnfs=require_epnfs,
            maxmcmc=maxmcmc,
            template_settings=template_settings,
            sine_gaussian_config=sine_gaussian_config,
            noise_only_inference=noise_only_inference,
            disable_calibration=disable_calibration,
            joint=joint,
        ),
        encoding="utf-8",
    )

    band_fragment = (
        f" N={run_band_count}"
        if parametric_noise_likelihood_selected(hypothesis)
        else ""
    )
    print(
        f"Prepared {hypothesis}{band_fragment} "
        f"({sine_gaussian_config.description}):"
    )
    print(f"  prior: {prior_path}")
    print(f"  ini:   {ini_path}")
    return ini_path


def submit_run(ini_path: Path, *, submit_directory: Path) -> None:
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
    if not args.dry_run:
        preflight_local_data(
            template_settings,
            event=args.event,
            working_directory=working_directory,
        )

    if args.waveform_approximant is not None:
        template_approximant = template_settings["waveform_approximant"]
        min_freq = template_settings["minimum_frequency"]
        if isinstance(min_freq, dict):
            detector_freqs = [v for k, v in min_freq.items() if k != "waveform"]
            min_freq = dict(min_freq, waveform=min(detector_freqs) if detector_freqs else 20.0)
        template_settings = dict(
            template_settings,
            waveform_approximant=args.waveform_approximant,
            minimum_frequency=min_freq,
        )
        approximant_suffix = (
            f"_{args.waveform_approximant}"
            if args.waveform_approximant != template_approximant
            else ""
        )
    else:
        approximant_suffix = ""

    sine_gaussian_configs = resolve_sine_gaussian_configurations(
        num_sine_gaussians=args.num_sine_gaussians,
        range_mode=args.sine_gaussian_range,
        mode=args.sine_gaussian_mode,
        incoherent_detectors=args.incoherent_detectors,
        incoherent_counts_spec=args.incoherent_sg_counts,
        detectors=detectors,
    )
    for sine_gaussian_config in sine_gaussian_configs:
        if args.noise_only_inference and sine_gaussian_config.enabled:
            raise ValueError(
                "--noise-only-inference cannot be combined with recovery "
                "sine-Gaussian parameters."
            )
        require_supported_sine_gaussian_source_model(
            template_settings,
            sine_gaussian_config,
        )

    ini_dir = args.ini_dir.expanduser().resolve()
    prior_dir = args.prior_dir.expanduser().resolve()
    ini_dir.mkdir(parents=True, exist_ok=True)
    prior_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        Path(outdir_base).mkdir(parents=True, exist_ok=True)
        Path(webdir_base).mkdir(parents=True, exist_ok=True)

    if args.range_mode:
        band_counts = range(1, args.num_frequency_bands + 1)
    else:
        band_counts = [args.num_frequency_bands]
    run_requests = build_run_requests(args, band_counts=band_counts)

    for sine_gaussian_config in sine_gaussian_configs:
        for hypothesis, band_count in run_requests:
            if (
                hypothesis == "gaussian"
                and args.likelihood in PARAMETRIC_NOISE_LIKELIHOODS
                and not args.dry_run
            ):
                waveform_suffix = (
                    sine_gaussian_config.label_suffix + approximant_suffix
                )
                run_directory = Path(outdir_base) / build_run_directory_name(
                    explicit_run_directory_stem(
                        hypothesis=hypothesis,
                        band_count=DEFAULT_NUM_FREQUENCY_BANDS,
                        detector_dependent_noise=False,
                        waveform_suffix=waveform_suffix,
                    ),
                    args.outdir_label,
                )
                if run_directory.exists():
                    print(
                        "Skipping Gaussian companion because its output "
                        f"directory already exists: {run_directory}"
                    )
                    continue
            ini_path = prepare_run(
                hypothesis=hypothesis,
                band_count=band_count,
                ini_template=ini_template,
                prior_template=prior_template,
                template_settings=template_settings,
                ini_dir=ini_dir,
                prior_dir=prior_dir,
                detector_dependent_noise=args.detector_dependent_noise,
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
                maxmcmc=args.maxmcmc,
                sine_gaussian_config=sine_gaussian_config,
                noise_only_inference=args.noise_only_inference,
                disable_calibration=args.disable_calibration,
                approximant_suffix=approximant_suffix,
                joint=args.joint,
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
