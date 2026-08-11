#!/usr/bin/env python3

"""Stage a GW231123 Student-t injection and generate bilby_pipe configs.

The generated ini/prior files are rendered from the same GW231123 template files
used for the real-data analyses in this directory. Only the path- and
injection-specific settings are replaced, so the resulting configs stay as close
as possible to the production templates. `--injection-noise` chooses the staged
noise model, `--likelihood` chooses the recovery likelihood, and
Student-t runs also generate a single-band Gaussian companion run by default.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bilby

from container_creation.submission_container_utils import (
    add_container_arguments,
    resolve_container_image,
)
from submission_sine_gaussian_utils import (
    SINE_GAUSSIAN_HRSS_BOUNDS,
    SINE_GAUSSIAN_Q_BOUNDS,
    SINE_GAUSSIAN_TIME_OFFSET_BOUNDS,
    add_sine_gaussian_arguments,
    apply_sine_gaussian_waveform_settings,
    build_sine_gaussian_prior_block,
    combine_prior_blocks,
    effective_nlive,
    parse_ini_dict_string,
    parse_template_value,
    positive_int,
    read_template_settings,
    require_supported_sine_gaussian_source_model,
    resolve_sine_gaussian_configurations,
    sine_gaussian_frequency_bounds,
    validate_submission_local_paths,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONTAINER_IMAGES_FILE = (
    SCRIPT_DIR / "container_creation" / "container_images.json"
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
DEFAULT_BASE_SUBDIR = (
    Path("public_html") / "GW231123" / "t_Student" / "Runs_injections"
)
DEFAULT_ENVIRONMENT_VARIABLES = {
    "HDF5_USE_FILE_LOCKING": False,
    "NUMBA_CACHE_DIR": "/tmp",
    "OMP_NUM_THREADS": 1,
    "OMP_PROC_BIND": False,
    "LAL_DATA_PATH": "/scratch/lalsimulation:/opt/lalsimulation-data",
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
INI_TEMPLATE_PATH = (
    SCRIPT_DIR / "Initialisation_file_templates" / "GW231123_t_student_template.ini"
)
PRIOR_TEMPLATE_PATH = SCRIPT_DIR / "Prior_templates" / "GW231123_template.prior"
DEFAULT_POSTERIOR_PATH = (
    SCRIPT_DIR / "LVK_posteriors" / "GW231123" / "posterior_samples.h5"
)
LEGACY_POSTERIOR_PATHS = (
    SCRIPT_DIR / "LVK_posterior" / "posterior_samples.h5",
    SCRIPT_DIR
    / "Data"
    / "LVK_run"
    / "bilby-NRSur7dq4"
    / "samples"
    / "posterior_samples.h5",
)
DEFAULT_STAGING_RANDOM_SEED = 12345
TEST_INJECTION_CHIRP_MASS_CREDIBLE_INTERVAL = 0.99
TEST_INJECTION_NU_MAX = 100.0
DEFAULT_NLIVE = 2000
TEST_INJECTION_NLIVE = 256
DEFAULT_NUM_FREQUENCY_BANDS = 1
DEFAULT_INJECTION_NOISE = "student"
FD_DATA_FORMAT = "bilby_frequency_domain_hdf5"
INJECTED_SINE_GAUSSIAN_VALUES_PATH = (
    SCRIPT_DIR / "runbooks" / "injected_sine_gaussian_values.json"
)
INJECTED_SINE_GAUSSIAN_COMPONENT_KEYS = (
    "hrss",
    "Q",
    "frequency",
    "time_offset",
    "phase_offset",
)

INJECTION_KEYS = (
    "mass_1",
    "mass_2",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "luminosity_distance",
    "theta_jn",
    "psi",
    "phase",
    "geocent_time",
    "ra",
    "dec",
)
TEST_INJECTION_FIXED_KEYS = (
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "luminosity_distance",
    "theta_jn",
    "psi",
    "phase",
    "ra",
    "dec",
)
def outdir_label(value: str) -> str:
    label = value.strip()
    if not label:
        raise argparse.ArgumentTypeError("outdir label must not be empty")
    if any(separator and separator in label for separator in (os.sep, os.altsep)):
        raise argparse.ArgumentTypeError("outdir label must not contain path separators")
    return label


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home-dir",
        type=Path,
        default=DEFAULT_HOME_DIR,
        help=(
            "Base home directory containing public_html, used to build the "
            "default --base-dir when --base-dir is not provided."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Root directory where staged data, generated ini/prior files, and "
            "run/web folders are written. Defaults to "
            "<home-dir>/public_html/GW231123/t_Student/Runs_injections."
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
        default="GW231123_student_t_injection",
        help=(
            "Prefix used to name the Gaussian/Student runs and the staged-data files."
        ),
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
        "--nu-injection",
        default="2.1",
        help=(
            "Student-t nu specification used when --injection-noise student is "
            "selected. Accepts a scalar, a per-band list, or a detector "
            "dictionary (values may be scalar or per-band lists). The same "
            "values are also used to seed Student recovery runs."
        ),
    )
    parser.add_argument(
        "--injection-noise",
        choices=("student", "gaussian", "zero-gaussian"),
        default=DEFAULT_INJECTION_NOISE,
        help=(
            "Noise model used when staging the injected data. "
            "`zero-gaussian` means zero Gaussian noise. "
            f"Default: {DEFAULT_INJECTION_NOISE}."
        ),
    )
    parser.add_argument(
        "--noise-generation-seed",
        type=int,
        default=None,
        help=(
            "Random seed used to generate the staged injection noise. "
            "Defaults to the template sampling-seed when it is an integer; "
            f"otherwise defaults to {DEFAULT_STAGING_RANDOM_SEED}."
        ),
    )
    parser.add_argument(
        "--injection-duration",
        type=positive_float,
        default=None,
        help=(
            "Duration in seconds used to stage injected data and write the "
            "generated recovery ini files. Defaults to the template duration."
        ),
    )
    parser.add_argument(
        "--frequency-domain-injection",
        "--fd-injection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Stage injected data as frequency-domain strain and make the "
            "generated bilby_pipe jobs load it directly into the likelihood. "
            "This avoids the time-domain HDF5 round-trip and Tukey window."
        ),
    )
    parser.add_argument(
        "--num-frequency-bands",
        type=positive_int,
        default=None,
        help=(
            "Number of frequency bands for Student-t noise generation and "
            "Student likelihood nu parameterization. "
            f"Defaults to {DEFAULT_NUM_FREQUENCY_BANDS}."
        ),
    )
    parser.add_argument(
        "--detector-dependent-nu",
        action="store_true",
        help=(
            "Use detector-dependent nu in the Student likelihood. If --nu-injection "
            "is not a detector dictionary, the same nu specification is repeated "
            "for each detector in the staged injection."
        ),
    )
    parser.add_argument(
        "--nu-min",
        type=float,
        default=2.1,
        help="Lower prior bound for the inferred Student-t nu parameter.",
    )
    parser.add_argument(
        "--nu-max",
        type=float,
        default=1000.0,
        help="Upper prior bound for the inferred Student-t nu parameter.",
    )
    parser.add_argument(
        "--nlive",
        type=int,
        default=None,
        help=(
            "Base nested-sampler live points written into sampler-kwargs before "
            "the automatic sine-Gaussian uplift: +500 for one recovered SG, "
            "+1000 for two or more, plus a further +500 for "
            "coherent-independent recovery. Defaults to "
            f"{TEST_INJECTION_NLIVE} for --test-injection and "
            f"{DEFAULT_NLIVE} otherwise."
        ),
    )
    parser.add_argument(
        "--naccept",
        type=int,
        default=60,
        help=(
            "Dynesty acceptance-walk target naccept to write into sampler-kwargs."
        ),
    )
    parser.add_argument(
        "--maxmcmc",
        type=positive_int,
        default=None,
        help=(
            "Optional Dynesty maxmcmc value written into sampler-kwargs. "
            "Defaults to the value in the ini template."
        ),
    )
    parser.add_argument(
        "--local-posterior",
        action="store_true",
        help=(
            "Use a legacy posterior layout if available (first existing file in "
            "Cluster_runs_and_utils/LVK_posterior/posterior_samples.h5 or "
            "Cluster_runs_and_utils/Data/LVK_run/bilby-NRSur7dq4/samples/"
            "posterior_samples.h5). By default, the posterior is read from "
            "Cluster_runs_and_utils/LVK_posteriors/GW231123/posterior_samples.h5."
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
        "--test-injection",
        action="store_true",
        help=(
            "Generate the standard staged injection runs, but edit the standard "
            "prior template to fix the maximum-likelihood injection parameters "
            "read by this script. In this mode, only nu (for Student-t runs) "
            "and chirp_mass are sampled."
        ),
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
        "--submit",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bilby-pipe-executable",
        default="bilby_pipe",
        help="Executable name or absolute path used to call bilby_pipe.",
    )
    add_container_arguments(
        parser,
        default_image_file=DEFAULT_CONTAINER_IMAGES_FILE,
    )
    add_sine_gaussian_arguments(parser)
    add_sine_gaussian_arguments(
        parser,
        prefix="injection",
        subject="injected signal",
    )
    return parser


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    if args.likelihood == "student":
        return ["student"]
    raise ValueError(f"Unknown likelihood selection: {args.likelihood}")


def load_template(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing template: {path}")
    return path.read_text(encoding="utf-8")


def resolve_posterior_path(args: argparse.Namespace) -> Path:
    if args.local_posterior:
        for legacy_path in LEGACY_POSTERIOR_PATHS:
            if legacy_path.is_file():
                return legacy_path
        checked_legacy = "\n".join(f"  - {path}" for path in LEGACY_POSTERIOR_PATHS)
        raise FileNotFoundError(
            "Requested --local-posterior, but no legacy posterior was found. "
            f"Checked:\n{checked_legacy}"
        )
    else:
        posterior_path = DEFAULT_POSTERIOR_PATH
    if not posterior_path.is_file():
        legacy_locations = ", ".join(str(path) for path in LEGACY_POSTERIOR_PATHS)
        raise FileNotFoundError(
            "Default posterior not found at "
            f"{DEFAULT_POSTERIOR_PATH}. "
            "Pass --local-posterior to use a legacy path if present "
            f"({legacy_locations})."
        )
    return posterior_path


def _coerce_nu_per_band(nu_value, num_frequency_bands: int) -> list[float]:
    nu_array = np.asarray(nu_value, dtype=float)
    if nu_array.ndim == 0:
        nu_array = np.repeat(nu_array[None], num_frequency_bands)
    elif nu_array.ndim == 1:
        if len(nu_array) == 1:
            nu_array = np.repeat(nu_array, num_frequency_bands)
        elif len(nu_array) != num_frequency_bands:
            raise ValueError(
                "nu list must contain one entry or exactly one entry per frequency band"
            )
    else:
        raise ValueError("nu values must be scalar or one-dimensional")

    if not np.all(np.isfinite(nu_array)) or np.any(nu_array <= 0):
        raise ValueError("All nu values must be positive and finite")
    return [float(value) for value in nu_array]


def parse_nu_injection_spec(raw_value: str):
    parsed = parse_template_value(raw_value)
    if parsed is None:
        raise ValueError("nu-injection cannot be None")
    return parsed


def resolve_nu_configuration(
    *,
    raw_nu_injection: str,
    detectors: tuple[str, ...],
    num_frequency_bands: int,
    detector_dependent_nu: bool,
) -> tuple[object, object, bool]:
    if int(num_frequency_bands) < 1:
        raise ValueError("num-frequency-bands must be a positive integer")
    parsed_nu = parse_nu_injection_spec(raw_nu_injection)
    if isinstance(parsed_nu, dict):
        resolved_detector_nu = {}
        for detector in detectors:
            if detector not in parsed_nu:
                raise ValueError(
                    f"nu-injection dictionary is missing detector '{detector}'"
                )
            resolved_detector_nu[detector] = _coerce_nu_per_band(
                parsed_nu[detector], num_frequency_bands
            )
        effective_detector_dependent_nu = True
    else:
        shared_nu = _coerce_nu_per_band(parsed_nu, num_frequency_bands)
        if detector_dependent_nu:
            resolved_detector_nu = {
                detector: list(shared_nu) for detector in detectors
            }
            effective_detector_dependent_nu = True
        else:
            resolved_detector_nu = list(shared_nu)
            effective_detector_dependent_nu = False

    if effective_detector_dependent_nu:
        if num_frequency_bands == 1:
            likelihood_nu = [resolved_detector_nu[detector][0] for detector in detectors]
            noise_nu = {detector: values[0] for detector, values in resolved_detector_nu.items()}
        else:
            likelihood_nu = [resolved_detector_nu[detector] for detector in detectors]
            noise_nu = resolved_detector_nu
    else:
        if num_frequency_bands == 1:
            likelihood_nu = resolved_detector_nu[0]
            noise_nu = resolved_detector_nu[0]
        else:
            likelihood_nu = resolved_detector_nu
            noise_nu = resolved_detector_nu

    return noise_nu, likelihood_nu, effective_detector_dependent_nu


def load_maximum_likelihood_injection(posterior_path: Path) -> tuple[dict, float, int]:
    with h5py.File(posterior_path, "r") as posterior_file:
        posterior_samples = posterior_file["C00:NRSur7dq4/posterior_samples"]
        maxl_index = int(posterior_samples["log_likelihood"][:].argmax())
        maxl_parameters = {
            key: float(posterior_samples[key][maxl_index]) for key in INJECTION_KEYS
        }
        maxl_log_likelihood = float(posterior_samples["log_likelihood"][maxl_index])
    return maxl_parameters, maxl_log_likelihood, maxl_index


def load_test_injection_chirp_mass_bounds(
    posterior_path: Path,
    credible_interval: float = TEST_INJECTION_CHIRP_MASS_CREDIBLE_INTERVAL,
) -> tuple[float, float]:
    if not 0 < credible_interval < 1:
        raise ValueError("credible_interval must lie strictly between 0 and 1")

    tail_probability = 0.5 * (1 - credible_interval)
    with h5py.File(posterior_path, "r") as posterior_file:
        posterior_samples = posterior_file["C00:NRSur7dq4/posterior_samples"]
        dtype_names = posterior_samples.dtype.names or ()
        if "chirp_mass" in dtype_names:
            chirp_mass_samples = np.asarray(posterior_samples["chirp_mass"][:], dtype=float)
        else:
            mass_1 = np.asarray(posterior_samples["mass_1"][:], dtype=float)
            mass_2 = np.asarray(posterior_samples["mass_2"][:], dtype=float)
            chirp_mass_samples = bilby.gw.conversion.component_masses_to_chirp_mass(
                mass_1, mass_2
            )

    lower, upper = np.quantile(
        chirp_mass_samples,
        [tail_probability, 1 - tail_probability],
    )
    return float(lower), float(upper)


def load_psds(
    posterior_path: Path,
    detectors: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with h5py.File(posterior_path, "r") as posterior_file:
        return {
            detector: tuple(
                posterior_file[f"C00:NRSur7dq4/psds/{detector}"][:].T
            )
            for detector in detectors
        }


def template_settings_with_injection_duration(
    template_settings: dict[str, object],
    injection_duration: float | None,
) -> dict[str, object]:
    if injection_duration is None:
        return template_settings
    updated_settings = dict(template_settings)
    updated_settings["duration"] = float(injection_duration)
    return updated_settings


def build_waveform_generator(
    template_settings: dict[str, object],
    *,
    sine_gaussian_config=None,
) -> bilby.gw.LALCBCWaveformGenerator:
    if sine_gaussian_config is not None and sine_gaussian_config.enabled:
        source_model = bilby.gw.source.cbc_plus_sine_gaussians
        parameter_conversion = (
            bilby.gw.conversion.convert_to_cbc_plus_sine_gaussian_parameters
        )
    else:
        source_model = bilby.gw.source.lal_binary_black_hole
        parameter_conversion = (
            bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
        )

    return bilby.gw.LALCBCWaveformGenerator(
        duration=template_settings["duration"],
        sampling_frequency=template_settings["sampling_frequency"],
        start_time=(
            template_settings["trigger_time"]
            + template_settings["post_trigger_duration"]
            - template_settings["duration"]
        ),
        frequency_domain_source_model=source_model,
        parameter_conversion=parameter_conversion,
        waveform_arguments=dict(
            waveform_approximant=template_settings["waveform_approximant"],
            reference_frequency=template_settings["reference_frequency"],
            minimum_frequency=template_settings["minimum_frequency"]["waveform"],
            maximum_frequency=template_settings["maximum_frequency"],
            catch_waveform_errors=True,
            pn_spin_order=-1,
            pn_tidal_order=-1,
            pn_phase_order=-1,
            pn_amplitude_order=0,
            mode_array=None,
        ),
    )


def build_injected_label_prefix(label_prefix: str, sine_gaussian_config) -> str:
    if not sine_gaussian_config.enabled:
        return label_prefix
    return f"{label_prefix}_injected{sine_gaussian_config.label_suffix}"


def build_stage_directory_name(sine_gaussian_config) -> str:
    if not sine_gaussian_config.enabled:
        return "staged_data"
    return f"staged_data{sine_gaussian_config.label_suffix}"


def serialize_sine_gaussian_configuration(sine_gaussian_config) -> dict[str, object]:
    return dict(
        enabled=sine_gaussian_config.enabled,
        mode=sine_gaussian_config.mode,
        total_components=sine_gaussian_config.total_components,
        detector_counts=dict(sine_gaussian_config.detector_counts),
    )


@lru_cache(maxsize=1)
def load_injected_sine_gaussian_values() -> dict[str, object]:
    if not INJECTED_SINE_GAUSSIAN_VALUES_PATH.is_file():
        raise FileNotFoundError(
            f"Missing injected SG values file: {INJECTED_SINE_GAUSSIAN_VALUES_PATH}"
        )
    with INJECTED_SINE_GAUSSIAN_VALUES_PATH.open(encoding="utf-8") as stream:
        raw_values = json.load(stream)

    if not isinstance(raw_values, dict):
        raise ValueError(
            f"Injected SG values file must contain a JSON object: {INJECTED_SINE_GAUSSIAN_VALUES_PATH}"
        )

    coherent_raw = raw_values.get("coherent")
    if not isinstance(coherent_raw, dict):
        raise ValueError(
            "Injected SG values file must define a 'coherent' object keyed by component count."
        )

    incoherent_raw = raw_values.get("incoherent")
    if not isinstance(incoherent_raw, dict):
        raise ValueError(
            "Injected SG values file must define an 'incoherent' object keyed by detector."
        )

    independent_sky_raw = raw_values.get("coherent-independent")
    if not isinstance(independent_sky_raw, dict):
        raise ValueError(
            "Injected SG values file must define a 'coherent-independent' sky object."
        )
    independent_sky = {}
    for key, bounds in {
        "ra": (0.0, 2.0 * float(np.pi)),
        "dec": (-0.5 * float(np.pi), 0.5 * float(np.pi)),
        "psi": (0.0, float(np.pi)),
    }.items():
        try:
            value = float(independent_sky_raw[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"coherent-independent.{key} must be a finite numeric value."
            ) from exc
        if not np.isfinite(value) or not bounds[0] <= value <= bounds[1]:
            raise ValueError(
                f"coherent-independent.{key}={value} is outside {bounds}."
            )
        independent_sky[key] = value

    def parse_count(raw_count, *, context: str) -> int:
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context} count key must be an integer, got {raw_count!r}."
            ) from exc
        if count < 1:
            raise ValueError(f"{context} count key must be >= 1, got {count}.")
        return count

    def parse_component(raw_component, *, context: str) -> dict[str, float]:
        if not isinstance(raw_component, dict):
            raise ValueError(f"{context} must be a JSON object.")

        missing = [
            key
            for key in INJECTED_SINE_GAUSSIAN_COMPONENT_KEYS
            if key not in raw_component
        ]
        if missing:
            raise ValueError(f"{context} is missing keys: {', '.join(missing)}.")

        unexpected = sorted(
            set(raw_component).difference(INJECTED_SINE_GAUSSIAN_COMPONENT_KEYS)
        )
        if unexpected:
            raise ValueError(
                f"{context} has unexpected keys: {', '.join(unexpected)}."
            )

        component = {}
        for key in INJECTED_SINE_GAUSSIAN_COMPONENT_KEYS:
            try:
                component[key] = float(raw_component[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{context}.{key} must be a finite numeric value."
                ) from exc
            if not np.isfinite(component[key]):
                raise ValueError(f"{context}.{key} must be finite.")
        return component

    def parse_component_series(
        raw_components,
        *,
        expected_count: int,
        context: str,
    ) -> tuple[dict[str, float], ...]:
        if not isinstance(raw_components, list):
            raise ValueError(f"{context} must be a JSON array of components.")
        components = tuple(
            parse_component(raw_component, context=f"{context}[{index}]")
            for index, raw_component in enumerate(raw_components)
        )
        if len(components) != expected_count:
            raise ValueError(
                f"{context} must contain exactly {expected_count} component(s), got {len(components)}."
            )
        return components

    coherent = {}
    for raw_count, raw_components in coherent_raw.items():
        count = parse_count(raw_count, context="coherent")
        coherent[count] = parse_component_series(
            raw_components,
            expected_count=count,
            context=f"coherent[{count}]",
        )

    incoherent = {}
    for detector, detector_raw in incoherent_raw.items():
        if not isinstance(detector_raw, dict):
            raise ValueError(
                f"incoherent[{detector!r}] must be an object keyed by component count."
            )
        detector_components = {}
        for raw_count, raw_components in detector_raw.items():
            count = parse_count(raw_count, context=f"incoherent[{detector}]")
            detector_components[count] = parse_component_series(
                raw_components,
                expected_count=count,
                context=f"incoherent[{detector}][{count}]",
            )
        incoherent[str(detector)] = detector_components

    return dict(
        coherent=coherent,
        coherent_independent=independent_sky,
        incoherent=incoherent,
    )


def load_injected_sine_gaussian_component_series(
    *,
    mode: str,
    count: int,
    detector: str | None = None,
) -> list[dict[str, float]]:
    if count < 1:
        raise ValueError(f"Sine-Gaussian component count must be >= 1, got {count}.")

    injected_values = load_injected_sine_gaussian_values()
    if mode in {"coherent", "coherent-independent"}:
        if detector is not None:
            raise ValueError("Coherent SG injections do not take a detector selector.")
        component_series = injected_values["coherent"].get(count)
        if component_series is None:
            raise ValueError(
                "Injected SG values file does not define a coherent configuration "
                f"with {count} component(s): {INJECTED_SINE_GAUSSIAN_VALUES_PATH}"
            )
    elif mode == "incoherent":
        if detector is None:
            raise ValueError("Incoherent SG injections require a detector name.")
        detector_values = injected_values["incoherent"].get(detector)
        if detector_values is None:
            raise ValueError(
                "Injected SG values file does not define incoherent SG values "
                f"for detector {detector!r}: {INJECTED_SINE_GAUSSIAN_VALUES_PATH}"
            )
        component_series = detector_values.get(count)
        if component_series is None:
            raise ValueError(
                "Injected SG values file does not define an incoherent configuration "
                f"for detector {detector!r} with {count} component(s): "
                f"{INJECTED_SINE_GAUSSIAN_VALUES_PATH}"
            )
    else:
        raise ValueError(f"Unknown SG injection mode: {mode}")

    return [dict(component) for component in component_series]


def validate_injected_sine_gaussian_component(
    component: dict[str, float],
    *,
    frequency_minimum: float,
    frequency_maximum: float,
    context: str = "component",
) -> None:
    bounds = {
        "hrss": SINE_GAUSSIAN_HRSS_BOUNDS,
        "Q": SINE_GAUSSIAN_Q_BOUNDS,
        "frequency": (frequency_minimum, frequency_maximum),
        "time_offset": SINE_GAUSSIAN_TIME_OFFSET_BOUNDS,
        "phase_offset": (-float(np.pi), float(np.pi)),
    }
    for key, (minimum, maximum) in bounds.items():
        value = float(component[key])
        if value < minimum or value > maximum:
            raise ValueError(
                "Injected sine-Gaussian {} has {}={} outside prior bounds [{}, {}].".format(
                    context, key, value, minimum, maximum
                )
            )


def flatten_sine_gaussian_component(
    index: int,
    component: dict[str, float],
    *,
    detector: str | None = None,
    independent: bool = False,
) -> dict[str, float]:
    prefix = (
        f"independent_sine_gaussian_{index}_"
        if independent
        else f"sine_gaussian_{index}_"
    )
    if detector is not None:
        prefix += f"{detector}_"
    return {
        f"{prefix}hrss": component["hrss"],
        f"{prefix}Q": component["Q"],
        f"{prefix}frequency": component["frequency"],
        f"{prefix}time_offset": component["time_offset"],
        f"{prefix}phase_offset": component["phase_offset"],
    }


def add_injected_sine_gaussians(
    injection_parameters: dict[str, float],
    *,
    template_settings: dict[str, object],
    sine_gaussian_config,
) -> dict[str, float]:
    if not sine_gaussian_config.enabled:
        return injection_parameters

    frequency_minimum, frequency_maximum = sine_gaussian_frequency_bounds(
        template_settings["minimum_frequency"],
        template_settings["maximum_frequency"],
    )
    updated_parameters = dict(injection_parameters)

    if sine_gaussian_config.mode == "coherent":
        components = load_injected_sine_gaussian_component_series(
            mode="coherent",
            count=sine_gaussian_config.total_components,
        )
        for index, component in enumerate(components):
            validate_injected_sine_gaussian_component(
                component,
                frequency_minimum=frequency_minimum,
                frequency_maximum=frequency_maximum,
                context=f"coherent[{sine_gaussian_config.total_components}][{index}]",
            )
            updated_parameters.update(
                flatten_sine_gaussian_component(
                    index,
                    component,
                )
            )
        return updated_parameters

    if sine_gaussian_config.mode == "coherent-independent":
        components = load_injected_sine_gaussian_component_series(
            mode="coherent-independent",
            count=sine_gaussian_config.total_components,
        )
        for index, component in enumerate(components):
            validate_injected_sine_gaussian_component(
                component,
                frequency_minimum=frequency_minimum,
                frequency_maximum=frequency_maximum,
                context=(
                    "coherent-independent"
                    f"[{sine_gaussian_config.total_components}][{index}]"
                ),
            )
            updated_parameters.update(
                flatten_sine_gaussian_component(
                    index,
                    component,
                    independent=True,
                )
            )
        independent_sky = load_injected_sine_gaussian_values()[
            "coherent_independent"
        ]
        updated_parameters.update({
            f"independent_sine_gaussian_{key}": value
            for key, value in independent_sky.items()
        })
        return updated_parameters

    component_index = 0
    for detector, count in sine_gaussian_config.detector_counts:
        components = load_injected_sine_gaussian_component_series(
            mode="incoherent",
            detector=detector,
            count=count,
        )
        for local_index, component in enumerate(components):
            validate_injected_sine_gaussian_component(
                component,
                frequency_minimum=frequency_minimum,
                frequency_maximum=frequency_maximum,
                context=f"incoherent[{detector}][{count}][{local_index}]",
            )
            updated_parameters.update(
                flatten_sine_gaussian_component(
                    component_index,
                    component,
                    detector=detector,
                )
            )
            component_index += 1
    return updated_parameters


def build_interferometers(
    psds: dict[str, tuple[np.ndarray, np.ndarray]],
    template_settings: dict[str, object],
    *,
    noise_model: str,
    nu_injection,
    num_frequency_bands: int,
) -> bilby.gw.detector.InterferometerList:
    interferometers = bilby.gw.detector.InterferometerList([])
    for detector in template_settings["detectors"]:
        ifo = bilby.gw.detector.get_empty_interferometer(detector)
        frequency_array, psd_array = psds[detector]
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=frequency_array,
            psd_array=psd_array,
        )
        ifo.minimum_frequency = template_settings["minimum_frequency"][detector]
        ifo.maximum_frequency = template_settings["maximum_frequency"]
        interferometers.append(ifo)

    start_time = (
        template_settings["trigger_time"]
        + template_settings["post_trigger_duration"]
        - template_settings["duration"]
    )
    if noise_model == "student":
        interferometers.set_strain_data_from_power_spectral_densities_student_t(
            sampling_frequency=template_settings["sampling_frequency"],
            duration=template_settings["duration"],
            nu=nu_injection,
            start_time=start_time,
            num_frequency_bands=num_frequency_bands,
        )
    elif noise_model == "gaussian":
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=template_settings["sampling_frequency"],
            duration=template_settings["duration"],
            start_time=start_time,
        )
    elif noise_model == "zero-gaussian":
        interferometers.set_strain_data_from_zero_noise(
            sampling_frequency=template_settings["sampling_frequency"],
            duration=template_settings["duration"],
            start_time=start_time,
        )
    else:
        raise ValueError(f"Unknown injection noise model: {noise_model}")
    return interferometers


def write_time_series(
    path: Path,
    detector: str,
    strain: np.ndarray,
    start_time: float,
    sampling_frequency: float,
) -> None:
    from gwpy.timeseries import TimeSeries

    series = TimeSeries(
        strain,
        t0=start_time,
        dt=1.0 / sampling_frequency,
        name=f"{detector}_SIM",
    )
    series.write(str(path), format="hdf5", overwrite=True)


def write_frequency_domain_strain(
    path: Path,
    detector: str,
    frequency_domain_strain: np.ndarray,
    frequencies: np.ndarray,
    start_time: float,
    duration: float,
    sampling_frequency: float,
) -> None:
    if np.shape(frequency_domain_strain) != np.shape(frequencies):
        raise ValueError("Frequency-domain strain and frequency array shapes differ")
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset(
            "frequency_array",
            data=np.asarray(frequencies, dtype=float),
        )
        h5_file.create_dataset(
            "frequency_domain_strain",
            data=np.asarray(frequency_domain_strain, dtype=complex),
        )
        h5_file.attrs["detector"] = detector
        h5_file.attrs["duration"] = float(duration)
        h5_file.attrs["sampling_frequency"] = float(sampling_frequency)
        h5_file.attrs["start_time"] = float(start_time)
        h5_file.attrs["data_format"] = FD_DATA_FORMAT


def write_psd(path: Path, frequencies: np.ndarray, psd: np.ndarray) -> None:
    if np.shape(frequencies) != np.shape(psd):
        raise ValueError("PSD and frequency array shapes differ")
    np.savetxt(
        path,
        np.column_stack([frequencies, psd]),
        header="f psd(f)",
    )


def stage_injection_bundle(
    base_dir: Path,
    args: argparse.Namespace,
    template_settings: dict[str, object],
    posterior_path: Path,
    injected_sine_gaussian_config,
) -> dict[str, object]:
    staged_label_prefix = build_injected_label_prefix(
        args.label_prefix,
        injected_sine_gaussian_config,
    )
    stage_dir = ensure_dir(
        base_dir / build_stage_directory_name(injected_sine_gaussian_config)
    )
    data_dir = ensure_dir(stage_dir / "data")
    psd_dir = ensure_dir(stage_dir / "psds")

    noise_generation_seed = getattr(args, "noise_generation_seed", None)
    if noise_generation_seed is not None:
        staging_seed = int(noise_generation_seed)
    else:
        template_seed = template_settings.get("sampling_seed")
        staging_seed = (
            int(template_seed)
            if isinstance(template_seed, (int, np.integer))
            else DEFAULT_STAGING_RANDOM_SEED
        )
    bilby.core.utils.random.seed(staging_seed)

    injection_parameters, maxl_log_likelihood, maxl_index = (
        load_maximum_likelihood_injection(posterior_path)
    )
    injection_parameters = add_injected_sine_gaussians(
        injection_parameters,
        template_settings=template_settings,
        sine_gaussian_config=injected_sine_gaussian_config,
    )
    need_nu_configuration = (
        args.injection_noise == "student" or args.likelihood == "student"
    )
    if need_nu_configuration:
        configured_noise_nu, likelihood_nu, effective_detector_dependent_nu = (
            resolve_nu_configuration(
                raw_nu_injection=args.nu_injection,
                detectors=template_settings["detectors"],
                num_frequency_bands=args.num_frequency_bands,
                detector_dependent_nu=args.detector_dependent_nu,
            )
        )
    else:
        configured_noise_nu = None
        likelihood_nu = None
        effective_detector_dependent_nu = False

    injected_noise_nu = (
        configured_noise_nu if args.injection_noise == "student" else None
    )
    psds = load_psds(posterior_path, template_settings["detectors"])
    interferometers = build_interferometers(
        psds,
        template_settings,
        noise_model=args.injection_noise,
        nu_injection=injected_noise_nu,
        num_frequency_bands=args.num_frequency_bands,
    )
    waveform_generator = build_waveform_generator(
        template_settings,
        sine_gaussian_config=injected_sine_gaussian_config,
    )
    interferometers.inject_signal(
        parameters=injection_parameters,
        waveform_generator=waveform_generator,
    )

    data_paths = {}
    psd_paths = {}
    for interferometer in interferometers:
        detector = interferometer.name
        if args.frequency_domain_injection:
            data_path = data_dir / f"{detector}_{staged_label_prefix}_fd.hdf5"
        else:
            data_path = data_dir / f"{detector}_{staged_label_prefix}.hdf5"
        psd_path = psd_dir / f"{detector}_{staged_label_prefix}_psd.dat"
        if args.frequency_domain_injection:
            write_frequency_domain_strain(
                data_path,
                detector,
                interferometer.frequency_domain_strain,
                interferometer.frequency_array,
                interferometer.start_time,
                interferometer.duration,
                interferometer.sampling_frequency,
            )
        else:
            write_time_series(
                data_path,
                detector,
                interferometer.time_domain_strain,
                interferometer.start_time,
                interferometer.sampling_frequency,
            )
        write_psd(
            psd_path,
            interferometer.frequency_array,
            interferometer.power_spectral_density_array,
        )
        data_paths[detector] = str(data_path.resolve())
        psd_paths[detector] = str(psd_path.resolve())

    metadata = dict(
        maxl_index=maxl_index,
        maxl_log_likelihood=maxl_log_likelihood,
        injection_noise_model=args.injection_noise,
        injection_data_domain=(
            "frequency" if args.frequency_domain_injection else "time"
        ),
        injection_data_format=(
            FD_DATA_FORMAT if args.frequency_domain_injection else "hdf5"
        ),
        nu_injection=injected_noise_nu,
        likelihood_nu=likelihood_nu,
        num_frequency_bands=args.num_frequency_bands,
        injection_duration=template_settings["duration"],
        injection_start_time=(
            template_settings["trigger_time"]
            + template_settings["post_trigger_duration"]
            - template_settings["duration"]
        ),
        detector_dependent_nu=effective_detector_dependent_nu,
        posterior_path=str(posterior_path.resolve()),
        waveform_approximant=template_settings["waveform_approximant"],
        sampling_seed=staging_seed,
        injection_parameters=injection_parameters,
        injected_sine_gaussian_configuration=serialize_sine_gaussian_configuration(
            injected_sine_gaussian_config
        ),
        data_paths=data_paths,
        psd_paths=psd_paths,
    )
    metadata_path = stage_dir / f"{staged_label_prefix}_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return dict(
        stage_dir=stage_dir,
        staged_label_prefix=staged_label_prefix,
        metadata_path=metadata_path,
        data_paths=data_paths,
        psd_paths=psd_paths,
        likelihood_nu=likelihood_nu,
        detector_dependent_nu=effective_detector_dependent_nu,
        injection_parameters=injection_parameters,
        injected_sine_gaussian_configuration=injected_sine_gaussian_config,
        maxl_index=maxl_index,
        maxl_log_likelihood=maxl_log_likelihood,
        test_injection_chirp_mass_bounds=load_test_injection_chirp_mass_bounds(
            posterior_path
        ),
    )


def format_ini_dict(mapping: dict[str, str], *, quote_values: bool = False) -> str:
    items = []
    for key, value in mapping.items():
        rendered = f"'{value}'" if quote_values else value
        items.append(f"{key}: {rendered}")
    return "{ " + ", ".join(items) + ", }"


def apply_frequency_domain_injection_ini_settings(text: str) -> str:
    rendered = replace_line(text, "data-format", FD_DATA_FORMAT)
    rendered = replace_or_append_line(rendered, "gaussian-noise", "False")
    rendered = replace_or_append_line(rendered, "zero-noise", "False")
    rendered = replace_or_append_line(rendered, "injection", "False")
    rendered = replace_or_append_line(rendered, "plot-data", "False")
    rendered = replace_or_append_line(rendered, "plot-spectrogram", "False")
    return rendered


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
    rendered = replace_or_append_line(
        rendered,
        "calibration-correction-type",
        "None",
        insert_after="calibration-model",
    )
    rendered = replace_line(rendered, "spline-calibration-envelope-dict", "None")
    return rendered


def replace_or_append_prior_line(
    text: str,
    key: str,
    line: str,
    *,
    insert_after: str | None = None,
) -> str:
    lines = text.splitlines()
    prefix = f"{key} ="
    for index, existing in enumerate(lines):
        if existing.startswith(prefix):
            lines[index] = line
            return "\n".join(lines) + "\n"
    if insert_after is not None:
        anchor = f"{insert_after} ="
        for index, existing in enumerate(lines):
            if existing.startswith(anchor):
                lines.insert(index + 1, line)
                return "\n".join(lines) + "\n"
    lines.append(line)
    return "\n".join(lines) + "\n"


def format_prior_value(value: float) -> str:
    return repr(float(value))


def build_nu_priors(
    args: argparse.Namespace,
    *,
    include_nu_prior: bool,
    detectors: tuple[str, ...],
    num_frequency_bands: int,
    detector_dependent_nu: bool,
) -> str:
    if not include_nu_prior:
        return ""

    nu_maximum = (
        min(args.nu_max, TEST_INJECTION_NU_MAX)
        if args.test_injection
        else args.nu_max
    )

    if detector_dependent_nu:
        if num_frequency_bands == 1:
            keys = [f"nu_{detector}" for detector in detectors]
        else:
            keys = [
                f"nu_{detector}_{index}"
                for detector in detectors
                for index in range(1, num_frequency_bands + 1)
            ]
    else:
        if num_frequency_bands == 1:
            keys = ["nu"]
        else:
            keys = [f"nu_{index}" for index in range(1, num_frequency_bands + 1)]

    return "\n".join(
        f"{key} = Uniform(name='{key}', minimum={args.nu_min}, maximum={nu_maximum})"
        for key in keys
    )


def render_prior(
    prior_template: str,
    *,
    args: argparse.Namespace,
    include_nu_prior: bool,
    detectors: tuple[str, ...],
    num_frequency_bands: int,
    detector_dependent_nu: bool,
    template_settings: dict[str, object],
    sine_gaussian_config,
    bundle: dict[str, object],
) -> str:
    nu_prior_block = build_nu_priors(
        args,
        include_nu_prior=include_nu_prior,
        detectors=detectors,
        num_frequency_bands=num_frequency_bands,
        detector_dependent_nu=detector_dependent_nu,
    )
    sine_gaussian_prior_block = build_sine_gaussian_prior_block(
        sine_gaussian_config,
        minimum_frequency=template_settings["minimum_frequency"],
        maximum_frequency=template_settings["maximum_frequency"],
    )
    rendered = prior_template.replace(
        "__NU_PRIORS__",
        combine_prior_blocks(nu_prior_block, sine_gaussian_prior_block),
    )
    if not args.test_injection:
        return rendered

    injection_parameters = bundle["injection_parameters"]
    mass_ratio = injection_parameters["mass_2"] / injection_parameters["mass_1"]
    chirp_mass_minimum, chirp_mass_maximum = bundle["test_injection_chirp_mass_bounds"]
    rendered = replace_or_append_prior_line(
        rendered,
        "chirp_mass",
        (
            "chirp_mass = bilby.gw.prior.UniformInComponentsChirpMass("
            f"name='chirp_mass', minimum={format_prior_value(chirp_mass_minimum)}, "
            f"maximum={format_prior_value(chirp_mass_maximum)}, "
            "unit='$M_{\\{\\odot\\}}')"
        ),
    )
    rendered = replace_or_append_prior_line(
        rendered,
        "mass_ratio",
        (
            "mass_ratio = DeltaFunction("
            f"name='mass_ratio', peak={format_prior_value(mass_ratio)})"
        ),
    )
    for key in TEST_INJECTION_FIXED_KEYS:
        rendered = replace_or_append_prior_line(
            rendered,
            key,
            (
                f"{key} = DeltaFunction("
                f"name='{key}', peak={format_prior_value(injection_parameters[key])})"
            ),
        )
    rendered = replace_or_append_prior_line(
        rendered,
        "geocent_time",
        (
            "geocent_time = DeltaFunction("
            f"name='geocent_time', peak={format_prior_value(injection_parameters['geocent_time'])})"
        ),
        insert_after="ra",
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


def build_pesummary_arguments(
    template_settings: dict[str, object],
    *,
    psd_paths: dict[str, str],
) -> dict[str, object]:
    arguments = dict(DEFAULT_PESUMMARY_ARGUMENTS)
    f_low = minimum_frequency_for_pesummary(template_settings["minimum_frequency"])
    f_ref = template_settings["reference_frequency"]
    arguments.update(
        f_low=f_low,
        f_start=f_ref,
        f_ref=f_ref,
        f_final=maximum_frequency_for_pesummary(template_settings["maximum_frequency"]),
        approximant=[template_settings["waveform_approximant"]],
        psd=psd_paths,
    )
    return arguments


def render_ini(
    ini_template: str,
    *,
    args: argparse.Namespace,
    template_settings: dict[str, object],
    num_frequency_bands: int,
    detector_dependent_nu: bool,
    likelihood_nu,
    label: str,
    outdir: Path,
    webdir: Path,
    prior_path: Path,
    data_paths: dict[str, str],
    psd_paths: dict[str, str],
    stage_dir: Path,
    hypothesis: str,
    sine_gaussian_config,
) -> str:
    rendered = ini_template
    placeholders = {
        "__LABEL__": label,
        "__OUTDIR__": str(outdir.resolve()),
        "__WEBDIR__": str(webdir.resolve()),
        "__PRIOR_FILE__": str(prior_path.resolve()),
        "__NUM_FREQUENCY_BANDS__": str(num_frequency_bands),
        "__DETECTOR_DEPENDENT_NU__": str(detector_dependent_nu),
    }
    for placeholder, value in placeholders.items():
        rendered = rendered.replace(placeholder, value)

    if "duration" in template_settings:
        rendered = replace_or_append_line(
            rendered,
            "duration",
            repr(float(template_settings["duration"])),
        )
    rendered = replace_line(rendered, "accounting-user", args.accounting_user)
    rendered = replace_or_append_line(
        rendered,
        "container",
        getattr(args, "container_image", None) or "None",
    )
    rendered = replace_or_append_line(rendered, "transfer-files", "True")
    rendered = replace_or_append_line(rendered, "osg", "True")
    rendered = replace_or_append_line(rendered, "desired-sites", "None")
    if args.require_epnfs:
        rendered = replace_line(rendered, "queue", "EPNFS")
    rendered = replace_line(rendered, "create-summary", "True")
    rendered = replace_line(
        rendered,
        "environment-variables",
        repr(DEFAULT_ENVIRONMENT_VARIABLES),
    )
    rendered = replace_line(
        rendered,
        "summarypages-arguments",
        repr(build_pesummary_arguments(template_settings, psd_paths=psd_paths)),
    )
    rendered = replace_line(rendered, "data-dict", format_ini_dict(data_paths))
    if args.frequency_domain_injection:
        rendered = apply_frequency_domain_injection_ini_settings(rendered)
    else:
        rendered = replace_line(rendered, "data-format", "hdf5")
    rendered = disable_calibration_settings(rendered)
    rendered = replace_line(
        rendered,
        "channel-dict",
        format_ini_dict(
            {detector: "SIM" for detector in template_settings["detectors"]},
            quote_values=True,
        ),
    )
    rendered = replace_line(rendered, "psd-dict", format_ini_dict(psd_paths))
    rendered = replace_line(
        rendered,
        "additional-transfer-paths",
        f"[{stage_dir.resolve()}]",
    )
    if args.test_injection:
        rendered = replace_line(rendered, "reference-frame", "sky")
        rendered = replace_line(rendered, "time-reference", "geocenter")

    sampler_kwargs = dict(template_settings["sampler_kwargs"])
    sampler_kwargs["nlive"] = effective_nlive(args.nlive, sine_gaussian_config)
    sampler_kwargs["naccept"] = args.naccept
    if getattr(args, "maxmcmc", None) is not None:
        sampler_kwargs["maxmcmc"] = args.maxmcmc
    rendered = replace_line(
        rendered,
        "sampler-kwargs",
        repr(sampler_kwargs),
    )

    if hypothesis == "gaussian":
        rendered = replace_line(
            rendered,
            "likelihood-type",
            "bilby.gw.likelihood.GravitationalWaveTransient",
        )
        rendered = replace_line(rendered, "extra-likelihood-kwargs", "None")
    elif hypothesis == "student":
        rendered = replace_line(
            rendered,
            "likelihood-type",
            "bilby.gw.likelihood.StudentTGravitationalWaveTransient",
        )
        rendered = replace_line(
            rendered,
            "extra-likelihood-kwargs",
            (
                "{'nu': "
                f"{repr(likelihood_nu)}, 'infer_nu': True, "
                f"'num_frequency_bands': {num_frequency_bands}, "
                f"'detector_dependent_nu': {detector_dependent_nu}"
                "}"
            ),
        )
    else:
        raise ValueError(f"Unknown hypothesis '{hypothesis}'")

    rendered = apply_sine_gaussian_waveform_settings(
        rendered,
        sine_gaussian_config,
        replace_line=replace_line,
    )
    return rendered


def build_run_label(label_prefix: str, hypothesis: str, sine_gaussian_config) -> str:
    return f"{label_prefix}_{hypothesis}{sine_gaussian_config.label_suffix}"


def build_run_directory_name(label: str, outdir_label: str | None) -> str:
    if outdir_label is None:
        return label
    return f"{label}_{outdir_label}"


def write_run_files(
    *,
    base_dir: Path,
    args: argparse.Namespace,
    template_settings: dict[str, object],
    hypothesis: str,
    sine_gaussian_config,
    ini_template: str,
    prior_template: str,
    bundle: dict[str, object],
) -> tuple[Path, Path]:
    prior_dir = ensure_dir(base_dir / "Priors")
    ini_dir = ensure_dir(base_dir / "ini_files")
    run_dir = ensure_dir(base_dir / "Runs")

    label = build_run_label(
        bundle["staged_label_prefix"],
        hypothesis,
        sine_gaussian_config,
    )
    run_num_frequency_bands = (
        args.num_frequency_bands
        if hypothesis == "student"
        else DEFAULT_NUM_FREQUENCY_BANDS
    )
    run_detector_dependent_nu = (
        bundle["detector_dependent_nu"] if hypothesis == "student" else False
    )
    run_likelihood_nu = bundle["likelihood_nu"] if hypothesis == "student" else None
    run_directory_name = build_run_directory_name(label, args.outdir_label)
    prior_path = prior_dir / f"{label}.prior"
    ini_path = ini_dir / f"{label}.ini"
    outdir = ensure_dir(run_dir / run_directory_name)
    webdir = ensure_dir(outdir / "web")

    prior_path.write_text(
        render_prior(
            prior_template,
            args=args,
            include_nu_prior=(hypothesis == "student"),
            detectors=template_settings["detectors"],
            num_frequency_bands=run_num_frequency_bands,
            detector_dependent_nu=run_detector_dependent_nu,
            template_settings=template_settings,
            sine_gaussian_config=sine_gaussian_config,
            bundle=bundle,
        ),
        encoding="utf-8",
    )
    ini_path.write_text(
        render_ini(
            ini_template,
            args=args,
            template_settings=template_settings,
            num_frequency_bands=run_num_frequency_bands,
            detector_dependent_nu=run_detector_dependent_nu,
            likelihood_nu=run_likelihood_nu,
            label=label,
            outdir=outdir,
            webdir=webdir,
            prior_path=prior_path,
            data_paths=bundle["data_paths"],
            psd_paths=bundle["psd_paths"],
            stage_dir=bundle["stage_dir"],
            hypothesis=hypothesis,
            sine_gaussian_config=sine_gaussian_config,
        ),
        encoding="utf-8",
    )
    return ini_path, prior_path


def prepare_runs(args: argparse.Namespace) -> list[Path]:
    if args.base_dir is None:
        base_dir = args.home_dir.expanduser() / DEFAULT_BASE_SUBDIR
    else:
        base_dir = args.base_dir.expanduser().resolve()
    base_dir = ensure_dir(base_dir)
    ini_template = load_template(INI_TEMPLATE_PATH)
    prior_template = load_template(PRIOR_TEMPLATE_PATH)
    template_settings = read_template_settings(ini_template)
    template_settings = template_settings_with_injection_duration(
        template_settings,
        args.injection_duration,
    )
    sine_gaussian_configs = resolve_sine_gaussian_configurations(
        num_sine_gaussians=args.num_sine_gaussians,
        range_mode=args.sine_gaussian_range,
        mode=args.sine_gaussian_mode,
        incoherent_detectors=args.incoherent_detectors,
        incoherent_counts_spec=args.incoherent_sg_counts,
        detectors=template_settings["detectors"],
    )
    injected_sine_gaussian_configs = resolve_sine_gaussian_configurations(
        num_sine_gaussians=args.injection_num_sine_gaussians,
        range_mode=args.injection_sine_gaussian_range,
        mode=args.injection_sine_gaussian_mode,
        incoherent_detectors=args.injection_incoherent_detectors,
        incoherent_counts_spec=args.injection_incoherent_sg_counts,
        detectors=template_settings["detectors"],
    )
    if args.test_injection and any(config.enabled for config in sine_gaussian_configs):
        raise ValueError(
            "--test-injection cannot be combined with recovery sine-Gaussian "
            "parameters. Disable --test-injection or set --num-sine-gaussians 0."
        )
    for sine_gaussian_config in [
        *sine_gaussian_configs,
        *injected_sine_gaussian_configs,
    ]:
        require_supported_sine_gaussian_source_model(
            template_settings,
            sine_gaussian_config,
        )
    posterior_path = resolve_posterior_path(args)

    ini_paths = []
    for injected_sine_gaussian_config in injected_sine_gaussian_configs:
        bundle = stage_injection_bundle(
            base_dir,
            args,
            template_settings,
            posterior_path,
            injected_sine_gaussian_config,
        )
        for sine_gaussian_config in sine_gaussian_configs:
            for hypothesis in hypothesis_list(args):
                ini_path, prior_path = write_run_files(
                    base_dir=base_dir,
                    args=args,
                    template_settings=template_settings,
                    hypothesis=hypothesis,
                    sine_gaussian_config=sine_gaussian_config,
                    ini_template=ini_template,
                    prior_template=prior_template,
                    bundle=bundle,
                )
                ini_paths.append(ini_path)
                print(
                    "Prepared {} (injection: {}; recovery: {}):".format(
                        hypothesis,
                        injected_sine_gaussian_config.description,
                        sine_gaussian_config.description,
                    )
                )
                print(f"  prior: {prior_path}")
                print(f"  ini:   {ini_path}")

        print(f"Staged injection data in: {bundle['stage_dir']}")
        print(f"Metadata written to:      {bundle['metadata_path']}")
    return ini_paths


def submit_runs(ini_paths: list[Path], executable: str) -> None:
    if shutil.which(executable) is None:
        raise FileNotFoundError(f"Unable to find executable: {executable}")
    for ini_path in ini_paths:
        run_base_dir = ini_path.resolve().parents[1]
        validate_submission_local_paths(
            ini_path.read_text(encoding="utf-8"),
            base_directory=run_base_dir,
        )
        subprocess.run(
            [executable, str(ini_path), "--submit"],
            check=True,
            cwd=str(run_base_dir),
        )


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.container_image = resolve_container_image(
            use_container=args.container,
            container_image=args.container_image,
            default_image_file=DEFAULT_CONTAINER_IMAGES_FILE,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    args.num_frequency_bands_was_explicit = args.num_frequency_bands is not None
    if args.num_frequency_bands is None:
        args.num_frequency_bands = DEFAULT_NUM_FREQUENCY_BANDS
    if args.nlive is None:
        args.nlive = TEST_INJECTION_NLIVE if args.test_injection else DEFAULT_NLIVE
    try:
        ini_paths = prepare_runs(args)
        if not args.dry_run:
            submit_runs(ini_paths, args.bilby_pipe_executable)
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
