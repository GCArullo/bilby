#!/usr/bin/env python3

"""Stage a GW231123 Student-t injection and generate bilby_pipe configs.

The generated ini/prior files are rendered from the same GW231123 template files
used for the real-data analyses in this directory. Only the path- and
injection-specific settings are replaced, so the resulting configs stay as close
as possible to the production templates. The staged simulated noise is always
Student-t; --gaussian-only/--student-only only choose which recovery likelihood
jobs are generated.
"""

from __future__ import annotations

import argparse
import ast
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np
from gwpy.timeseries import TimeSeries

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bilby


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HOME_DIR = Path.home()
DEFAULT_ACCOUNTING_USER = getpass.getuser()
DEFAULT_BASE_SUBDIR = Path("GW231123") / "t_Student" / "Runs_injections"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home-dir",
        type=Path,
        default=DEFAULT_HOME_DIR,
        help=(
            "Base home directory used to build the default --base-dir when "
            "--base-dir is not provided."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Root directory where staged data, generated ini/prior files, and "
            "run/web folders are written. Defaults to "
            "<home-dir>/GW231123/t_Student/Runs_injections."
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
        "--nu-injection",
        default="2.1",
        help=(
            "Injected Student-t nu specification for staged noise generation. "
            "Accepts a scalar, a per-band list, or a detector dictionary "
            "(values may be scalar or per-band lists)."
        ),
    )
    parser.add_argument(
        "--num-frequency-bands",
        type=int,
        default=1,
        help=(
            "Number of frequency bands for Student-t noise generation and "
            "Student likelihood nu parameterization."
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
        default=1000,
        help=(
            "Nested-sampler live points to write into sampler-kwargs in the generated ini."
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
        "--student-only",
        action="store_true",
        help=(
            "Generate only the Student-t likelihood recovery run. "
            "This does not change the injected noise model."
        ),
    )
    parser.add_argument(
        "--gaussian-only",
        action="store_true",
        help=(
            "Generate only the Gaussian likelihood recovery run. "
            "This does not change the injected noise model."
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="After generating ini/prior files, invoke bilby_pipe --submit for each run.",
    )
    parser.add_argument(
        "--bilby-pipe-executable",
        default="bilby_pipe",
        help="Executable name or absolute path used to call bilby_pipe.",
    )
    return parser


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def hypothesis_list(args: argparse.Namespace) -> list[str]:
    if args.student_only and args.gaussian_only:
        raise ValueError("Choose at most one of --student-only and --gaussian-only")
    if args.student_only:
        return ["student"]
    if args.gaussian_only:
        return ["gaussian"]
    return ["gaussian", "student"]


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


def parse_template_value(raw_value: str):
    raw_value = raw_value.strip()
    if raw_value in {"None", ""}:
        return None
    if raw_value in {"True", "False"}:
        return raw_value == "True"
    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def parse_ini_dict_string(raw_value: str) -> dict[str, object]:
    normalized = raw_value.strip()
    normalized = normalized.replace("=", ":")
    normalized = normalized.replace(" ", "")
    normalized = re.sub(
        r'([A-Za-z/\.0-9\-\+][^\[\],:"}]*)',
        r'"\g<1>"',
        normalized,
    )
    normalized = normalized.replace('""', '"')
    parsed = ast.literal_eval(normalized)
    if not isinstance(parsed, dict):
        raise ValueError(f"Unable to parse ini dict: {raw_value}")
    return parsed


def read_template_settings(ini_template: str) -> dict[str, object]:
    parsed = {}
    for line in ini_template.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        parsed[key.strip()] = parse_template_value(value)

    required_keys = (
        "detectors",
        "trigger-time",
        "duration",
        "post-trigger-duration",
        "sampling-frequency",
        "maximum-frequency",
        "minimum-frequency",
        "reference-frequency",
        "waveform-approximant",
        "sampler-kwargs",
    )
    missing = [key for key in required_keys if key not in parsed]
    if missing:
        raise ValueError(
            f"Template ini is missing required keys: {', '.join(missing)}"
        )

    sampler_kwargs = parsed["sampler-kwargs"]
    if not isinstance(sampler_kwargs, dict):
        raise ValueError("sampler-kwargs must parse to a dictionary")

    calibration_envelopes = parsed.get("spline-calibration-envelope-dict")
    if isinstance(calibration_envelopes, str):
        calibration_envelopes = parse_ini_dict_string(calibration_envelopes)

    return dict(
        detectors=tuple(parsed["detectors"]),
        trigger_time=float(parsed["trigger-time"]),
        duration=float(parsed["duration"]),
        post_trigger_duration=float(parsed["post-trigger-duration"]),
        sampling_frequency=float(parsed["sampling-frequency"]),
        maximum_frequency=float(parsed["maximum-frequency"]),
        minimum_frequency=parsed["minimum-frequency"],
        reference_frequency=float(parsed["reference-frequency"]),
        waveform_approximant=str(parsed["waveform-approximant"]),
        sampler_kwargs=sampler_kwargs,
        sampling_seed=parsed.get("sampling-seed"),
        spline_calibration_envelope_dict=calibration_envelopes,
    )


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


def build_waveform_generator(
    template_settings: dict[str, object],
) -> bilby.gw.LALCBCWaveformGenerator:
    return bilby.gw.LALCBCWaveformGenerator(
        duration=template_settings["duration"],
        sampling_frequency=template_settings["sampling_frequency"],
        start_time=(
            template_settings["trigger_time"]
            + template_settings["post_trigger_duration"]
            - template_settings["duration"]
        ),
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
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


def build_interferometers(
    psds: dict[str, tuple[np.ndarray, np.ndarray]],
    template_settings: dict[str, object],
    *,
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
    interferometers.set_strain_data_from_power_spectral_densities_student_t(
        sampling_frequency=template_settings["sampling_frequency"],
        duration=template_settings["duration"],
        nu=nu_injection,
        start_time=start_time,
        num_frequency_bands=num_frequency_bands,
    )
    return interferometers


def write_time_series(
    path: Path,
    detector: str,
    strain: np.ndarray,
    start_time: float,
    sampling_frequency: float,
) -> None:
    series = TimeSeries(
        strain,
        t0=start_time,
        dt=1.0 / sampling_frequency,
        name=f"{detector}_SIM",
    )
    series.write(str(path), format="hdf5", overwrite=True)


def write_psd(path: Path, frequencies: np.ndarray, psd: np.ndarray) -> None:
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
) -> dict[str, object]:
    stage_dir = ensure_dir(base_dir / "staged_data")
    data_dir = ensure_dir(stage_dir / "data")
    psd_dir = ensure_dir(stage_dir / "psds")

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
    noise_nu, likelihood_nu, effective_detector_dependent_nu = resolve_nu_configuration(
        raw_nu_injection=args.nu_injection,
        detectors=template_settings["detectors"],
        num_frequency_bands=args.num_frequency_bands,
        detector_dependent_nu=args.detector_dependent_nu,
    )
    psds = load_psds(posterior_path, template_settings["detectors"])
    interferometers = build_interferometers(
        psds,
        template_settings,
        nu_injection=noise_nu,
        num_frequency_bands=args.num_frequency_bands,
    )
    waveform_generator = build_waveform_generator(template_settings)
    interferometers.inject_signal(
        parameters=injection_parameters,
        waveform_generator=waveform_generator,
    )

    data_paths = {}
    psd_paths = {}
    for interferometer in interferometers:
        detector = interferometer.name
        data_path = data_dir / f"{detector}_{args.label_prefix}.hdf5"
        psd_path = psd_dir / f"{detector}_{args.label_prefix}_psd.dat"
        write_time_series(
            data_path,
            detector,
            interferometer.time_domain_strain,
            interferometer.start_time,
            interferometer.sampling_frequency,
        )
        frequencies, psd_array = psds[detector]
        write_psd(psd_path, frequencies, psd_array)
        data_paths[detector] = str(data_path.resolve())
        psd_paths[detector] = str(psd_path.resolve())

    metadata = dict(
        maxl_index=maxl_index,
        maxl_log_likelihood=maxl_log_likelihood,
        nu_injection=noise_nu,
        likelihood_nu=likelihood_nu,
        num_frequency_bands=args.num_frequency_bands,
        detector_dependent_nu=effective_detector_dependent_nu,
        posterior_path=str(posterior_path.resolve()),
        waveform_approximant=template_settings["waveform_approximant"],
        sampling_seed=staging_seed,
        injection_parameters=injection_parameters,
        data_paths=data_paths,
        psd_paths=psd_paths,
    )
    metadata_path = stage_dir / f"{args.label_prefix}_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return dict(
        stage_dir=stage_dir,
        metadata_path=metadata_path,
        data_paths=data_paths,
        psd_paths=psd_paths,
        likelihood_nu=likelihood_nu,
        detector_dependent_nu=effective_detector_dependent_nu,
    )


def format_ini_dict(mapping: dict[str, str], *, quote_values: bool = False) -> str:
    items = []
    for key, value in mapping.items():
        rendered = f"'{value}'" if quote_values else value
        items.append(f"{key}: {rendered}")
    return "{ " + ", ".join(items) + ", }"


def replace_line(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    prefix = f"{key}="
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f"{key}={value}"
            return "\n".join(lines) + "\n"
    raise ValueError(f"Unable to find config key '{key}' in template")


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
        f"{key} = Uniform(name='{key}', minimum={args.nu_min}, maximum={args.nu_max})"
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
) -> str:
    nu_prior_block = build_nu_priors(
        args,
        include_nu_prior=include_nu_prior,
        detectors=detectors,
        num_frequency_bands=num_frequency_bands,
        detector_dependent_nu=detector_dependent_nu,
    )
    return prior_template.replace("__NU_PRIORS__", nu_prior_block)


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

    rendered = replace_line(rendered, "accounting-user", args.accounting_user)
    rendered = replace_line(rendered, "data-dict", format_ini_dict(data_paths))
    rendered = replace_line(rendered, "data-format", "hdf5")
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

    sampler_kwargs = dict(template_settings["sampler_kwargs"])
    sampler_kwargs["nlive"] = args.nlive
    sampler_kwargs["naccept"] = args.naccept
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

    return rendered


def build_run_label(label_prefix: str, hypothesis: str) -> str:
    return f"{label_prefix}_{hypothesis}"


def write_run_files(
    *,
    base_dir: Path,
    args: argparse.Namespace,
    template_settings: dict[str, object],
    hypothesis: str,
    ini_template: str,
    prior_template: str,
    bundle: dict[str, object],
) -> tuple[Path, Path]:
    prior_dir = ensure_dir(base_dir / "Priors")
    ini_dir = ensure_dir(base_dir / "ini_files")
    run_dir = ensure_dir(base_dir / "Runs")
    web_dir = ensure_dir(base_dir / "web")

    label = build_run_label(args.label_prefix, hypothesis)
    prior_path = prior_dir / f"{label}.prior"
    ini_path = ini_dir / f"{label}.ini"
    outdir = ensure_dir(run_dir / label)
    webdir = ensure_dir(web_dir / label)

    prior_path.write_text(
        render_prior(
            prior_template,
            args=args,
            include_nu_prior=(hypothesis == "student"),
            detectors=template_settings["detectors"],
            num_frequency_bands=args.num_frequency_bands,
            detector_dependent_nu=bundle["detector_dependent_nu"],
        ),
        encoding="utf-8",
    )
    ini_path.write_text(
        render_ini(
            ini_template,
            args=args,
            template_settings=template_settings,
            num_frequency_bands=args.num_frequency_bands,
            detector_dependent_nu=bundle["detector_dependent_nu"],
            likelihood_nu=bundle["likelihood_nu"],
            label=label,
            outdir=outdir,
            webdir=webdir,
            prior_path=prior_path,
            data_paths=bundle["data_paths"],
            psd_paths=bundle["psd_paths"],
            stage_dir=bundle["stage_dir"],
            hypothesis=hypothesis,
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
    posterior_path = resolve_posterior_path(args)
    bundle = stage_injection_bundle(
        base_dir,
        args,
        template_settings,
        posterior_path,
    )

    ini_paths = []
    for hypothesis in hypothesis_list(args):
        ini_path, prior_path = write_run_files(
            base_dir=base_dir,
            args=args,
            template_settings=template_settings,
            hypothesis=hypothesis,
            ini_template=ini_template,
            prior_template=prior_template,
            bundle=bundle,
        )
        ini_paths.append(ini_path)
        print(f"Prepared {hypothesis}:")
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
        subprocess.run(
            [executable, str(ini_path), "--submit"],
            check=True,
            cwd=str(run_base_dir),
        )


def main() -> int:
    args = build_parser().parse_args()
    try:
        ini_paths = prepare_runs(args)
        if args.submit:
            submit_runs(ini_paths, args.bilby_pipe_executable)
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
