#!/usr/bin/env python3

"""Reproduce the Student-t nu shift introduced by bilby conditioning.

This script reconstructs the GW231123-like Student-t injection used by
``submit_runs_injection.py``, then applies the same extra conditioning steps
that bilby/bilby_pipe use when time-domain strain is staged and read
back for analysis.

For the supplied run directory it compares five cases:

1. the raw frequency-domain Student-t draw used at staging time
2. a pure TD/FD round-trip with no extra windowing
3. bilby's TD->FD conditioning with a configurable Tukey roll-off
4. an explicit gwpy HDF5 write/read followed by the same conditioning
5. the stored bilby_pipe generation data dump used by the analysis run

It also scans the Tukey roll-off and the analysed segment duration to show how
the effective inferred ``nu`` changes with conditioning strength, then writes
one overview figure that summarizes the posterior bias.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.integrate import cumulative_trapezoid

REPO_ROOT = Path(__file__).resolve().parents[4]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bilby


DEFAULT_POSTERIOR_PATH = (
    Path(__file__).resolve().parent
    / "LVK_posteriors"
    / "GW231123"
    / "posterior_samples.h5"
)
DEFAULT_STAGING_SEED = 12345
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


@dataclass
class RunSettings:
    detectors: tuple[str, ...]
    trigger_time: float
    duration: float
    post_trigger_duration: float
    sampling_frequency: float
    maximum_frequency: float
    minimum_frequency: dict[str, float]
    reference_frequency: float
    waveform_approximant: str
    tukey_roll_off: float
    start_time: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the bilby_pipe Student-t injection run directory.",
    )
    parser.add_argument(
        "--posterior-path",
        type=Path,
        default=DEFAULT_POSTERIOR_PATH,
        help=(
            "LVK GW231123 posterior used by submit_runs_injection.py to define "
            "the injected maximum-likelihood parameters."
        ),
    )
    parser.add_argument(
        "--nu-injection",
        type=float,
        default=3.0,
        help="Injected single-band Student-t nu used during staging.",
    )
    parser.add_argument(
        "--staging-seed",
        type=int,
        default=DEFAULT_STAGING_SEED,
        help=(
            "Random seed used by submit_runs_injection.py when staging the "
            "injection. Default matches the script constant."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory where plots and summary files are written.",
    )
    parser.add_argument(
        "--rolloff-values",
        default="0,0.1,0.2,0.4,1.0",
        help="Comma-separated Tukey roll-off values to scan.",
    )
    parser.add_argument(
        "--duration-values",
        default=None,
        help=(
            "Comma-separated analysis durations [s] to scan at fixed roll-off. "
            "Defaults to the run duration, 1.5x, 2x, and 4x."
        ),
    )
    parser.add_argument(
        "--nu-grid-max",
        type=float,
        default=50.0,
        help="Upper end of the dense nu grid used for posterior scans.",
    )
    return parser


def parse_scalar(value) -> float:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        value = value.strip()
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (int, float)):
            return float(parsed)
        raise ValueError(f"Unable to parse scalar value {value!r}")
    return float(value)


def parse_literal(value):
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        return ast.literal_eval(value.strip())
    return value


def parse_float_list(value: str | None) -> list[float]:
    if value is None:
        return []
    return [float(entry) for entry in value.split(",") if entry.strip()]


def find_unique_file(directory: Path, pattern: str, description: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {description} in {directory} matching "
            f"{pattern!r}, found {len(matches)}"
        )
    return matches[0]


def load_data_dump(run_dir: Path):
    data_dump_path = find_unique_file(
        run_dir / "data", "*generation_data_dump.pickle", "generation data dump"
    )
    with data_dump_path.open("rb") as file_object:
        data_dump = pickle.load(file_object)
    return data_dump_path, data_dump


def build_run_settings(data_dump) -> RunSettings:
    command_line_args = data_dump.meta_data["command_line_args"]
    detectors = tuple(parse_literal(command_line_args["detectors"]))
    minimum_frequency = parse_literal(command_line_args["minimum_frequency"])
    settings = RunSettings(
        detectors=tuple(detector.strip("'") for detector in detectors),
        trigger_time=parse_scalar(command_line_args["trigger_time"]),
        duration=parse_scalar(command_line_args["duration"]),
        post_trigger_duration=parse_scalar(command_line_args["post_trigger_duration"]),
        sampling_frequency=parse_scalar(command_line_args["sampling_frequency"]),
        maximum_frequency=parse_scalar(command_line_args["maximum_frequency"]),
        minimum_frequency={
            key: float(value) for key, value in minimum_frequency.items()
        },
        reference_frequency=parse_scalar(command_line_args["reference_frequency"]),
        waveform_approximant=str(command_line_args["waveform_approximant"]),
        tukey_roll_off=parse_scalar(command_line_args["tukey_roll_off"]),
        start_time=0.0,
    )
    settings.start_time = (
        settings.trigger_time
        + settings.post_trigger_duration
        - settings.duration
    )
    return settings


def with_duration(settings: RunSettings, duration: float) -> RunSettings:
    duration = float(duration)
    if duration <= 0:
        raise ValueError("Duration must be positive")
    return replace(
        settings,
        duration=duration,
        start_time=settings.trigger_time + settings.post_trigger_duration - duration,
    )


def default_duration_values(settings: RunSettings) -> list[float]:
    candidates = [
        settings.duration,
        1.5 * settings.duration,
        2.0 * settings.duration,
        4.0 * settings.duration,
    ]
    return sorted({float(candidate) for candidate in candidates})


def load_maximum_likelihood_injection(
    posterior_path: Path,
) -> tuple[dict[str, float], dict[str, tuple[np.ndarray, np.ndarray]], float, int]:
    with h5py.File(posterior_path, "r") as posterior_file:
        posterior_samples = posterior_file["C00:NRSur7dq4/posterior_samples"]
        maxl_index = int(posterior_samples["log_likelihood"][:].argmax())
        injection_parameters = {
            key: float(posterior_samples[key][maxl_index]) for key in INJECTION_KEYS
        }
        maxl_log_likelihood = float(posterior_samples["log_likelihood"][maxl_index])
        psds = {
            detector: tuple(
                posterior_file[f"C00:NRSur7dq4/psds/{detector}"][:].T
            )
            for detector in ("H1", "L1")
        }
    return injection_parameters, psds, maxl_log_likelihood, maxl_index


def build_waveform_generator(settings: RunSettings) -> bilby.gw.LALCBCWaveformGenerator:
    return bilby.gw.LALCBCWaveformGenerator(
        duration=settings.duration,
        sampling_frequency=settings.sampling_frequency,
        start_time=settings.start_time,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant=settings.waveform_approximant,
            reference_frequency=settings.reference_frequency,
            minimum_frequency=settings.minimum_frequency["waveform"],
            maximum_frequency=settings.maximum_frequency,
            catch_waveform_errors=True,
            pn_spin_order=-1,
            pn_tidal_order=-1,
            pn_phase_order=-1,
            pn_amplitude_order=0,
            mode_array=None,
        ),
    )


def build_empty_interferometer(
    detector: str,
    settings: RunSettings,
    psds: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    roll_off: float,
):
    interferometer = bilby.gw.detector.get_empty_interferometer(detector)
    frequency_array, psd_array = psds[detector]
    interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=frequency_array,
        psd_array=psd_array,
    )
    interferometer.minimum_frequency = settings.minimum_frequency[detector]
    interferometer.maximum_frequency = settings.maximum_frequency
    interferometer.strain_data.roll_off = roll_off
    return interferometer


def build_raw_interferometers(
    settings: RunSettings,
    psds: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    nu_injection: float,
    staging_seed: int,
    injection_parameters: dict[str, float],
    waveform_generator,
) -> bilby.gw.detector.InterferometerList:
    bilby.core.utils.random.seed(staging_seed)
    interferometers = bilby.gw.detector.InterferometerList(
        [
            build_empty_interferometer(
                detector,
                settings,
                psds,
                roll_off=settings.tukey_roll_off,
            )
            for detector in settings.detectors
        ]
    )
    interferometers.set_strain_data_from_power_spectral_densities_student_t(
        sampling_frequency=settings.sampling_frequency,
        duration=settings.duration,
        nu=nu_injection,
        start_time=settings.start_time,
        num_frequency_bands=1,
    )
    interferometers.inject_signal(
        parameters=injection_parameters,
        waveform_generator=waveform_generator,
    )
    return interferometers


def clone_with_frequency_domain_strain(
    settings: RunSettings,
    psds: dict[str, tuple[np.ndarray, np.ndarray]],
    frequency_domain_strain: dict[str, np.ndarray],
    *,
    roll_off: float,
) -> bilby.gw.detector.InterferometerList:
    interferometers = bilby.gw.detector.InterferometerList([])
    for detector in settings.detectors:
        interferometer = build_empty_interferometer(
            detector, settings, psds, roll_off=roll_off
        )
        interferometer.set_strain_data_from_frequency_domain_strain(
            frequency_domain_strain=frequency_domain_strain[detector],
            sampling_frequency=settings.sampling_frequency,
            duration=settings.duration,
            start_time=settings.start_time,
        )
        interferometers.append(interferometer)
    return interferometers


def clone_with_time_domain_strain(
    settings: RunSettings,
    psds: dict[str, tuple[np.ndarray, np.ndarray]],
    time_domain_strain: dict[str, np.ndarray],
    *,
    roll_off: float,
) -> bilby.gw.detector.InterferometerList:
    interferometers = bilby.gw.detector.InterferometerList([])
    for detector in settings.detectors:
        interferometer = build_empty_interferometer(
            detector, settings, psds, roll_off=roll_off
        )
        interferometer.strain_data.set_from_time_domain_strain(
            time_domain_strain[detector],
            sampling_frequency=settings.sampling_frequency,
            duration=settings.duration,
            start_time=settings.start_time,
        )
        interferometers.append(interferometer)
    return interferometers


def gwpy_roundtrip_time_domain_strain(
    settings: RunSettings, time_domain_strain: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], float]:
    max_abs_difference = 0.0
    reread = {}
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)
        for detector in settings.detectors:
            filename = temporary_directory / f"{detector}.hdf5"
            timeseries = TimeSeries(
                time_domain_strain[detector],
                t0=settings.start_time,
                dt=1.0 / settings.sampling_frequency,
                name=f"{detector}_SIM",
            )
            timeseries.write(str(filename), format="hdf5", overwrite=True)
            reread_series = TimeSeries.read(
                str(filename),
                start=settings.start_time,
                end=settings.start_time + settings.duration,
                format="hdf5",
            )
            reread[detector] = reread_series.value
            max_abs_difference = max(
                max_abs_difference,
                float(
                    np.max(
                        np.abs(time_domain_strain[detector] - reread_series.value)
                    )
                ),
            )
    return reread, max_abs_difference


def build_nu_grid(nu_grid_max: float) -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                np.linspace(2.1, 10.0, 5000),
                np.linspace(10.0, min(50.0, nu_grid_max), 2500),
                np.linspace(max(50.0, min(50.0, nu_grid_max)), nu_grid_max, 2000),
            ]
        )
    )


def posterior_curve_from_log_likelihood(
    nu_grid: np.ndarray, log_likelihood_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    log_posterior = log_likelihood_values - np.max(log_likelihood_values)
    posterior_density = np.exp(log_posterior)
    posterior_density /= np.trapezoid(posterior_density, nu_grid)
    cumulative_density = np.concatenate(
        [[0.0], cumulative_trapezoid(posterior_density, nu_grid)]
    )
    cumulative_density /= cumulative_density[-1]
    return posterior_density, cumulative_density


def summarize_curve(
    nu_grid: np.ndarray,
    posterior_density: np.ndarray,
    cumulative_density: np.ndarray,
) -> dict[str, float]:
    quantile = lambda probability: float(
        np.interp(probability, cumulative_density, nu_grid)
    )
    return dict(
        map=float(nu_grid[np.argmax(posterior_density)]),
        mean=float(np.trapezoid(nu_grid * posterior_density, nu_grid)),
        median=quantile(0.5),
        q05=quantile(0.05),
        q16=quantile(0.16),
        q84=quantile(0.84),
        q95=quantile(0.95),
        p_nu_lt_3=float(np.interp(3.0, nu_grid, cumulative_density)),
    )


def compute_fixed_signal_nu_posterior(
    interferometers,
    waveform_generator,
    injection_parameters: dict[str, float],
    nu_grid: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    waveform_polarizations = waveform_generator.frequency_domain_strain(
        injection_parameters
    )
    likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
        interferometers=interferometers,
        waveform_generator=waveform_generator,
        nu=3.0,
        infer_nu=True,
        num_frequency_bands=1,
        detector_dependent_nu=False,
        phase_marginalization=False,
        time_marginalization=False,
        distance_marginalization=False,
        calibration_marginalization=False,
        reference_frame="sky",
        time_reference="geocent",
    )

    log_likelihood_values = np.array(
        [
            sum(
                likelihood._compute_detector_log_likelihood(
                    interferometer=interferometer,
                    nu_values=np.array([float(nu)]),
                    parameters=injection_parameters,
                    waveform_polarizations=waveform_polarizations,
                )
                for interferometer in interferometers
            )
            for nu in nu_grid
        ],
        dtype=float,
    )
    posterior_density, cumulative_density = posterior_curve_from_log_likelihood(
        nu_grid, log_likelihood_values
    )
    return (
        summarize_curve(nu_grid, posterior_density, cumulative_density),
        posterior_density,
    )


def load_full_run_posterior_summary(run_dir: Path) -> dict[str, float] | None:
    final_result_dir = run_dir / "final_result"
    if not final_result_dir.is_dir():
        return None
    try:
        result_path = find_unique_file(
            final_result_dir, "*merge_result.hdf5", "merged final result"
        )
    except RuntimeError:
        return None

    with h5py.File(result_path, "r") as result_file:
        if "posterior" not in result_file or "nu" not in result_file["posterior"]:
            return None
        nu_samples = np.asarray(result_file["posterior"]["nu"][:], dtype=float)
    return dict(
        source=str(result_path),
        median=float(np.median(nu_samples)),
        q05=float(np.quantile(nu_samples, 0.05)),
        q16=float(np.quantile(nu_samples, 0.16)),
        q84=float(np.quantile(nu_samples, 0.84)),
        q95=float(np.quantile(nu_samples, 0.95)),
        p_nu_lt_3=float(np.mean(nu_samples < 3.0)),
    )


def max_abs_frequency_difference(
    interferometers_a, interferometers_b
) -> float:
    maximum = 0.0
    for interferometer_a, interferometer_b in zip(interferometers_a, interferometers_b):
        difference = np.max(
            np.abs(
                interferometer_a.frequency_domain_strain
                - interferometer_b.frequency_domain_strain
            )
        )
        maximum = max(maximum, float(difference))
    return maximum


def plot_stage_posteriors(
    nu_grid: np.ndarray,
    stage_posteriors: dict[str, np.ndarray],
    outpath: Path,
) -> None:
    plt.figure(figsize=(9, 5.5))
    colors = {
        "raw_fd_direct": "#1f77b4",
        "td_roundtrip_no_window": "#2ca02c",
        "set_from_time_domain_rolloff_1p0": "#d62728",
        "gwpy_hdf5_roundtrip_rolloff_1p0": "#9467bd",
        "stored_data_dump": "#ff7f0e",
    }
    labels = {
        "raw_fd_direct": "Raw FD Student-t draw",
        "td_roundtrip_no_window": "TD/FD round-trip, no window",
        "set_from_time_domain_rolloff_1p0": "bilby TD conditioning, roll-off 1.0",
        "gwpy_hdf5_roundtrip_rolloff_1p0": "gwpy HDF5 round-trip + bilby conditioning",
        "stored_data_dump": "Stored bilby_pipe data dump",
    }
    for key, density in stage_posteriors.items():
        plt.plot(
            nu_grid,
            density,
            linewidth=2,
            color=colors.get(key),
            label=labels.get(key, key),
        )
    plt.axvline(3.0, color="black", linestyle="--", linewidth=1.5, label="Injected nu = 3")
    plt.xlim(2.1, 5.0)
    plt.xlabel("nu")
    plt.ylabel("Posterior density")
    plt.title("Student-t nu posterior after each conditioning stage")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_rolloff_scan(
    rolloff_values: list[float],
    medians: list[float],
    outpath: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(rolloff_values, medians, marker="o", linewidth=2, color="#1f77b4")
    plt.axhline(3.0, color="black", linestyle="--", linewidth=1.5, label="Injected nu = 3")
    plt.xlabel("Tukey roll-off [s]")
    plt.ylabel("Median inferred nu")
    plt.title("Effective nu shift from bilby windowing")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_duration_scan(duration_scan: list[dict[str, float]], outpath: Path) -> None:
    durations = [entry["duration"] for entry in duration_scan]
    raw_medians = [entry["raw_summary"]["median"] for entry in duration_scan]
    conditioned_medians = [
        entry["conditioned_summary"]["median"] for entry in duration_scan
    ]
    plt.figure(figsize=(7.5, 4.75))
    plt.plot(
        durations,
        raw_medians,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Raw FD Student-t draw",
    )
    plt.plot(
        durations,
        conditioned_medians,
        marker="o",
        linewidth=2,
        color="#d62728",
        label="After bilby TD conditioning",
    )
    plt.axhline(3.0, color="black", linestyle="--", linewidth=1.5, label="Injected nu = 3")
    plt.xlabel("Analysis duration [s]")
    plt.ylabel("Median inferred nu")
    plt.title("Duration dependence of the conditioning-induced nu shift")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_bias_overview(
    nu_grid: np.ndarray,
    stage_curves: dict[str, np.ndarray],
    stage_summaries: dict[str, dict[str, float]],
    rolloff_scan: list[dict[str, float]],
    duration_scan: list[dict[str, float]],
    outpath: Path,
    *,
    injected_nu: float,
) -> None:
    figure = plt.figure(figsize=(13, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.15, 1.0])

    posterior_axis = figure.add_subplot(grid[0, :])
    colors = {
        "raw_fd_direct": "#1f77b4",
        "set_from_time_domain_rolloff_1p0": "#d62728",
        "stored_data_dump": "#ff7f0e",
    }
    labels = {
        "raw_fd_direct": (
            "Raw FD Student-t draw "
            f"(median={stage_summaries['raw_fd_direct']['median']:.3f})"
        ),
        "set_from_time_domain_rolloff_1p0": (
            "bilby TD conditioning "
            f"(median={stage_summaries['set_from_time_domain_rolloff_1p0']['median']:.3f})"
        ),
        "stored_data_dump": (
            "Stored bilby_pipe data dump "
            f"(median={stage_summaries['stored_data_dump']['median']:.3f})"
        ),
    }
    for key in ("raw_fd_direct", "set_from_time_domain_rolloff_1p0", "stored_data_dump"):
        posterior_axis.plot(
            nu_grid,
            stage_curves[key],
            linewidth=2.25,
            color=colors[key],
            label=labels[key],
        )
    posterior_axis.axvline(
        injected_nu,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Injected nu = {injected_nu:g}",
    )
    posterior_axis.set_xlim(2.1, 4.8)
    posterior_axis.set_xlabel("nu")
    posterior_axis.set_ylabel("Posterior density")
    posterior_axis.set_title("Posterior shift from bilby time-domain conditioning")
    posterior_axis.legend(frameon=False, fontsize=10)

    rolloff_axis = figure.add_subplot(grid[1, 0])
    rolloff_values = [entry["roll_off"] for entry in rolloff_scan]
    rolloff_bias = [entry["median"] - injected_nu for entry in rolloff_scan]
    rolloff_yerr = np.array(
        [
            [entry["median"] - entry["q05"] for entry in rolloff_scan],
            [entry["q95"] - entry["median"] for entry in rolloff_scan],
        ]
    )
    rolloff_axis.errorbar(
        rolloff_values,
        rolloff_bias,
        yerr=rolloff_yerr,
        marker="o",
        linewidth=2,
        color="#d62728",
        capsize=3,
    )
    rolloff_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.25)
    rolloff_axis.set_xlabel("Tukey roll-off [s]")
    rolloff_axis.set_ylabel("Posterior median bias in nu")
    rolloff_axis.set_title("Bias vs roll-off")

    duration_axis = figure.add_subplot(grid[1, 1])
    durations = [entry["duration"] for entry in duration_scan]
    raw_bias = [entry["raw_summary"]["median"] - injected_nu for entry in duration_scan]
    conditioned_bias = [
        entry["conditioned_summary"]["median"] - injected_nu for entry in duration_scan
    ]
    raw_yerr = np.array(
        [
            [
                entry["raw_summary"]["median"] - entry["raw_summary"]["q05"]
                for entry in duration_scan
            ],
            [
                entry["raw_summary"]["q95"] - entry["raw_summary"]["median"]
                for entry in duration_scan
            ],
        ]
    )
    conditioned_yerr = np.array(
        [
            [
                entry["conditioned_summary"]["median"] - entry["conditioned_summary"]["q05"]
                for entry in duration_scan
            ],
            [
                entry["conditioned_summary"]["q95"] - entry["conditioned_summary"]["median"]
                for entry in duration_scan
            ],
        ]
    )
    duration_axis.errorbar(
        durations,
        raw_bias,
        yerr=raw_yerr,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        capsize=3,
        label="Raw FD draw",
    )
    duration_axis.errorbar(
        durations,
        conditioned_bias,
        yerr=conditioned_yerr,
        marker="o",
        linewidth=2,
        color="#d62728",
        capsize=3,
        label="After bilby TD conditioning",
    )
    duration_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.25)
    duration_axis.set_xlabel("Analysis duration [s]")
    duration_axis.set_ylabel("Posterior median bias in nu")
    duration_axis.set_title("Bias vs duration at fixed roll-off")
    duration_axis.legend(frameon=False, fontsize=9)

    figure.tight_layout()
    figure.savefig(outpath, dpi=220)
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    posterior_path = args.posterior_path.expanduser().resolve()
    outdir = (
        args.outdir.expanduser().resolve()
        if args.outdir is not None
        else (run_dir / "conditioning_shift_reproduction").resolve()
    )
    outdir.mkdir(parents=True, exist_ok=True)

    data_dump_path, data_dump = load_data_dump(run_dir)
    settings = build_run_settings(data_dump)
    injection_parameters, psds, maxl_log_likelihood, maxl_index = (
        load_maximum_likelihood_injection(posterior_path)
    )
    waveform_generator = build_waveform_generator(settings)

    raw_ifos = build_raw_interferometers(
        settings,
        psds,
        nu_injection=args.nu_injection,
        staging_seed=args.staging_seed,
        injection_parameters=injection_parameters,
        waveform_generator=waveform_generator,
    )
    raw_frequency_domain_strain = {
        interferometer.name: interferometer.strain_data._frequency_domain_strain.copy()
        for interferometer in raw_ifos
    }
    raw_time_domain_strain = {
        detector: bilby.core.utils.infft(
            frequency_domain_strain, settings.sampling_frequency
        )
        for detector, frequency_domain_strain in raw_frequency_domain_strain.items()
    }
    reread_time_domain_strain, max_td_diff_after_hdf5 = gwpy_roundtrip_time_domain_strain(
        settings, raw_time_domain_strain
    )

    stage_interferometers = {
        "raw_fd_direct": clone_with_frequency_domain_strain(
            settings,
            psds,
            raw_frequency_domain_strain,
            roll_off=settings.tukey_roll_off,
        ),
        "td_roundtrip_no_window": clone_with_frequency_domain_strain(
            settings,
            psds,
            {
                detector: bilby.core.utils.nfft(
                    time_domain_strain, settings.sampling_frequency
                )[0]
                for detector, time_domain_strain in raw_time_domain_strain.items()
            },
            roll_off=settings.tukey_roll_off,
        ),
        "set_from_time_domain_rolloff_1p0": clone_with_time_domain_strain(
            settings,
            psds,
            raw_time_domain_strain,
            roll_off=settings.tukey_roll_off,
        ),
        "gwpy_hdf5_roundtrip_rolloff_1p0": clone_with_time_domain_strain(
            settings,
            psds,
            reread_time_domain_strain,
            roll_off=settings.tukey_roll_off,
        ),
        "stored_data_dump": data_dump.interferometers,
    }

    nu_grid = build_nu_grid(args.nu_grid_max)
    stage_summaries: dict[str, dict[str, float]] = {}
    stage_curves: dict[str, np.ndarray] = {}
    for name, interferometers in stage_interferometers.items():
        summary, density = compute_fixed_signal_nu_posterior(
            interferometers,
            waveform_generator,
            injection_parameters,
            nu_grid,
        )
        stage_summaries[name] = summary
        stage_curves[name] = density

    rolloff_values = parse_float_list(args.rolloff_values)
    rolloff_scan = []
    for rolloff in rolloff_values:
        summary, _ = compute_fixed_signal_nu_posterior(
            clone_with_time_domain_strain(
                settings,
                psds,
                raw_time_domain_strain,
                roll_off=rolloff,
            ),
            waveform_generator,
            injection_parameters,
            nu_grid,
        )
        rolloff_scan.append(dict(roll_off=rolloff, **summary))

    duration_values = (
        parse_float_list(args.duration_values)
        if args.duration_values is not None
        else default_duration_values(settings)
    )
    duration_scan = []
    for duration in duration_values:
        duration_settings = with_duration(settings, duration)
        duration_waveform_generator = build_waveform_generator(duration_settings)
        duration_raw_ifos = build_raw_interferometers(
            duration_settings,
            psds,
            nu_injection=args.nu_injection,
            staging_seed=args.staging_seed,
            injection_parameters=injection_parameters,
            waveform_generator=duration_waveform_generator,
        )
        duration_raw_frequency_domain_strain = {
            interferometer.name: interferometer.strain_data._frequency_domain_strain.copy()
            for interferometer in duration_raw_ifos
        }
        duration_raw_time_domain_strain = {
            detector: bilby.core.utils.infft(
                frequency_domain_strain, duration_settings.sampling_frequency
            )
            for detector, frequency_domain_strain in duration_raw_frequency_domain_strain.items()
        }
        raw_summary, _ = compute_fixed_signal_nu_posterior(
            clone_with_frequency_domain_strain(
                duration_settings,
                psds,
                duration_raw_frequency_domain_strain,
                roll_off=duration_settings.tukey_roll_off,
            ),
            duration_waveform_generator,
            injection_parameters,
            nu_grid,
        )
        conditioned_summary, _ = compute_fixed_signal_nu_posterior(
            clone_with_time_domain_strain(
                duration_settings,
                psds,
                duration_raw_time_domain_strain,
                roll_off=duration_settings.tukey_roll_off,
            ),
            duration_waveform_generator,
            injection_parameters,
            nu_grid,
        )
        duration_scan.append(
            dict(
                duration=duration,
                roll_off=duration_settings.tukey_roll_off,
                alpha=2.0 * duration_settings.tukey_roll_off / duration,
                raw_summary=raw_summary,
                conditioned_summary=conditioned_summary,
                delta_median=conditioned_summary["median"] - raw_summary["median"],
            )
        )

    overview_plot_path = outdir / "conditioning_bias_overview.png"
    plot_bias_overview(
        nu_grid,
        stage_curves,
        stage_summaries,
        rolloff_scan,
        duration_scan,
        overview_plot_path,
        injected_nu=args.nu_injection,
    )

    np.savez(
        outdir / "conditioning_stage_posteriors.npz",
        nu_grid=nu_grid,
        **stage_curves,
    )

    full_run_summary = load_full_run_posterior_summary(run_dir)
    direct_vs_dump_fd_difference = max_abs_frequency_difference(
        stage_interferometers["set_from_time_domain_rolloff_1p0"],
        stage_interferometers["stored_data_dump"],
    )

    print(f"Wrote bias overview plot to {overview_plot_path}")
    print(
        "Stored the posterior curves for reuse in "
        f"{outdir / 'conditioning_stage_posteriors.npz'}"
    )
    print(
        "Stored-data dump matches the bilby-conditioned strain: "
        f"{direct_vs_dump_fd_difference < 1e-12}"
    )
    if full_run_summary is not None:
        print(
            "Full Student-t run median nu = "
            f"{full_run_summary['median']:.4f}; conditioned rerun median nu = "
            f"{stage_summaries['stored_data_dump']['median']:.4f}"
        )


if __name__ == "__main__":
    main()
