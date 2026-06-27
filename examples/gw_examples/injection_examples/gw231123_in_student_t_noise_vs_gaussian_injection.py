#!/usr/bin/env python3

"""Run a short local GW231123 Student-t injection study.

This script uses the LVK NRSur7dq4 maximum-likelihood sample for GW231123 as the
injection parameters, generates detector noise from a single-band Student-t model
with one shared ``nu`` across H1/L1, and then runs two deliberately small local
inference jobs:

1. Gaussian likelihood, sampling only ``luminosity_distance``
2. Student-t likelihood, sampling ``luminosity_distance`` and ``nu``

All other CBC parameters are fixed to the injected values with delta-function priors.
The defaults are tuned to finish quickly on a local machine while still showing the
expected Gaussian bias on Student-t noise. If the bias is not visible at the default
distance scale, the script automatically increases the SNR and retries.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.special import gammaln

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import bilby


SCRIPT_DIR = Path(__file__).resolve().parent
POSTERIOR_PATH_CANDIDATES = (
    SCRIPT_DIR / "LVK_posterior" / "posterior_samples.h5",
    SCRIPT_DIR.parent
    / "data_examples"
    / "Cluster_runs_and_utils"
    / "LVK_posteriors"
    / "GW231123"
    / "posterior_samples.h5",
)

TEMPLATE_SETTINGS = dict(
    detectors=("H1", "L1"),
    trigger_time=1384782888.634277,
    duration=8.0,
    post_trigger_duration=2.0,
    sampling_frequency=1024.0,
    maximum_frequency=448.0,
    minimum_frequency={"H1": 20.0, "L1": 20.0, "waveform": 0.0},
    reference_frequency=10.0,
    waveform_approximant="NRSur7dq4",
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

DEFAULT_DISTANCE_SCALES = (1.0, 0.5, 0.25, 0.15, 0.1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="outdir/gw231123_student_t_injection_local")
    parser.add_argument("--label", default="GW231123_student_t_local")
    parser.add_argument("--nu-injection", type=float, default=2.1)
    parser.add_argument("--nu-min", type=float, default=2.1)
    parser.add_argument("--nu-max", type=float, default=10.0)
    parser.add_argument("--nlive", type=int, default=40)
    parser.add_argument("--maxcall", type=int, default=1000)
    parser.add_argument("--dlogz", type=float, default=5.0)
    parser.add_argument("--npool", type=int, default=1)
    parser.add_argument("--sampling-seed", type=int, default=12345)
    parser.add_argument(
        "--distance-prior-fraction",
        type=float,
        default=0.3,
        help="Use a symmetric +/- fraction around the injected luminosity distance.",
    )
    parser.add_argument(
        "--fix-distance",
        action="store_true",
        help="Keep luminosity_distance fixed to the injected value instead of sampling it.",
    )
    parser.add_argument(
        "--required-bias-improvement",
        type=float,
        default=0.05,
        help="Require the Student-t distance bias to improve on Gaussian by at least this fraction.",
    )
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=None,
        help="Override automatic SNR escalation and use one fixed luminosity-distance scale.",
    )
    parser.add_argument("--skip-gaussian", action="store_true")
    parser.add_argument("--skip-student", action="store_true")
    parser.add_argument("--plot-data", action="store_true")
    parser.add_argument(
        "--waveform-plot-samples",
        type=int,
        default=50,
        help="Maximum number of posterior samples used in waveform reconstruction plots.",
    )
    parser.add_argument(
        "--waveform-approximant",
        default=TEMPLATE_SETTINGS["waveform_approximant"],
        help="Defaults to the NRSur7dq4 approximant used in the LVK run.",
    )
    return parser


def resolve_posterior_path() -> Path:
    for posterior_path in POSTERIOR_PATH_CANDIDATES:
        if posterior_path.exists():
            return posterior_path
    checked_paths = "\n".join(str(path) for path in POSTERIOR_PATH_CANDIDATES)
    raise FileNotFoundError(
        "Could not locate GW231123 posterior_samples.h5. Checked:\n"
        f"{checked_paths}"
    )


def load_maximum_likelihood_injection(posterior_path: Path) -> tuple[dict, float, int]:
    with h5py.File(posterior_path, "r") as posterior_file:
        posterior_samples = posterior_file["C00:NRSur7dq4/posterior_samples"]
        maxl_index = int(posterior_samples["log_likelihood"][:].argmax())
        maxl_parameters = {
            key: float(posterior_samples[key][maxl_index]) for key in INJECTION_KEYS
        }
        maxl_log_likelihood = float(posterior_samples["log_likelihood"][maxl_index])

    return maxl_parameters, maxl_log_likelihood, maxl_index


def load_psds(posterior_path: Path) -> dict[str, tuple]:
    with h5py.File(posterior_path, "r") as posterior_file:
        return {
            detector: tuple(
                posterior_file[f"C00:NRSur7dq4/psds/{detector}"][:].T
            )
            for detector in TEMPLATE_SETTINGS["detectors"]
        }


def build_waveform_generator(waveform_approximant: str) -> bilby.gw.LALCBCWaveformGenerator:
    return bilby.gw.LALCBCWaveformGenerator(
        duration=TEMPLATE_SETTINGS["duration"],
        sampling_frequency=TEMPLATE_SETTINGS["sampling_frequency"],
        start_time=(
            TEMPLATE_SETTINGS["trigger_time"]
            + TEMPLATE_SETTINGS["post_trigger_duration"]
            - TEMPLATE_SETTINGS["duration"]
        ),
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant=waveform_approximant,
            reference_frequency=TEMPLATE_SETTINGS["reference_frequency"],
            minimum_frequency=TEMPLATE_SETTINGS["minimum_frequency"]["waveform"],
            maximum_frequency=TEMPLATE_SETTINGS["maximum_frequency"],
            catch_waveform_errors=True,
            pn_spin_order=-1,
            pn_tidal_order=-1,
            pn_phase_order=-1,
            pn_amplitude_order=0,
            mode_array=None,
        ),
    )


def build_interferometers(
    psds: dict[str, tuple],
    *,
    nu_injection: float,
) -> bilby.gw.detector.InterferometerList:
    interferometers = bilby.gw.detector.InterferometerList([])
    for detector in TEMPLATE_SETTINGS["detectors"]:
        ifo = bilby.gw.detector.get_empty_interferometer(detector)
        frequency_array, psd_array = psds[detector]
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=frequency_array,
            psd_array=psd_array,
        )
        ifo.minimum_frequency = TEMPLATE_SETTINGS["minimum_frequency"][detector]
        ifo.maximum_frequency = TEMPLATE_SETTINGS["maximum_frequency"]
        interferometers.append(ifo)

    start_time = (
        TEMPLATE_SETTINGS["trigger_time"]
        + TEMPLATE_SETTINGS["post_trigger_duration"]
        - TEMPLATE_SETTINGS["duration"]
    )
    interferometers.set_strain_data_from_power_spectral_densities_student_t(
        sampling_frequency=TEMPLATE_SETTINGS["sampling_frequency"],
        duration=TEMPLATE_SETTINGS["duration"],
        nu=nu_injection,
        start_time=start_time,
    )
    return interferometers


def get_trial_distance_scales(args: argparse.Namespace) -> list[float]:
    if args.distance_scale is not None:
        return [float(args.distance_scale)]
    return list(DEFAULT_DISTANCE_SCALES)


def build_injection_parameters(base_parameters: dict, distance_scale: float) -> dict:
    injection_parameters = copy.deepcopy(base_parameters)
    injection_parameters["luminosity_distance"] *= float(distance_scale)
    return injection_parameters


def compute_network_optimal_snr(
    interferometers: bilby.gw.detector.InterferometerList,
    waveform_generator: bilby.gw.LALCBCWaveformGenerator,
    injection_parameters: dict,
) -> float:
    waveform_polarizations = waveform_generator.frequency_domain_strain(injection_parameters)
    optimal_snr_squared = 0.0
    for interferometer in interferometers:
        signal = interferometer.get_detector_response(
            waveform_polarizations, injection_parameters
        )
        optimal_snr_squared += interferometer.optimal_snr_squared(signal).real
    return float(np.sqrt(optimal_snr_squared))


def build_priors(
    injection_parameters: dict,
    *,
    hypothesis: str,
    distance_prior_fraction: float,
    fix_distance: bool,
    nu_min: float,
    nu_max: float,
) -> bilby.core.prior.PriorDict:
    priors = bilby.core.prior.PriorDict()
    for key, value in injection_parameters.items():
        if key == "luminosity_distance":
            continue
        priors[key] = bilby.core.prior.DeltaFunction(value, name=key)

    if fix_distance:
        priors["luminosity_distance"] = bilby.core.prior.DeltaFunction(
            injection_parameters["luminosity_distance"],
            name="luminosity_distance",
        )
    else:
        priors["luminosity_distance"] = bilby.core.prior.Uniform(
            injection_parameters["luminosity_distance"] * (1.0 - distance_prior_fraction),
            injection_parameters["luminosity_distance"] * (1.0 + distance_prior_fraction),
            name="luminosity_distance",
        )

    if hypothesis == "student":
        priors["nu"] = bilby.core.prior.Uniform(nu_min, nu_max, name="nu")

    return priors


def build_likelihood(
    hypothesis: str,
    interferometers: bilby.gw.detector.InterferometerList,
    waveform_generator: bilby.gw.LALCBCWaveformGenerator,
    *,
    nu_injection: float,
):
    likelihood_kwargs = dict(
        interferometers=interferometers,
        waveform_generator=waveform_generator,
        phase_marginalization=False,
        time_marginalization=False,
        distance_marginalization=False,
        calibration_marginalization=False,
        reference_frame="sky",
        time_reference="geocent",
    )

    if hypothesis == "gaussian":
        return bilby.gw.likelihood.GravitationalWaveTransient(**likelihood_kwargs)
    if hypothesis == "student":
        return bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            **likelihood_kwargs,
            nu=nu_injection,
            infer_nu=True,
            num_frequency_bands=1,
            detector_dependent_noise=False,
        )
    raise ValueError(f"Unknown hypothesis: {hypothesis}")


def get_waveform_parameters(parameters: dict) -> dict:
    return {key: parameters[key] for key in INJECTION_KEYS}


def get_result_median_parameters(
    result, fallback_parameters: dict | None = None
) -> dict:
    parameters = dict(fallback_parameters or {})
    injection_parameters = getattr(result, "injection_parameters", None) or {}
    parameters.update(injection_parameters)
    parameters.update(result.posterior.median(numeric_only=True).to_dict())
    return parameters


def gaussian_component_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def student_t_component_pdf(x: np.ndarray, nu: float) -> np.ndarray:
    if nu <= 0:
        raise ValueError("nu must be positive")
    log_norm = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
    )
    return np.exp(log_norm - 0.5 * (nu + 1.0) * np.log1p((x**2) / nu))


def build_whitened_residual_components(
    interferometers: bilby.gw.detector.InterferometerList,
    waveform_generator: bilby.gw.LALCBCWaveformGenerator,
    parameters: dict,
) -> dict[str, np.ndarray]:
    waveform_polarizations = waveform_generator.frequency_domain_strain(
        get_waveform_parameters(parameters)
    )
    components = {}
    for interferometer in interferometers:
        detector_response = interferometer.get_detector_response(
            waveform_polarizations, get_waveform_parameters(parameters)
        )
        whitened_residual = interferometer.whiten_frequency_series(
            interferometer.frequency_domain_strain - detector_response
        )[interferometer.frequency_mask]
        components[interferometer.name] = np.concatenate(
            [whitened_residual.real, whitened_residual.imag]
        )
    return components


def plot_bayes_factor_comparison(
    outdir: Path,
    label: str,
    gaussian_summary: dict | None,
    student_summary: dict | None,
) -> Path | None:
    if gaussian_summary is None and student_summary is None:
        return None

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        gridspec_kw=dict(height_ratios=[2.0, 1.0]),
    )

    hypotheses = []
    log_bayes_factors = []
    colors = []
    if gaussian_summary is not None:
        hypotheses.append("Gaussian")
        log_bayes_factors.append(gaussian_summary["log_bayes_factor"])
        colors.append("#ff7f0e")
    if student_summary is not None:
        hypotheses.append("Student-t")
        log_bayes_factors.append(student_summary["log_bayes_factor"])
        colors.append("#1f77b4")

    axes[0].bar(hypotheses, log_bayes_factors, color=colors)
    axes[0].set_ylabel(r"$\ln B_{\mathrm{signal/noise}}$")
    axes[0].set_title("Signal-vs-noise Bayes factors")

    for hypothesis, log_bf in zip(hypotheses, log_bayes_factors):
        axes[0].text(
            hypothesis,
            log_bf,
            f"{log_bf:.2f}",
            ha="center",
            va="bottom",
        )

    comparison_lines = []
    if gaussian_summary is not None:
        comparison_lines.append(
            "Gaussian: ln Z = "
            f"{gaussian_summary['log_evidence']:.3f}, ln Z_noise = {gaussian_summary['log_noise_evidence']:.3f}"
        )
    if student_summary is not None:
        comparison_lines.append(
            "Student-t: ln Z = "
            f"{student_summary['log_evidence']:.3f}, ln Z_noise = {student_summary['log_noise_evidence']:.3f}"
        )
    if gaussian_summary is not None and student_summary is not None:
        log_bf_student_over_gaussian = (
            student_summary["log_evidence"] - gaussian_summary["log_evidence"]
        )
        comparison_lines.append(
            "Student/gaussian model odds: "
            f"ln B = {log_bf_student_over_gaussian:.3f}, "
            f"log10 B = {log_bf_student_over_gaussian / np.log(10.0):.3f}"
        )

    axes[1].axis("off")
    axes[1].text(
        0.02,
        0.95,
        "\n".join(comparison_lines),
        va="top",
        ha="left",
        family="monospace",
    )

    figure_path = outdir / f"{label}_bayes_factor_comparison.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def plot_waveform_reconstructions(
    result,
    interferometers: bilby.gw.detector.InterferometerList,
    waveform_generator: bilby.gw.LALCBCWaveformGenerator,
    injection_parameters: dict,
    *,
    n_samples: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    posterior = result.posterior
    if len(posterior) > n_samples:
        sample_indices = np.linspace(0, len(posterior) - 1, n_samples, dtype=int)
        posterior = posterior.iloc[sample_indices]

    df = 1.0 / interferometers[0].duration
    if "geocent_time" in posterior:
        geocent_time = float(posterior["geocent_time"].mean())
    else:
        geocent_time = float(injection_parameters["geocent_time"])
    plot_start = geocent_time - 0.15
    plot_end = geocent_time + 0.05

    waveform_paths = []
    for interferometer in interferometers:
        frequency_mask = interferometer.frequency_mask
        plot_frequencies = interferometer.frequency_array[frequency_mask]

        time_mask = (
            (interferometer.time_array >= plot_start)
            & (interferometer.time_array <= plot_end)
        )
        plot_times = interferometer.time_array[time_mask] - geocent_time

        fd_waveforms = []
        td_waveforms = []
        for sample in posterior.to_dict(orient="records"):
            waveform_parameters = dict(injection_parameters)
            waveform_parameters.update(sample)
            waveform_parameters = get_waveform_parameters(waveform_parameters)
            waveform_polarizations = waveform_generator.frequency_domain_strain(
                waveform_parameters
            )
            detector_response = interferometer.get_detector_response(
                waveform_polarizations, waveform_parameters
            )
            fd_waveforms.append(
                bilby.gw.utils.asd_from_freq_series(
                    detector_response[frequency_mask], df
                )
            )
            whitened_fd = interferometer.whiten_frequency_series(detector_response)
            td_waveforms.append(
                interferometer.get_whitened_time_series_from_whitened_frequency_series(
                    whitened_fd
                )[time_mask]
            )

        fd_waveforms = np.asarray(fd_waveforms)
        td_waveforms = np.asarray(td_waveforms)
        fd_median = np.median(fd_waveforms, axis=0)
        fd_lower = np.percentile(fd_waveforms, 5, axis=0)
        fd_upper = np.percentile(fd_waveforms, 95, axis=0)
        td_median = np.median(td_waveforms, axis=0)
        td_lower = np.percentile(td_waveforms, 5, axis=0)
        td_upper = np.percentile(td_waveforms, 95, axis=0)

        injection_polarizations = waveform_generator.frequency_domain_strain(
            get_waveform_parameters(injection_parameters)
        )
        injection_response = interferometer.get_detector_response(
            injection_polarizations, get_waveform_parameters(injection_parameters)
        )
        injection_fd = bilby.gw.utils.asd_from_freq_series(
            injection_response[frequency_mask], df
        )
        injection_whitened_td = (
            interferometer.get_whitened_time_series_from_whitened_frequency_series(
                interferometer.whiten_frequency_series(injection_response)
            )[time_mask]
        )

        data_asd = bilby.gw.utils.asd_from_freq_series(
            interferometer.frequency_domain_strain[frequency_mask], df
        )
        data_whitened_td = interferometer.whitened_time_domain_strain[time_mask]

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            gridspec_kw=dict(height_ratios=[1.4, 1.0]),
        )
        axes[0].loglog(plot_frequencies, data_asd, color="#999999", alpha=0.7, label="Data")
        axes[0].loglog(plot_frequencies, fd_median, color="#1f77b4", label="Median reconstruction")
        axes[0].fill_between(
            plot_frequencies,
            fd_lower,
            fd_upper,
            color="#1f77b4",
            alpha=0.2,
            label="90% band",
        )
        axes[0].loglog(
            plot_frequencies,
            injection_fd,
            color="#000000",
            linestyle="--",
            label="Injection",
        )
        axes[0].set_xlim(
            interferometer.minimum_frequency, interferometer.maximum_frequency
        )
        axes[0].set_ylabel(r"ASD [Hz$^{-1/2}$]")
        axes[0].legend(loc="lower left", ncol=2)

        axes[1].plot(plot_times, data_whitened_td, color="#999999", alpha=0.7, label="Data")
        axes[1].plot(plot_times, td_median, color="#1f77b4", label="Median reconstruction")
        axes[1].fill_between(
            plot_times,
            td_lower,
            td_upper,
            color="#1f77b4",
            alpha=0.2,
            label="90% band",
        )
        axes[1].plot(
            plot_times,
            injection_whitened_td,
            color="#000000",
            linestyle="--",
            label="Injection",
        )
        axes[1].set_xlabel(r"$t - t_c$ [s]")
        axes[1].set_ylabel("Whitened strain")
        axes[1].set_xlim(plot_times[0], plot_times[-1])

        fig.suptitle(f"{result.label}: {interferometer.name}")
        figure_path = Path(result.outdir) / f"{result.label}_{interferometer.name}_waveform.png"
        fig.tight_layout()
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)
        waveform_paths.append(figure_path)

    return waveform_paths


def plot_whitened_residual_histograms(
    outdir: Path,
    label: str,
    interferometers: bilby.gw.detector.InterferometerList,
    waveform_generator: bilby.gw.LALCBCWaveformGenerator,
    *,
    gaussian_result=None,
    student_result=None,
    nu_injection: float,
) -> Path | None:
    plot_inputs = []
    if gaussian_result is not None:
        plot_inputs.append(("Gaussian median residual", gaussian_result))
    if student_result is not None:
        plot_inputs.append(("Student-t median residual", student_result))
    if not plot_inputs:
        return None

    import matplotlib.pyplot as plt

    component_sets = []
    for title, result in plot_inputs:
        component_sets.append(
            (
                title,
                build_whitened_residual_components(
                    interferometers,
                    waveform_generator,
                    get_result_median_parameters(result, injection_parameters),
                ),
            )
        )

    detector_names = [interferometer.name for interferometer in interferometers]
    all_components = [
        np.abs(components[detector_name])
        for _, components in component_sets
        for detector_name in detector_names
    ]
    x_limit = float(
        min(
            40.0,
            max(6.0, np.quantile(np.concatenate(all_components), 0.995)),
        )
    )
    x_values = np.linspace(-x_limit, x_limit, 1000)

    fig, axes = plt.subplots(
        len(detector_names),
        len(component_sets),
        figsize=(6.0 * len(component_sets), 3.8 * len(detector_names)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row_index, detector_name in enumerate(detector_names):
        for column_index, (title, components) in enumerate(component_sets):
            ax = axes[row_index][column_index]
            component_values = components[detector_name]
            ax.hist(
                component_values,
                bins=80,
                range=(-x_limit, x_limit),
                density=True,
                alpha=0.55,
                color="#4c72b0",
                label="Whitened residual",
            )
            ax.plot(
                x_values,
                gaussian_component_pdf(x_values),
                color="#dd8452",
                linewidth=2.0,
                label="Gaussian",
            )
            ax.plot(
                x_values,
                student_t_component_pdf(x_values, nu_injection),
                color="#55a868",
                linewidth=2.0,
                linestyle="--",
                label=f"Student-t (nu={nu_injection:.2f})",
            )
            ax.set_yscale("log")
            ax.set_title(f"{detector_name}: {title}")
            ax.set_xlabel("Whitened frequency-bin component")
            if column_index == 0:
                ax.set_ylabel("Density")
            if row_index == 0 and column_index == 0:
                ax.legend(loc="upper right")

    figure_path = outdir / f"{label}_whitened_residual_histograms.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def run_inference(
    hypothesis: str,
    interferometers: bilby.gw.detector.InterferometerList,
    waveform_generator: bilby.gw.LALCBCWaveformGenerator,
    injection_parameters: dict,
    args: argparse.Namespace,
    *,
    distance_scale: float,
):
    priors = build_priors(
        injection_parameters,
        hypothesis=hypothesis,
        distance_prior_fraction=args.distance_prior_fraction,
        fix_distance=args.fix_distance,
        nu_min=args.nu_min,
        nu_max=args.nu_max,
    )
    likelihood = build_likelihood(
        hypothesis,
        interferometers,
        waveform_generator,
        nu_injection=args.nu_injection,
    )

    run_label = f"{args.label}_{hypothesis}_scale_{distance_scale:g}".replace(".", "p")
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=args.nlive,
        bound="multi",
        sample="unif",
        maxcall=args.maxcall,
        dlogz=args.dlogz,
        npool=args.npool,
        outdir=str(Path(args.outdir).resolve()),
        label=run_label,
        injection_parameters=injection_parameters,
        clean=True,
        check_point=False,
        verbose=False,
    )

    if "luminosity_distance" in result.posterior:
        distance_median = float(result.posterior["luminosity_distance"].median())
        distance_bias_fraction = (
            distance_median / injection_parameters["luminosity_distance"] - 1.0
        )
    else:
        distance_median = float(injection_parameters["luminosity_distance"])
        distance_bias_fraction = 0.0

    summary = dict(
        label=run_label,
        hypothesis=hypothesis,
        distance_scale=float(distance_scale),
        distance_fixed=bool(args.fix_distance),
        true_luminosity_distance=float(injection_parameters["luminosity_distance"]),
        distance_median=distance_median,
        distance_bias_fraction=float(distance_bias_fraction),
        log_evidence=float(result.log_evidence),
        log_evidence_err=float(result.log_evidence_err),
        log_noise_evidence=float(result.log_noise_evidence),
        log_bayes_factor=float(result.log_bayes_factor),
        nsamples=int(len(result.posterior)),
    )
    if "nu" in result.posterior:
        summary["nu_median"] = float(result.posterior["nu"].median())

    bilby.core.utils.logger.info(
        "%s median luminosity distance = %.6f (bias %.3f%%)",
        hypothesis.capitalize(),
        distance_median,
        100.0 * distance_bias_fraction,
    )
    if "nu_median" in summary:
        bilby.core.utils.logger.info(
            "Student median nu = %.6f", summary["nu_median"]
        )

    return result, summary


def is_bias_visible(
    gaussian_summary: dict | None,
    student_summary: dict | None,
    *,
    required_improvement: float,
) -> bool:
    if gaussian_summary is None or student_summary is None:
        return False

    gaussian_bias = abs(gaussian_summary["distance_bias_fraction"])
    student_bias = abs(student_summary["distance_bias_fraction"])
    return gaussian_bias - student_bias >= required_improvement


def get_bias_improvement(
    gaussian_summary: dict | None,
    student_summary: dict | None,
) -> float | None:
    if gaussian_summary is None or student_summary is None:
        return None
    gaussian_bias = abs(gaussian_summary["distance_bias_fraction"])
    student_bias = abs(student_summary["distance_bias_fraction"])
    return float(gaussian_bias - student_bias)


def write_summary(
    outdir: Path,
    *,
    injection_parameters: dict,
    maxl_index: int,
    maxl_log_likelihood: float,
    network_optimal_snr: float,
    chosen_scale: float,
    bias_improvement: float | None,
    model_comparison: dict | None,
    generated_files: dict,
    gaussian_summary: dict | None,
    student_summary: dict | None,
) -> Path:
    summary = dict(
        maxl_index=int(maxl_index),
        maxl_log_likelihood=float(maxl_log_likelihood),
        chosen_distance_scale=float(chosen_scale),
        network_optimal_snr=float(network_optimal_snr),
        bias_improvement=bias_improvement,
        model_comparison=model_comparison,
        generated_files=generated_files,
        injection_parameters=injection_parameters,
        gaussian=gaussian_summary,
        student=student_summary,
    )
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def build_model_comparison(
    gaussian_summary: dict | None,
    student_summary: dict | None,
) -> dict | None:
    if gaussian_summary is None or student_summary is None:
        return None

    log_bf_student_over_gaussian = (
        student_summary["log_evidence"] - gaussian_summary["log_evidence"]
    )
    return dict(
        log_bayes_factor_student_over_gaussian=float(log_bf_student_over_gaussian),
        log10_bayes_factor_student_over_gaussian=float(
            log_bf_student_over_gaussian / np.log(10.0)
        ),
        delta_log_bayes_factor_to_noise=float(
            student_summary["log_bayes_factor"] - gaussian_summary["log_bayes_factor"]
        ),
    )


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir).resolve()
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(str(outdir))

    posterior_path = resolve_posterior_path()
    base_injection_parameters, maxl_log_likelihood, maxl_index = (
        load_maximum_likelihood_injection(posterior_path)
    )
    bilby.core.utils.logger.info(
        "Using GW231123 NRSur7dq4 maximum-likelihood sample %d with LVK log-likelihood %.6f from %s",
        maxl_index,
        maxl_log_likelihood,
        posterior_path,
    )

    waveform_generator = build_waveform_generator(args.waveform_approximant)
    psds = load_psds(posterior_path)

    selected_outputs = None
    for distance_scale in get_trial_distance_scales(args):
        bilby.core.utils.random.seed(args.sampling_seed)
        injection_parameters = build_injection_parameters(
            base_injection_parameters, distance_scale
        )
        interferometers = build_interferometers(
            psds, nu_injection=args.nu_injection
        )
        interferometers.inject_signal(
            parameters=injection_parameters,
            waveform_generator=waveform_generator,
        )
        network_optimal_snr = compute_network_optimal_snr(
            interferometers, waveform_generator, injection_parameters
        )
        bilby.core.utils.logger.info(
            "Trying distance scale %.3f, injected luminosity distance %.6f Mpc, network optimal SNR %.3f",
            distance_scale,
            injection_parameters["luminosity_distance"],
            network_optimal_snr,
        )

        if args.plot_data:
            plot_label = f"{args.label}_scale_{distance_scale:g}".replace(".", "p")
            interferometers.plot_data(outdir=str(outdir), label=plot_label)

        gaussian_result = gaussian_summary = None
        student_result = student_summary = None

        if not args.skip_gaussian:
            gaussian_result, gaussian_summary = run_inference(
                "gaussian",
                copy.deepcopy(interferometers),
                waveform_generator,
                injection_parameters,
                args,
                distance_scale=distance_scale,
            )
        if not args.skip_student:
            student_result, student_summary = run_inference(
                "student",
                copy.deepcopy(interferometers),
                waveform_generator,
                injection_parameters,
                args,
                distance_scale=distance_scale,
            )

        selected_outputs = dict(
            chosen_scale=distance_scale,
            injection_parameters=injection_parameters,
            network_optimal_snr=network_optimal_snr,
            gaussian_result=gaussian_result,
            gaussian_summary=gaussian_summary,
            student_result=student_result,
            student_summary=student_summary,
        )

        if (
            not args.fix_distance
            and is_bias_visible(
            gaussian_summary,
            student_summary,
            required_improvement=args.required_bias_improvement,
            )
        ):
            bilby.core.utils.logger.info(
                "Student-t outperformed Gaussian at distance scale %.3f",
                distance_scale,
            )
            break

        if args.distance_scale is None:
            bilby.core.utils.logger.warning(
                "Bias improvement was not yet visible at distance scale %.3f; increasing SNR.",
                distance_scale,
            )

    if selected_outputs is None:
        raise RuntimeError("No inference runs were executed")

    generated_files = {}
    if selected_outputs["gaussian_result"] is not None:
        generated_files["gaussian_waveform_plots"] = [
            str(path)
            for path in plot_waveform_reconstructions(
                selected_outputs["gaussian_result"],
                copy.deepcopy(interferometers),
                waveform_generator,
                selected_outputs["injection_parameters"],
                n_samples=args.waveform_plot_samples,
            )
        ]
    if selected_outputs["student_result"] is not None:
        generated_files["student_waveform_plots"] = [
            str(path)
            for path in plot_waveform_reconstructions(
                selected_outputs["student_result"],
                copy.deepcopy(interferometers),
                waveform_generator,
                selected_outputs["injection_parameters"],
                n_samples=args.waveform_plot_samples,
            )
        ]

    bayes_factor_plot = plot_bayes_factor_comparison(
        outdir,
        f"{args.label}_scale_{selected_outputs['chosen_scale']:g}".replace(".", "p"),
        selected_outputs["gaussian_summary"],
        selected_outputs["student_summary"],
    )
    if bayes_factor_plot is not None:
        generated_files["bayes_factor_plot"] = str(bayes_factor_plot)

    histogram_plot = plot_whitened_residual_histograms(
        outdir,
        f"{args.label}_scale_{selected_outputs['chosen_scale']:g}".replace(".", "p"),
        copy.deepcopy(interferometers),
        waveform_generator,
        gaussian_result=selected_outputs["gaussian_result"],
        student_result=selected_outputs["student_result"],
        nu_injection=args.nu_injection,
    )
    if histogram_plot is not None:
        generated_files["whitened_residual_histogram_plot"] = str(histogram_plot)

    model_comparison = build_model_comparison(
        selected_outputs["gaussian_summary"],
        selected_outputs["student_summary"],
    )

    summary_path = write_summary(
        outdir,
        injection_parameters=selected_outputs["injection_parameters"],
        maxl_index=maxl_index,
        maxl_log_likelihood=maxl_log_likelihood,
        network_optimal_snr=selected_outputs["network_optimal_snr"],
        chosen_scale=selected_outputs["chosen_scale"],
        bias_improvement=get_bias_improvement(
            selected_outputs["gaussian_summary"],
            selected_outputs["student_summary"],
        ),
        model_comparison=model_comparison,
        generated_files=generated_files,
        gaussian_summary=selected_outputs["gaussian_summary"],
        student_summary=selected_outputs["student_summary"],
    )
    bilby.core.utils.logger.info("Wrote summary to %s", summary_path)

    if (
        not args.skip_gaussian
        and not args.skip_student
        and not args.fix_distance
        and not is_bias_visible(
            selected_outputs["gaussian_summary"],
            selected_outputs["student_summary"],
            required_improvement=args.required_bias_improvement,
        )
    ):
        raise RuntimeError(
            "Student-t did not improve on the Gaussian distance bias by the required amount"
        )


if __name__ == "__main__":
    main()
