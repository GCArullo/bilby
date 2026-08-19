#!/usr/bin/env python
"""Reweight a Gaussian LVK posterior onto a parametric-noise likelihood.

Given a posterior produced with the ordinary Whittle likelihood, this answers
"would a heavy-tailed or rescaled noise model have changed the answer?" in
minutes rather than in a rerun, by importance reweighting:

    w_i = exp( logL_new(theta_i) - logL_gaussian(theta_i) ),

with the new likelihood's noise parameters marginalised over their priors.  Only
the residual changes between the two likelihoods, so the reweighting is exact up
to Monte Carlo error, and its effective sample size says whether the answer can
be trusted.

Reading the result
------------------

``ESS/N`` high, posterior unchanged
    The parameters are robust to that noise model.  Concluded; no rerun needed.
``ESS/N`` low
    The noise model genuinely moves the posterior and reweighting cannot resolve
    where to.  This, and only this, justifies spending cluster time on a rerun.

Reweighting can only lose information, never discover a new mode, so a low ESS
is a trigger to run, never evidence that anything has been healed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LOG_TWO_PI = np.log(2.0 * np.pi)

# Prior support matching build_hyperbolic_priors / build_nu_priors /
# build_log_psd_scale_priors in submit_runs_real_data.py.
DEFAULT_PRIORS = {
    "hyperbolic": dict(alpha=(1e-6, 30.0), delta=(1e-6, 30.0)),
    "student": dict(nu=(2.1, 1000.0)),
    "psd-scale": dict(log_psd_scale=(-1.0, 1.0)),
}


# --------------------------------------------------------------------- densities
def gaussian_log_density(q):
    return -0.5 * q - LOG_TWO_PI


def psd_scale_log_density(q, log_psd_scale):
    level = 10.0 ** log_psd_scale
    return -q / (2.0 * level) - LOG_TWO_PI - np.log(level)


def student_log_density(q, nu):
    return -LOG_TWO_PI - 0.5 * (nu + 2.0) * np.log1p(q / nu)


def _log_scaled_bessel_three_halves(value):
    return 0.5 * np.log(np.pi / (2.0 * value)) + np.log1p(1.0 / value)


def hyperbolic_log_density(q, alpha, delta):
    radial_shift = q / (np.sqrt(delta ** 2 + q) + delta)
    constant = (
        1.5 * np.log(alpha / delta)
        - 0.5 * LOG_TWO_PI
        - np.log(2.0 * alpha)
        - _log_scaled_bessel_three_halves(alpha * delta)
    )
    return -alpha * radial_shift + constant


def _log_mean_exp(values, axis=0):
    peak = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        peak + np.log(np.mean(np.exp(values - peak), axis=axis, keepdims=True)),
        axis=axis,
    )


# ------------------------------------------------------------------ marginalising
def marginal_delta(q_region, family, grid_size, priors):
    """log E_prior[exp(sum ell_new - sum ell_gaussian)] per sample.

    ``q_region`` is (n_samples, n_bins) restricted to the region the noise model
    is active in; everything outside stays Gaussian and cancels.
    """
    gaussian_sum = -0.5 * q_region.sum(axis=1)
    n_bins = q_region.shape[1]

    if family == "psd-scale":
        low, high = priors["log_psd_scale"]
        grid = np.linspace(low, high, grid_size)[:, None]
        total = q_region.sum(axis=1)[None, :]
        values = (
            -total / (2.0 * 10.0 ** grid) - n_bins * np.log(10.0 ** grid)
            + total / 2.0
        )
        return _log_mean_exp(values, axis=0)

    if family == "student":
        low, high = priors["nu"]
        grid = np.linspace(low, high, grid_size)
        values = np.empty((len(grid), q_region.shape[0]))
        for index, nu in enumerate(grid):
            values[index] = (
                -0.5 * (nu + 2.0) * np.log1p(q_region / nu)
            ).sum(axis=1) - gaussian_sum
        return _log_mean_exp(values, axis=0)

    if family == "hyperbolic":
        # ell_hyp = -alpha * S(delta) + n c(alpha, delta): the radial shift
        # depends only on delta, so a 2-D grid collapses to one reduction per
        # delta value instead of one per pair.
        alpha_grid = np.linspace(*priors["alpha"], grid_size)
        delta_grid = np.linspace(*priors["delta"], grid_size)
        blocks = []
        for delta in delta_grid:
            shift = (
                q_region / (np.sqrt(delta ** 2 + q_region) + delta)
            ).sum(axis=1)
            alpha = alpha_grid[:, None]
            constant = (
                1.5 * np.log(alpha / delta)
                - 0.5 * LOG_TWO_PI
                - np.log(2.0 * alpha)
                - _log_scaled_bessel_three_halves(alpha * delta)
            )
            blocks.append(
                -alpha * shift[None, :] + n_bins * constant - gaussian_sum[None, :]
            )
        return _log_mean_exp(np.concatenate(blocks, axis=0), axis=0)

    raise ValueError(f"Unknown family '{family}'")


# ------------------------------------------------------------------ detectability
def shape_statistic(q_region):
    """kappa = <q^2>/<q>^2, a scale-free measure of non-Gaussianity.

    For Gaussian noise ``q`` is exponential with mean 2, so

        kappa = E[q^2] / E[q]^2 = 8 / 4 = 2

    exactly, *whatever the PSD normalisation*: rescaling ``q -> c q`` leaves the
    ratio bit-identically unchanged.  That is the point.  ``Lambda`` responds to
    the overall level and so cannot tell a genuinely heavy-tailed region from one
    where the PSD is merely mis-estimated; ``kappa`` is blind to the level and
    responds only to the shape.

    Together they separate the regimes:

    ==============  ==============  ======================================
    Lambda          kappa           interpretation
    ==============  ==============  ======================================
    high            ~ 2             PSD mis-estimated; use a free scale
    high            > 2             genuine heavy tail
    low             > 2             tail carrying no excess power
    low             ~ 2             nothing an amplitude model can model
    ==============  ==============  ======================================

    The null scatter is ``sigma = 2 / sqrt(n)`` by the delta method, confirmed by
    simulation to better than 1% for n >= 100.  The estimator is biased low at
    small ``n`` (1.96 at n = 50, 1.86 at n = 13), which makes the reported z
    conservative for detecting non-Gaussianity in small tiles.
    """
    mean_q = q_region.mean(axis=1)
    mean_q_squared = (q_region ** 2).mean(axis=1)
    n_bins = q_region.shape[1]
    kappa = mean_q_squared / np.clip(mean_q ** 2, 1e-30, None)
    sigma = 2.0 / np.sqrt(n_bins)
    median_kappa = float(np.median(kappa))
    z = (median_kappa - 2.0) / sigma
    return {
        "kappa": median_kappa,
        "kappa_null": 2.0,
        "kappa_null_sigma": float(sigma),
        "kappa_z": float(z),
        "shape_verdict": (
            "GAUSSIAN SHAPE. Any excess here is a level error, not a tail; "
            "prefer a free PSD scale, which a heavy tail would only duplicate."
            if z < 3.0 else
            "HEAVY TAILED. The excess is not a level error, so a free scale "
            "cannot absorb it; prefer Student-t or hyperbolic."
        ),
    }


def detectability(q_region):
    """Can an amplitude model see anything in this region at all?

    From notes/robust_likelihood_regimes.tex, Eq. (2). An artefact depositing
    whitened power ``rho^2`` into a region of ``n`` bins whose expected power is
    ``2n`` gives a fractional excess ``eps = rho^2 / 2n``; profiling a free scale
    yields ``L* = 1 + eps`` and

        Lambda = n (L* - 1 - ln L*)  ~=  n eps^2 / 2  =  rho^4 / 8n   (eps << 1).

    Detectability grows as the *fourth* power of the artefact's SNR and falls
    *linearly* with the size of the region the parameter is shared over.  Loud
    artefacts are trivially found, quiet ones are invisible, and diluting a
    feature across a wide band destroys sensitivity to it.

    The exact profile form is reported, not the small-excess expansion: applied
    to a +77% excess the expansion overpredicts Lambda by roughly a factor of two.

    This is logically prior to the reweighting result.  If Lambda is below
    threshold the region holds nothing an amplitude model can engage with, and a
    null reweighting says only that -- not that the data are clean.
    """
    mean_q = q_region.mean(axis=1)
    n_bins = q_region.shape[1]
    level = np.clip(mean_q / 2.0, 1e-12, None)
    lam = n_bins * (level - 1.0 - np.log(level))
    excess = level - 1.0
    # rho^2 = 2 n eps, negative when the region sits below the nominal PSD
    rho_squared = 2.0 * n_bins * excess
    implied_snr = np.sign(rho_squared) * np.sqrt(np.abs(rho_squared))

    median_lambda = float(np.median(lam))
    if median_lambda < 1.0:
        verdict = (
            "BELOW THRESHOLD. Nothing here for an amplitude model to engage "
            "with; a null reweighting result says only that, not that the data "
            "are clean. A coherent (subtracting) model may still be warranted."
        )
    elif median_lambda < 5.0:
        verdict = (
            "MARGINAL. A weak excess is present. Expect the noise parameters to "
            "move a little and the posterior barely at all."
        )
    else:
        verdict = (
            "DETECTABLE. The region carries a clear excess, so the noise model "
            "will engage. Whether that *matters* is a separate question: an "
            "artefact disjoint from the signal biases nothing."
        )
    return {
        "n_bins": int(n_bins),
        "median_mean_q": float(np.median(mean_q)),
        **shape_statistic(q_region),
        "median_fractional_excess": float(np.median(excess)),
        "implied_artefact_snr": float(np.median(implied_snr)),
        "lambda_profile": median_lambda,
        "lambda_small_excess_approximation": float(
            np.median(n_bins * excess ** 2 / 2.0)
        ),
        "verdict": verdict,
    }


# ----------------------------------------------------------------------- reporting
def summarise(log_weights, samples, parameters):
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    ess = float(1.0 / np.sum(weights ** 2))
    fraction = ess / len(weights)

    report = {
        "n_samples": int(len(weights)),
        "effective_sample_size": ess,
        "ess_fraction": fraction,
        "max_log_weight_spread": float(log_weights.max() - log_weights.min()),
        "parameters": {},
    }
    for name in parameters:
        if name not in samples:
            continue
        values = np.asarray(samples[name], dtype=float)
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order])
        q05, q50, q95 = np.interp([0.05, 0.5, 0.95], cumulative, values[order])
        base = np.percentile(values, [5, 50, 95])
        spread = float(values.std())
        report["parameters"][name] = {
            "baseline": [float(v) for v in base],
            "reweighted": [float(q05), float(q50), float(q95)],
            "median_shift": float(q50 - base[1]),
            "median_shift_in_sigma": float((q50 - base[1]) / spread) if spread else 0.0,
        }

    if fraction >= 0.5:
        verdict = (
            "ROBUST. The noise model barely perturbs the likelihood, and the "
            "reweighted posterior is reliable. No rerun is justified."
        )
    elif fraction >= 0.1:
        verdict = (
            "USABLE BUT MARGINAL. The reweighted quantiles are indicative, not "
            "precise. Confirm any shift that matters with a direct rerun."
        )
    else:
        verdict = (
            "UNRELIABLE. The noise model moves the posterior far enough that "
            "reweighting cannot resolve it. This is the case that justifies a "
            "full rerun with the new likelihood."
        )
    report["recommendation"] = verdict
    return report, weights


# ------------------------------------------------------------------------- data
def load_analysis_data(path):
    """Analysis arrays, either from a npz export or a bilby_pipe data dump."""
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        detectors = [str(name) for name in data["detectors"]]
        return {
            name: dict(
                frequency_array=data[f"{name}_frequency_array"],
                strain=data[f"{name}_frequency_domain_strain"],
                psd=data[f"{name}_psd_array"],
                mask=data[f"{name}_frequency_mask"],
                duration=float(data[f"{name}_duration"]),
                sampling_frequency=float(data[f"{name}_sampling_frequency"]),
            )
            for name in detectors
        }
    import dill
    with open(path, "rb") as stream:
        dump = dill.load(stream)
    return {
        ifo.name: dict(
            frequency_array=ifo.frequency_array,
            strain=ifo.frequency_domain_strain,
            psd=ifo.power_spectral_density_array,
            mask=ifo.frequency_mask,
            duration=ifo.strain_data.duration,
            sampling_frequency=ifo.strain_data.sampling_frequency,
        )
        for ifo in dump.interferometers
    }


def load_posterior(path):
    """Posterior samples as a dict of arrays, from pesummary or plain HDF5/npz."""
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return {key: data[key] for key in data.files}
    import h5py
    with h5py.File(path, "r") as handle:
        candidates = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith("posterior_samples"):
                candidates.append(name)

        handle.visititems(visit)
        if not candidates:
            raise SystemExit(f"no posterior_samples dataset found in {path}")
        samples = handle[candidates[0]][()]
        return {name: np.asarray(samples[name], dtype=float)
                for name in samples.dtype.names}


def whitened_series(entry):
    mask = entry["mask"]
    scale2 = entry["psd"] * entry["duration"] / 4.0
    whitened = np.zeros(len(entry["strain"]), dtype=complex)
    whitened[mask] = entry["strain"][mask] / np.sqrt(scale2[mask])
    n = int(round(entry["duration"] * entry["sampling_frequency"]))
    series = np.fft.irfft(whitened, n=n)
    target = float(np.sum(np.abs(whitened) ** 2))
    current = float(np.sum(series ** 2))
    return series * np.sqrt(target / current) if current > 0 else series


def region_quadratic_forms(residual_series, entry, time_band, frequency_band):
    """q for the bins inside one time chunk and frequency band."""
    sampling = entry["sampling_frequency"]
    n = len(residual_series)
    lo = 0 if time_band is None else int(round(time_band[0] * sampling))
    hi = n if time_band is None else int(round(time_band[1] * sampling))
    chunk = residual_series[lo:hi]
    coefficients = np.fft.rfft(chunk)
    weights = np.full(len(coefficients), 2.0)
    weights[0] = 1.0
    if len(chunk) % 2 == 0:
        weights[-1] = 1.0
    q = weights * np.abs(coefficients) ** 2 / len(chunk)
    frequencies = np.fft.rfftfreq(len(chunk), 1.0 / sampling)
    selection = weights == 2.0
    if frequency_band is not None:
        selection &= (frequencies >= frequency_band[0]) & (
            frequencies <= frequency_band[1]
        )
    return q[selection]


def build_residual_series(arguments, data, samples, indices):
    """Whitened time-domain residual per detector per sample."""
    import bilby
    from importlib import import_module

    bilby.core.utils.setup_logger(log_level="ERROR")
    module_name, _, function_name = arguments.frequency_domain_source_model.rpartition(".")
    source_model = getattr(import_module(module_name), function_name)

    names = list(data)
    interferometers = bilby.gw.detector.InterferometerList(names)
    reference = data[names[0]]
    for ifo in interferometers:
        entry = data[ifo.name]
        ifo.strain_data.set_from_frequency_domain_strain(
            frequency_domain_strain=entry["strain"],
            frequency_array=entry["frequency_array"],
            start_time=arguments.start_time,
        )
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=entry["frequency_array"], psd_array=entry["psd"]
        )
        active = entry["frequency_array"][entry["mask"]]
        ifo.minimum_frequency = float(active[0])
        ifo.maximum_frequency = float(active[-1])

    generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=reference["duration"],
        sampling_frequency=reference["sampling_frequency"],
        start_time=arguments.start_time,
        frequency_domain_source_model=source_model,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            reference_frequency=arguments.reference_frequency,
            waveform_approximant=arguments.approximant,
            minimum_frequency=interferometers[0].minimum_frequency,
            maximum_frequency=interferometers[0].maximum_frequency,
            catch_waveform_errors=True,
        ),
    )
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=interferometers, waveform_generator=generator,
        reference_frame=arguments.reference_frame, time_reference="geocent",
        jitter_time=False,
    )

    series = {name: [] for name in names}
    for count, index in enumerate(indices):
        parameters = {key: float(samples[key][index]) for key in samples}
        parameters.update(likelihood.get_sky_frame_parameters(parameters))
        polarisations = generator.frequency_domain_strain(parameters)
        for ifo in interferometers:
            entry = data[ifo.name]
            response = ifo.get_detector_response(polarisations, parameters)
            series[ifo.name].append(
                whitened_series({**entry, "strain": entry["strain"] - response})
            )
        if count % 500 == 0:
            print(f"  residuals {count}/{len(indices)}", file=sys.stderr, flush=True)
    return {name: np.array(values) for name, values in series.items()}


def parse_band(text):
    if text is None:
        return None
    values = [float(v) for v in text.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("bands are given as 'low,high'")
    return values


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--posterior", required=True,
                        help="pesummary/bilby HDF5, or an npz of sample arrays")
    parser.add_argument("--data", required=True,
                        help="bilby_pipe data dump pickle, or an analysis npz")
    parser.add_argument("--family", required=True,
                        choices=["hyperbolic", "student", "psd-scale"])
    parser.add_argument("--frequency-band", type=parse_band, default=None,
                        help="'low,high' in Hz; omit for the whole band")
    parser.add_argument("--time-band", type=parse_band, default=None,
                        help="'start,end' in seconds from the segment start; "
                             "omit for the whole segment")
    parser.add_argument("--detectors", nargs="+", default=None,
                        help="restrict the noise model to these detectors")
    parser.add_argument("--parameters", nargs="+",
                        default=["chirp_mass", "mass_ratio", "chi_eff"],
                        help="posterior parameters to report")
    parser.add_argument("--approximant", default="IMRPhenomXPHM")
    parser.add_argument("--frequency-domain-source-model",
                        default="bilby.gw.source.lal_binary_black_hole")
    parser.add_argument("--reference-frequency", type=float, default=20.0)
    parser.add_argument("--reference-frame", default="sky")
    parser.add_argument("--start-time", type=float, required=True,
                        help="segment start time (GPS)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="thin the posterior (0 uses all samples)")
    parser.add_argument("--seed", type=int, default=20260819,
                        help="seed for random thinning when --max-samples is set")
    parser.add_argument("--grid-size", type=int, default=60,
                        help="quadrature points per noise parameter")
    parser.add_argument("--output", default="reweighted_posterior.json")
    arguments = parser.parse_args()

    samples = load_posterior(arguments.posterior)
    data = load_analysis_data(arguments.data)
    n_total = len(next(iter(samples.values())))
    if arguments.max_samples and arguments.max_samples < n_total:
        # Random, not strided: posterior files are not randomly ordered, so
        # strided thinning would sample the nested-sampling history rather than
        # the posterior.
        rng = np.random.default_rng(arguments.seed)
        indices = np.sort(
            rng.choice(n_total, size=arguments.max_samples, replace=False)
        )
    else:
        indices = np.arange(n_total)

    series = build_residual_series(arguments, data, samples, indices)
    active = arguments.detectors or list(data)

    log_weights = np.zeros(len(indices))
    n_bins_total = 0
    detectability_report = {}
    for name in active:
        q = np.array([
            region_quadratic_forms(row, data[name], arguments.time_band,
                                   arguments.frequency_band)
            for row in series[name]
        ])
        n_bins_total += q.shape[1]
        detectability_report[name] = detectability(q)
        log_weights += marginal_delta(
            q, arguments.family, arguments.grid_size, DEFAULT_PRIORS[arguments.family]
        )

    thinned = {key: np.asarray(value)[indices] for key, value in samples.items()}
    report, weights = summarise(log_weights, thinned, arguments.parameters)
    report["detectability"] = detectability_report
    report["configuration"] = dict(
        family=arguments.family, frequency_band=arguments.frequency_band,
        time_band=arguments.time_band, detectors=active,
        n_bins_per_detector=int(n_bins_total / max(1, len(active))),
    )

    output = Path(arguments.output)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    np.savez(output.with_suffix(".npz"), weights=weights,
             **{key: value for key, value in thinned.items()})

    print(json.dumps(
        {k: v for k, v in report.items()
         if k not in ("parameters", "detectability")}, indent=2))
    print("\ndetectability (can an amplitude model see anything here?)")
    for name, entry in report["detectability"].items():
        print(f"  {name}: {entry['n_bins']:5d} bins, excess "
              f"{100 * entry['median_fractional_excess']:+7.2f}%, implied artefact "
              f"SNR {entry['implied_artefact_snr']:6.2f}, Lambda "
              f"{entry['lambda_profile']:8.2f}")
        print(f"      kappa = {entry['kappa']:.3f} "
              f"(Gaussian 2, sigma {entry['kappa_null_sigma']:.3f}, "
              f"z = {entry['kappa_z']:+.1f})")
        print(f"      {entry['verdict']}")
        print(f"      {entry['shape_verdict']}")
    print()
    for name, entry in report["parameters"].items():
        base, new = entry["baseline"], entry["reweighted"]
        print(f"  {name:24s} {base[0]:8.3f} {base[1]:8.3f} {base[2]:8.3f}  ->  "
              f"{new[0]:8.3f} {new[1]:8.3f} {new[2]:8.3f}  "
              f"({entry['median_shift_in_sigma']:+.2f} sigma)")


if __name__ == "__main__":
    main()
