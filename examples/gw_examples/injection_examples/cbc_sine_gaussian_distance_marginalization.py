#!/usr/bin/env python
"""Compare explicit and marginalized distance for a small CBC+SG injection.

Only luminosity distance and one coherent sine-Gaussian's amplitude and phase
are varied. Dynesty always calls the production Bilby likelihood. An independent
three-template quadratic likelihood supplies converged evidence/posterior
quadrature; it is checked against the production unmarginalized likelihood.

Run from the installed checkout, for example::

    python cbc_sine_gaussian_distance_marginalization.py --outdir validation

Completed seed/method files are reused. Use a new output directory after any
change to the code, injection, prior, or sampler settings.
"""

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import bilby
import dynesty
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.special import i0e, logsumexp
from scipy.stats import vonmises

DISTANCE = "luminosity_distance"
AMPLITUDE = "sine_gaussian_0_hrss"
PHASE = "sine_gaussian_0_phase_offset"
NAMES = [DISTANCE, AMPLITUDE, PHASE]
REFERENCE_DISTANCE = 1000.0
REFERENCE_AMPLITUDE = 3e-23


def setup(outdir):
    parameters = dict(
        mass_1=36.0,
        mass_2=29.0,
        a_1=0.4,
        a_2=0.3,
        tilt_1=0.5,
        tilt_2=1.0,
        phi_12=1.7,
        phi_jl=0.3,
        luminosity_distance=REFERENCE_DISTANCE,
        theta_jn=0.4,
        psi=2.659,
        phase=1.3,
        geocent_time=0.0,
        ra=1.375,
        dec=-1.2108,
        sine_gaussian_0_hrss=REFERENCE_AMPLITUDE,
        sine_gaussian_0_Q=4.0,
        sine_gaussian_0_frequency=100.0,
        sine_gaussian_0_time_offset=-0.015,
        sine_gaussian_0_phase_offset=1.0,
    )
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=512,
        duration=4,
        start_time=-2,
    )
    generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=4,
        sampling_frequency=512,
        start_time=-2,
        frequency_domain_source_model=bilby.gw.source.cbc_plus_sine_gaussians,
        parameter_conversion=bilby.gw.conversion.convert_to_cbc_plus_sine_gaussian_parameters,
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=50,
            minimum_frequency=20,
            maximum_frequency=256,
        ),
    )
    ifos.inject_signal(waveform_generator=generator, parameters=parameters)
    priors = bilby.core.prior.PriorDict(parameters.copy())
    priors[DISTANCE] = bilby.gw.prior.UniformSourceFrame(
        minimum=200,
        maximum=3000,
        name=DISTANCE,
    )
    priors[AMPLITUDE] = bilby.core.prior.LogUniform(1e-24, 2e-22, name=AMPLITUDE)
    priors[PHASE] = bilby.core.prior.Uniform(
        0, 2 * np.pi, name=PHASE, boundary="periodic"
    )
    explicit = bilby.gw.likelihood.GravitationalWaveTransient(ifos, generator)
    marginalized = bilby.gw.likelihood.GravitationalWaveTransient(
        ifos,
        generator,
        priors=priors.copy(),
        distance_marginalization=True,
        distance_marginalization_lookup_table=str(outdir / "distance_lookup.npz"),
    )
    return parameters, priors, explicit, marginalized


def reference_inner_products(parameters, likelihood):
    """Build CBC, phase-zero SG and phase-pi/2 SG detector templates."""
    generator = likelihood.waveform_generator
    templates = []
    for amplitude, phase in [
        (0.0, 0.0),
        (REFERENCE_AMPLITUDE, 0.0),
        (REFERENCE_AMPLITUDE, np.pi / 2),
    ]:
        point = dict(parameters, **{AMPLITUDE: amplitude, PHASE: phase})
        polarizations = generator.frequency_domain_strain(point)
        templates.append(
            [
                ifo.get_detector_response(polarizations, point)
                for ifo in likelihood.interferometers
            ]
        )
    templates = np.asarray(templates)
    templates[1:] -= templates[0]
    gram = np.zeros((3, 3))
    data = np.zeros(3)
    for index, ifo in enumerate(likelihood.interferometers):
        mask = ifo.frequency_mask
        weight = 4 / ifo.duration / ifo.power_spectral_density_array[mask]
        basis = templates[:, index, mask]
        gram += np.real(np.einsum("if,jf,f->ij", basis.conj(), basis, weight))
        data += np.real(
            np.einsum(
                "if,f,f->i", basis.conj(), ifo.frequency_domain_strain[mask], weight
            )
        )
    # Phase marginalization below is analytic because SG phase rotates the
    # complex detector strain: its two quadratures are orthogonal/equal-norm.
    np.testing.assert_allclose(gram[1, 1], gram[2, 2], rtol=1e-12)
    assert abs(gram[1, 2]) < 1e-10
    return gram, data


def reference_log_likelihood(point, gram, data):
    coefficients = np.array(
        [
            REFERENCE_DISTANCE / point[DISTANCE],
            point[AMPLITUDE] / REFERENCE_AMPLITUDE * np.cos(point[PHASE]),
            point[AMPLITUDE] / REFERENCE_AMPLITUDE * np.sin(point[PHASE]),
        ]
    )
    return coefficients @ data - coefficients @ gram @ coefficients / 2


def reference_posterior(priors, gram, data, size):
    distances = np.linspace(200, 3000, size)
    log_amplitudes = np.linspace(np.log(1e-24), np.log(2e-22), size)
    amplitudes = np.exp(log_amplitudes)
    scale = REFERENCE_DISTANCE / distances[:, None]
    amplitude = amplitudes[None, :] / REFERENCE_AMPLITUDE
    phase_strength = amplitude * np.hypot(
        data[1] - scale * gram[0, 1], data[2] - scale * gram[0, 2]
    )
    # Integrate uniform SG phase analytically, retaining the CBC--SG term.
    phase_integrated = (
        scale * data[0]
        - scale**2 * gram[0, 0] / 2
        - amplitude**2 * gram[1, 1] / 2
        + np.log(i0e(phase_strength))
        + phase_strength
    )
    maximum = phase_integrated.max()
    density = np.exp(phase_integrated - maximum) * priors[DISTANCE].prob(
        distances[:, None]
    )
    density /= log_amplitudes[-1] - log_amplitudes[0]
    distance_pdf = trapezoid(density, log_amplitudes, axis=1)
    amplitude_pdf = trapezoid(density, distances, axis=0)
    normalization = trapezoid(distance_pdf, distances)
    distance_cdf = (
        cumulative_trapezoid(distance_pdf, distances, initial=0) / normalization
    )
    amplitude_cdf = (
        cumulative_trapezoid(amplitude_pdf, log_amplitudes, initial=0) / normalization
    )
    joint_cdf = (
        cumulative_trapezoid(
            cumulative_trapezoid(density, distances, axis=0, initial=0),
            log_amplitudes,
            axis=1,
            initial=0,
        )
        / normalization
    )
    joint_thresholds = np.array(
        [
            np.interp([0.1, 0.25, 0.5, 0.75, 0.9], cdf, grid)
            for grid, cdf in [(distances, distance_cdf), (amplitudes, amplitude_cdf)]
        ]
    )
    threshold_distance, threshold_log_amplitude = np.meshgrid(
        joint_thresholds[0],
        np.log(joint_thresholds[1]),
        indexing="ij",
    )
    joint_probabilities = RegularGridInterpolator(
        (distances, log_amplitudes), joint_cdf
    )(np.stack([threshold_distance, threshold_log_amplitude], axis=-1))

    # The phase marginal uses independent Gauss--Legendre distance/log-hrss
    # quadrature. This also provides a second evidence integration.
    nodes, weights = np.polynomial.legendre.leggauss(128 if size < 800 else 256)
    quad_distance = 1600 + 1400 * nodes
    quad_log_amplitude = (log_amplitudes[-1] + log_amplitudes[0]) / 2 + np.ptp(
        log_amplitudes
    ) * nodes / 2
    scale = REFERENCE_DISTANCE / quad_distance[:, None]
    amplitude = np.exp(quad_log_amplitude)[None, :] / REFERENCE_AMPLITUDE
    quadrature_weights = (
        1400
        * weights[:, None]
        * priors[DISTANCE].prob(quad_distance[:, None])
        * weights[None, :]
        / 2
    )
    base = (
        scale * data[0] - scale**2 * gram[0, 0] / 2 - amplitude**2 * gram[1, 1] / 2
    )
    phases = np.linspace(0, 2 * np.pi, 257 if size < 800 else 513)
    phase_pdf = []
    first_moments = []
    second_moments = []
    coordinate_distance = np.broadcast_to(
        quad_distance[:, None] / REFERENCE_DISTANCE, base.shape
    )
    coordinate_amplitude = np.broadcast_to(amplitude, base.shape)
    for phase in phases:
        log_likelihood = base + amplitude * (
            (data[1] - scale * gram[0, 1]) * np.cos(phase)
            + (data[2] - scale * gram[0, 2]) * np.sin(phase)
        )
        phase_density = (
            np.exp(log_likelihood - maximum) * quadrature_weights / (2 * np.pi)
        )
        phase_pdf.append(np.sum(phase_density))
        coordinates = [
            coordinate_distance,
            coordinate_amplitude,
            np.full_like(base, phase),
        ]
        first_moments.append([np.sum(value * phase_density) for value in coordinates])
        second_moments.append(
            [
                [np.sum(left * right * phase_density) for left in coordinates]
                for right in coordinates
            ]
        )
    phase_pdf = np.asarray(phase_pdf)
    phase_normalization = trapezoid(phase_pdf, phases)
    phase_cdf = cumulative_trapezoid(phase_pdf, phases, initial=0) / phase_normalization
    phase_thresholds = np.interp([0.1, 0.25, 0.5, 0.75, 0.9], phase_cdf, phases)
    scale = REFERENCE_DISTANCE / distances[:, None]
    phase_location = np.arctan2(
        data[2] - scale * gram[0, 2], data[1] - scale * gram[0, 1]
    )
    phase_cdf_zero = vonmises.cdf(0, phase_strength, loc=phase_location)
    joint_distance_phase = []
    for phase in phase_thresholds:
        conditional_cdf = (
            vonmises.cdf(phase, phase_strength, loc=phase_location) - phase_cdf_zero
        )
        distance_subdensity = trapezoid(
            density * conditional_cdf, log_amplitudes, axis=1
        )
        distance_subcdf = (
            cumulative_trapezoid(distance_subdensity, distances, initial=0)
            / normalization
        )
        joint_distance_phase.append(
            np.interp(joint_thresholds[0], distances, distance_subcdf)
        )
    mean = trapezoid(np.asarray(first_moments), phases, axis=0) / phase_normalization
    covariance = trapezoid(
        np.asarray(second_moments), phases, axis=0
    ) / phase_normalization - np.outer(mean, mean)
    correlation = covariance / np.sqrt(
        np.outer(np.diag(covariance), np.diag(covariance))
    )
    return dict(
        log_bayes_factor=float(maximum + np.log(normalization)),
        phase_quadrature_log_bayes_factor=float(maximum + np.log(phase_normalization)),
        grids=[distances, amplitudes, phases],
        cdfs=[distance_cdf, amplitude_cdf, phase_cdf],
        joint_distance_amplitude_thresholds=joint_thresholds.tolist(),
        joint_distance_amplitude_cdf=joint_probabilities.tolist(),
        joint_distance_phase_thresholds=[
            joint_thresholds[0].tolist(),
            phase_thresholds.tolist(),
        ],
        joint_distance_phase_cdf=np.asarray(joint_distance_phase).T.tolist(),
        correlation=correlation.tolist(),
        quantiles=[
            np.interp([0.05, 0.5, 0.95], cdf, grid).tolist()
            for grid, cdf in zip(
                [distances, amplitudes, phases],
                [distance_cdf, amplitude_cdf, phase_cdf],
            )
        ],
    )


def run_one(method, seed, nlive, outdir, parameters, priors, likelihood):
    filename = outdir / f"{method}_{seed}.npz"
    source = Path(bilby.gw.likelihood.base.__file__).read_text()
    signature = json.dumps(
        dict(
            method=method,
            seed=seed,
            nlive=nlive,
            injection=parameters,
            priors={name: repr(priors[name]) for name in NAMES},
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            source_ast_sha256=hashlib.sha256(
                ast.dump(ast.parse(source)).encode()
            ).hexdigest(),
            dynesty_version=dynesty.__version__,
            sample="rwalk",
            walks=20,
            dlogz=0.05,
        ),
        sort_keys=True,
    )
    if filename.exists():
        previous = dict(np.load(filename))
        previous_signature = json.loads(str(previous.get("signature", "{}")))
        current_signature = json.loads(signature)
        # Keep exact launch provenance, while allowing whitespace-only changes.
        previous_signature.pop("source_sha256", None)
        current_signature.pop("source_sha256")
        if previous_signature != current_signature:
            raise ValueError(
                f"{filename} has different or unknown provenance; choose a new output directory."
            )
        return previous
    names = NAMES if method == "explicit" else NAMES[1:]

    def log_likelihood(values):
        point = dict(parameters)
        point.update(zip(names, values))
        if method == "marginalized":
            point[DISTANCE] = likelihood._ref_dist
        return likelihood.log_likelihood_ratio(point)

    def prior_transform(unit_cube):
        return np.array(
            [priors[name].rescale(value) for name, value in zip(names, unit_cube)]
        )

    start = time.monotonic()
    sampler = dynesty.NestedSampler(
        log_likelihood,
        prior_transform,
        len(names),
        nlive=nlive,
        sample="rwalk",
        walks=20,
        periodic=[len(names) - 1],
        rstate=np.random.default_rng(seed),
    )
    sampler.run_nested(dlogz=0.05, print_progress=False)
    result = sampler.results
    weights = np.exp(result.logwt - logsumexp(result.logwt))
    # Equal-weight draws allow the production conditional distance reconstruction
    # to be tested along with the actual marginalized sampling calculation.
    rng = np.random.default_rng(seed + 10000)
    selected = rng.choice(len(weights), size=10000, p=weights)
    posterior = np.zeros((len(selected), 3))
    posterior[:, 3 - len(names) :] = result.samples[selected]
    bilby.core.utils.random.seed(seed + 20000)
    if method == "marginalized":
        for row in posterior:
            point = dict(parameters, **dict(zip(NAMES[1:], row[1:])))
            point[DISTANCE] = likelihood._ref_dist
            row[0] = likelihood.generate_posterior_sample_from_marginalized_likelihood(
                point
            )[DISTANCE]
    output = dict(
        signature=signature,
        posterior=posterior,
        samples=result.samples,
        weights=weights,
        log_bayes_factor=result.logz[-1],
        log_bayes_factor_error=result.logzerr[-1],
        ncall=np.sum(result.ncall),
        elapsed=time.monotonic() - start,
        effective_samples=1 / np.sum(weights**2),
    )
    np.savez(filename, **output)
    print(
        json.dumps(
            {
                "method": method,
                "seed": seed,
                **{
                    key: float(value)
                    for key, value in output.items()
                    if np.ndim(value) == 0 and key != "signature"
                },
            }
        ),
        flush=True,
    )
    return output


def posterior_diagnostics(posterior, reference):
    cdf_distances = []
    for index, (grid, cdf) in enumerate(zip(reference["grids"], reference["cdfs"])):
        ordered = np.sort(posterior[:, index])
        expected = np.interp(ordered, grid, cdf)
        measured = np.arange(1, len(ordered) + 1) / len(ordered)
        cdf_distances.append(
            float(
                max(
                    np.max(abs(measured - expected)),
                    np.max(abs(measured - 1 / len(ordered) - expected)),
                )
            )
        )
    thresholds = reference["joint_distance_amplitude_thresholds"]
    joint = np.array(
        [
            [
                np.mean((posterior[:, 0] <= distance) & (posterior[:, 1] <= amplitude))
                for amplitude in thresholds[1]
            ]
            for distance in thresholds[0]
        ]
    )
    correlation = np.corrcoef(posterior.T)
    phase_thresholds = reference["joint_distance_phase_thresholds"]
    joint_phase = np.array(
        [
            [
                np.mean((posterior[:, 0] <= distance) & (posterior[:, 2] <= phase))
                for phase in phase_thresholds[1]
            ]
            for distance in phase_thresholds[0]
        ]
    )
    return dict(
        quantiles=np.quantile(posterior, [0.05, 0.5, 0.95], axis=0).T.tolist(),
        cdf_distances=cdf_distances,
        correlation=correlation.tolist(),
        maximum_correlation_residual=float(
            np.max(abs(correlation - np.asarray(reference["correlation"])))
        ),
        joint_distance_amplitude_cdf=joint.tolist(),
        maximum_joint_cdf_residual=float(
            np.max(abs(joint - np.asarray(reference["joint_distance_amplitude_cdf"])))
        ),
        joint_distance_phase_cdf=joint_phase.tolist(),
        maximum_joint_distance_phase_cdf_residual=float(
            np.max(abs(joint_phase - np.asarray(reference["joint_distance_phase_cdf"])))
        ),
    )


def check_distance_quadrature(posterior, parameters, priors, likelihood, gram, data):
    distances = np.linspace(200, 3000, 20001)
    scale = REFERENCE_DISTANCE / distances
    weights = priors[DISTANCE].prob(distances) * (distances[1] - distances[0])
    weights[[0, -1]] *= 0.5
    selected = posterior[
        np.random.default_rng(7).choice(len(posterior), 64, replace=False)
    ]
    points = [(row[1], row[2]) for row in selected]
    points += [
        (REFERENCE_AMPLITUDE * amplitude, phase)
        for amplitude, phase in [
            (0, 0),
            (0.5, 1),
            (1, 1),
            (1.5, 1),
            (1, 1 + np.pi),
            (3, 1 + np.pi),
        ]
    ]
    rows = []
    for amplitude, phase in points:
        coefficients = np.array(
            [
                scale,
                np.full_like(scale, amplitude / REFERENCE_AMPLITUDE * np.cos(phase)),
                np.full_like(scale, amplitude / REFERENCE_AMPLITUDE * np.sin(phase)),
            ]
        ).T
        raw_likelihood = np.einsum("ni,i->n", coefficients, data) - 0.5 * np.einsum(
            "ni,ij,nj->n", coefficients, gram, coefficients
        )
        quadrature = logsumexp(raw_likelihood, b=weights)
        point = dict(
            parameters,
            **{DISTANCE: likelihood._ref_dist, AMPLITUDE: amplitude, PHASE: phase},
        )
        production = likelihood.log_likelihood_ratio(point)
        rows.append(
            dict(
                hrss=amplitude,
                phase=phase,
                quadrature=float(quadrature),
                production=float(production),
                residual=float(production - quadrature),
            )
        )
    maximum = max(abs(row["residual"]) for row in rows)
    assert maximum < 1e-3
    return dict(
        posterior_points=64,
        maximum_absolute_residual=maximum,
        tolerance=1e-3,
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--nlive", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[121, 432, 789])
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    parameters, priors, explicit, marginalized = setup(args.outdir)
    source_hash = hashlib.sha256(
        Path(bilby.gw.likelihood.base.__file__).read_bytes()
    ).hexdigest()
    summary_path = args.outdir / "summary.json"
    gram, data = reference_inner_products(parameters, explicit)
    rng = np.random.default_rng(67)
    residuals = []
    for _ in range(24):
        point = dict(
            parameters, **{name: priors[name].rescale(rng.uniform()) for name in NAMES}
        )
        residuals.append(
            explicit.log_likelihood_ratio(point)
            - reference_log_likelihood(point, gram, data)
        )
    assert max(abs(np.asarray(residuals))) < 1e-8
    coarse = reference_posterior(priors, gram, data, 401)
    reference = reference_posterior(priors, gram, data, 801)
    assert abs(coarse["log_bayes_factor"] - reference["log_bayes_factor"]) < 1e-5
    assert (
        abs(
            reference["phase_quadrature_log_bayes_factor"]
            - reference["log_bayes_factor"]
        )
        < 1e-5
    )
    np.savez(
        args.outdir / "reference.npz",
        gram=gram,
        data=data,
        **{f"grid_{name}": grid for name, grid in zip(NAMES, reference["grids"])},
        **{f"cdf_{name}": cdf for name, cdf in zip(NAMES, reference["cdfs"])},
    )
    summary = dict(
        injection=parameters,
        priors={name: repr(priors[name]) for name in NAMES},
        bilby_version=bilby.__version__,
        dynesty_version=dynesty.__version__,
        validation_source_sha256=source_hash,
        nlive=args.nlive,
        seeds=args.seeds,
        reference={
            key: value
            for key, value in reference.items()
            if key not in ["grids", "cdfs"]
        },
        reference_convergence=reference["log_bayes_factor"]
        - coarse["log_bayes_factor"],
        maximum_raw_likelihood_residual=float(np.max(np.abs(residuals))),
        gram=gram.tolist(),
        data_inner_products=data.tolist(),
        runs=[],
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not args.reference_only:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        colors = {"explicit": "C0", "marginalized": "C1"}
        summary["pooled"] = {}
        for method, likelihood in [
            ("explicit", explicit),
            ("marginalized", marginalized),
        ]:
            method_posteriors = []
            for seed in args.seeds:
                result = run_one(
                    method,
                    seed,
                    args.nlive,
                    args.outdir,
                    parameters,
                    priors,
                    likelihood,
                )
                posterior = result["posterior"]
                method_posteriors.append(posterior)
                entry = dict(
                    method=method,
                    seed=seed,
                    source_sha256=json.loads(str(result["signature"]))["source_sha256"],
                    **{
                        key: float(value)
                        for key, value in result.items()
                        if np.ndim(value) == 0 and key != "signature"
                    },
                    **posterior_diagnostics(posterior, reference),
                )
                entry["reference_evidence_zscore"] = (
                    entry["log_bayes_factor"] - reference["log_bayes_factor"]
                ) / entry["log_bayes_factor_error"]
                summary["runs"].append(entry)
            pooled = np.concatenate(method_posteriors)
            if method == "explicit":
                quadrature_checks = check_distance_quadrature(
                    pooled, parameters, priors, marginalized, gram, data
                )
                (args.outdir / "likelihood_quadrature.json").write_text(
                    json.dumps(quadrature_checks, indent=2) + "\n"
                )
                summary["maximum_distance_quadrature_residual"] = quadrature_checks[
                    "maximum_absolute_residual"
                ]
            summary["pooled"][method] = posterior_diagnostics(pooled, reference)
            method_results = [
                entry for entry in summary["runs"] if entry["method"] == method
            ]
            evidences = [entry["log_bayes_factor"] for entry in method_results]
            errors = [entry["log_bayes_factor_error"] for entry in method_results]
            summary["pooled"][method].update(
                mean_log_bayes_factor=float(np.mean(evidences)),
                mean_log_bayes_factor_error=float(np.linalg.norm(errors) / len(errors)),
                seed_log_bayes_factor_scatter=float(np.std(evidences, ddof=1))
                if len(evidences) > 1
                else None,
            )
            for index, scale in enumerate([1, 1e-23, 1]):
                ordered = np.sort(pooled[:, index])
                axes[index].plot(
                    ordered / scale,
                    np.arange(1, len(ordered) + 1) / len(ordered),
                    color=colors[method],
                    label=method,
                )
        labels = [r"$d_L$ [Mpc]", r"$h_{\rm rss}$ [$10^{-23}$]", r"SG phase [rad]"]
        for index, (name, grid, cdf, scale, label) in enumerate(
            zip(NAMES, reference["grids"], reference["cdfs"], [1, 1e-23, 1], labels)
        ):
            axes[index].plot(grid / scale, cdf, "k--", label="quadrature")
            axes[index].set_xlabel(label)
            axes[index].axvline(parameters[name] / scale, color="0.5", linewidth=0.8)
            axes[index].set_xlim(np.interp([0.001, 0.999], cdf, grid) / scale)
            axes[index].set_ylim(0, 1)
        axes[0].set_ylabel("Posterior CDF")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(args.outdir / "posterior_cdfs.png", dpi=180)
        fig.savefig(args.outdir / "posterior_cdfs.pdf")
        plt.close(fig)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
