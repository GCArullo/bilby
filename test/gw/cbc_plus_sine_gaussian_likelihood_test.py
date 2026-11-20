import numpy as np
import pytest
from scipy.integrate import quad

import bilby


def _setup():
    bilby.core.utils.random.seed(1)
    interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
    interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=1024,
        duration=4,
        start_time=-2,
    )
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=4,
        sampling_frequency=1024,
        frequency_domain_source_model=bilby.gw.source.cbc_plus_sine_gaussians,
        parameter_conversion=bilby.gw.conversion.convert_to_cbc_plus_sine_gaussian_parameters,
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=50,
            minimum_frequency=20,
        ),
    )
    parameters = dict(
        mass_1=36.0, mass_2=29.0, a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
        phi_12=1.7, phi_jl=0.3, luminosity_distance=1000.0, theta_jn=0.4,
        psi=2.659, phase=1.3, geocent_time=0.0, ra=1.375, dec=-1.2108,
        sine_gaussian_0_hrss=3e-23, sine_gaussian_0_Q=8.0,
        sine_gaussian_0_frequency=120.0, sine_gaussian_0_time_offset=0.01,
        sine_gaussian_0_phase_offset=0.4,
    )
    interferometers.inject_signal(
        waveform_generator=waveform_generator, parameters=parameters
    )
    return interferometers, waveform_generator, parameters


@pytest.mark.parametrize("case", ["ordinary", "cbc_null", "negative_overlap"])
def test_cbc_plus_sine_gaussians_distance_marginalization(tmp_path, case):
    interferometers, waveform_generator, parameters = _setup()
    if case == "cbc_null":
        interferometers = bilby.gw.detector.InterferometerList(interferometers[:1])
        interferometer = interferometers[0]
        parameters.update(a_1=0.0, a_2=0.0, theta_jn=np.pi / 2)
        plus = interferometer.antenna_response(
            parameters["ra"], parameters["dec"], parameters["geocent_time"], 0, "plus"
        )
        cross = interferometer.antenna_response(
            parameters["ra"], parameters["dec"], parameters["geocent_time"], np.pi / 4, "plus"
        )
        # An edge-on, nonspinning CBC is invisible at this polarization angle;
        # the SG's circular polarization still has a nonzero detector response.
        parameters["psi"] = np.arctan2(-plus, cross) / 2
        interferometers.set_strain_data_from_zero_noise(
            sampling_frequency=1024, duration=4, start_time=-2
        )
        interferometers.inject_signal(
            waveform_generator=waveform_generator, parameters=parameters
        )
    distance_prior = bilby.gw.prior.UniformSourceFrame(
        minimum=200, maximum=3000, name="luminosity_distance"
    )
    if case == "negative_overlap":
        parameters.update(sine_gaussian_0_hrss=5e-21, sine_gaussian_0_phase_offset=0.0)
        # This also exercises accurate integration near a distance endpoint.
        distance_prior = bilby.core.prior.TruncatedGaussian(
            mu=1000, sigma=100, minimum=200, maximum=3000, name="luminosity_distance"
        )
    priors = bilby.gw.prior.BBHPriorDict()
    priors["luminosity_distance"] = distance_prior
    marginalized = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=interferometers,
        waveform_generator=waveform_generator,
        priors=priors,
        distance_marginalization=True,
        distance_marginalization_lookup_table=str(tmp_path / "lookup.npz"),
    )
    non_marginalized = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=interferometers,
        waveform_generator=waveform_generator,
    )

    distances = np.linspace(200, 3000, 1000)
    ln_likes = np.array([
        non_marginalized.log_likelihood_ratio(
            dict(parameters, luminosity_distance=distance)
        )
        for distance in distances
    ])
    log_post_peak = max(ln_likes + distance_prior.ln_prob(distances))

    def integrand(distance):
        return np.exp(
            non_marginalized.log_likelihood_ratio(
                dict(parameters, luminosity_distance=distance)
            ) + distance_prior.ln_prob(distance) - log_post_peak
        )

    expected = np.log(quad(integrand, 200, 3000, epsabs=1e-10, epsrel=1e-10)[0]) + log_post_peak

    ln_like = marginalized.log_likelihood_ratio(
        dict(parameters, luminosity_distance=marginalized._ref_dist)
    )
    assert ln_like == pytest.approx(expected, abs=0.005 if case == "ordinary" else 1e-6)


def test_distance_lookup_rejects_mismatched_prior(monkeypatch):
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient.__new__(
        bilby.gw.likelihood.GravitationalWaveTransient
    )
    likelihood._distance_array = np.linspace(200, 3000, 10)
    likelihood.distance_prior_array = np.ones(10) / 2800
    likelihood._ref_dist = 1000.0
    likelihood.phase_marginalization = False
    lookup = dict(
        distance_array=likelihood._distance_array,
        prior_array=likelihood.distance_prior_array * np.linspace(1, 2, 10),
        reference_distance=likelihood._ref_dist,
        phase_marginalization=False,
        lookup_table=np.zeros((4, 8)),
    )
    rebuilt = []

    def rebuild():
        rebuilt.append(True)
        likelihood._dist_margd_loglikelihood_array = np.ones((4, 8))

    monkeypatch.setattr(likelihood, "_create_lookup_table", rebuild)
    likelihood._setup_distance_marginalization(lookup)
    assert rebuilt == [True]


def test_cbc_plus_sine_gaussians_rejects_distance_marginalization():
    interferometers, waveform_generator, _ = _setup()
    with pytest.raises(
        ValueError,
        match="distance_marginalization=True is not supported",
    ):
        bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            priors=bilby.gw.prior.BBHPriorDict(),
            distance_marginalization=True,
            nu=8.0,
        )
