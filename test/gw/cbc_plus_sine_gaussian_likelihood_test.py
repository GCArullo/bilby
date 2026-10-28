import numpy as np
import pytest
from scipy.integrate import trapezoid

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


def test_cbc_plus_sine_gaussians_distance_marginalization(tmp_path):
    interferometers, waveform_generator, parameters = _setup()
    distance_prior = bilby.gw.prior.UniformSourceFrame(
        minimum=200, maximum=3000, name="luminosity_distance"
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
    expected = np.log(trapezoid(
        np.exp(ln_likes - max(ln_likes)) * distance_prior.prob(distances),
        distances,
    )) + max(ln_likes)

    ln_like = marginalized.log_likelihood_ratio(
        dict(parameters, luminosity_distance=marginalized._ref_dist)
    )
    assert ln_like == pytest.approx(expected, abs=0.05)


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
