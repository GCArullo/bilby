import numpy as np
import pandas as pd
import pytest

import bilby


@pytest.mark.parametrize("incoherent", [False, True])
def test_sg_only_matches_cbc_plus_sg_with_zero_cbc(monkeypatch, incoherent):
    frequencies = np.arange(0, 512, 0.25)
    component = dict(hrss=2e-22, Q=9, frequency=80, time_offset=0.07, phase_offset=0.4)
    sg_parameters = (
        dict(incoherent_sine_gaussian_parameters={"H1": [component, component]})
        if incoherent else dict(sine_gaussian_parameters=[component, component])
    )
    cbc_parameters = dict(
        mass_1=100, mass_2=80, luminosity_distance=1000, a_1=0, a_2=0,
        tilt_1=0, tilt_2=0, phi_12=0, phi_jl=0, theta_jn=0, phase=0,
    )
    monkeypatch.setattr(
        bilby.gw.source, "_base_lal_cbc_fd_waveform",
        lambda **kwargs: dict(plus=np.zeros(len(frequencies), complex),
                              cross=np.zeros(len(frequencies), complex)),
    )
    expected = bilby.gw.source.cbc_plus_sine_gaussians(
        frequencies, **cbc_parameters, **sg_parameters,
    )
    monkeypatch.setattr(
        bilby.gw.source, "_base_lal_cbc_fd_waveform",
        lambda **kwargs: pytest.fail("SG-only must not evaluate a CBC"),
    )
    actual = bilby.gw.source.sine_gaussians(frequencies, **sg_parameters)
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("incoherent", [False, True])
def test_sg_only_likelihood_and_posterior_generation(incoherent):
    interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
    interferometers.set_strain_data_from_zero_noise(
        sampling_frequency=1024, duration=4, start_time=1384782886,
    )
    generator = bilby.gw.WaveformGenerator(
        duration=4, sampling_frequency=1024, start_time=1384782886,
        frequency_domain_source_model=bilby.gw.source.sine_gaussians,
    )
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers, generator, reference_frame="L1H1", time_reference="L1",
    )
    prefix = "sine_gaussian_0_H1_" if incoherent else "sine_gaussian_0_"
    sample = dict(zenith=1.2, azimuth=0.9, psi=0.4, L1_time=1384782888.6)
    sample.update({prefix + key: value for key, value in dict(
        hrss=1e-22, Q=9, frequency=80, time_offset=0.07, phase_offset=0.4,
    ).items()})
    assert np.isfinite(likelihood.log_likelihood_ratio(sample))
    posterior = bilby.gw.conversion.identity_map_generation(pd.DataFrame([sample]), likelihood)
    assert {"ra", "dec", "geocent_time", "H1_optimal_snr"} <= set(posterior)
    assert not {"mass_1", "phi_jl", "luminosity_distance", "redshift"} & set(posterior)
    assert prefix + "time_offset" in posterior
    assert posterior.H1_cbc_optimal_snr.iloc[0] == 0
    assert posterior.H1_sine_gaussian_optimal_snr.iloc[0] == posterior.H1_optimal_snr.iloc[0]
    if incoherent:
        assert posterior.L1_optimal_snr.iloc[0] == 0
        assert posterior.L1_matched_filter_snr.iloc[0] == 0
