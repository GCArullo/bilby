import pytest

import bilby


@pytest.mark.parametrize(
    ("likelihood_class", "extra_kwargs"),
    [
        (bilby.gw.likelihood.GravitationalWaveTransient, {}),
        (bilby.gw.likelihood.StudentTGravitationalWaveTransient, {"nu": 8.0}),
    ],
)
def test_cbc_plus_sine_gaussians_rejects_distance_marginalization(
    likelihood_class,
    extra_kwargs,
):
    interferometers = bilby.gw.detector.InterferometerList(["H1"])
    interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=1024,
        duration=4,
    )
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=4,
        sampling_frequency=1024,
        frequency_domain_source_model=bilby.gw.source.cbc_plus_sine_gaussians,
    )

    with pytest.raises(
        ValueError,
        match="distance_marginalization=True is not supported",
    ):
        likelihood_class(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            distance_marginalization=True,
            **extra_kwargs,
        )
