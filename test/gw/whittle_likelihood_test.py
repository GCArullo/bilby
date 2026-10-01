import unittest

import numpy as np
from scipy.integrate import quad

import bilby


def constant_frequency_domain_source(frequency_array, amplitude=0.0):
    plus = amplitude * np.ones_like(frequency_array, dtype=complex)
    cross = np.zeros_like(frequency_array, dtype=complex)
    return dict(plus=plus, cross=cross)


class TestWhittleGWTransient(unittest.TestCase):
    def setUp(self):
        bilby.core.utils.random.seed(500)
        self.duration = 2
        self.sampling_frequency = 128
        self.parameters = dict(
            amplitude=1e-23,
            geocent_time=0.0,
            ra=1.1,
            dec=-0.7,
            psi=0.4,
        )
        self.interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
        )
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=constant_frequency_domain_source,
            parameter_conversion=bilby.gw.conversion.identity_map_conversion,
        )

    def test_fixed_unit_scale_matches_gaussian_likelihood(self):
        gaussian_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
        )
        whittle_likelihood = bilby.gw.likelihood.WhittleGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            log_psd_scale=0.0,
            infer_log_psd_scale=False,
        )

        self.assertAlmostEqual(
            whittle_likelihood.noise_log_likelihood(),
            gaussian_likelihood.noise_log_likelihood(),
            7,
        )
        self.assertAlmostEqual(
            whittle_likelihood.log_likelihood(self.parameters),
            gaussian_likelihood.log_likelihood(self.parameters),
            7,
        )
        self.assertAlmostEqual(
            whittle_likelihood.log_likelihood_ratio(self.parameters),
            gaussian_likelihood.log_likelihood_ratio(self.parameters),
            7,
        )

    def test_sampled_log_psd_scale_matches_manual_whittle_likelihood(self):
        likelihood = bilby.gw.likelihood.WhittleGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            log_psd_scale=[0.0, 0.0],
            infer_log_psd_scale=True,
            num_psd_frequency_bands=2,
        )
        parameters = self.parameters.copy()
        parameters.update(
            log_psd_scale_1=np.log10(2.0),
            log_psd_scale_2=np.log10(0.5),
        )

        resolved_parameters = likelihood._resolve_likelihood_parameters(parameters)
        waveform_polarizations = likelihood.waveform_generator.frequency_domain_strain(
            resolved_parameters
        )

        manual = 0.0
        for interferometer in self.interferometers:
            mask = interferometer.frequency_mask
            frequencies = interferometer.frequency_array[mask]
            scale2 = (
                interferometer.power_spectral_density_array[mask]
                * likelihood.waveform_generator.duration
                / 4.0
            )
            h_f = interferometer.get_detector_response(
                waveform_polarizations,
                resolved_parameters,
            )
            residual = interferometer.frequency_domain_strain[mask] - h_f[mask]
            abs2 = residual.real ** 2 + residual.imag ** 2
            for log_psd_scale, band_mask in zip(
                [parameters["log_psd_scale_1"], parameters["log_psd_scale_2"]],
                likelihood._get_frequency_band_masks(frequencies),
            ):
                band_scale2 = scale2[band_mask] * 10.0 ** log_psd_scale
                manual += np.sum(
                    -np.log(2.0 * np.pi * band_scale2)
                    - abs2[band_mask] / (2.0 * band_scale2)
                )

        self.assertEqual(
            likelihood.noise_parameter_keys,
            ["log_psd_scale", "log_psd_scale_1", "log_psd_scale_2"],
        )
        self.assertAlmostEqual(likelihood.log_likelihood(parameters), manual, 7)

    def test_noise_log_evidence_integrates_sampled_psd_scale(self):
        likelihood = bilby.gw.likelihood.WhittleGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            infer_log_psd_scale=True,
        )
        likelihood._noise_log_likelihood_from_parameters = (
            lambda parameters: -parameters["log_psd_scale"] ** 2
        )
        priors = bilby.core.prior.PriorDict(
            dict(
                log_psd_scale=bilby.core.prior.Uniform(
                    0.0,
                    1.0,
                    name="log_psd_scale",
                )
            )
        )

        integral, _ = quad(lambda value: np.exp(-value ** 2), 0.0, 1.0)
        self.assertAlmostEqual(
            likelihood.noise_log_evidence(priors=priors),
            np.log(integral),
            7,
        )

    def test_noise_log_evidence_uses_fixed_psd_scale_prior(self):
        likelihood = bilby.gw.likelihood.WhittleGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            infer_log_psd_scale=True,
            num_psd_frequency_bands=2,
        )
        likelihood._noise_log_likelihood_from_parameters = (
            lambda parameters: np.sum(
                likelihood._get_log_psd_scale_values(parameters)
            )
        )
        priors = bilby.core.prior.PriorDict(
            dict(
                log_psd_scale=bilby.core.prior.DeltaFunction(
                    0.25,
                    name="log_psd_scale",
                )
            )
        )

        self.assertIn("log_psd_scale", likelihood.noise_parameter_keys)
        self.assertEqual(likelihood.noise_log_evidence(priors=priors), 0.5)


if __name__ == "__main__":
    unittest.main()
