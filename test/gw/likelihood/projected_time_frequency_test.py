import unittest

import numpy as np

import bilby


def constant_frequency_domain_source(frequency_array, amplitude, **kwargs):
    return dict(
        plus=amplitude * np.ones_like(frequency_array, dtype=complex),
        cross=np.zeros_like(frequency_array, dtype=complex),
    )


class TestProjectedTimeFrequency(unittest.TestCase):
    def setUp(self):
        bilby.core.utils.random.seed(500)
        self.parameters = dict(
            amplitude=1e-23, geocent_time=0.0, ra=1.1, dec=-0.7, psi=0.4
        )
        self.interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=256, duration=4
        )
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=4,
            sampling_frequency=256,
            frequency_domain_source_model=constant_frequency_domain_source,
            parameter_conversion=bilby.gw.conversion.identity_map_conversion,
        )
        self.reference = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
        )
        self.target = dict(
            target_time_intervals={"L1": [1.5, 2.0]},
            target_frequency_intervals={"L1": [28.0, 60.0]},
            minimum_concentration=0.5,
        )

    def _likelihood(self, **kwargs):
        target = {**self.target, **kwargs}
        return bilby.gw.likelihood.ProjectedTimeFrequencyGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            **target,
        )

    def test_gaussian_model_reproduces_absolute_and_ratio_likelihoods(self):
        likelihood = self._likelihood(noise_model="gaussian")
        self.assertAlmostEqual(
            likelihood.log_likelihood_ratio(self.parameters),
            self.reference.log_likelihood_ratio(self.parameters),
            places=10,
        )
        self.assertAlmostEqual(
            likelihood.log_likelihood(self.parameters),
            self.reference.log_likelihood(self.parameters),
            places=10,
        )
        self.assertAlmostEqual(
            likelihood.noise_log_likelihood(),
            self.reference.noise_log_likelihood(),
            places=10,
        )

    def test_projector_is_orthonormal_and_uses_only_the_target_detector(self):
        likelihood = self._likelihood(noise_model="gaussian-parametric")
        self.assertEqual(set(likelihood._projectors), {"L1"})
        projector = likelihood._projectors["L1"]["projector"]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gram = projector @ projector.T
        np.testing.assert_allclose(gram, np.eye(len(projector)), atol=1e-12)
        self.assertGreater(len(projector), 0)

    def test_noise_parameters_enter_signal_and_noise_with_the_same_value(self):
        likelihood = self._likelihood(
            noise_model="gaussian-parametric",
            infer_log_projected_variance=True,
        )
        key = likelihood.log_projected_variance_parameter_keys[0]
        parameters = {**self.parameters, key: 0.3}
        expected = likelihood.log_likelihood(parameters) - (
            likelihood._noise_log_likelihood_from_parameters(parameters)
        )
        self.assertAlmostEqual(
            likelihood.log_likelihood_ratio(parameters), expected, places=10
        )
        self.assertIn(key, likelihood.noise_parameter_keys)

        fixed_prior = bilby.core.prior.PriorDict(
            {key: bilby.core.prior.DeltaFunction(peak=0.3, name=key)}
        )
        self.assertAlmostEqual(
            likelihood.noise_log_evidence(fixed_prior),
            likelihood._noise_log_likelihood_from_parameters({key: 0.3}),
            places=10,
        )

        for value in [-np.inf, -400.0, np.inf, 400.0, np.nan]:
            self.assertEqual(
                likelihood._noise_log_likelihood_from_parameters({key: value}),
                -np.inf,
            )
        unbounded_prior = bilby.core.prior.PriorDict(
            {key: bilby.core.prior.Gaussian(mu=0.0, sigma=1.0, name=key)}
        )
        self.assertTrue(np.isfinite(likelihood.noise_log_evidence(unbounded_prior)))

    def test_hyperbolic_gaussian_limit(self):
        likelihood = self._likelihood(
            noise_model="hyperbolic", alpha=1e4, delta=1e4
        )
        self.assertAlmostEqual(
            likelihood.log_likelihood_ratio(self.parameters),
            self.reference.log_likelihood_ratio(self.parameters),
            places=3,
        )

    def test_parameter_keys_are_derived_from_the_target(self):
        likelihood = self._likelihood(
            noise_model="hyperbolic", infer_alpha=True, infer_delta=True
        )
        self.assertEqual(
            likelihood.alpha_parameter_keys,
            ["alpha_L1_1p5_2_28_60"],
        )
        self.assertEqual(
            likelihood.delta_parameter_keys,
            ["delta_L1_1p5_2_28_60"],
        )
        self.assertEqual(len(likelihood.noise_parameter_keys), 2)

    def test_invalid_or_unresolved_targets_raise(self):
        with self.assertRaises(ValueError):
            bilby.gw.likelihood.ProjectedTimeFrequencyGravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                target_time_intervals={"L1": [1.5, 2.0]},
                target_frequency_intervals={"H1": [28.0, 60.0]},
            )
        with self.assertRaises(ValueError):
            self._likelihood(minimum_concentration=1.1, noise_model="gaussian")
        with self.assertRaises(ValueError):
            self._likelihood(
                target_time_intervals={"H1": [1.5, 2.0], "L1": [1.5, 2.0]},
                target_frequency_intervals={
                    "H1": [28.0, 60.0],
                    "L1": [28.0, 60.0],
                },
            )


if __name__ == "__main__":
    unittest.main()
