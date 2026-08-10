import inspect
import unittest
from unittest.mock import MagicMock, patch

import bilby
import numpy as np
from scipy.special import kve


class TestHyperbolicGWTransient(unittest.TestCase):
    def setUp(self):
        bilby.core.utils.random.seed(500)
        self.duration = 4
        self.sampling_frequency = 2048
        self.parameters = dict(
            mass_1=31.0,
            mass_2=29.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=1.7,
            phi_jl=0.3,
            luminosity_distance=4000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3,
            geocent_time=1126259642.413,
            ra=1.375,
            dec=-1.2108,
        )
        self.interferometers = bilby.gw.detector.InterferometerList(["H1"])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        )

    def test_distance_marginalization_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "HyperbolicGravitationalWaveTransient does not support "
            "distance marginalization",
        ):
            bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                distance_marginalization=True,
            )

    @classmethod
    def _manual_log_density(cls, quadratic_forms, dimensions, alpha, delta):
        quadratic_forms = np.asarray(quadratic_forms, dtype=float)
        dimensions = np.asarray(dimensions, dtype=int)
        log_density = np.empty_like(quadratic_forms, dtype=float)

        for dimension in np.unique(dimensions):
            order = 0.5 * (dimension + 1.0)
            mask = dimensions == dimension
            log_density[mask] = (
                order * np.log(alpha / delta)
                + 0.5 * (1.0 - dimension) * np.log(2.0 * np.pi)
                - np.log(2.0 * alpha)
                - np.log(kve(order, alpha * delta))
                - alpha * (np.sqrt(delta ** 2 + quadratic_forms[mask]) - delta)
            )

        return log_density

    def test_log_likelihood_matches_direct_calculation(self):
        alpha = 10.0
        delta = 2.0
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=alpha,
            delta=delta,
        )
        parameters = self.parameters.copy()

        calculated = likelihood.log_likelihood(parameters)

        pols = self.waveform_generator.frequency_domain_strain(self.parameters)
        manual = 0.0
        for ifo in self.interferometers:
            h_f = ifo.get_detector_response(pols, self.parameters)
            mask = ifo.frequency_mask
            r = ifo.frequency_domain_strain[mask] - h_f[mask]
            scale2 = ifo.power_spectral_density_array[mask] * self.duration / 4.0
            quadratic_forms = (r.real ** 2 + r.imag ** 2) / scale2
            manual += np.sum(
                self._manual_log_density(
                    quadratic_forms=quadratic_forms,
                    dimensions=np.full(len(quadratic_forms), 2, dtype=int),
                    alpha=alpha,
                    delta=delta,
                )
                - np.log(scale2)
            )

        self.assertAlmostEqual(calculated, float(manual), 7)

    def test_multiband_log_likelihood_matches_direct_calculation(self):
        band_alphas = [4.0, 10.0, 25.0]
        band_deltas = [0.7, 1.3, 2.0]
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=band_alphas,
            delta=band_deltas,
            num_frequency_bands=len(band_alphas),
        )
        parameters = self.parameters.copy()

        calculated = likelihood.log_likelihood(parameters)

        pols = self.waveform_generator.frequency_domain_strain(self.parameters)
        manual = 0.0
        band_edges = likelihood._frequency_band_edges
        for ifo in self.interferometers:
            h_f = ifo.get_detector_response(pols, self.parameters)
            mask = ifo.frequency_mask
            frequencies = ifo.frequency_array[mask]
            r = ifo.frequency_domain_strain[mask] - h_f[mask]
            scale2 = ifo.power_spectral_density_array[mask] * self.duration / 4.0
            quadratic_forms = (r.real ** 2 + r.imag ** 2) / scale2

            for band_index, (alpha, delta) in enumerate(zip(band_alphas, band_deltas)):
                lower = band_edges[band_index]
                upper = band_edges[band_index + 1]
                if band_index == len(band_alphas) - 1:
                    band_mask = (frequencies >= lower) & (frequencies <= upper)
                else:
                    band_mask = (frequencies >= lower) & (frequencies < upper)

                manual += np.sum(
                    self._manual_log_density(
                        quadratic_forms=quadratic_forms[band_mask],
                        dimensions=np.full(np.sum(band_mask), 2, dtype=int),
                        alpha=alpha,
                        delta=delta,
                    )
                    - np.log(scale2[band_mask])
                )

        self.assertAlmostEqual(calculated, float(manual), 7)

    def test_frequency_band_edges_override_equal_width_bands(self):
        band_alphas = [4.0, 10.0]
        band_deltas = [0.7, 1.3]
        equal_width = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=band_alphas,
            delta=band_deltas,
            num_frequency_bands=2,
        )
        explicit = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=band_alphas,
            delta=band_deltas,
            frequency_band_edges=equal_width._frequency_band_edges,
        )
        parameters = self.parameters.copy()
        self.assertEqual(explicit.num_frequency_bands, 2)
        self.assertEqual(
            equal_width.log_likelihood(parameters), explicit.log_likelihood(parameters)
        )

        network = bilby.gw.detector.InterferometerList(["H1", "L1"])
        network.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        per_detector = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=network,
            waveform_generator=self.waveform_generator,
            detector_dependent_noise=True,
            frequency_band_edges={"H1": [20.0, 30.0, 512.0], "L1": [20.0, 60.0, 512.0]},
        )
        frequencies = np.array([25.0, 45.0])
        self.assertEqual(
            [mask.tolist() for mask in per_detector._get_frequency_band_masks(frequencies, "H1")],
            [[True, False], [False, True]],
        )
        self.assertEqual(
            [mask.tolist() for mask in per_detector._get_frequency_band_masks(frequencies, "L1")],
            [[True, True], [False, False]],
        )
        with self.assertRaises(ValueError):
            bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
                interferometers=network,
                waveform_generator=self.waveform_generator,
                detector_dependent_noise=True,
                frequency_band_edges={"H1": [20.0, 30.0, 512.0], "L1": [20.0, 512.0]},
            )

    def test_log_likelihood_handles_different_detector_masks(self):
        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        active_l1_frequencies = interferometers[1].frequency_array[interferometers[1].frequency_mask]
        cutoff_frequency = active_l1_frequencies[len(active_l1_frequencies) // 2]
        interferometers[1].frequency_mask = (
            interferometers[1].frequency_mask
            & (interferometers[1].frequency_array <= cutoff_frequency)
        )

        alpha = 10.0
        delta = 1.5
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=self.waveform_generator,
            alpha=alpha,
            delta=delta,
            joint=True,
        )
        parameters = self.parameters.copy()

        calculated = likelihood.log_likelihood(parameters)

        pols = self.waveform_generator.frequency_domain_strain(self.parameters)
        network_frequencies = np.unique(
            np.concatenate(
                [
                    interferometer.frequency_array[interferometer.frequency_mask]
                    for interferometer in interferometers
                ]
            )
        )
        quadratic_forms = np.zeros(len(network_frequencies), dtype=float)
        log_scale2_terms = np.zeros(len(network_frequencies), dtype=float)
        active_counts = np.zeros(len(network_frequencies), dtype=int)

        for interferometer in interferometers:
            mask = interferometer.frequency_mask
            frequencies = interferometer.frequency_array[mask]
            h_f = interferometer.get_detector_response(pols, self.parameters)
            residual = interferometer.frequency_domain_strain[mask] - h_f[mask]
            scale2 = interferometer.power_spectral_density_array[mask] * self.duration / 4.0
            q_contribution = (residual.real ** 2 + residual.imag ** 2) / scale2
            indices = np.searchsorted(network_frequencies, frequencies)
            quadratic_forms[indices] += q_contribution
            log_scale2_terms[indices] += np.log(scale2)
            active_counts[indices] += 1

        manual = np.sum(
            self._manual_log_density(
                quadratic_forms=quadratic_forms,
                dimensions=2 * active_counts,
                alpha=alpha,
                delta=delta,
            )
            - log_scale2_terms
        )

        self.assertAlmostEqual(calculated, float(manual), 7)

    def test_detector_independent_likelihood_is_factorized_by_default(self):
        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        alpha = 10.0
        delta = 1.5
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=self.waveform_generator,
            alpha=alpha,
            delta=delta,
        )
        parameters = self.parameters.copy()

        calculated = likelihood.log_likelihood(parameters)

        pols = self.waveform_generator.frequency_domain_strain(self.parameters)
        manual = 0.0
        manual_noise = 0.0
        for interferometer in interferometers:
            mask = interferometer.frequency_mask
            h_f = interferometer.get_detector_response(pols, self.parameters)
            residual = interferometer.frequency_domain_strain[mask] - h_f[mask]
            scale2 = (
                interferometer.power_spectral_density_array[mask]
                * self.duration
                / 4.0
            )
            quadratic_forms = (
                residual.real ** 2 + residual.imag ** 2
            ) / scale2
            manual += np.sum(
                self._manual_log_density(
                    quadratic_forms=quadratic_forms,
                    dimensions=np.full(len(quadratic_forms), 2, dtype=int),
                    alpha=alpha,
                    delta=delta,
                )
                - np.log(scale2)
            )
            noise_quadratic_forms = (
                interferometer.frequency_domain_strain[mask].real ** 2
                + interferometer.frequency_domain_strain[mask].imag ** 2
            ) / scale2
            manual_noise += np.sum(
                self._manual_log_density(
                    quadratic_forms=noise_quadratic_forms,
                    dimensions=np.full(len(noise_quadratic_forms), 2, dtype=int),
                    alpha=alpha,
                    delta=delta,
                )
                - np.log(scale2)
            )

        self.assertFalse(likelihood.joint)
        self.assertAlmostEqual(calculated, float(manual), 7)
        self.assertAlmostEqual(
            likelihood.noise_log_likelihood(), float(manual_noise), 7
        )
        self.assertAlmostEqual(
            likelihood.log_likelihood_ratio(parameters),
            float(manual - manual_noise),
            7,
        )

    def test_infer_alpha_uses_parameter_dict(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            infer_alpha=True,
        )
        self.assertIn("alpha", likelihood.noise_parameter_keys)

        parameters = self.parameters.copy()
        parameters["alpha"] = 4.0
        self.assertEqual(likelihood.alpha(parameters), 4.0)

        logl_alpha4 = likelihood.log_likelihood(parameters)
        parameters["alpha"] = 40.0
        logl_alpha40 = likelihood.log_likelihood(parameters)
        self.assertNotEqual(logl_alpha4, logl_alpha40)

    def test_infer_delta_uses_parameter_dict(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
            infer_delta=True,
        )
        self.assertIn("delta", likelihood.noise_parameter_keys)

        parameters = self.parameters.copy()
        parameters["delta"] = 0.5
        self.assertEqual(likelihood.delta(parameters), 0.5)

        logl_delta05 = likelihood.log_likelihood(parameters)
        parameters["delta"] = 2.5
        logl_delta25 = likelihood.log_likelihood(parameters)
        self.assertNotEqual(logl_delta05, logl_delta25)

    def test_infer_alpha_and_delta_use_per_band_parameters(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
            infer_alpha=True,
            infer_delta=True,
            num_frequency_bands=2,
        )
        self.assertIn("alpha_1", likelihood.noise_parameter_keys)
        self.assertIn("alpha_2", likelihood.noise_parameter_keys)
        self.assertIn("delta_1", likelihood.noise_parameter_keys)
        self.assertIn("delta_2", likelihood.noise_parameter_keys)

        parameters = self.parameters.copy()
        parameters["alpha_1"] = 4.0
        parameters["alpha_2"] = 40.0
        parameters["delta_1"] = 0.5
        parameters["delta_2"] = 2.5

        np.testing.assert_allclose(likelihood.alpha(parameters), np.array([4.0, 40.0]))
        np.testing.assert_allclose(likelihood.delta(parameters), np.array([0.5, 2.5]))

        logl_split = likelihood.log_likelihood(parameters)
        parameters["alpha_1"] = 40.0
        parameters["alpha_2"] = 4.0
        parameters["delta_1"] = 2.5
        parameters["delta_2"] = 0.5
        logl_swapped = likelihood.log_likelihood(parameters)

        self.assertNotEqual(logl_split, logl_swapped)

    def test_calculate_snrs_applies_delta_over_alpha_scaling(self):
        gaussian = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
        )
        pols = self.waveform_generator.frequency_domain_strain(self.parameters)
        ifo = self.interferometers[0]
        raw = gaussian.calculate_snrs(
            pols, ifo, parameters=self.parameters.copy()
        ).optimal_snr_squared.real

        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=2.0,
            infer_alpha=True,
            infer_delta=True,
        )
        parameters = self.parameters.copy()
        parameters.update(alpha=3.0, delta=7.0)
        corrected = likelihood.calculate_snrs(
            pols, ifo, parameters=parameters
        ).optimal_snr_squared.real

        self.assertAlmostEqual(corrected / raw, 3.0 / 7.0, places=12)

    def test_noise_evidence_uses_shape_parameters_fixed_at_initialization(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=[10.0, 12.0],
            delta=[1.0, 1.5],
            infer_alpha=True,
            infer_delta=True,
            num_frequency_bands=2,
            noise_evidence_method="nested",
        )
        priors = bilby.core.prior.PriorDict(
            dict(
                alpha_1=bilby.core.prior.Uniform(2.0, 20.0, name="alpha_1"),
                alpha_2=bilby.core.prior.Uniform(2.0, 20.0, name="alpha_2"),
                delta_1=bilby.core.prior.Uniform(0.5, 5.0, name="delta_1"),
                delta_2=bilby.core.prior.Uniform(0.5, 5.0, name="delta_2"),
            )
        )
        with patch(
            "bilby.core.sampler.run_sampler",
            return_value=MagicMock(log_evidence=-10.0),
        ):
            noise_log_evidence = likelihood.noise_log_evidence(priors=priors)

        self.assertEqual(
            likelihood._get_default_shape_parameter_dict(),
            dict(alpha_1=10.0, alpha_2=12.0, delta_1=1.0, delta_2=1.5),
        )
        self.assertEqual(noise_log_evidence, -10.0)

    def test_meta_data_includes_hyperbolic_configuration(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=[10.0, 12.0],
            delta=[1.0, 1.5],
            infer_alpha=True,
            infer_delta=True,
            num_frequency_bands=2,
        )

        self.assertIs(
            likelihood.meta_data["likelihood_class"],
            bilby.gw.likelihood.HyperbolicGravitationalWaveTransient,
        )
        self.assertEqual(likelihood.meta_data["alpha"], [10.0, 12.0])
        self.assertEqual(likelihood.meta_data["delta"], [1.0, 1.5])
        self.assertTrue(likelihood.meta_data["infer_alpha"])
        self.assertTrue(likelihood.meta_data["infer_delta"])
        self.assertFalse(likelihood.meta_data["detector_dependent_noise"])
        self.assertFalse(likelihood.meta_data["joint"])
        self.assertEqual(likelihood.meta_data["num_frequency_bands"], 2)
        self.assertEqual(likelihood.meta_data["noise_evidence_method"], "quadrature")

    def test_gaussian_limit_diagnostic_uses_shape_prior_maxima(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            infer_alpha=True,
            infer_delta=True,
        )
        priors = bilby.core.prior.PriorDict(
            dict(
                alpha=bilby.core.prior.Uniform(1e-6, 30.0, name="alpha"),
                delta=bilby.core.prior.Uniform(1e-6, 30.0, name="delta"),
            )
        )

        with patch("builtins.print") as print_mock:
            likelihood.print_gaussian_limit_diagnostic(priors)

        output = "\n".join(call.args[0] for call in print_mock.call_args_list)
        self.assertIn("Hyperbolic Gaussian-limit diagnostic", output)
        self.assertIn("alpha = 30 (prior max)", output)
        self.assertIn("delta = 30 (prior max)", output)
        self.assertIn("variance scale = 1.00222 (+0.222%)", output)

    def test_detector_dependent_fixed_shape_matches_direct_calculation(self):
        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        detector_alphas = np.array([[4.0, 10.0], [8.0, 25.0]])
        detector_deltas = np.array([[0.7, 1.3], [1.1, 2.0]])
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=self.waveform_generator,
            alpha=detector_alphas,
            delta=detector_deltas,
            num_frequency_bands=2,
            detector_dependent_noise=True,
        )
        parameters = self.parameters.copy()

        calculated = likelihood.log_likelihood(parameters)

        pols = self.waveform_generator.frequency_domain_strain(self.parameters)
        manual = 0.0
        band_edges = likelihood._frequency_band_edges
        for detector_index, ifo in enumerate(interferometers):
            h_f = ifo.get_detector_response(pols, self.parameters)
            mask = ifo.frequency_mask
            frequencies = ifo.frequency_array[mask]
            r = ifo.frequency_domain_strain[mask] - h_f[mask]
            scale2 = ifo.power_spectral_density_array[mask] * self.duration / 4.0
            quadratic_forms = (r.real ** 2 + r.imag ** 2) / scale2

            for band_index, (alpha, delta) in enumerate(
                zip(detector_alphas[detector_index], detector_deltas[detector_index])
            ):
                lower = band_edges[band_index]
                upper = band_edges[band_index + 1]
                if band_index == detector_alphas.shape[1] - 1:
                    band_mask = (frequencies >= lower) & (frequencies <= upper)
                else:
                    band_mask = (frequencies >= lower) & (frequencies < upper)

                manual += np.sum(
                    self._manual_log_density(
                        quadratic_forms=quadratic_forms[band_mask],
                        dimensions=np.full(np.sum(band_mask), 2, dtype=int),
                        alpha=alpha,
                        delta=delta,
                    )
                    - np.log(scale2[band_mask])
                )

        self.assertAlmostEqual(calculated, float(manual), 7)

    def test_detector_dependent_infer_shape_uses_detector_specific_parameters(self):
        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
            infer_alpha=True,
            infer_delta=True,
            num_frequency_bands=2,
            detector_dependent_noise=True,
        )

        for key in [
            "alpha_H1_1",
            "alpha_H1_2",
            "alpha_L1_1",
            "alpha_L1_2",
            "delta_H1_1",
            "delta_H1_2",
            "delta_L1_1",
            "delta_L1_2",
        ]:
            self.assertIn(key, likelihood.noise_parameter_keys)

        parameters = self.parameters.copy()
        parameters.update(
            {
                "alpha_H1_1": 4.0,
                "alpha_H1_2": 40.0,
                "alpha_L1_1": 8.0,
                "alpha_L1_2": 15.0,
                "delta_H1_1": 0.5,
                "delta_H1_2": 2.5,
                "delta_L1_1": 1.0,
                "delta_L1_2": 1.8,
            }
        )

        np.testing.assert_allclose(
            likelihood.alpha(parameters),
            np.array([[4.0, 40.0], [8.0, 15.0]]),
        )
        np.testing.assert_allclose(
            likelihood.delta(parameters),
            np.array([[0.5, 2.5], [1.0, 1.8]]),
        )

        logl_a = likelihood.log_likelihood(parameters)
        parameters.update(
            {
                "alpha_H1_1": 40.0,
                "alpha_H1_2": 4.0,
                "delta_H1_1": 2.5,
                "delta_H1_2": 0.5,
            }
        )
        logl_b = likelihood.log_likelihood(parameters)

        self.assertNotEqual(logl_a, logl_b)

    def test_invalid_alpha_returns_negative_infinity(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            infer_alpha=True,
        )
        parameters = self.parameters.copy()
        parameters["alpha"] = -1.0

        self.assertEqual(likelihood.log_likelihood(parameters), np.nan_to_num(-np.inf))

    def test_invalid_delta_returns_negative_infinity(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
            infer_delta=True,
        )
        parameters = self.parameters.copy()
        parameters["delta"] = -1.0

        self.assertEqual(likelihood.log_likelihood(parameters), np.nan_to_num(-np.inf))

    def test_invalid_per_band_alpha_returns_negative_infinity(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            infer_alpha=True,
            num_frequency_bands=2,
        )
        parameters = self.parameters.copy()
        parameters["alpha_1"] = 10.0
        parameters["alpha_2"] = -1.0

        self.assertEqual(likelihood.log_likelihood(parameters), np.nan_to_num(-np.inf))

    def test_invalid_per_band_delta_returns_negative_infinity(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
            infer_delta=True,
            num_frequency_bands=2,
        )
        parameters = self.parameters.copy()
        parameters["delta_1"] = 1.0
        parameters["delta_2"] = -1.0

        self.assertEqual(likelihood.log_likelihood(parameters), np.nan_to_num(-np.inf))

    def test_noise_log_evidence_defaults_to_2d_quadrature_for_sampled_shape_parameters(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
            infer_alpha=True,
            infer_delta=True,
        )
        parameters = self.parameters.copy()
        parameters["alpha"] = 8.0
        parameters["delta"] = 1.5
        priors = bilby.core.prior.PriorDict(
            dict(
                alpha=bilby.core.prior.Uniform(2.0, 20.0, name="alpha"),
                delta=bilby.core.prior.Uniform(0.5, 5.0, name="delta"),
            )
        )
        unit_grid = np.linspace(0.0, 1.0, 5001)
        alpha_grid = priors["alpha"].rescale(unit_grid)
        delta_grid = priors["delta"].rescale(unit_grid)
        expected = (
            np.log(np.trapezoid(np.exp(-0.5 * (alpha_grid - 8.0) ** 2), unit_grid))
            + np.log(
                np.trapezoid(np.exp(-0.5 * (delta_grid - 1.5) ** 2), unit_grid)
            )
        )
        likelihood._noise_log_likelihood_from_parameters = (
            lambda parameters: (
                -0.5 * (parameters["alpha"] - 8.0) ** 2
                - 0.5 * (parameters["delta"] - 1.5) ** 2
            )
        )

        with patch("bilby.core.sampler.run_sampler") as mock_run_sampler:
            noise_log_evidence = likelihood.noise_log_evidence(priors=priors, npool=2)

        self.assertAlmostEqual(noise_log_evidence, expected, 6)
        mock_run_sampler.assert_not_called()

    def test_noise_log_evidence_runs_auxiliary_nested_sampling_when_requested(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
            infer_alpha=True,
            infer_delta=True,
            noise_evidence_method="nested",
        )
        parameters = self.parameters.copy()
        parameters["alpha"] = 8.0
        parameters["delta"] = 1.5
        priors = bilby.core.prior.PriorDict(
            dict(
                alpha=bilby.core.prior.Uniform(2.0, 20.0, name="alpha"),
                delta=bilby.core.prior.Uniform(0.5, 5.0, name="delta"),
            )
        )
        mock_result = MagicMock(log_evidence=-321.0)

        with patch("bilby.core.sampler.run_sampler", return_value=mock_result) as mock_run_sampler:
            noise_log_evidence = likelihood.noise_log_evidence(priors=priors, npool=2)

        self.assertEqual(noise_log_evidence, mock_result.log_evidence)
        self.assertEqual(mock_run_sampler.call_count, 1)
        self.assertEqual(mock_run_sampler.call_args.kwargs["sampler"], "dynesty")
        self.assertEqual(mock_run_sampler.call_args.kwargs["npool"], 1)
        self.assertListEqual(
            list(mock_run_sampler.call_args.kwargs["priors"].keys()),
            ["alpha", "delta"],
        )
        self.assertEqual(
            mock_run_sampler.call_args.kwargs["likelihood"].__class__.__name__,
            "_HyperbolicNoiseOnlyLikelihood",
        )

    def test_noise_log_evidence_uses_conditional_value_when_shape_parameters_are_not_sampled(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
            infer_alpha=True,
            infer_delta=True,
        )
        parameters = self.parameters.copy()
        parameters["alpha"] = 8.0
        parameters["delta"] = 1.5
        priors = bilby.core.prior.PriorDict(
            dict(
                alpha=bilby.core.prior.DeltaFunction(peak=8.0, name="alpha"),
                delta=bilby.core.prior.DeltaFunction(peak=1.5, name="delta"),
            )
        )

        with patch("bilby.core.sampler.run_sampler") as mock_run_sampler:
            noise_log_evidence = likelihood.noise_log_evidence(priors=priors)

        self.assertAlmostEqual(noise_log_evidence, likelihood.noise_log_likelihood(), 7)
        mock_run_sampler.assert_not_called()

    def test_noise_log_evidence_uses_custom_sampler_controls_for_hyperbolic(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
            infer_alpha=True,
            infer_delta=True,
            noise_evidence_method="nested",
            noise_evidence_nlive=192,
            dlogZ_noise=0.02,
        )
        parameters = self.parameters.copy()
        parameters["alpha"] = 8.0
        parameters["delta"] = 1.5
        priors = bilby.core.prior.PriorDict(
            dict(
                alpha=bilby.core.prior.Uniform(2.0, 20.0, name="alpha"),
                delta=bilby.core.prior.Uniform(0.5, 5.0, name="delta"),
            )
        )
        mock_result = MagicMock(log_evidence=-321.0)

        with patch("bilby.core.sampler.run_sampler", return_value=mock_result) as mock_run_sampler:
            likelihood.noise_log_evidence(priors=priors, npool=2)

        self.assertEqual(mock_run_sampler.call_args.kwargs["npool"], 1)
        self.assertEqual(mock_run_sampler.call_args.kwargs["nlive"], 192)
        self.assertEqual(mock_run_sampler.call_args.kwargs["dlogz"], 0.02)

    def test_noise_log_evidence_quadrature_falls_back_to_nested_above_2d(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=[10.0, 12.0],
            delta=[1.0, 1.5],
            infer_alpha=True,
            infer_delta=True,
            num_frequency_bands=2,
        )
        parameters = self.parameters.copy()
        parameters["alpha_1"] = 8.0
        parameters["alpha_2"] = 9.0
        parameters["delta_1"] = 1.5
        parameters["delta_2"] = 2.0
        priors = bilby.core.prior.PriorDict(
            dict(
                alpha_1=bilby.core.prior.Uniform(2.0, 20.0, name="alpha_1"),
                alpha_2=bilby.core.prior.Uniform(2.0, 20.0, name="alpha_2"),
                delta_1=bilby.core.prior.Uniform(0.5, 5.0, name="delta_1"),
                delta_2=bilby.core.prior.Uniform(0.5, 5.0, name="delta_2"),
            )
        )
        mock_result = MagicMock(log_evidence=-321.0)

        with patch("bilby.core.sampler.run_sampler", return_value=mock_result) as mock_run_sampler:
            noise_log_evidence = likelihood.noise_log_evidence(priors=priors, npool=2)

        self.assertEqual(noise_log_evidence, mock_result.log_evidence)
        self.assertListEqual(
            list(mock_run_sampler.call_args.kwargs["priors"].keys()),
            ["alpha_1", "alpha_2", "delta_1", "delta_2"],
        )

    def test_time_reference_agrees_with_default(self):
        default_likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
        )
        h1_time_likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
            time_reference="H1",
        )

        ifo = bilby.gw.detector.get_empty_interferometer("H1")
        time_delay = ifo.time_delay_from_geocenter(
            ra=self.parameters["ra"],
            dec=self.parameters["dec"],
            time=self.parameters["geocent_time"],
        )
        parameters = self.parameters.copy()
        parameters.pop("geocent_time")
        parameters["H1_time"] = self.parameters["geocent_time"] + time_delay

        self.assertEqual(
            h1_time_likelihood.log_likelihood(parameters),
            default_likelihood.log_likelihood(self.parameters),
        )

    def test_bilby_pipe_style_filtered_kwargs_preserve_frame_configuration(self):
        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        likelihood_kwargs = dict(
            interferometers=interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
            priors=None,
            time_marginalization=False,
            distance_marginalization=False,
            phase_marginalization=False,
            calibration_marginalization=False,
            distance_marginalization_lookup_table=None,
            calibration_lookup_table=None,
            number_of_response_curves=1000,
            starting_index=0,
            jitter_time=False,
            reference_frame="L1H1",
            time_reference="L1",
        )
        filtered_kwargs = {
            key: value
            for key, value in likelihood_kwargs.items()
            if key
            in inspect.getfullargspec(
                bilby.gw.likelihood.HyperbolicGravitationalWaveTransient.__init__
            ).args
        }

        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            **filtered_kwargs
        )
        self.assertEqual(likelihood._reference_frame_str, "L1H1")
        self.assertEqual(likelihood.time_reference, "L1")

        parameters = self.parameters.copy()
        parameters["zenith"] = 1.0
        parameters["azimuth"] = 1.0
        parameters["ra"], parameters["dec"] = bilby.gw.utils.zenith_azimuth_to_ra_dec(
            zenith=parameters["zenith"],
            azimuth=parameters["azimuth"],
            geocent_time=parameters["geocent_time"],
            ifos=likelihood.reference_frame,
        )
        l1 = bilby.gw.detector.get_empty_interferometer("L1")
        parameters["L1_time"] = parameters["geocent_time"] + l1.time_delay_from_geocenter(
            ra=parameters["ra"],
            dec=parameters["dec"],
            time=parameters["geocent_time"],
        )
        sampled_parameters = {
            key: value
            for key, value in parameters.items()
            if key not in {"ra", "dec", "geocent_time"}
        }

        self.assertAlmostEqual(
            likelihood.log_likelihood(sampled_parameters),
            likelihood.log_likelihood(parameters),
            7,
        )

    def test_log_likelihood_ratio_matches_noise_subtraction_and_per_detector_output(self):
        likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
            infer_alpha=True,
            infer_delta=True,
        )
        parameters = self.parameters.copy()
        parameters["alpha"] = 8.0
        parameters["delta"] = 1.5

        # noise_log_likelihood() takes no parameters, so it is evaluated at the
        # alpha and delta fixed at initialisation. The noise term subtracted by
        # log_likelihood_ratio is the one at the sampled values instead.
        sampled_shape_likelihood = (
            bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                alpha=8.0,
                delta=1.5,
            )
        )

        log_likelihood = likelihood.log_likelihood(parameters)
        noise_log_likelihood = sampled_shape_likelihood.noise_log_likelihood()
        log_likelihood_ratio = likelihood.log_likelihood_ratio(parameters)
        per_detector = likelihood.compute_per_detector_log_likelihood(parameters)

        self.assertAlmostEqual(
            log_likelihood - noise_log_likelihood, log_likelihood_ratio, 7
        )
        self.assertAlmostEqual(
            per_detector["H1_log_likelihood"], log_likelihood_ratio, 7
        )

    def test_large_alpha_delta_match_gaussian_likelihood_ratio_to_machine_precision(self):
        gaussian_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
        )
        hyperbolic_likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=1e8,
            delta=1e8,
        )

        parameter_points = [
            self.parameters.copy(),
            {**self.parameters, "luminosity_distance": 2000.0},
            {**self.parameters, "luminosity_distance": 8000.0},
            {**self.parameters, "phase": 0.0},
            {**self.parameters, "phase": 2.4},
        ]

        for parameters in parameter_points:
            self.assertAlmostEqual(
                hyperbolic_likelihood.log_likelihood_ratio(parameters),
                gaussian_likelihood.log_likelihood_ratio(parameters),
                delta=5e-11,
            )
