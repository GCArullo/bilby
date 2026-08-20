"""Tests for covariance-based time-domain likelihoods."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.linalg import solve_toeplitz, toeplitz

import bilby
from bilby.gw.likelihood.time_domain import (
    _GohbergSemenculToeplitzInverse,
    _factorized_noise_log_evidence_by_quadrature,
    _gaussian_log_likelihood_from_inner_product,
    _hyperbolic_log_likelihood_from_inner_product,
    _residuals_inner_product_from_cache,
    _student_t_log_likelihood_from_inner_product,
)


def constant_frequency_domain_source(frequency_array, amplitude=0.0):
    plus = amplitude * np.ones_like(frequency_array, dtype=complex)
    cross = np.zeros_like(frequency_array, dtype=complex)
    return dict(plus=plus, cross=cross)


def constant_time_domain_source(time_array, amplitude=0.0):
    plus = amplitude * np.ones_like(time_array, dtype=float)
    cross = np.zeros_like(time_array, dtype=float)
    return dict(plus=plus, cross=cross)


def _direct_network_residual_statistics(likelihood, residuals_by_detector):
    statistics = []
    for band_index in range(likelihood._number_of_time_bands):
        residuals_inner_product = 0.0
        logdet = 0.0
        dimension = 0
        for interferometer in likelihood.interferometers:
            detector_cache = likelihood._detector_likelihood_caches[
                interferometer.name
            ]
            if likelihood._number_of_time_bands > 1:
                cache = detector_cache["time_bands"][band_index]
            else:
                cache = detector_cache["full"]
            residuals = residuals_by_detector[interferometer.name]
            residuals_inner_product += _residuals_inner_product_from_cache(
                residuals[cache.start : cache.end],
                cache,
                likelihood.likelihood_method,
            )
            logdet += cache.logdet
            dimension += cache.end - cache.start
        statistics.append((residuals_inner_product, logdet, dimension))
    return statistics


class TestTimeDomainGWTransient(unittest.TestCase):
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

    def test_gaussian_log_likelihood_matches_direct_calculation(self):
        likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
        )
        parameters = likelihood._resolve_signal_likelihood_parameters(
            self.parameters.copy()
        )
        waveform_polarizations = likelihood._waveform_polarizations_frequency_domain(
            parameters
        )

        calculated = likelihood.log_likelihood(self.parameters.copy())

        manual = 0.0
        for interferometer in self.interferometers:
            residuals = likelihood._residual_time_domain(
                interferometer=interferometer,
                parameters=parameters,
                waveform_polarizations=waveform_polarizations,
            )
            cache = likelihood._detector_likelihood_caches[interferometer.name]["full"]
            residuals_inner_product = _residuals_inner_product_from_cache(
                residuals,
                cache,
                likelihood.likelihood_method,
            )
            manual += _gaussian_log_likelihood_from_inner_product(
                residuals_inner_product=residuals_inner_product,
                log_normalisation=cache.log_normalisation,
            )

        self.assertAlmostEqual(calculated, float(manual), 7)

    def test_data_and_signal_use_the_analysis_frequency_mask(self):
        interferometer = self.interferometers[0]
        interferometer.minimum_frequency = 20.0
        interferometer.maximum_frequency = 40.0
        likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            interferometers=bilby.gw.detector.InterferometerList([interferometer]),
            waveform_generator=self.waveform_generator,
        )

        expected_data = np.real(
            bilby.core.utils.infft(
                np.where(
                    interferometer.frequency_mask,
                    interferometer.frequency_domain_strain,
                    0.0,
                ),
                interferometer.sampling_frequency,
            )
        )
        np.testing.assert_allclose(
            likelihood._data_time_domain(interferometer),
            expected_data,
            rtol=0.0,
            atol=0.0,
        )

        parameters = likelihood._resolve_signal_likelihood_parameters(
            self.parameters.copy()
        )
        waveform_polarizations = likelihood._waveform_polarizations_frequency_domain(
            parameters
        )
        signal_frequency_domain = likelihood._detector_response_frequency_domain(
            waveform_polarizations,
            interferometer,
            parameters,
        )
        expected_signal = np.real(
            bilby.core.utils.infft(
                np.where(
                    interferometer.frequency_mask,
                    signal_frequency_domain,
                    0.0,
                ),
                interferometer.sampling_frequency,
            )
        )
        np.testing.assert_allclose(
            likelihood._signal_time_domain(
                interferometer,
                waveform_polarizations,
                parameters,
            ),
            expected_signal,
            rtol=0.0,
            atol=0.0,
        )

    def test_student_t_detector_dependent_time_bands_match_direct_calculation(self):
        time_bands = [0.5]
        likelihood = bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            infer_nu=True,
            detector_dependent_noise=True,
            time_bands=time_bands,
        )
        parameters = self.parameters.copy()
        parameters.update(
            nu_H1_0_0p5=6.0,
            nu_H1_0p5_2=8.0,
            nu_L1_0_0p5=10.0,
            nu_L1_0p5_2=12.0,
        )
        signal_parameters = likelihood._resolve_signal_likelihood_parameters(
            parameters.copy()
        )
        waveform_polarizations = likelihood._waveform_polarizations_frequency_domain(
            signal_parameters
        )

        calculated = likelihood.log_likelihood(parameters.copy())

        manual = 0.0
        for interferometer in self.interferometers:
            residuals = likelihood._residual_time_domain(
                interferometer=interferometer,
                parameters=signal_parameters,
                waveform_polarizations=waveform_polarizations,
            )
            for band_index, cache in enumerate(
                likelihood._detector_likelihood_caches[interferometer.name]["time_bands"]
            ):
                band_residuals = residuals[cache.start : cache.end]
                residuals_inner_product = _residuals_inner_product_from_cache(
                    band_residuals,
                    cache,
                    likelihood.likelihood_method,
                )
                nu = parameters[
                    likelihood._detector_nu_parameter_key(
                        interferometer.name, band_index
                    )
                ]
                manual += _student_t_log_likelihood_from_inner_product(
                    residuals_inner_product=residuals_inner_product,
                    logdet=cache.logdet,
                    dimension=cache.end - cache.start,
                    nu=nu,
                )

        self.assertAlmostEqual(calculated, float(manual), 7)

    def test_time_band_boundaries_match_explicit_cut_times(self):
        parameters = self.parameters.copy()
        parameters.update(
            nu_H1_0_0p5=6.0,
            nu_H1_0p5_2=8.0,
            nu_L1_0_0p5=10.0,
            nu_L1_0p5_2=12.0,
        )

        list_likelihood = bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            infer_nu=True,
            detector_dependent_noise=True,
            time_bands=[0.5],
        )
        boundary_likelihood = bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            infer_nu=True,
            detector_dependent_noise=True,
            time_band_boundaries=[0.5],
        )

        self.assertEqual(boundary_likelihood.time_bands, [0.5])
        self.assertEqual(boundary_likelihood.time_band_boundaries, [0.5])
        self.assertEqual(
            boundary_likelihood.nu_parameter_keys,
            [
                "nu_H1_0_0p5",
                "nu_H1_0p5_2",
                "nu_L1_0_0p5",
                "nu_L1_0p5_2",
            ],
        )
        self.assertAlmostEqual(
            boundary_likelihood.log_likelihood(parameters.copy()),
            list_likelihood.log_likelihood(parameters.copy()),
            10,
        )

    def test_time_band_boundaries_reject_conflicting_time_bands(self):
        with self.assertRaisesRegex(
            ValueError,
            "time_bands and time_band_boundaries must describe the same number of bands",
        ):
            bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                time_bands=3,
                time_band_boundaries=[0.5],
            )

        with self.assertRaisesRegex(
            ValueError,
            "time_bands and time_band_boundaries must match when both are provided",
        ):
            bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                time_bands=[0.25],
                time_band_boundaries=[0.5],
            )

    def test_detector_specific_time_band_boundaries(self):
        boundaries = {"H1": [0.25], "L1": [0.5]}
        likelihood = bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            infer_nu=True,
            detector_dependent_noise=True,
            time_bands=2,
            time_band_boundaries=boundaries,
        )

        self.assertEqual(likelihood.time_band_boundaries, boundaries)
        self.assertEqual(
            likelihood.nu_parameter_keys,
            [
                "nu_H1_0_0p25",
                "nu_H1_0p25_2",
                "nu_L1_0_0p5",
                "nu_L1_0p5_2",
            ],
        )
        self.assertEqual(
            likelihood._detector_likelihood_caches["H1"]["time_bands"][0].end,
            32,
        )
        self.assertEqual(
            likelihood._detector_likelihood_caches["L1"]["time_bands"][0].end,
            64,
        )
        parameters = self.parameters.copy()
        parameters.update(
            dict(zip(likelihood.nu_parameter_keys, [6.0, 8.0, 10.0, 12.0]))
        )
        self.assertTrue(np.isfinite(likelihood.log_likelihood(parameters)))

        with self.assertRaisesRegex(
            ValueError, "Per-detector time_band_boundaries requires"
        ):
            bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                time_band_boundaries=boundaries,
            )

        with self.assertRaisesRegex(ValueError, "unknown V1"):
            bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                detector_dependent_noise=True,
                time_band_boundaries={**boundaries, "V1": [0.5]},
            )

    def test_hyperbolic_time_bands_match_direct_calculation(self):
        time_bands = [0.5]
        likelihood = bilby.gw.likelihood.HyperbolicTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            infer_alpha=True,
            infer_delta=True,
            detector_dependent_noise=False,
            time_bands=time_bands,
        )
        parameters = self.parameters.copy()
        parameters.update(
            alpha_0_0p5=6.0,
            alpha_0p5_2=12.0,
            delta_0_0p5=0.8,
            delta_0p5_2=1.4,
        )
        signal_parameters = likelihood._resolve_signal_likelihood_parameters(
            parameters.copy()
        )
        waveform_polarizations = likelihood._waveform_polarizations_frequency_domain(
            signal_parameters
        )

        calculated = likelihood.log_likelihood(parameters.copy())

        manual = 0.0
        for interferometer in self.interferometers:
            residuals = likelihood._residual_time_domain(
                interferometer=interferometer,
                parameters=signal_parameters,
                waveform_polarizations=waveform_polarizations,
            )
            for band_index, cache in enumerate(
                likelihood._detector_likelihood_caches[interferometer.name]["time_bands"]
            ):
                band_residuals = residuals[cache.start : cache.end]
                residuals_inner_product = _residuals_inner_product_from_cache(
                    band_residuals,
                    cache,
                    likelihood.likelihood_method,
                )
                manual += _hyperbolic_log_likelihood_from_inner_product(
                    residuals_inner_product=residuals_inner_product,
                    logdet=cache.logdet,
                    dimension=cache.end - cache.start,
                    alpha=parameters[likelihood.alpha_parameter_keys[band_index]],
                    delta=parameters[likelihood.delta_parameter_keys[band_index]],
                )

        self.assertAlmostEqual(calculated, float(manual), 7)

    def test_joint_network_likelihoods_match_direct_calculation(self):
        cases = [
            (
                bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient,
                dict(nu=[6.0, 12.0]),
                _student_t_log_likelihood_from_inner_product,
                [dict(nu=6.0), dict(nu=12.0)],
            ),
            (
                bilby.gw.likelihood.HyperbolicTimeDomainGravitationalWaveTransient,
                dict(alpha=[6.0, 12.0], delta=[0.8, 1.4]),
                _hyperbolic_log_likelihood_from_inner_product,
                [
                    dict(alpha=6.0, delta=0.8),
                    dict(alpha=12.0, delta=1.4),
                ],
            ),
        ]

        for likelihood_class, kwargs, density, band_parameters in cases:
            with self.subTest(likelihood=likelihood_class.__name__):
                likelihood = likelihood_class(
                    interferometers=self.interferometers,
                    waveform_generator=self.waveform_generator,
                    joint=True,
                    time_band_boundaries=[0.5],
                    **kwargs,
                )
                parameters = likelihood._resolve_signal_likelihood_parameters(
                    self.parameters
                )
                waveform_polarizations = (
                    likelihood._waveform_polarizations_frequency_domain(parameters)
                )
                signal_residuals = {
                    interferometer.name: likelihood._residual_time_domain(
                        interferometer=interferometer,
                        parameters=parameters,
                        waveform_polarizations=waveform_polarizations,
                    )
                    for interferometer in self.interferometers
                }
                noise_residuals = {
                    interferometer.name: likelihood._data_time_domain(
                        interferometer
                    )
                    for interferometer in self.interferometers
                }

                expected_signal = sum(
                    density(
                        residuals_inner_product=statistics[0],
                        logdet=statistics[1],
                        dimension=statistics[2],
                        **band_parameters[band_index],
                    )
                    for band_index, statistics in enumerate(
                        _direct_network_residual_statistics(
                            likelihood, signal_residuals
                        )
                    )
                )
                expected_noise = sum(
                    density(
                        residuals_inner_product=statistics[0],
                        logdet=statistics[1],
                        dimension=statistics[2],
                        **band_parameters[band_index],
                    )
                    for band_index, statistics in enumerate(
                        _direct_network_residual_statistics(
                            likelihood, noise_residuals
                        )
                    )
                )

                self.assertAlmostEqual(
                    likelihood.log_likelihood(self.parameters),
                    expected_signal,
                    10,
                )
                self.assertAlmostEqual(
                    likelihood.noise_log_likelihood(), expected_noise, 10
                )
                self.assertAlmostEqual(
                    likelihood.log_likelihood_ratio(self.parameters),
                    expected_signal - expected_noise,
                    10,
                )
                per_detector = likelihood.compute_per_detector_log_likelihood(
                    self.parameters
                )
                diagnostic_total = sum(
                    per_detector[f"{interferometer.name}_log_likelihood"]
                    for interferometer in self.interferometers
                )
                self.assertNotAlmostEqual(
                    diagnostic_total,
                    expected_signal - expected_noise,
                    7,
                )
                self.assertTrue(likelihood.meta_data["joint"])

    def test_joint_student_t_noise_evidence_matches_direct_integral(self):
        likelihood = bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=[8.0, 9.0],
            infer_nu=True,
            joint=True,
            time_band_boundaries=[0.5],
        )
        sampled_key, fixed_key = likelihood.nu_parameter_keys
        prior = bilby.core.prior.Uniform(4.0, 12.0, name=sampled_key)
        priors = bilby.core.prior.PriorDict(
            {
                sampled_key: prior,
                fixed_key: bilby.core.prior.DeltaFunction(
                    9.0, name=fixed_key
                ),
            }
        )
        statistics_by_band = _direct_network_residual_statistics(
            likelihood,
            {
                interferometer.name: likelihood._data_time_domain(interferometer)
                for interferometer in self.interferometers
            },
        )
        sampled_statistics = statistics_by_band[0]

        def log_likelihood(unit_value):
            return _student_t_log_likelihood_from_inner_product(
                residuals_inner_product=sampled_statistics[0],
                logdet=sampled_statistics[1],
                dimension=sampled_statistics[2],
                nu=float(prior.rescale(unit_value)),
            )

        reference = max(
            log_likelihood(unit_value)
            for unit_value in np.linspace(0.0, 1.0, 101)
        )
        integral, _ = quad(
            lambda unit_value: np.exp(
                log_likelihood(unit_value) - reference
            ),
            0.0,
            1.0,
            epsabs=0.0,
            epsrel=1e-10,
            limit=200,
        )
        fixed_statistics = statistics_by_band[1]
        fixed_log_likelihood = _student_t_log_likelihood_from_inner_product(
            residuals_inner_product=fixed_statistics[0],
            logdet=fixed_statistics[1],
            dimension=fixed_statistics[2],
            nu=9.0,
        )
        expected = reference + np.log(integral) + fixed_log_likelihood

        self.assertAlmostEqual(
            likelihood.noise_log_evidence(priors=priors), expected, 8
        )

    def test_joint_network_validation(self):
        positional_likelihood = (
            bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
                self.interferometers,
                self.waveform_generator,
                8.0,
                False,
                False,
                "toeplitz-inversion",
            )
        )
        self.assertFalse(positional_likelihood.joint)
        self.assertEqual(
            positional_likelihood.likelihood_method, "toeplitz-inversion"
        )

        for likelihood_class in [
            bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient,
            bilby.gw.likelihood.HyperbolicTimeDomainGravitationalWaveTransient,
        ]:
            with self.subTest(likelihood=likelihood_class.__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "joint=True requires detector_dependent_noise=False",
                ):
                    likelihood_class(
                        interferometers=self.interferometers,
                        waveform_generator=self.waveform_generator,
                        joint=True,
                        detector_dependent_noise=True,
                    )

        with self.assertRaisesRegex(ValueError, "does not support joint"):
            bilby.gw.likelihood.MixedTimeDomainGravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                joint=True,
            )

    def test_student_t_noise_log_evidence_factorizes_over_detector_time_bands(self):
        likelihood = bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            infer_nu=True,
            detector_dependent_noise=True,
            time_bands=[0.5],
        )
        priors = bilby.core.prior.PriorDict(
            dict(
                nu_H1_0_0p5=bilby.core.prior.Uniform(
                    2.0, 20.0, name="nu_H1_0_0p5"
                ),
                nu_H1_0p5_2=bilby.core.prior.Uniform(
                    3.0, 21.0, name="nu_H1_0p5_2"
                ),
                nu_L1_0_0p5=bilby.core.prior.Uniform(
                    4.0, 22.0, name="nu_L1_0_0p5"
                ),
                nu_L1_0p5_2=bilby.core.prior.Uniform(
                    5.0, 23.0, name="nu_L1_0p5_2"
                ),
            )
        )
        centers = dict(
            nu_H1_0_0p5=6.0,
            nu_H1_0p5_2=8.0,
            nu_L1_0_0p5=10.0,
            nu_L1_0p5_2=12.0,
        )
        unit_grid = np.linspace(0.0, 1.0, 5001)
        expected = 0.0
        for key, center in centers.items():
            expected += np.log(
                np.trapezoid(
                    np.exp(-0.5 * (priors[key].rescale(unit_grid) - center) ** 2),
                    unit_grid,
                )
            )

        likelihood._noise_block_log_likelihood = lambda block, parameters: (
            -0.5 * (parameters.get(block["keys"][0], block["default_nu"]) - centers[block["keys"][0]]) ** 2
        )

        with patch("bilby.core.sampler.run_sampler") as mock_run_sampler:
            noise_log_evidence = likelihood.noise_log_evidence(priors=priors)

        self.assertAlmostEqual(noise_log_evidence, expected, 6)
        mock_run_sampler.assert_not_called()

    def test_hyperbolic_noise_log_evidence_factorizes_over_detector_time_bands(self):
        likelihood = bilby.gw.likelihood.HyperbolicTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            infer_alpha=True,
            infer_delta=True,
            detector_dependent_noise=True,
            time_bands=[0.5],
        )
        priors = bilby.core.prior.PriorDict(
            dict(
                alpha_H1_0_0p5=bilby.core.prior.Uniform(
                    2.0, 20.0, name="alpha_H1_0_0p5"
                ),
                delta_H1_0_0p5=bilby.core.prior.Uniform(
                    0.5, 5.0, name="delta_H1_0_0p5"
                ),
                alpha_H1_0p5_2=bilby.core.prior.Uniform(
                    3.0, 21.0, name="alpha_H1_0p5_2"
                ),
                delta_H1_0p5_2=bilby.core.prior.Uniform(
                    0.6, 5.1, name="delta_H1_0p5_2"
                ),
                alpha_L1_0_0p5=bilby.core.prior.Uniform(
                    4.0, 22.0, name="alpha_L1_0_0p5"
                ),
                delta_L1_0_0p5=bilby.core.prior.Uniform(
                    0.7, 5.2, name="delta_L1_0_0p5"
                ),
                alpha_L1_0p5_2=bilby.core.prior.Uniform(
                    5.0, 23.0, name="alpha_L1_0p5_2"
                ),
                delta_L1_0p5_2=bilby.core.prior.Uniform(
                    0.8, 5.3, name="delta_L1_0p5_2"
                ),
            )
        )
        centers = dict(
            alpha_H1_0_0p5=6.0,
            delta_H1_0_0p5=0.9,
            alpha_H1_0p5_2=8.0,
            delta_H1_0p5_2=1.1,
            alpha_L1_0_0p5=10.0,
            delta_L1_0_0p5=1.3,
            alpha_L1_0p5_2=12.0,
            delta_L1_0p5_2=1.5,
        )
        unit_grid = np.linspace(0.0, 1.0, 5001)
        expected = 0.0
        for alpha_key, delta_key in (
            ("alpha_H1_0_0p5", "delta_H1_0_0p5"),
            ("alpha_H1_0p5_2", "delta_H1_0p5_2"),
            ("alpha_L1_0_0p5", "delta_L1_0_0p5"),
            ("alpha_L1_0p5_2", "delta_L1_0p5_2"),
        ):
            expected += np.log(
                np.trapezoid(
                    np.exp(
                        -0.5 * (priors[alpha_key].rescale(unit_grid) - centers[alpha_key]) ** 2
                    ),
                    unit_grid,
                )
            )
            expected += np.log(
                np.trapezoid(
                    np.exp(
                        -0.5 * (priors[delta_key].rescale(unit_grid) - centers[delta_key]) ** 2
                    ),
                    unit_grid,
                )
            )

        likelihood._noise_block_log_likelihood = lambda block, parameters: (
            -0.5
            * (
                parameters.get(block["alpha_key"], block["default_alpha"])
                - centers[block["alpha_key"]]
            )
            ** 2
            - 0.5
            * (
                parameters.get(block["delta_key"], block["default_delta"])
                - centers[block["delta_key"]]
            )
            ** 2
        )

        with patch("bilby.core.sampler.run_sampler") as mock_run_sampler:
            noise_log_evidence = likelihood.noise_log_evidence(priors=priors)

        self.assertAlmostEqual(noise_log_evidence, expected, 6)
        mock_run_sampler.assert_not_called()

    def test_student_t_noise_evidence_uses_fixed_prior_peaks(self):
        likelihood = bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=[8.0, 9.0],
            infer_nu=True,
            time_bands=2,
        )
        priors = bilby.core.prior.PriorDict(
            dict(
                nu_1=bilby.core.prior.DeltaFunction(5.0, name="nu_1"),
                nu_2=bilby.core.prior.DeltaFunction(6.0, name="nu_2"),
            )
        )

        expected = likelihood._noise_log_likelihood_from_parameters(
            dict(nu_1=5.0, nu_2=6.0)
        )
        self.assertEqual(likelihood.noise_log_evidence(priors=priors), expected)

    def test_hyperbolic_noise_evidence_uses_fixed_prior_peaks(self):
        likelihood = bilby.gw.likelihood.HyperbolicTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=10.0,
            delta=1.0,
            infer_alpha=True,
            infer_delta=True,
        )
        priors = bilby.core.prior.PriorDict(
            dict(
                alpha=bilby.core.prior.DeltaFunction(6.0, name="alpha"),
                delta=bilby.core.prior.DeltaFunction(1.5, name="delta"),
            )
        )

        expected = likelihood._noise_log_likelihood_from_parameters(
            dict(alpha=6.0, delta=1.5)
        )
        self.assertEqual(likelihood.noise_log_evidence(priors=priors), expected)

    def test_student_t_shared_nu_prior_is_integrated_once(self):
        likelihood = bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            infer_nu=True,
            detector_dependent_noise=True,
            time_bands=2,
        )
        prior = bilby.core.prior.Uniform(2.0, 10.0, name="nu")
        priors = bilby.core.prior.PriorDict(dict(nu=prior))
        likelihood._noise_log_likelihood_from_parameters = lambda parameters: (
            -0.5 * (parameters["nu"] - 6.0) ** 2
        )
        unit_grid = np.linspace(0.0, 1.0, 5001)
        expected = np.log(
            np.trapezoid(
                np.exp(-0.5 * (prior.rescale(unit_grid) - 6.0) ** 2),
                unit_grid,
            )
        )

        self.assertAlmostEqual(
            likelihood.noise_log_evidence(priors=priors), expected, 7
        )
        np.testing.assert_array_equal(
            likelihood._get_nu_values(dict(nu=6.0)), np.full((2, 2), 6.0)
        )

        priors["nu_H1_1"] = bilby.core.prior.Uniform(
            2.0, 10.0, name="nu_H1_1"
        )
        with self.assertRaisesRegex(ValueError, "either the shared nu prior"):
            likelihood.noise_log_evidence(priors=priors)

    def test_prefer_time_domain_waveform_path_runs(self):
        waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            time_domain_source_model=constant_time_domain_source,
            parameter_conversion=bilby.gw.conversion.identity_map_conversion,
        )
        likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=waveform_generator,
            prefer_time_domain_waveform=True,
        )

        calculated = likelihood.log_likelihood(self.parameters.copy())
        self.assertTrue(np.isfinite(calculated))

    def test_build_finite_psd_array_patches_outside_active_band(self):
        interferometer = self.interferometers[0]
        interferometer.minimum_frequency = 20.0
        interferometer.maximum_frequency = 40.0

        source_frequencies = np.asarray(
            interferometer.power_spectral_density.frequency_array, dtype=float
        )
        source_psd = np.asarray(interferometer.power_spectral_density.psd_array, dtype=float)
        analysis_frequencies = np.asarray(interferometer.frequency_array, dtype=float)
        active_frequencies = analysis_frequencies[interferometer.frequency_mask]

        interpolation = interp1d(
            source_frequencies,
            source_psd,
            bounds_error=False,
            fill_value=(float(source_psd[0]), float(source_psd[-1])),
        )
        expected = np.asarray(interpolation(analysis_frequencies), dtype=float)
        finite_mask = np.isfinite(expected) & (expected > 0.0)
        if not np.all(finite_mask):
            fill_value = float(
                np.max(source_psd[np.isfinite(source_psd) & (source_psd > 0.0)])
            )
            expected[~finite_mask] = fill_value

        patch_value = 10.0 * float(
            np.max(
                expected[
                    (analysis_frequencies >= active_frequencies[0])
                    & (analysis_frequencies <= active_frequencies[-1])
                ]
            )
        )
        expected[analysis_frequencies < active_frequencies[0]] = patch_value
        expected[analysis_frequencies > active_frequencies[-1]] = patch_value

        actual = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient._build_finite_psd_array(
            interferometer
        )

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_psd_patch_stays_conditioned_when_psd_file_stops_below_nyquist(self):
        """A PSD tabulated only up to the analysis band must not blow up the patch.

        Above ``maximum_frequency`` bilby leaves the PSD array non-finite when the
        source file does not reach Nyquist, and ``_build_finite_psd_array``
        substitutes ``max(source_psd)`` there.  Released LVK PSD files carry a
        large sentinel at DC, so that maximum is the sentinel, not a noise level:
        referring the upper patch to it rather than to the in-band level inflates
        the patch by tens of orders of magnitude, which makes the Toeplitz
        covariance indefinite and stops the likelihood being constructed at all.
        """
        interferometer = self.interferometers[0]
        interferometer.minimum_frequency = 20.0
        interferometer.maximum_frequency = 40.0

        # Reproduce what bilby_pipe hands the likelihood for a released LVK PSD
        # file: a large sentinel at DC to suppress it, a physical spectrum in
        # band, and non-finite entries above maximum_frequency because the file
        # is tabulated only up to the analysis band, not up to Nyquist.
        source_frequencies = np.asarray(interferometer.frequency_array, dtype=float)
        source_psd = np.full_like(source_frequencies, np.inf)
        physical = (source_frequencies > 0.0) & (source_frequencies <= 40.0)
        source_psd[physical] = 1e-46 * (20.0 / source_frequencies[physical]) ** 4
        source_psd[source_frequencies == 0.0] = 0.5
        interferometer.power_spectral_density = (
            bilby.gw.detector.PowerSpectralDensity(
                frequency_array=source_frequencies, psd_array=source_psd
            )
        )

        analysis_frequencies = np.asarray(interferometer.frequency_array, dtype=float)
        self.assertTrue(
            np.any(~np.isfinite(interferometer.power_spectral_density_array)),
            msg="the test needs a PSD that does not reach Nyquist",
        )

        patched = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient._build_finite_psd_array(
            interferometer
        )
        active = analysis_frequencies[interferometer.frequency_mask]
        in_band = patched[
            (analysis_frequencies >= active[0]) & (analysis_frequencies <= active[-1])
        ]
        above = patched[analysis_frequencies > active[-1]]

        self.assertTrue(np.all(np.isfinite(patched)))
        np.testing.assert_allclose(above, 10.0 * np.max(in_band))

        acf = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient._acf_from_psd(
            patched, interferometer
        )
        eigenvalues = np.linalg.eigvalsh(toeplitz(acf))
        self.assertGreater(eigenvalues.min(), 0.0)

        likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            interferometers=bilby.gw.detector.InterferometerList([interferometer]),
            waveform_generator=self.waveform_generator,
            likelihood_method="gohberg-semencul",
        )
        self.assertTrue(np.isfinite(likelihood.log_likelihood(self.parameters.copy())))

    def test_gohberg_semencul_inverse_matches_solve_toeplitz(self):
        acf = 1.7 * 0.7 ** np.arange(128, dtype=float)
        vector = np.sin(np.linspace(0.0, 5.0 * np.pi, len(acf)))

        expected = solve_toeplitz(acf, vector, check_finite=False)
        actual = _GohbergSemenculToeplitzInverse.from_acf(acf).matvec(vector)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_gohberg_semencul_likelihood_matches_toeplitz_likelihood(self):
        toeplitz_likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            likelihood_method="toeplitz-inversion",
        )
        gohberg_likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            likelihood_method="gohberg-semencul",
        )

        toeplitz_value = toeplitz_likelihood.log_likelihood(self.parameters.copy())
        gohberg_value = gohberg_likelihood.log_likelihood(self.parameters.copy())

        self.assertAlmostEqual(gohberg_value, toeplitz_value, places=10)

    def test_gohberg_semencul_banded_student_t_matches_toeplitz_likelihood(self):
        parameters = self.parameters.copy()
        parameters.update(
            nu_H1_0_0p5=6.0,
            nu_H1_0p5_2=8.0,
            nu_L1_0_0p5=10.0,
            nu_L1_0p5_2=12.0,
        )

        toeplitz_likelihood = (
            bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                infer_nu=True,
                detector_dependent_noise=True,
                time_bands=[0.5],
                likelihood_method="toeplitz-inversion",
            )
        )
        gohberg_likelihood = (
            bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                infer_nu=True,
                detector_dependent_noise=True,
                time_bands=[0.5],
                likelihood_method="gohberg-semencul",
            )
        )

        toeplitz_value = toeplitz_likelihood.log_likelihood(parameters.copy())
        gohberg_value = gohberg_likelihood.log_likelihood(parameters.copy())

        self.assertAlmostEqual(gohberg_value, toeplitz_value, places=10)

    def test_mixed_detector_likelihood_matches_manual_sum(self):
        likelihood = bilby.gw.likelihood.MixedTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            likelihood_type={"H1": "gaussian", "L1": "hyperbolic"},
            infer_alpha=True,
            infer_delta=True,
            detector_dependent_noise=True,
        )
        parameters = self.parameters.copy()
        parameters.update(alpha_L1=6.0, delta_L1=1.4)
        signal_parameters = likelihood._resolve_signal_likelihood_parameters(
            parameters.copy()
        )
        waveform_polarizations = likelihood._waveform_polarizations_frequency_domain(
            signal_parameters
        )

        calculated = likelihood.log_likelihood(parameters.copy())

        manual = 0.0
        for interferometer in self.interferometers:
            residuals = likelihood._residual_time_domain(
                interferometer=interferometer,
                parameters=signal_parameters,
                waveform_polarizations=waveform_polarizations,
            )
            cache = likelihood._detector_likelihood_caches[interferometer.name]["full"]
            residuals_inner_product = _residuals_inner_product_from_cache(
                residuals,
                cache,
                likelihood.likelihood_method,
            )
            if interferometer.name == "H1":
                manual += _gaussian_log_likelihood_from_inner_product(
                    residuals_inner_product=residuals_inner_product,
                    log_normalisation=cache.log_normalisation,
                )
            else:
                manual += _hyperbolic_log_likelihood_from_inner_product(
                    residuals_inner_product=residuals_inner_product,
                    logdet=cache.logdet,
                    dimension=len(residuals),
                    alpha=parameters["alpha_L1"],
                    delta=parameters["delta_L1"],
                )

        self.assertAlmostEqual(calculated, float(manual), 7)

    def test_mixed_detector_noise_parameter_keys_only_include_active_non_gaussian_detectors(self):
        likelihood = bilby.gw.likelihood.MixedTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            likelihood_type={"H1": "gaussian", "L1": "student-t"},
            infer_nu=True,
            detector_dependent_noise=True,
        )

        self.assertEqual(likelihood.noise_parameter_keys, ["nu", "nu_L1"])

    def test_parameter_api_is_explicit_and_noise_is_evaluation_order_independent(self):
        likelihood = bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
        )

        with self.assertRaises(TypeError):
            likelihood.log_likelihood()

        fixed_noise_log_likelihood = likelihood.noise_log_likelihood()
        parameters = self.parameters.copy()
        parameters["nu"] = 3.0
        likelihood.log_likelihood(parameters)

        self.assertFalse(hasattr(likelihood, "_parameters"))
        self.assertEqual(
            likelihood.noise_log_likelihood(), fixed_noise_log_likelihood
        )

    def test_invalid_noise_parameters_return_minus_infinity(self):
        likelihoods_and_parameters = [
            (
                bilby.gw.likelihood.StudentTTimeDomainGravitationalWaveTransient(
                    interferometers=self.interferometers,
                    waveform_generator=self.waveform_generator,
                    infer_nu=True,
                    joint=True,
                ),
                {"nu": -1.0},
            ),
            (
                bilby.gw.likelihood.HyperbolicTimeDomainGravitationalWaveTransient(
                    interferometers=self.interferometers,
                    waveform_generator=self.waveform_generator,
                    infer_alpha=True,
                    joint=True,
                ),
                {"alpha": -1.0},
            ),
            (
                bilby.gw.likelihood.MixedTimeDomainGravitationalWaveTransient(
                    interferometers=self.interferometers,
                    waveform_generator=self.waveform_generator,
                    likelihood_type={"H1": "student-t", "L1": "gaussian"},
                    infer_nu=True,
                    detector_dependent_noise=True,
                ),
                {"nu_H1": -1.0},
            ),
        ]
        minus_infinity = np.nan_to_num(-np.inf)

        for likelihood, noise_parameters in likelihoods_and_parameters:
            with self.subTest(likelihood=type(likelihood).__name__):
                parameters = self.parameters.copy()
                parameters.update(noise_parameters)
                self.assertEqual(
                    likelihood.log_likelihood_ratio(parameters), minus_infinity
                )
                result = likelihood.compute_per_detector_log_likelihood(parameters)
                for interferometer in self.interferometers:
                    self.assertEqual(
                        result[f"{interferometer.name}_log_likelihood"],
                        minus_infinity,
                    )

    def test_quadrature_reference_ignores_default_outside_prior(self):
        noise_priors = bilby.core.prior.PriorDict(
            dict(x=bilby.core.prior.Uniform(1.0, 2.0))
        )
        calculated = _factorized_noise_log_evidence_by_quadrature(
            blocks=[{"keys": ("x",)}],
            noise_priors=noise_priors,
            base_parameters={"x": 0.0},
            block_log_likelihood=lambda block, parameters: (
                -1000.0 * parameters["x"]
            ),
            epsabs=1e-10,
            epsrel=1e-10,
            limit=100,
            error_label="Test",
        )

        self.assertAlmostEqual(calculated, -1000.0 - np.log(1000.0), 7)

    def test_mixed_detector_values_are_subset_and_per_detector_terms_sum(self):
        likelihood = bilby.gw.likelihood.MixedTimeDomainGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            likelihood_type={"H1": "student-t", "L1": "hyperbolic"},
            nu=[[6.0, 7.0], [8.0, 9.0]],
            alpha=[[10.0, 11.0], [12.0, 13.0]],
            delta=[[1.0, 1.1], [1.5, 1.6]],
            detector_dependent_noise=True,
            time_band_boundaries={"H1": [0.25], "L1": [0.5]},
        )

        self.assertIsInstance(
            likelihood._student_t_likelihood.interferometers,
            bilby.gw.detector.InterferometerList,
        )
        np.testing.assert_array_equal(
            likelihood._student_t_likelihood._fixed_nu, [[6.0, 7.0]]
        )
        np.testing.assert_array_equal(
            likelihood._hyperbolic_likelihood._fixed_alpha, [[12.0, 13.0]]
        )
        np.testing.assert_array_equal(
            likelihood._hyperbolic_likelihood._fixed_delta, [[1.5, 1.6]]
        )

        per_detector = likelihood.compute_per_detector_log_likelihood(
            self.parameters.copy()
        )
        per_detector_total = sum(
            per_detector[f"{interferometer.name}_log_likelihood"]
            for interferometer in self.interferometers
        )
        self.assertAlmostEqual(
            per_detector_total,
            likelihood.log_likelihood_ratio(self.parameters.copy()),
            10,
        )

    def test_mixed_detector_mapping_rejects_unknown_detector(self):
        with self.assertRaisesRegex(ValueError, "unknown V1"):
            bilby.gw.likelihood.MixedTimeDomainGravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                likelihood_type={
                    "H1": "gaussian",
                    "L1": "student-t",
                    "V1": "hyperbolic",
                },
            )


if __name__ == "__main__":
    unittest.main()
