import unittest

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import solve_toeplitz

import bilby
from bilby.gw.likelihood.time_domain import (
    _GohbergSemenculToeplitzInverse,
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
            nu_H1_1=6.0,
            nu_H1_2=8.0,
            nu_L1_1=10.0,
            nu_L1_2=12.0,
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
                nu = parameters[f"nu_{interferometer.name}_{band_index + 1}"]
                manual += _student_t_log_likelihood_from_inner_product(
                    residuals_inner_product=residuals_inner_product,
                    logdet=cache.logdet,
                    dimension=cache.end - cache.start,
                    nu=nu,
                )

        self.assertAlmostEqual(calculated, float(manual), 7)

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
        parameters.update(alpha_1=6.0, alpha_2=12.0, delta_1=0.8, delta_2=1.4)
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
                    alpha=parameters[f"alpha_{band_index + 1}"],
                    delta=parameters[f"delta_{band_index + 1}"],
                )

        self.assertAlmostEqual(calculated, float(manual), 7)

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

        low_patch_value = 10.0 * float(
            np.max(
                expected[
                    (analysis_frequencies >= active_frequencies[0])
                    & (analysis_frequencies <= active_frequencies[-1])
                ]
            )
        )
        high_patch_value = 10.0 * float(
            np.max(expected[analysis_frequencies >= active_frequencies[-1]])
        )
        expected[analysis_frequencies < active_frequencies[0]] = low_patch_value
        expected[analysis_frequencies > active_frequencies[-1]] = high_patch_value

        actual = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient._build_finite_psd_array(
            interferometer
        )

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

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
            nu_H1_1=6.0,
            nu_H1_2=8.0,
            nu_L1_1=10.0,
            nu_L1_2=12.0,
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

        self.assertEqual(likelihood.noise_parameter_keys, ["nu_L1"])


if __name__ == "__main__":
    unittest.main()
