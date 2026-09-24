import unittest

import bilby
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln


class TestStudentTGWTransient(unittest.TestCase):
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

    def test_log_likelihood_matches_direct_calculation(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
        )
        calculated = likelihood.log_likelihood(self.parameters)

        pols = self.waveform_generator.frequency_domain_strain(self.parameters)
        manual = 0.0
        nu = 8.0
        for ifo in self.interferometers:
            h_f = ifo.get_detector_response(pols, self.parameters)
            mask = ifo.frequency_mask
            r = ifo.frequency_domain_strain[mask] - h_f[mask]
            scale2 = ifo.power_spectral_density_array[mask] * self.duration / 4.0
            abs2 = r.real ** 2 + r.imag ** 2
            const = (
                gammaln((nu + 2.0) / 2.0)
                - gammaln(nu / 2.0)
                - np.log(nu * np.pi * scale2)
            )
            manual += np.sum(
                const - 0.5 * (nu + 2.0) * np.log1p(abs2 / (nu * scale2))
            )

        self.assertAlmostEqual(calculated, float(manual), 7)

    def test_multiband_log_likelihood_matches_direct_calculation(self):
        band_nus = [6.0, 12.0, 20.0]
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=band_nus,
            num_frequency_bands=len(band_nus),
        )
        calculated = likelihood.log_likelihood(self.parameters)

        pols = self.waveform_generator.frequency_domain_strain(self.parameters)
        manual = 0.0
        band_edges = likelihood._frequency_band_edges
        for ifo in self.interferometers:
            h_f = ifo.get_detector_response(pols, self.parameters)
            mask = ifo.frequency_mask
            frequencies = ifo.frequency_array[mask]
            r = ifo.frequency_domain_strain[mask] - h_f[mask]
            scale2 = ifo.power_spectral_density_array[mask] * self.duration / 4.0
            abs2 = r.real ** 2 + r.imag ** 2

            for band_index, nu in enumerate(band_nus):
                lower = band_edges[band_index]
                upper = band_edges[band_index + 1]
                if band_index == len(band_nus) - 1:
                    band_mask = (frequencies >= lower) & (frequencies <= upper)
                else:
                    band_mask = (frequencies >= lower) & (frequencies < upper)

                const = (
                    gammaln((nu + 2.0) / 2.0)
                    - gammaln(nu / 2.0)
                    - np.log(nu * np.pi * scale2[band_mask])
                )
                manual += np.sum(
                    const
                    - 0.5
                    * (nu + 2.0)
                    * np.log1p(abs2[band_mask] / (nu * scale2[band_mask]))
                )

        self.assertAlmostEqual(calculated, float(manual), 7)

    def test_large_nu_matches_gaussian_absolute_likelihood(self):
        gaussian_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
        )
        student_likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=1e15,
        )

        parameter_points = [
            self.parameters.copy(),
            {**self.parameters, "luminosity_distance": 2000.0},
            {**self.parameters, "luminosity_distance": 8000.0},
        ]
        for parameters in parameter_points:
            self.assertAlmostEqual(
                student_likelihood.log_likelihood(parameters),
                gaussian_likelihood.log_likelihood(parameters),
                delta=1e-9,
            )

    def test_large_nu_matches_gaussian_likelihood_ratio_to_machine_precision(self):
        gaussian_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
        )
        student_likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=1e15,
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
                student_likelihood.log_likelihood_ratio(parameters),
                gaussian_likelihood.log_likelihood_ratio(parameters),
                delta=5e-11,
            )

    def test_infer_nu_uses_parameter_dict(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
        )
        parameters = self.parameters.copy()
        parameters["nu"] = 5.0
        self.assertEqual(likelihood.nu(parameters), 5.0)

        logl_nu5 = likelihood.log_likelihood(parameters)
        parameters["nu"] = 30.0
        logl_nu30 = likelihood.log_likelihood(parameters)
        self.assertNotEqual(logl_nu5, logl_nu30)

    def test_infer_nu_uses_per_band_parameters(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
            num_frequency_bands=2,
        )
        self.assertEqual(likelihood.nu_parameter_keys, ["nu_1", "nu_2"])

        parameters = self.parameters.copy()
        parameters["nu_1"] = 4.0
        parameters["nu_2"] = 30.0

        np.testing.assert_allclose(likelihood.nu(parameters), np.array([4.0, 30.0]))

        logl_split = likelihood.log_likelihood(parameters)
        parameters["nu_1"] = 30.0
        parameters["nu_2"] = 4.0
        logl_swapped = likelihood.log_likelihood(parameters)

        self.assertNotEqual(logl_split, logl_swapped)

    def test_noise_log_evidence_uses_fixed_nu_prior(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
            num_frequency_bands=2,
        )
        likelihood._noise_log_likelihood_from_parameters = (
            lambda parameters: -np.sum(
                likelihood._get_nu_values(parameters) ** 2
            )
        )
        priors = bilby.core.prior.PriorDict(
            dict(nu=bilby.core.prior.DeltaFunction(5.0, name="nu"))
        )

        self.assertIn("nu", likelihood.noise_parameter_keys)
        self.assertEqual(likelihood.noise_log_evidence(priors=priors), -50.0)

    def test_noise_log_evidence_uses_fixed_nu_in_quadrature(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
            num_frequency_bands=2,
        )
        likelihood._noise_log_likelihood_from_parameters = lambda parameters: (
            -parameters["nu_1"] ** 2 - parameters["nu_2"] ** 2
        )
        priors = bilby.core.prior.PriorDict(
            dict(
                nu_1=bilby.core.prior.DeltaFunction(5.0, name="nu_1"),
                nu_2=bilby.core.prior.Uniform(0.0, 1.0, name="nu_2"),
            )
        )
        integral, _ = quad(lambda value: np.exp(-value ** 2), 0.0, 1.0)

        self.assertAlmostEqual(
            likelihood.noise_log_evidence(priors=priors),
            -25.0 + np.log(integral),
            7,
        )

    def test_frequency_bands_cover_all_interferometer_analysis_bins(self):
        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        interferometers.set_strain_data_from_zero_noise(
            sampling_frequency=128,
            duration=2,
        )
        interferometers[0].minimum_frequency = 30
        interferometers[0].maximum_frequency = 40
        interferometers[1].minimum_frequency = 20
        interferometers[1].maximum_frequency = 50
        waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=2,
            sampling_frequency=128,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        )
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            nu=8.0,
            num_frequency_bands=2,
        )

        np.testing.assert_allclose(
            likelihood._frequency_band_edges,
            [20.0, 35.0, 50.0],
        )
        for interferometer in interferometers:
            coverage = np.sum(
                likelihood._get_frequency_band_masks(interferometer),
                axis=0,
            )
            np.testing.assert_array_equal(
                coverage,
                np.ones(np.sum(interferometer.frequency_mask), dtype=int),
            )

    def test_invalid_nu_returns_negative_infinity(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
        )
        parameters = self.parameters.copy()
        parameters["nu"] = -1

        self.assertEqual(
            likelihood.log_likelihood(parameters), np.nan_to_num(-np.inf)
        )

    def test_invalid_per_band_nu_returns_negative_infinity(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
            num_frequency_bands=2,
        )
        parameters = self.parameters.copy()
        parameters["nu_1"] = 8.0
        parameters["nu_2"] = -1.0

        self.assertEqual(
            likelihood.log_likelihood(parameters), np.nan_to_num(-np.inf)
        )

    def test_time_reference_agrees_with_default(self):
        default_likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
        )
        h1_time_likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
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

    def test_log_likelihood_ratio_matches_noise_subtraction_and_per_detector_output(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
        )
        parameters = self.parameters.copy()
        parameters["nu"] = 5.0

        log_likelihood_ratio = likelihood.log_likelihood_ratio(parameters)
        per_detector = likelihood.compute_per_detector_log_likelihood(parameters)

        self.assertAlmostEqual(
            likelihood.log_likelihood(parameters)
            - likelihood._noise_log_likelihood_from_parameters(parameters),
            log_likelihood_ratio,
            7,
        )
        self.assertAlmostEqual(
            per_detector["H1_log_likelihood"], log_likelihood_ratio, 7
        )
