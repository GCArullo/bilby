import inspect
import unittest
from unittest.mock import MagicMock, patch

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

    def test_distance_marginalization_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "StudentTGravitationalWaveTransient does not support "
            "distance marginalization",
        ):
            bilby.gw.likelihood.StudentTGravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                distance_marginalization=True,
            )

    def test_log_likelihood_matches_direct_calculation(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
        )
        parameters = self.parameters.copy()

        calculated = likelihood.log_likelihood(parameters)

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
            manual += np.sum(const - 0.5 * (nu + 2.0) * np.log1p(abs2 / (nu * scale2)))

        self.assertAlmostEqual(calculated, float(manual), 7)

    def test_multiband_log_likelihood_matches_direct_calculation(self):
        band_nus = [6.0, 12.0, 20.0]
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=band_nus,
            num_frequency_bands=len(band_nus),
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
                    - 0.5 * (nu + 2.0) * np.log1p(abs2[band_mask] / (nu * scale2[band_mask]))
                )

        self.assertAlmostEqual(calculated, float(manual), 7)

    def test_joint_log_likelihood_matches_direct_calculation(self):
        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        nu = 8.0
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=self.waveform_generator,
            nu=nu,
            joint=True,
        )
        parameters = self.parameters.copy()

        pols = self.waveform_generator.frequency_domain_strain(self.parameters)
        signal_quadratic_form = 0.0
        noise_quadratic_form = 0.0
        log_scale2 = 0.0
        for interferometer in interferometers:
            mask = interferometer.frequency_mask
            h_f = interferometer.get_detector_response(pols, self.parameters)
            scale2 = (
                interferometer.power_spectral_density_array[mask]
                * self.duration
                / 4.0
            )
            signal_residual = interferometer.frequency_domain_strain[mask] - h_f[mask]
            noise_residual = interferometer.frequency_domain_strain[mask]
            signal_quadratic_form += (
                signal_residual.real ** 2 + signal_residual.imag ** 2
            ) / scale2
            noise_quadratic_form += (
                noise_residual.real ** 2 + noise_residual.imag ** 2
            ) / scale2
            log_scale2 += np.log(scale2)

        dimension = 4
        constant = (
            gammaln((nu + dimension) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * dimension * np.log(nu * np.pi)
            - log_scale2
        )
        manual_signal = np.sum(
            constant
            - 0.5
            * (nu + dimension)
            * np.log1p(signal_quadratic_form / nu)
        )
        manual_noise = np.sum(
            constant
            - 0.5
            * (nu + dimension)
            * np.log1p(noise_quadratic_form / nu)
        )

        self.assertAlmostEqual(
            likelihood.log_likelihood(parameters), float(manual_signal), 7
        )
        self.assertAlmostEqual(
            likelihood.noise_log_likelihood(), float(manual_noise), 7
        )
        self.assertAlmostEqual(
            likelihood.log_likelihood_ratio(parameters),
            float(manual_signal - manual_noise),
            7,
        )

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
        self.assertIn("nu", likelihood.noise_parameter_keys)

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
        self.assertIn("nu_1", likelihood.noise_parameter_keys)
        self.assertIn("nu_2", likelihood.noise_parameter_keys)

        parameters = self.parameters.copy()
        parameters["nu_1"] = 4.0
        parameters["nu_2"] = 30.0

        np.testing.assert_allclose(likelihood.nu(parameters), np.array([4.0, 30.0]))

        logl_split = likelihood.log_likelihood(parameters)
        parameters["nu_1"] = 30.0
        parameters["nu_2"] = 4.0
        logl_swapped = likelihood.log_likelihood(parameters)

        self.assertNotEqual(logl_split, logl_swapped)

    def test_meta_data_includes_studentt_configuration(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=[8.0, 12.0],
            infer_nu=True,
            num_frequency_bands=2,
        )

        self.assertIs(
            likelihood.meta_data["likelihood_class"],
            bilby.gw.likelihood.StudentTGravitationalWaveTransient,
        )
        self.assertEqual(likelihood.meta_data["nu"], [8.0, 12.0])
        self.assertTrue(likelihood.meta_data["infer_nu"])
        self.assertFalse(likelihood.meta_data["detector_dependent_noise"])
        self.assertFalse(likelihood.meta_data["detector_dependent_nu"])
        self.assertFalse(likelihood.meta_data["joint"])
        self.assertEqual(likelihood.meta_data["num_frequency_bands"], 2)
        self.assertEqual(likelihood.meta_data["noise_evidence_method"], "quadrature")

    def test_gaussian_limit_diagnostic_uses_nu_prior_maximum(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            infer_nu=True,
            num_frequency_bands=2,
        )
        priors = bilby.core.prior.PriorDict(
            dict(nu=bilby.core.prior.Uniform(2.1, 1000.0, name="nu"))
        )

        with patch("builtins.print") as print_mock:
            likelihood.print_gaussian_limit_diagnostic(priors)

        output = "\n".join(call.args[0] for call in print_mock.call_args_list)
        self.assertIn("Student-t Gaussian-limit diagnostic", output)
        self.assertIn("nu = 1000 (prior max)", output)
        self.assertIn("variance scale = 1.002 (+0.200%)", output)

    def test_invalid_nu_returns_negative_infinity(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
        )
        parameters = self.parameters.copy()
        parameters["nu"] = -1

        self.assertEqual(likelihood.log_likelihood(parameters), np.nan_to_num(-np.inf))

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

        self.assertEqual(likelihood.log_likelihood(parameters), np.nan_to_num(-np.inf))

    def test_noise_log_evidence_defaults_to_quadrature_for_sampled_nu(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
        )
        parameters = self.parameters.copy()
        parameters["nu"] = 5.0
        priors = bilby.core.prior.PriorDict(
            dict(nu=bilby.core.prior.Uniform(2.1, 50.0, name="nu"))
        )
        unit_grid = np.linspace(0.0, 1.0, 5001)
        nu_grid = priors["nu"].rescale(unit_grid)
        expected = np.log(
            np.trapezoid(np.exp(-0.5 * (nu_grid - 5.0) ** 2), unit_grid)
        )
        likelihood._noise_log_likelihood_from_parameters = (
            lambda parameters: -0.5 * (parameters["nu"] - 5.0) ** 2
        )

        with patch("bilby.core.sampler.run_sampler") as mock_run_sampler:
            noise_log_evidence = likelihood.noise_log_evidence(priors=priors, npool=3)

        self.assertAlmostEqual(noise_log_evidence, expected, 6)
        mock_run_sampler.assert_not_called()

    def test_noise_log_evidence_runs_auxiliary_nested_sampling_when_requested(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
            noise_evidence_method="nested",
        )
        parameters = self.parameters.copy()
        parameters["nu"] = 5.0
        priors = bilby.core.prior.PriorDict(
            dict(nu=bilby.core.prior.Uniform(2.1, 50.0, name="nu"))
        )
        mock_result = MagicMock(log_evidence=-123.4)

        with patch("bilby.core.sampler.run_sampler", return_value=mock_result) as mock_run_sampler:
            noise_log_evidence = likelihood.noise_log_evidence(priors=priors, npool=3)

        self.assertEqual(noise_log_evidence, mock_result.log_evidence)
        self.assertEqual(mock_run_sampler.call_args.kwargs["sampler"], "dynesty")
        self.assertEqual(mock_run_sampler.call_args.kwargs["npool"], 1)
        self.assertListEqual(
            list(mock_run_sampler.call_args.kwargs["priors"].keys()),
            ["nu"],
        )
        self.assertEqual(
            mock_run_sampler.call_args.kwargs["likelihood"].__class__.__name__,
            "_StudentTNoiseOnlyLikelihood",
        )

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
                likelihood._get_interferometer_frequency_band_masks(
                    interferometer
                ),
                axis=0,
            )
            np.testing.assert_array_equal(
                coverage,
                np.ones(np.sum(interferometer.frequency_mask), dtype=int),
            )

    def test_noise_log_evidence_uses_custom_sampler_controls_for_studentt(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
            noise_evidence_method="nested",
            noise_evidence_nlive=256,
            dlogz_noise=0.03,
        )
        parameters = self.parameters.copy()
        parameters["nu"] = 5.0
        priors = bilby.core.prior.PriorDict(
            dict(nu=bilby.core.prior.Uniform(2.1, 50.0, name="nu"))
        )
        mock_result = MagicMock(log_evidence=-123.4)

        with patch("bilby.core.sampler.run_sampler", return_value=mock_result) as mock_run_sampler:
            likelihood.noise_log_evidence(priors=priors, npool=3)

        self.assertEqual(mock_run_sampler.call_args.kwargs["npool"], 1)
        self.assertEqual(mock_run_sampler.call_args.kwargs["nlive"], 256)
        self.assertEqual(mock_run_sampler.call_args.kwargs["dlogz"], 0.03)

    def test_noise_log_evidence_uses_2d_quadrature_for_two_sampled_nu(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=[8.0, 12.0],
            infer_nu=True,
            num_frequency_bands=2,
        )
        parameters = self.parameters.copy()
        parameters["nu_1"] = 5.0
        parameters["nu_2"] = 10.0
        priors = bilby.core.prior.PriorDict(
            dict(
                nu_1=bilby.core.prior.Uniform(2.1, 50.0, name="nu_1"),
                nu_2=bilby.core.prior.Uniform(2.1, 50.0, name="nu_2"),
            )
        )
        unit_grid = np.linspace(0.0, 1.0, 5001)
        nu_1_grid = priors["nu_1"].rescale(unit_grid)
        nu_2_grid = priors["nu_2"].rescale(unit_grid)
        expected = (
            np.log(np.trapezoid(np.exp(-0.5 * (nu_1_grid - 5.0) ** 2), unit_grid))
            + np.log(np.trapezoid(np.exp(-0.5 * (nu_2_grid - 10.0) ** 2), unit_grid))
        )
        likelihood._noise_log_likelihood_from_parameters = (
            lambda parameters: (
                -0.5 * (parameters["nu_1"] - 5.0) ** 2
                -0.5 * (parameters["nu_2"] - 10.0) ** 2
            )
        )

        with patch("bilby.core.sampler.run_sampler") as mock_run_sampler:
            noise_log_evidence = likelihood.noise_log_evidence(priors=priors, npool=3)

        self.assertAlmostEqual(noise_log_evidence, expected, 6)
        mock_run_sampler.assert_not_called()

    def test_noise_log_evidence_quadrature_falls_back_to_nested_above_2d(self):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=[8.0, 12.0, 16.0],
            infer_nu=True,
            num_frequency_bands=3,
        )
        parameters = self.parameters.copy()
        parameters["nu_1"] = 5.0
        parameters["nu_2"] = 10.0
        parameters["nu_3"] = 20.0
        priors = bilby.core.prior.PriorDict(
            dict(
                nu_1=bilby.core.prior.Uniform(2.1, 50.0, name="nu_1"),
                nu_2=bilby.core.prior.Uniform(2.1, 50.0, name="nu_2"),
                nu_3=bilby.core.prior.Uniform(2.1, 50.0, name="nu_3"),
            )
        )
        mock_result = MagicMock(log_evidence=-123.4)

        with patch("bilby.core.sampler.run_sampler", return_value=mock_result) as mock_run_sampler:
            noise_log_evidence = likelihood.noise_log_evidence(priors=priors, npool=3)

        self.assertEqual(noise_log_evidence, mock_result.log_evidence)
        self.assertListEqual(
            list(mock_run_sampler.call_args.kwargs["priors"].keys()),
            ["nu_1", "nu_2", "nu_3"],
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

    def test_bilby_pipe_style_filtered_kwargs_preserve_frame_configuration(self):
        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        likelihood_kwargs = dict(
            interferometers=interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            joint=True,
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
                bilby.gw.likelihood.StudentTGravitationalWaveTransient.__init__
            ).args
        }

        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            **filtered_kwargs
        )
        self.assertEqual(likelihood._reference_frame_str, "L1H1")
        self.assertEqual(likelihood.time_reference, "L1")
        self.assertTrue(likelihood.joint)

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
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
        )
        parameters = self.parameters.copy()
        parameters["nu"] = 5.0

        log_likelihood = likelihood.log_likelihood(parameters)
        noise_log_likelihood = likelihood._noise_log_likelihood_from_parameters(
            parameters
        )
        log_likelihood_ratio = likelihood.log_likelihood_ratio(parameters)
        per_detector = likelihood.compute_per_detector_log_likelihood(parameters)

        self.assertAlmostEqual(
            log_likelihood - noise_log_likelihood, log_likelihood_ratio, 7
        )
        self.assertAlmostEqual(
            per_detector["H1_log_likelihood"], log_likelihood_ratio, 7
        )

    def test_detector_dependent_fixed_nu_matches_direct_calculation(self):
        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        detector_nus = np.array([[5.0, 10.0], [12.0, 20.0]])
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=self.waveform_generator,
            nu=detector_nus,
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
            abs2 = r.real ** 2 + r.imag ** 2

            for band_index, nu in enumerate(detector_nus[detector_index]):
                lower = band_edges[band_index]
                upper = band_edges[band_index + 1]
                if band_index == detector_nus.shape[1] - 1:
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
                    - 0.5 * (nu + 2.0) * np.log1p(abs2[band_mask] / (nu * scale2[band_mask]))
                )

        self.assertAlmostEqual(calculated, float(manual), 7)

    def test_detector_dependent_infer_nu_uses_detector_specific_parameters(self):
        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
            num_frequency_bands=2,
            detector_dependent_noise=True,
        )

        for key in ["nu_H1_1", "nu_H1_2", "nu_L1_1", "nu_L1_2"]:
            self.assertIn(key, likelihood.noise_parameter_keys)

        parameters = self.parameters.copy()
        parameters.update(
            {"nu_H1_1": 4.0, "nu_H1_2": 30.0, "nu_L1_1": 8.0, "nu_L1_2": 15.0}
        )

        np.testing.assert_allclose(
            likelihood.nu(parameters),
            np.array([[4.0, 30.0], [8.0, 15.0]]),
        )

        logl_a = likelihood.log_likelihood(parameters)
        parameters.update(
            {"nu_H1_1": 30.0, "nu_H1_2": 4.0, "nu_L1_1": 8.0, "nu_L1_2": 15.0}
        )
        logl_b = likelihood.log_likelihood(parameters)

        self.assertNotEqual(logl_a, logl_b)

    def test_default_infer_nu_parameter_names_remain_unchanged(self):
        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=self.waveform_generator,
            nu=8.0,
            infer_nu=True,
            num_frequency_bands=2,
        )

        self.assertIn("nu_1", likelihood.noise_parameter_keys)
        self.assertIn("nu_2", likelihood.noise_parameter_keys)
        self.assertNotIn("nu_H1_1", likelihood.noise_parameter_keys)
        self.assertNotIn("nu_L1_1", likelihood.noise_parameter_keys)

    def test_gw231123_style_student_t_injection_prefers_student_model(self):
        duration = 8.0
        sampling_frequency = 1024.0
        trigger_time = 1384782888.634277
        start_time = trigger_time + 2.0 - duration
        nu = 4.5
        injection_parameters = dict(
            mass_1=32.0,
            mass_2=28.0,
            a_1=0.4,
            a_2=0.2,
            tilt_1=0.3,
            tilt_2=0.4,
            phi_12=1.1,
            phi_jl=0.2,
            luminosity_distance=1800.0,
            theta_jn=0.5,
            psi=2.1,
            phase=1.2,
            geocent_time=trigger_time,
            ra=1.3,
            dec=-1.0,
        )

        bilby.core.utils.random.seed(12345)
        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        for interferometer in interferometers:
            interferometer.minimum_frequency = 20.0
            interferometer.maximum_frequency = 448.0
        interferometers.set_strain_data_from_power_spectral_densities_student_t(
            sampling_frequency=sampling_frequency,
            duration=duration,
            nu=nu,
            start_time=start_time,
        )

        waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(
                waveform_approximant="IMRPhenomXPHM",
                reference_frequency=20.0,
                minimum_frequency=20.0,
                maximum_frequency=448.0,
                catch_waveform_errors=True,
                pn_spin_order=-1,
                pn_tidal_order=-1,
                pn_phase_order=-1,
                pn_amplitude_order=0,
                mode_array=None,
            ),
        )
        interferometers.inject_signal(
            parameters=injection_parameters, waveform_generator=waveform_generator
        )

        gaussian_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
        )
        student_likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            nu=nu,
            num_frequency_bands=1,
            detector_dependent_noise=False,
        )

        shifted_parameters = injection_parameters.copy()
        shifted_parameters["luminosity_distance"] *= 1.8
        shifted_parameters["phase"] = (
            shifted_parameters["phase"] + 1.1
        ) % (2.0 * np.pi)

        self.assertGreater(
            gaussian_likelihood.log_likelihood_ratio(injection_parameters),
            gaussian_likelihood.log_likelihood_ratio(shifted_parameters),
        )
        self.assertGreater(
            student_likelihood.log_likelihood_ratio(injection_parameters),
            student_likelihood.log_likelihood_ratio(shifted_parameters),
        )
        self.assertGreater(
            student_likelihood.noise_log_likelihood(),
            gaussian_likelihood.noise_log_likelihood(),
        )
        self.assertGreater(
            student_likelihood.log_likelihood(injection_parameters),
            gaussian_likelihood.log_likelihood(injection_parameters),
        )
