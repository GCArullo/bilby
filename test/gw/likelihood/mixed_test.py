import unittest

import bilby


class TestMixedGWTransient(unittest.TestCase):
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
        self.interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        )

    def test_gaussian_and_hyperbolic_detectors_match_manual_sum(self):
        mixed_likelihood = bilby.gw.likelihood.MixedGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            detector_likelihoods={"H1": "gaussian", "L1": "hyperbolic"},
            infer_alpha=True,
            infer_delta=True,
            detector_dependent_noise=True,
        )
        parameters = dict(self.parameters, alpha_L1=6.0, delta_L1=1.4)

        gaussian_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=bilby.gw.detector.InterferometerList(
                [self.interferometers[0]]
            ),
            waveform_generator=self.waveform_generator,
        )
        hyperbolic_likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=bilby.gw.detector.InterferometerList(
                [self.interferometers[1]]
            ),
            waveform_generator=self.waveform_generator,
            alpha=6.0,
            delta=1.4,
        )

        self.assertAlmostEqual(
            mixed_likelihood.log_likelihood(parameters),
            gaussian_likelihood.log_likelihood(parameters)
            + hyperbolic_likelihood.log_likelihood(parameters),
            7,
        )
        per_detector = mixed_likelihood.compute_per_detector_log_likelihood(parameters)
        self.assertAlmostEqual(
            mixed_likelihood.log_likelihood_ratio(parameters),
            per_detector["H1_log_likelihood"] + per_detector["L1_log_likelihood"],
            7,
        )
        self.assertEqual(mixed_likelihood.noise_parameter_keys, ["alpha_L1", "delta_L1"])

    def test_all_gaussian_detectors_match_standard_likelihood(self):
        mixed_likelihood = bilby.gw.likelihood.MixedGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            detector_likelihoods="gaussian",
        )
        gaussian_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
        )

        self.assertAlmostEqual(
            mixed_likelihood.log_likelihood(self.parameters),
            gaussian_likelihood.log_likelihood(self.parameters),
            7,
        )
        self.assertAlmostEqual(
            mixed_likelihood.noise_log_evidence(),
            mixed_likelihood.noise_log_likelihood(),
            7,
        )

    def test_all_hyperbolic_detectors_match_network_likelihood(self):
        mixed_likelihood = bilby.gw.likelihood.MixedGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            detector_likelihoods="hyperbolic",
            alpha=8.0,
            delta=1.5,
        )
        hyperbolic_likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            alpha=8.0,
            delta=1.5,
        )

        self.assertAlmostEqual(
            mixed_likelihood.log_likelihood(self.parameters),
            hyperbolic_likelihood.log_likelihood(self.parameters),
            7,
        )

    def test_student_t_parameters_are_limited_to_student_t_detectors(self):
        likelihood = bilby.gw.likelihood.MixedGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            detector_likelihoods={"H1": "student-t", "L1": "gaussian"},
            infer_nu=True,
            detector_dependent_noise=True,
        )

        self.assertEqual(likelihood.noise_parameter_keys, ["nu_H1"])
