import unittest

import numpy as np

import bilby


def constant_frequency_domain_source(frequency_array, amplitude, **kwargs):
    plus = amplitude * np.ones_like(frequency_array, dtype=complex)
    cross = np.zeros_like(frequency_array, dtype=complex)
    return dict(plus=plus, cross=cross)


class TestTimeFrequencyTiled(unittest.TestCase):
    def setUp(self):
        bilby.core.utils.random.seed(500)
        self.duration = 4
        self.sampling_frequency = 256
        self.parameters = dict(
            amplitude=1e-23, geocent_time=0.0, ra=1.1, dec=-0.7, psi=0.4
        )
        self.interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=constant_frequency_domain_source,
            parameter_conversion=bilby.gw.conversion.identity_map_conversion,
        )
        self.reference = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
        )

    def _tiled(self, **kwargs):
        return bilby.gw.likelihood.TimeFrequencyTiledGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            **kwargs,
        )

    def test_gaussian_limit_reproduces_whittle_for_every_tiling(self):
        """Whitening globally before tiling makes the Gaussian limit exact.

        The residual is whitened with the segment PSD, so the tiled series is
        white and Parseval returns the full-segment quadratic form for any set of
        chunk boundaries.  Any departure here means the tiling has become an
        approximation rather than a reparameterisation.
        """
        expected = self.reference.log_likelihood_ratio(self.parameters.copy())
        for boundaries, edges in (
            (None, None),
            ([2.0], None),
            ([1.0, 2.0, 3.0], None),
            ([1.9, 2.1], [20.0, 40.0, 128.0]),
            ([0.5, 1.5, 2.5, 3.5], [20.0, 30.0, 60.0, 128.0]),
        ):
            with self.subTest(boundaries=boundaries, edges=edges):
                likelihood = self._tiled(
                    time_band_boundaries=boundaries,
                    frequency_band_edges=edges,
                    noise_model="gaussian",
                )
                self.assertAlmostEqual(
                    likelihood.log_likelihood_ratio(self.parameters.copy()),
                    expected,
                    places=6,
                )

    def test_hyperbolic_and_student_recover_the_gaussian_limit(self):
        expected = self.reference.log_likelihood_ratio(self.parameters.copy())
        hyperbolic = self._tiled(
            time_band_boundaries=[1.9, 2.1],
            frequency_band_edges=[20.0, 40.0, 128.0],
            noise_model="hyperbolic", alpha=1e4, delta=1e4,
        )
        student = self._tiled(
            time_band_boundaries=[1.9, 2.1],
            frequency_band_edges=[20.0, 40.0, 128.0],
            noise_model="student", nu=1e6,
        )
        self.assertAlmostEqual(
            hyperbolic.log_likelihood_ratio(self.parameters.copy()), expected, places=1
        )
        self.assertAlmostEqual(
            student.log_likelihood_ratio(self.parameters.copy()), expected, places=1
        )

    def test_tile_keys_name_every_tile_by_time_and_frequency(self):
        likelihood = self._tiled(
            time_band_boundaries=[1.9, 2.1],
            frequency_band_edges=[20.0, 40.0, 128.0],
            noise_model="hyperbolic", infer_alpha=True, infer_delta=True,
        )
        keys = [key for key, *_ in likelihood.tile_keys("alpha")]
        self.assertEqual(len(keys), 3 * 2)
        self.assertIn("alpha_1p9_2p1_20_40", keys)
        self.assertEqual(len(likelihood.noise_parameter_keys), 12)

    def test_downweighting_one_tile_only_affects_that_tile(self):
        """A tile pushed to large variance must drop out, leaving the rest intact."""
        likelihood = self._tiled(
            time_band_boundaries=[1.9, 2.1],
            frequency_band_edges=[20.0, 40.0, 128.0],
            noise_model="hyperbolic", alpha=1e4, delta=1e4,
            infer_alpha=True, infer_delta=True,
        )
        baseline = likelihood.log_likelihood(self.parameters.copy())
        deleted = likelihood.log_likelihood(
            {**self.parameters, "alpha_1p9_2p1_20_40": 1e-3,
             "delta_1p9_2p1_20_40": 1e3}
        )
        self.assertLess(deleted, baseline)
        self.assertTrue(np.isfinite(deleted))

    def test_per_detector_edges_require_detector_dependent_noise(self):
        with self.assertRaises(ValueError):
            self._tiled(
                frequency_band_edges={"H1": [20.0, 128.0], "L1": [20.0, 128.0]},
                noise_model="hyperbolic",
            )

    def test_boundaries_must_lie_inside_the_segment_and_increase(self):
        for boundaries in ([0.0], [self.duration], [2.0, 1.0]):
            with self.subTest(boundaries=boundaries):
                with self.assertRaises(ValueError):
                    self._tiled(time_band_boundaries=boundaries)


if __name__ == "__main__":
    unittest.main()
