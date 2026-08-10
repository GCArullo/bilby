import unittest

import bilby
import numpy as np
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
            scale2 = ifo.power_spectral_density_array[mask] / 2.0
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
