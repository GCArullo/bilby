import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bilby.gw import source
from bilby.gw.waveform_generator import WaveformGenerator


PARAMETERS = dict(
    mass_1=180., mass_2=120., luminosity_distance=2000.,
    a_1=.6, a_2=.5, tilt_1=.8, tilt_2=1.2,
    phi_12=.3, phi_jl=.4, theta_jn=.7, phase=.8,
)


def test_dali_fft_preserves_long_waveform_and_epoch(monkeypatch):
    lal = pytest.importorskip('lal')
    lalsimulation = pytest.importorskip('lalsimulation')
    time = np.arange(6144) / 4096 - .73
    strain = np.sin(2 * np.pi * 32 * time) * np.exp(-time**2 / .05)
    received = {}

    def generate(parameters):
        received.update(parameters)
        return time, strain, 2 * strain

    monkeypatch.setitem(sys.modules, 'EOBRun_module', SimpleNamespace(EOBRunPy=generate))
    frequencies = np.arange(513.)
    waveform = source.teobresums_dali_binary_black_hole(frequencies, **PARAMETERS)
    tapered = lal.CreateREAL8Vector(len(strain))
    tapered.data[:] = strain
    lalsimulation.SimInspiralREAL8WaveTaper(tapered, lalsimulation.SIM_INSPIRAL_TAPER_START)
    for frequency in (20, 32, 41):
        expected = np.sum(tapered.data * np.exp(-2j * np.pi * frequency * time)) / 4096
        np.testing.assert_allclose(waveform['plus'][frequency], expected, atol=1e-14)
    np.testing.assert_allclose(waveform['cross'], 2 * waveform['plus'])
    assert received['q'] == 1.5
    assert received['use_mode_lm'] == list(range(9))
    assert received['inclination'] != PARAMETERS['theta_jn']


def test_real_dali_eccentricity_anomaly_and_sg_composition():
    pytest.importorskip('EOBRun_module')
    generator = WaveformGenerator(
        duration=8, sampling_frequency=1024,
        frequency_domain_source_model=source.cbc_plus_sine_gaussians,
        waveform_arguments=dict(waveform_approximant='TEOBResumS_Dali_Ecc',
                                reference_frequency=10., minimum_frequency=20.,
                                maximum_frequency=448.),
    )
    parameters = dict(PARAMETERS, eccentricity=.3, true_anomaly=.4)
    baseline = generator.frequency_domain_strain(parameters)
    circular = generator.frequency_domain_strain(dict(parameters, eccentricity=0.))
    other_anomaly = generator.frequency_domain_strain(dict(parameters, true_anomaly=2.))
    for polarization in baseline:
        assert np.isfinite(baseline[polarization]).all()
        assert np.linalg.norm(baseline[polarization] - circular[polarization]) > 1e-23
        assert np.linalg.norm(baseline[polarization] - other_anomaly[polarization]) > 1e-23
    sg = dict(hrss=1e-22, Q=5., frequency=40., time_offset=.03, phase_offset=.1)
    combined = generator.frequency_domain_strain(dict(parameters, sine_gaussian_parameters=[sg]))
    expected = source._add_sine_gaussians(generator.frequency_array, baseline, [sg])
    for polarization in baseline:
        np.testing.assert_allclose(combined[polarization], expected[polarization], atol=1e-35)
