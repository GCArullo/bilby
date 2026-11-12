"""Exercise the installed Dali source and CBC+SG dispatch before publication."""
import numpy as np

from bilby.gw.source import cbc_plus_sine_gaussians, teobresums_dali_binary_black_hole
from bilby.gw.waveform_generator import WaveformGenerator


parameters = dict(
    mass_1=180., mass_2=120., luminosity_distance=2000.,
    a_1=.6, a_2=.5, tilt_1=.8, tilt_2=1.2,
    phi_12=.3, phi_jl=.4, theta_jn=.7, phase=.8,
)
for source in (teobresums_dali_binary_black_hole, cbc_plus_sine_gaussians):
    generator = WaveformGenerator(
        duration=8, sampling_frequency=1024, frequency_domain_source_model=source,
        waveform_arguments=dict(waveform_approximant='TEOBResumS_Dali_Ecc',
                                reference_frequency=10., minimum_frequency=20.,
                                maximum_frequency=448.),
    )
    waveforms = []
    for eccentricity, anomaly in ((0., 0.), (.3, .4), (.3, 2.), (.5, 3.)):
        waveform = generator.frequency_domain_strain(
            dict(parameters, eccentricity=eccentricity, true_anomaly=anomaly))
        assert waveform is not None
        assert all(len(h) == 4097 and np.isfinite(h).all() and np.any(h) for h in waveform.values())
        waveforms.append(waveform['plus'])
    assert all(np.linalg.norm(h - waveforms[0]) > 1e-23 for h in waveforms[1:])
    assert np.linalg.norm(waveforms[1] - waveforms[2]) > 1e-23
print('Dali circular/eccentric, precession and anomaly validation passed')
