#!/usr/bin/env python
"""
Reduced-parameter CBC example using the time-domain likelihood.

This example keeps the inference problem small enough to run locally while
still using:

- a CBC waveform model
- H1/L1 design-sensitivity data
- a native time-domain waveform generator path
- the new covariance-based time-domain likelihood
"""

import bilby


duration = 2.0
sampling_frequency = 1024.0
minimum_frequency = 20.0

outdir = "outdir"
label = "time_domain_cbc_example"
bilby.core.utils.setup_logger(outdir=outdir, label=label, log_level="INFO")
bilby.core.utils.random.seed(123456)

injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.1,
    a_2=0.1,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,
    luminosity_distance=1500.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

waveform_arguments = dict(
    waveform_approximant="SEOBNRv5HM",
    reference_frequency=20.0,
    minimum_frequency=minimum_frequency,
    maximum_frequency=sampling_frequency / 2.0,
    catch_waveform_errors=True,
    lmax_nyquist=2,
)

waveform_generator = bilby.gw.waveform_generator.GWSignalWaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=injection_parameters["geocent_time"] - duration / 2.0,
    waveform_arguments=waveform_arguments,
)

ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - duration / 2.0,
)
ifos.inject_signal(
    waveform_generator=waveform_generator,
    parameters=injection_parameters,
    raise_error=False,
)

priors = bilby.gw.prior.BBHPriorDict()
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "psi",
    "ra",
    "dec",
    "geocent_time",
    "phase",
]:
    priors[key] = injection_parameters[key]

priors["chirp_mass"] = bilby.core.prior.Uniform(25.0, 35.0, name="chirp_mass")
priors["mass_ratio"] = bilby.core.prior.Uniform(0.6, 1.0, name="mass_ratio")
priors["luminosity_distance"] = bilby.core.prior.Uniform(
    1000.0, 2000.0, name="luminosity_distance"
)
priors.validate_prior(duration, minimum_frequency)

likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    likelihood_method="cholesky-solve-triangular",
    prefer_time_domain_waveform=True,
)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=32,
    walks=5,
    maxmcmc=200,
    dlogz=40,
    maxcall=1500,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    result_class=bilby.gw.result.CBCResult,
)

if len(result.posterior) > len(result.search_parameter_keys):
    result.plot_corner()
else:
    bilby.core.utils.logger.info(
        "Skipping corner plot because the smoke run returned too few posterior samples."
    )
