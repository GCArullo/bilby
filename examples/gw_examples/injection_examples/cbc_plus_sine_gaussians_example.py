#!/usr/bin/env python

"""

Example injection of a CBC signal with two sine-Gaussian bursts.

This script demonstrates how to run parameter estimation on a compact
binary coalescence (CBC) signal augmented with a set of sine-Gaussian bursts.
By default the sine-Gaussians are treated coherently (projected through the
antenna patterns like the CBC). Activating ``incoherent`` assigns detector-
local burst components that are added directly to each interferometer's strain.

"""

###########
# Imports # 
###########

import bilby
from bilby.core.utils.random import seed


###################
# Utils functions # 
###################

def flatten_sine_gaussians(components, detector=None):

    """Translate sine-Gaussian dicts into flat parameter names."""

    flat = {}
    for index, component in enumerate(components):
        prefix = f"sine_gaussian_{index}_"
        if detector is not None:
            prefix += f"{detector}_"
        flat.update({
            f"{prefix}hrss": component["hrss"],
            f"{prefix}Q": component["Q"],
            f"{prefix}frequency": component["frequency"],
            f"{prefix}time_offset": component["time_offset"],
            f"{prefix}phase_offset": component["phase_offset"],
        })

    return flat

def populate_sine_gaussian_priors(components, detector=None):

    for index, component in enumerate(components):
        prefix = f"sine_gaussian_{index}_"
        if detector is not None:
            prefix += f"{detector}_"
        add_sine_gaussian_priors(prefix, component)

    return


###############
# User inputs # 
###############

# Miscellaneous parameters
incoherent = False # If True, use detector-local sine-Gaussians
zero_noise = True
outdir     = "outdir"
nlive      = 64
seed_value = 123

# Detectors and time-series parameters
detectors_list       = ["H1", "L1"]
DURATION             = 4.0
SAMPLING_FREQUENCY   = 1024.0
f_min_bp, f_max_bp   = 30.0, 300.0 # Bandpassing frequencies for plotting
waveform_approximant = "IMRPhenomXPHM"
f_min                = 20.0   # Waveform minimum frequency
f_ref                = 50.0   # Waveform reference frequency

########################
# Injection parameters # 
########################

# CBC parameters are shared between the coherent and incoherent setups
cbc_parameters = dict(
    mass_1              = 36.0,
    mass_2              = 29.0,
    a_1                 = 0.3,
    a_2                 = 0.2,
    tilt_1              = 0.5,
    tilt_2              = 1.1,
    phi_12              = 1.7,
    phi_jl              = 0.3,
    luminosity_distance = 1200.0,
    theta_jn            = 0.4,
    psi                 = 2.659,
    phase               = 1.3,
    geocent_time        = 1126259642.413,
    ra                  = 1.375,
    dec                 =-1.2108,
)

# Two representative sine-Gaussian bursts when treated coherently
coherent_sine_gaussians = [
        dict(hrss=1e-22, Q=8.0, frequency=70.0, time_offset=-0.07, phase_offset=0.0),
        # dict(hrss=5e-23, Q=9.0, frequency=120.0, time_offset=0.02, phase_offset=1.0),
]

# Detector-local sine-Gaussians for the incoherent case (two bursts per detector)
incoherent_sine_gaussians = {
    "H1": [
        dict(hrss=1e-22, Q=8.0, frequency=70.0, time_offset=-0.07, phase_offset=0.0),
        # dict(hrss=4.0e-23, Q=9.0, frequency=115.0, time_offset= 0.03, phase_offset= 0.8),
    ],
    "L1": [
        dict(hrss=1e-22, Q=8.0, frequency=75.0, time_offset=-0.07, phase_offset=0.0),
        # dict(hrss=4.5e-23, Q=9.0, frequency=125.0, time_offset= 0.01, phase_offset= 1.2),
    ],
}

# Check that the detectors list matches the incoherent sine-Gaussians keys
if incoherent: assert set(detectors_list) == set(incoherent_sine_gaussians.keys()), ("Detectors list does not match incoherent sine-Gaussians keys.")

#############
# I/O setup # 
#############

label  = ("cbc_plus_sine_gaussians_incoherent" if incoherent else "cbc_plus_sine_gaussians")
bilby.core.utils.setup_logger(outdir=outdir, label=label)

seed(seed_value) # Set seed to ensure reproducibility


###################
# Injected signal # 
###################

# Build the injection parameters in the sampling space so they are converted by the waveform generator into the structured input expected by the source model.
injection_parameters = cbc_parameters.copy()

if incoherent:
    for detector, components in incoherent_sine_gaussians.items():
        injection_parameters.update(flatten_sine_gaussians(components, detector=detector))
else:
    injection_parameters.update(flatten_sine_gaussians(coherent_sine_gaussians))

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant = waveform_approximant,
    reference_frequency  = f_ref,
    minimum_frequency    = f_min,
)

waveform_generator = bilby.gw.WaveformGenerator(
    duration           = DURATION,
    sampling_frequency = SAMPLING_FREQUENCY,
    waveform_arguments = waveform_arguments,

    # CBC + sinegaussians case
    frequency_domain_source_model = bilby.gw.source.cbc_plus_sine_gaussians,
    parameter_conversion          = bilby.gw.conversion.convert_to_cbc_plus_sine_gaussian_parameters,

    # BBH case
    # frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    # parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
)


###############################################
# Construct simulated strain (noise + signal) # 
###############################################

ifos = bilby.gw.detector.InterferometerList(detectors_list)

# Set noise realization
if(zero_noise):
    ifos.set_strain_data_from_zero_noise(
    sampling_frequency=SAMPLING_FREQUENCY,
    duration=DURATION,
    start_time=cbc_parameters["geocent_time"] - DURATION / 2,
)
else:
    ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=SAMPLING_FREQUENCY,
    duration=DURATION,
    start_time=cbc_parameters["geocent_time"] - DURATION / 2,
)

# Add signal
ifos.inject_signal(
    waveform_generator=waveform_generator,
    parameters=injection_parameters,
)


##########
# Priors # 
##########

#----------------------#
# Extrinsic parameters # 
#----------------------#

priors = bilby.gw.prior.BBHPriorDict(dictionary={})
for fixed_key in ["psi", "ra", "dec", "geocent_time", "theta_jn"]:
    priors[fixed_key] = injection_parameters[fixed_key]

#----------------#
# CBC parameters # 
#----------------#

priors["luminosity_distance"] = injection_parameters["luminosity_distance"]
priors["mass_1"]              = injection_parameters["mass_1"]
priors["mass_2"]              = injection_parameters["mass_2"]
priors["a_1"]                 = injection_parameters["a_1"]
priors["a_2"]                 = injection_parameters["a_2"]
priors["tilt_1"]              = injection_parameters["tilt_1"]
priors["tilt_2"]              = injection_parameters["tilt_2"]
priors["phi_12"]              = injection_parameters["phi_12"]
priors["phi_jl"]              = injection_parameters["phi_jl"]
priors["phase"]               = injection_parameters["phase"]

# priors["mass_1"] = bilby.core.prior.Uniform(25, 45, name="mass_1", unit="$M_\odot$")
# priors["mass_2"] = bilby.core.prior.Uniform(20, 40, name="mass_2", unit="$M_\odot$")
# priors["a_1"]    = bilby.core.prior.Uniform(0, 0.8, name="a_1")
# priors["a_2"]    = bilby.core.prior.Uniform(0, 0.8, name="a_2")
# priors["tilt_1"] = bilby.core.prior.Sine(name="tilt_1")
# priors["tilt_2"] = bilby.core.prior.Sine(name="tilt_2")
# priors["phi_12"] = bilby.core.prior.Uniform(
#     minimum=0, maximum=2 * bilby.core.utils.np.pi, boundary="periodic", name="phi_12"
# )
# priors["phi_jl"] = bilby.core.prior.Uniform(
#     minimum=0, maximum=2 * bilby.core.utils.np.pi, boundary="periodic", name="phi_jl"
# )
# priors["phase"] = bilby.core.prior.Uniform(
#     minimum=0, maximum=2 * bilby.core.utils.np.pi, boundary="periodic", name="phase"
# )

#---------------------------#
# Sine-gaussians parameters # 
#---------------------------#

def add_sine_gaussian_priors(prefix, component):

    priors[f"{prefix}frequency"] = bilby.core.prior.Uniform(
        minimum=40, maximum=200, name=f"{prefix}frequency", unit="Hz"
    )
    # priors[f"{prefix}Q"] = bilby.core.prior.Uniform(
    #     minimum=5.0, maximum=15.0, name=f"{prefix}Q"
    # )
    # priors[f"{prefix}time_offset"] = bilby.core.prior.Uniform(
    #     minimum=-0.1, maximum=0.1, name=f"{prefix}time_offset", unit="s"
    # )
    # priors[f"{prefix}phase_offset"] = bilby.core.prior.Uniform(
    #     minimum=-bilby.core.utils.np.pi,
    #     maximum=bilby.core.utils.np.pi,
    #     name=f"{prefix}phase_offset",
    #     boundary="periodic",
    # )

    # priors[f"{prefix}frequency"]    = injection_parameters[f"{prefix}frequency"]
    priors[f"{prefix}hrss"]         = injection_parameters[f"{prefix}hrss"]
    priors[f"{prefix}Q"]            = injection_parameters[f"{prefix}Q"]
    priors[f"{prefix}time_offset"]  = injection_parameters[f"{prefix}time_offset"]
    priors[f"{prefix}phase_offset"] = injection_parameters[f"{prefix}phase_offset"]

    return

if incoherent:
    for detector, components in incoherent_sine_gaussians.items():
        populate_sine_gaussian_priors(components, detector=detector)
else:
    populate_sine_gaussian_priors(coherent_sine_gaussians)


##############
# Likelihood # 
##############

# Initialise the likelihood by passing in the interferometer data (IFOs) and the
# waveform generator
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers    = ifos,
    waveform_generator = waveform_generator,
)


############
# Sampling # 
############

result = bilby.run_sampler(
    likelihood = likelihood,
    priors     = priors,
    sampler    = "dynesty",
    nlive      = nlive, 

    # sinegaussians main defaults
    # walks=10,
    # nact=5,

    # New default in bilby tutorials
    # naccept=60,
    # sample="acceptance-walk",

    # Bilby TGR tutorial defaults
    sample               = 'rslice',
    slices               = 20,

    n_check_point        = 200, 
    check_point_plot     = True,
    injection_parameters = injection_parameters,
    outdir               = outdir,
    label                = label,
    result_class         = bilby.gw.result.CBCResult,
)


###################
# Post-processing # 
###################

# Make some plots of the outputs
result.plot_corner()
result.plot_waveform_posterior(interferometers=ifos)
ifos.plot_time_domain_data(
    outdir               = outdir,
    label                = label,
    t0                   = cbc_parameters["geocent_time"],
    bandpass_frequencies = (f_min_bp, f_max_bp),
)