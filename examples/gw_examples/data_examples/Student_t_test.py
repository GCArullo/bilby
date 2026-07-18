#!/usr/bin/env python
import bilby

from gwpy.timeseries import TimeSeries

"""

Tutorial to demonstrate running parameter estimation on GW150914 using either a
Student-t or hyperbolic likelihood instead of the standard Gaussian likelihood.

"""

########################
# Standard user inputs #
########################

event                = "GW150914"                   # Available options: ["GW150914", "GW231123"]
likelihood_type      = "Hyperbolic"                    # Available options: ["Gaussian", "Student", "Hyperbolic"]
outdir_label         = "test_multiband_N2"     # Label for output directory, e.g. "test", "test_fixed_nu", "test_GW231123"
single_par           = True           

waveform_approximant = "IMRPhenomXPHM" # Waveform approximant to use. Must be supported by your version of LALSimulation. Examples: "IMRPhenomD", "IMRPhenomPv2", "IMRPhenomXPHM", "SEOBNRv4_ROM", etc.

nu_min, nu_max       = 2.1, 1000            # Range for uniform prior on nu (if infer_nu=True). Must be > 2 for finite variance in 2D.
alpha_min, alpha_max = 1e-6, 30             # HyperWave-style uniform prior range for alpha (if infer_alpha=True).
delta_min, delta_max = 1e-6, 30             # HyperWave-style uniform prior range for delta (if infer_delta=True).
num_frequency_bands  = 2                    # Number of frequency bands. For N > 1, sample nu_i or alpha_i for each band.

location_type        = "sky" # Available options: ["sky", "L1"]. This sets the reference frame and time reference for the likelihood. "sky" uses the standard geocentric frame and time, while "L1" uses the L1 frame and time. The latter is a non-inertial frame which can cause issues with the standard bilby likelihood, but should work fine with the heavy-tailed likelihoods used here.

if(location_type == "sky"):

    reference_frame="sky"
    time_reference="geocent"

elif(location_type == "L1"):

    reference_frame="L1H1"
    time_reference="L1"

else:
    raise ValueError("location_type must be one of ['sky', 'L1']")

############################
# End standard user inputs #
############################

#============================================================================================================================================

##################################
# Data settings and conditioning #
##################################

# Note you can get trigger times using the gwosc package, e.g.:
# > from gwosc import datasets
# > datasets.event_gps("GW150914")
trigger_times         = {"GW150914": 1126259462.41, "GW231123": 1384782888.634}
trigger_time          = trigger_times[event]

detectors             = ["H1", "L1"]
minimum_frequency     = 20
maximum_frequency     = 512
duration              = 4  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
roll_off              = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
end_time              = trigger_time + post_trigger_duration
start_time            = end_time - duration

psd_duration   = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time   = start_time


################
# I/O settings #
################

logger = bilby.core.utils.logger
outdir = f"Runs/{event}_{likelihood_type}_{outdir_label}_{waveform_approximant}_single_par_{single_par}"
label  = f"{event}"


################
# Data and PSD #
################

# We now use gwpy to obtain analysis and psd data and create the ifo_list
ifo_list = bilby.gw.detector.InterferometerList([])

for det in detectors:

    logger.info("Downloading analysis data for ifo {}".format(det))

    ifo  = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info("Downloading psd data for ifo {}".format(det))

    psd_data  = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time, cache=True)
    psd_alpha = 2 * roll_off / duration
    psd       = psd_data.psd(fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median")

    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=psd.frequencies.value, psd_array=psd.value)
    ifo.maximum_frequency      = maximum_frequency
    ifo.minimum_frequency      = minimum_frequency
    ifo_list.append(ifo)

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)


##########
# Priors #
##########

# We have defined our prior distribution in a local file, GW150914.prior
# The prior is printed to the terminal at run-time.
# You can overwrite this using the syntax below in the file,
# or choose a fixed value by just providing a float value as the prior.

# # Add the geocent time prior
if(single_par): priors = bilby.gw.prior.BBHPriorDict(filename=f"Priors/{event}_single_par.prior")
else          : priors = bilby.gw.prior.BBHPriorDict(filename=f"Priors/{event}.prior")
# priors["geocent_time"] = bilby.core.prior.Uniform(trigger_time - 0.1, trigger_time + 0.1, name="geocent_time")

############
# Waveform #
############

# In this step we define a `waveform_generator`. This is the object which
# creates the frequency-domain strain. In this instance, we are using the
# `lal_binary_black_hole model` source model. We also pass other parameters:
# the waveform approximant and reference frequency and a parameter conversion
# which allows us to sample in chirp mass and ratio rather than component mass
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,
    parameter_conversion          = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments            = {"waveform_approximant": waveform_approximant, "reference_frequency": 50})

##############
# Likelihood #
##############

if(likelihood_type == "Gaussian"):
    # In this step, we define the likelihood. Here we use the standard likelihood
    # function, passing it the data and the waveform generator.
    # Note, phase_marginalization is formally invalid with a precessing waveform such as IMRPhenomPv2
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator,
        priors                      = priors,

        # time_marginalization=True,
        # distance_marginalization=True,

        phase_marginalization       = False,
        time_marginalization        = False,
        distance_marginalization    = False,
        calibration_marginalization = False,

        reference_frame             = reference_frame,
        time_reference              = time_reference,

    )

elif("Student" in likelihood_type):

    # Prior must be > 2 for finite variance in 2D
    if num_frequency_bands == 1:
        priors["nu"]                   = bilby.core.prior.Uniform(nu_min, nu_max, name="nu"              )
    else:
        for band_index in range(1, num_frequency_bands + 1):
            priors[f"nu_{band_index}"] = bilby.core.prior.Uniform(nu_min, nu_max, name=f"nu_{band_index}")

    likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
    
    interferometers             = ifo_list,
    waveform_generator          = waveform_generator,

    nu                          = 8.0,  # initial/fixed value
    infer_nu                    = True, # False to fix it
    num_frequency_bands         = num_frequency_bands,

    phase_marginalization       = False,
    time_marginalization        = False,
    distance_marginalization    = False,
    calibration_marginalization = False,

    reference_frame=reference_frame,
    time_reference=time_reference,

    )

elif("Hyperbolic" in likelihood_type):

    if num_frequency_bands == 1:
        priors["alpha"] = bilby.core.prior.Uniform(alpha_min, alpha_max, name="alpha")
        priors["delta"] = bilby.core.prior.Uniform(delta_min, delta_max, name="delta")
    else:
        for band_index in range(1, num_frequency_bands + 1):
            priors[f"alpha_{band_index}"] = bilby.core.prior.Uniform(
                alpha_min, alpha_max, name=f"alpha_{band_index}"
            )
            priors[f"delta_{band_index}"] = bilby.core.prior.Uniform(
                delta_min, delta_max, name=f"delta_{band_index}"
            )

    likelihood = bilby.gw.likelihood.HyperbolicGravitationalWaveTransient(

    interferometers             = ifo_list,
    waveform_generator          = waveform_generator,

    alpha                       = 10.0,  # initial/fixed value
    delta                       = 1.0,   # initial/fixed value
    infer_alpha                 = True,  # False to fix it
    infer_delta                 = True,  # False to fix it
    num_frequency_bands         = num_frequency_bands,

    phase_marginalization       = False,
    time_marginalization        = False,
    distance_marginalization    = False,
    calibration_marginalization = False,

    reference_frame             = reference_frame,
    time_reference              = time_reference,

    )

else:
    raise ValueError("likelihood_type must be one of ['Gaussian', 'Student', 'Hyperbolic']")


###########
# Sampler #
###########

# Finally, we run the sampler. This function takes the likelihood and prior
# along with some options for how to do the sampling and how to save the data
result = bilby.run_sampler(
    likelihood,
    priors,
    sampler             = "dynesty",

    # sample              = 'rslice',
    # bound               = "balls",
    # slices              = 20,

    nlive               = 64,
    n_check_point       = 200,
    outdir              = outdir,
    label               = label,
    #check_point_delta_t = 600,
    check_point_plot    = True,
    npool               = 1,
    conversion_function = bilby.gw.conversion.generate_all_bbh_parameters,
    result_class        = bilby.gw.result.CBCResult,

    clean               = True, # Overwrite existing output directory if it exists. Set to False to append to existing directory (e.g. for multiple runs with different samplers or settings
)

#########
# Plots #
#########

result.plot_corner()
