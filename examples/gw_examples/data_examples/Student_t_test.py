#!/usr/bin/env python
import bilby, numpy as np

from gwpy.timeseries import TimeSeries
from scipy.special import gammaln

"""

Tutorial to demonstrate running parameter estimation on GW150914 using a Student-t likelihood instead of the standard Gaussian likelihood.

"""

# Local likelihood implementation
class StudentTGravitationalWaveTransient(bilby.gw.likelihood.base.GravitationalWaveTransient):

    """
    A simple heavy-tailed replacement for the standard Gaussian (Whittle) likelihood.

    Model: per-frequency-bin complex Student-t for the residual r_k = d_k - h_k, with
    scale set by the one-sided PSD S_n(f).
    """

    def __init__(self, interferometers, waveform_generator, nu=8.0, infer_nu=False, **kwargs):

        """
        Parameters
        ----------
        nu : float
            Student-t degrees of freedom. Smaller => heavier tails. nu -> infinity gives Gaussian.
        infer_nu : bool
            If True, treat 'nu' as a sampled parameter (you must add a prior for it).
        kwargs :
            Passed to GravitationalWaveTransient. (Note: time/distance/phase marginalization in
            the base class assumes Gaussian structure; leave those False unless you re-derive them.)
        """
        
        super().__init__(interferometers=interferometers, waveform_generator=waveform_generator, **kwargs)

        self._fixed_nu = float(nu)
        self.infer_nu  = bool(infer_nu)

        # Add 'nu' into bilby's parameter dict if we want to sample it
        if self.infer_nu and "nu" not in self.parameters:
            self.parameters["nu"] = self._fixed_nu

    @property
    def nu(self):
        return float(self.parameters["nu"]) if self.infer_nu else self._fixed_nu

    def log_likelihood(self):
        nu = self.nu

        # waveform polarizations (dict: 'plus','cross')
        pols = self.waveform_generator.frequency_domain_strain(self.parameters)

        logl = 0.0
        for ifo in self.interferometers:
            # detector response h(f) in this interferometer
            h_f = ifo.get_detector_response(pols, self.parameters)

            # data d(f), PSD S_n(f), mask to the analysis band
            d_f  = ifo.frequency_domain_strain
            psd  = ifo.power_spectral_density_array
            mask = ifo.frequency_mask

            r  = (d_f[mask] - h_f[mask])  # complex residual
            Sn = psd[mask]

            # Effective complex variance per bin under Gaussian noise:
            # E[|r|^2] ~ (Sn/2) * (duration)  in common GW conventions.
            # Bilby stores frequency domain strain consistent with its inner product;
            # using Sn/2 here is a standard choice for complex bins.
            scale2 = Sn / 2.0

            # For complex residuals, treat Re/Im as 2D Student-t, see: https://en.wikipedia.org/wiki/Multivariate_t-distribution
            # log p(r) = const - ((nu+2)/2) * log(1 + |r|^2/(nu*scale2))
            # with const = log Γ((nu+2)/2) - log Γ(nu/2) - log(νπ scale2)
            abs2 = (r.real ** 2 + r.imag ** 2)

            const = (
                  gammaln((nu + 2.0) / 2.0)
                - gammaln(nu / 2.0)
                - np.log(nu * np.pi * scale2)
            )
            logl += np.sum(const - 0.5 * (nu + 2.0) * np.log1p(abs2 / (nu * scale2)))

        return float(logl)


########################
# Standard user inputs #
########################

event           = "GW150914"      # Available options: ["GW150914", "GW231123_135430"]
likelihood_type = "Gaussian"      # Available options: ["Gaussian", "Student-bilby", "Student-local"]
outdir_label    = "Mc_tc_only"    # Label for output directory, e.g. "test", "test_fixed_nu", "test_GW231123"

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
trigger_times         = {"GW150914": 1126259462.41, "GW231123_135430": 1384782888.634}
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
outdir = f"Runs/{event}_{likelihood_type}_{outdir_label}"
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
priors = bilby.gw.prior.BBHPriorDict(filename=f"Priors/{event}_single_par.prior")
# priors["geocent_time"] = bilby.core.prior.Uniform(trigger_time - 0.01, trigger_time + 0.01, name="geocent_time")

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
    waveform_arguments            = {"waveform_approximant": "IMRPhenomXPHM", "reference_frequency": 50})


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
        priors =priors,

        # time_marginalization=True,
        # distance_marginalization=True,

        phase_marginalization       = False,
        time_marginalization        = False,
        distance_marginalization    = False,
        calibration_marginalization = False,

    )

elif("Student" in likelihood_type):

    # Prior must be > 2 for finite variance in 2D
    priors["nu"] = bilby.core.prior.LogUniform(2.1, 200, name="nu") # example

    if(likelihood_type == "Student-local"):

        likelihood = StudentTGravitationalWaveTransient(
        
        interferometers             = ifo_list,
        waveform_generator          = waveform_generator,

        nu                          = 8.0,  # initial/fixed value
        infer_nu                    = True, # False to fix it

        phase_marginalization       = False,
        time_marginalization        = False,
        distance_marginalization    = False,
        calibration_marginalization = False,
        )
    elif(likelihood_type == "Student-bilby"):
        likelihood = bilby.gw.likelihood.StudentTGravitationalWaveTransient(
        
        interferometers             = ifo_list,
        waveform_generator          = waveform_generator,

        nu                          = 8.0,  # initial/fixed value
        infer_nu                    = True, # False to fix it

        phase_marginalization       = False,
        time_marginalization        = False,
        distance_marginalization    = False,
        calibration_marginalization = False,
        )

else:
    raise ValueError("likelihood_type must be one of ['Gaussian', 'Student-bilby', 'Student-local']")


###########
# Sampler #
###########

# Finally, we run the sampler. This function takes the likelihood and prior
# along with some options for how to do the sampling and how to save the data
result = bilby.run_sampler(
    likelihood,
    priors,
    sampler             = "dynesty",
    sample              = 'rslice',
    nlive               = 128,
    slices              = 20,
    n_check_point       = 200,
    outdir              = outdir,
    label               = label,
    #check_point_delta_t = 600,
    check_point_plot    = True,
    npool               = 1,
    conversion_function = bilby.gw.conversion.generate_all_bbh_parameters,
    result_class        = bilby.gw.result.CBCResult,
)


#########
# Plots #
#########

result.plot_corner()