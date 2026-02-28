import numpy as np
from scipy.special import gammaln

from ...core.likelihood import _fallback_to_parameters
from ...core.utils import logger
from .base import GravitationalWaveTransient

class StudentTGravitationalWaveTransient(GravitationalWaveTransient):
    """
    A simple heavy-tailed replacement for the standard Gaussian (Whittle) likelihood.

    Model: per-frequency-bin complex Student-t for the residual r_k = d_k - h_k, with
    scale set by the one-sided PSD S_n(f).
    """

    def __init__(
        self,
        interferometers,
        waveform_generator,
        nu=8.0,
        infer_nu=False,
        num_frequency_bands=1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        nu : float, array-like
            Student-t degrees of freedom. Smaller => heavier tails. nu -> infinity gives Gaussian.
            For multiple frequency bands this may either be a scalar, which is broadcast to all
            bands, or an array with one value per band.
        infer_nu : bool
            If True, treat the Student-t degrees of freedom as sampled parameters. For a single
            frequency band this uses the parameter name 'nu'. For multiple bands this uses
            'nu_1', ..., 'nu_N'; you must add priors for each sampled parameter.
        num_frequency_bands : int
            Number of equispaced contiguous frequency bands spanning the total likelihood frequency range. Each band has
            its own Student-t degrees of freedom parameter.
        kwargs :
            Passed to GravitationalWaveTransient. (Note: time/distance/phase marginalization in
            the base class assumes Gaussian structure; leave those False unless you re-derive them.)
        """
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            **kwargs,
        )

        self.num_frequency_bands   = self._validate_num_frequency_bands(num_frequency_bands)
        self._fixed_nu             = self._coerce_nu_array(nu)
        self.infer_nu              = bool(infer_nu)
        self._frequency_band_edges = self._create_frequency_band_edges()

        if not self._valid_nu_values(self._fixed_nu): raise ValueError("All nu values must be positive and finite")

        if self.infer_nu:
            for key, value in zip(self.nu_parameter_keys, self._fixed_nu): self.parameters.setdefault(key, float(value))

        if (
            self.time_marginalization
            or self.distance_marginalization
            or self.phase_marginalization
        ):
            logger.warning(
                "StudentTGravitationalWaveTransient is being used with Gaussian "
                "marginalization settings. These marginalizations are derived for "
                "the Gaussian likelihood and may be inconsistent for Student-t noise."
            )

    @property
    def nu(self):

        values = self._get_nu_values(self.parameters)
        if self.num_frequency_bands == 1: return float(values[0])
        
        return values.copy()

    @property
    def nu_parameter_keys(self):

        if self.num_frequency_bands == 1: return ["nu"]

        return [f"nu_{index}" for index in range(1, self.num_frequency_bands + 1)]

    def _validate_num_frequency_bands(self, num_frequency_bands):

        try                                   : num_frequency_bands = int(num_frequency_bands)
        except (TypeError, ValueError) as exc : raise ValueError("num_frequency_bands must be a positive integer") from exc

        if num_frequency_bands < 1:  raise ValueError("num_frequency_bands must be a positive integer")

        return num_frequency_bands

    def _coerce_nu_array(self, nu):

        values = np.asarray(nu, dtype=float)

        if   values.ndim == 0                                             : values = np.repeat(values[None], self.num_frequency_bands)
        elif values.ndim == 1 and len(values) == 1                        : values = np.repeat(values,       self.num_frequency_bands)
        elif values.ndim != 1 or  len(values) != self.num_frequency_bands : raise ValueError( "nu must be a scalar or an array with one entry per frequency band")

        return values.astype(float, copy=False)

    @staticmethod
    def _valid_nu_values(values): return np.all(np.isfinite(values)) and np.all(values > 0)

    def _create_frequency_band_edges(self):

        frequencies = self.interferometers[0].frequency_array[self.interferometers[0].frequency_mask]

        if len(frequencies) == 0: raise ValueError("No active frequencies available to construct Student-t bands")
        
        return np.linspace(frequencies[0], frequencies[-1], self.num_frequency_bands + 1)

    def _get_nu_values(self, parameters):

        if not self.infer_nu            : return self._fixed_nu

        if self.num_frequency_bands == 1: return self._coerce_nu_array(parameters.get("nu", self._fixed_nu[0]))

        if "nu" in parameters           : return self._coerce_nu_array(parameters["nu"])

        return self._coerce_nu_array([parameters.get(key, default) for key, default in zip(self.nu_parameter_keys, self._fixed_nu)]
                                     
        )

    def _store_nu_values(self, nu_values):

        for key, value in zip(self.nu_parameter_keys, nu_values): self.parameters[key] = float(value)

    def _get_frequency_band_masks(self, interferometer):

        frequencies = interferometer.frequency_array[interferometer.frequency_mask]
        band_masks = []
        for index, (lower, upper) in enumerate(
            zip(self._frequency_band_edges[:-1], self._frequency_band_edges[1:])
        ):
            if index == self.num_frequency_bands - 1: band_mask = (frequencies >= lower) & (frequencies <= upper)
            else                                    : band_mask = (frequencies >= lower) & (frequencies <  upper)
            band_masks.append(band_mask)

        return band_masks

    def log_likelihood(self, parameters=None):

        parameters = _fallback_to_parameters(self, parameters)
        if parameters is self.parameters:
            parameters = parameters.copy()
        else:
            merged_parameters = self.parameters.copy()
            merged_parameters.update(parameters)
            parameters = merged_parameters
        parameters.update(self.get_sky_frame_parameters(parameters))

        nu_values = self._get_nu_values(parameters)
        if self.infer_nu                       : self._store_nu_values(nu_values)
        if not self._valid_nu_values(nu_values): return np.nan_to_num(-np.inf)

        # waveform polarizations (dict: 'plus','cross')
        pols = self.waveform_generator.frequency_domain_strain(parameters)
        if pols is None:
            return np.nan_to_num(-np.inf)

        logl = 0.0
        for ifo in self.interferometers:
            # detector response h(f) in this interferometer
            h_f = ifo.get_detector_response(pols, parameters)

            # data d(f), PSD S_n(f), mask to the analysis band
            mask = ifo.frequency_mask

            d_f  = ifo.frequency_domain_strain
            psd  = ifo.power_spectral_density_array
            r    = d_f[mask] - h_f[mask]

            band_masks = self._get_frequency_band_masks(ifo)

            # Effective complex variance per bin under Gaussian noise:
            # E[|r|^2] ~ (Sn/2) * (duration) in common GW conventions.
            # Bilby stores frequency domain strain consistent with its inner product;
            # using Sn/2 here is a standard choice for complex bins.
            scale2 = psd[mask] / 2.0

            if np.any(scale2 <= 0) or not np.all(np.isfinite(scale2)): return np.nan_to_num(-np.inf)

            # For complex residuals, treat Re/Im as 2D Student-t, see:
            # https://en.wikipedia.org/wiki/Multivariate_t-distribution
            # log p(r) = const - ((nu+2)/2) * log(1 + |r|^2/(nu*scale2))
            # with const = log Γ((nu+2)/2) - log Γ(nu/2) - log(νπ scale2)
            abs2 = r.real ** 2 + r.imag ** 2

            for nu, band_mask in zip(nu_values, band_masks):

                if not np.any(band_mask): continue

                band_scale2 = scale2[band_mask]
                band_abs2   =   abs2[band_mask]
                const = (
                    gammaln((nu + 2.0) / 2.0)
                    - gammaln(nu / 2.0)
                    - np.log(nu * np.pi * band_scale2)
                )

                logl += np.sum(
                    const - 0.5 * (nu + 2.0) * np.log1p(band_abs2 / (nu * band_scale2))
                )

        return float(logl)