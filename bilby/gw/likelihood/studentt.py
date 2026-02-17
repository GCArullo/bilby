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
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            **kwargs,
        )

        self._fixed_nu = float(nu)
        self.infer_nu = bool(infer_nu)

        if self._fixed_nu <= 0:
            raise ValueError("nu must be positive")

        if self.infer_nu and "nu" not in self.parameters:
            self.parameters["nu"] = self._fixed_nu

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
        if self.infer_nu:
            return float(self.parameters["nu"])
        return self._fixed_nu

    def log_likelihood(self, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)
        parameters = parameters.copy()
        parameters.update(self.get_sky_frame_parameters(parameters))

        if self.infer_nu and "nu" in parameters:
            self.parameters["nu"] = parameters["nu"]

        nu = self.nu
        if nu <= 0 or not np.isfinite(nu):
            return np.nan_to_num(-np.inf)

        # waveform polarizations (dict: 'plus','cross')
        pols = self.waveform_generator.frequency_domain_strain(parameters)
        if pols is None:
            return np.nan_to_num(-np.inf)

        logl = 0.0
        for ifo in self.interferometers:
            # detector response h(f) in this interferometer
            h_f = ifo.get_detector_response(pols, parameters)

            # data d(f), PSD S_n(f), mask to the analysis band
            d_f = ifo.frequency_domain_strain
            psd = ifo.power_spectral_density_array
            mask = ifo.frequency_mask

            r = d_f[mask] - h_f[mask]

            # Effective complex variance per bin under Gaussian noise:
            # E[|r|^2] ~ (Sn/2) * (duration) in common GW conventions.
            # Bilby stores frequency domain strain consistent with its inner product;
            # using Sn/2 here is a standard choice for complex bins.
            scale2 = psd[mask] / 2.0

            if np.any(scale2 <= 0) or not np.all(np.isfinite(scale2)):
                return np.nan_to_num(-np.inf)

            # For complex residuals, treat Re/Im as 2D Student-t, see:
            # https://en.wikipedia.org/wiki/Multivariate_t-distribution
            # log p(r) = const - ((nu+2)/2) * log(1 + |r|^2/(nu*scale2))
            # with const = log Γ((nu+2)/2) - log Γ(nu/2) - log(νπ scale2)
            abs2 = r.real ** 2 + r.imag ** 2

            const = (
                gammaln((nu + 2.0) / 2.0)
                - gammaln(nu / 2.0)
                - np.log(nu * np.pi * scale2)
            )

            logl += np.sum(const - 0.5 * (nu + 2.0) * np.log1p(abs2 / (nu * scale2)))

        return float(logl)
