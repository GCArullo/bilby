import numpy as np
from scipy.special import kve

from ...core.likelihood import _fallback_to_parameters
from ...core.utils import logger
from .base import GravitationalWaveTransient


class HyperbolicGravitationalWaveTransient(GravitationalWaveTransient):
    r"""
    A heavy-tailed likelihood based on the symmetric 2D hyperbolic distribution.

    This follows the distribution used in arXiv:2602.22074. Each complex frequency
    bin is treated as a two-dimensional real vector built from the real and imaginary
    residual parts.
    """

    def __init__(
        self,
        interferometers,
        waveform_generator,
        alpha=10.0,
        delta=1.0,
        infer_alpha=False,
        infer_delta=False,
        num_frequency_bands=1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        alpha : float, array-like
            Positive hyperbolic tail parameter. For multiple frequency bands this may
            either be a scalar, which is broadcast to all bands, or an array with one
            value per band.
        delta : float, array-like
            Positive hyperbolic scale parameter. For multiple frequency bands this may
            either be a scalar, which is broadcast to all bands, or an array with one
            value per band.
        infer_alpha : bool
            If True, treat the hyperbolic tail parameter as sampled. For a single band
            this uses the parameter name 'alpha'. For multiple bands this uses
            'alpha_1', ..., 'alpha_N'; you must add priors for each sampled parameter.
        infer_delta : bool
            If True, treat the hyperbolic scale parameter as sampled. For a single band
            this uses the parameter name 'delta'. For multiple bands this uses
            'delta_1', ..., 'delta_N'; you must add priors for each sampled parameter.
        num_frequency_bands : int
            Number of contiguous frequency bands spanning the active analysis range. Each
            band has its own hyperbolic `alpha` and `delta` parameters.
        kwargs :
            Passed to GravitationalWaveTransient.
        """
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            **kwargs,
        )

        self.num_frequency_bands = self._validate_num_frequency_bands(num_frequency_bands)
        self._fixed_alpha = self._coerce_parameter_array(alpha, "alpha")
        self._fixed_delta = self._coerce_parameter_array(delta, "delta")
        self.infer_alpha = bool(infer_alpha)
        self.infer_delta = bool(infer_delta)
        self._frequency_band_edges = self._create_frequency_band_edges()

        if not self._valid_positive_values(self._fixed_alpha):
            raise ValueError("All alpha values must be positive and finite")
        if not self._valid_positive_values(self._fixed_delta):
            raise ValueError("All delta values must be positive and finite")

        if self.infer_alpha:
            for key, value in zip(self.alpha_parameter_keys, self._fixed_alpha):
                self.parameters.setdefault(key, float(value))
        if self.infer_delta:
            for key, value in zip(self.delta_parameter_keys, self._fixed_delta):
                self.parameters.setdefault(key, float(value))

        if (
            self.time_marginalization
            or self.distance_marginalization
            or self.phase_marginalization
        ):
            logger.warning(
                "HyperbolicGravitationalWaveTransient is being used with Gaussian "
                "marginalization settings. These marginalizations are derived for "
                "the Gaussian likelihood and may be inconsistent for hyperbolic noise."
            )

    @property
    def alpha(self):
        values = self._get_alpha_values(self.parameters)
        if self.num_frequency_bands == 1:
            return float(values[0])
        return values.copy()

    @property
    def delta(self):
        values = self._get_delta_values(self.parameters)
        if self.num_frequency_bands == 1:
            return float(values[0])
        return values.copy()

    @property
    def alpha_parameter_keys(self):
        if self.num_frequency_bands == 1:
            return ["alpha"]
        return [f"alpha_{index}" for index in range(1, self.num_frequency_bands + 1)]

    @property
    def delta_parameter_keys(self):
        if self.num_frequency_bands == 1:
            return ["delta"]
        return [f"delta_{index}" for index in range(1, self.num_frequency_bands + 1)]

    def _validate_num_frequency_bands(self, num_frequency_bands):
        try:
            num_frequency_bands = int(num_frequency_bands)
        except (TypeError, ValueError) as exc:
            raise ValueError("num_frequency_bands must be a positive integer") from exc
        if num_frequency_bands < 1:
            raise ValueError("num_frequency_bands must be a positive integer")
        return num_frequency_bands

    def _coerce_parameter_array(self, values, name):
        values = np.asarray(values, dtype=float)
        if values.ndim == 0:
            values = np.repeat(values[None], self.num_frequency_bands)
        elif values.ndim == 1 and len(values) == 1:
            values = np.repeat(values, self.num_frequency_bands)
        elif values.ndim != 1 or len(values) != self.num_frequency_bands:
            raise ValueError(
                f"{name} must be a scalar or an array with one entry per frequency band"
            )
        return values.astype(float, copy=False)

    @staticmethod
    def _valid_positive_values(values):
        return np.all(np.isfinite(values)) and np.all(values > 0)

    def _create_frequency_band_edges(self):
        frequencies = self.interferometers[0].frequency_array[
            self.interferometers[0].frequency_mask
        ]
        if len(frequencies) == 0:
            raise ValueError("No active frequencies available to construct hyperbolic bands")
        return np.linspace(frequencies[0], frequencies[-1], self.num_frequency_bands + 1)

    def _get_alpha_values(self, parameters):
        if not self.infer_alpha:
            return self._fixed_alpha

        if self.num_frequency_bands == 1:
            return self._coerce_parameter_array(parameters.get("alpha", self._fixed_alpha[0]), "alpha")

        if "alpha" in parameters:
            return self._coerce_parameter_array(parameters["alpha"], "alpha")

        return self._coerce_parameter_array(
            [
                parameters.get(key, default)
                for key, default in zip(self.alpha_parameter_keys, self._fixed_alpha)
            ],
            "alpha",
        )

    def _get_delta_values(self, parameters):
        if not self.infer_delta:
            return self._fixed_delta

        if self.num_frequency_bands == 1:
            return self._coerce_parameter_array(parameters.get("delta", self._fixed_delta[0]), "delta")

        if "delta" in parameters:
            return self._coerce_parameter_array(parameters["delta"], "delta")

        return self._coerce_parameter_array(
            [
                parameters.get(key, default)
                for key, default in zip(self.delta_parameter_keys, self._fixed_delta)
            ],
            "delta",
        )

    def _store_alpha_values(self, alpha_values):
        for key, value in zip(self.alpha_parameter_keys, alpha_values):
            self.parameters[key] = float(value)

    def _store_delta_values(self, delta_values):
        for key, value in zip(self.delta_parameter_keys, delta_values):
            self.parameters[key] = float(value)

    def _get_frequency_band_masks(self, interferometer):
        frequencies = interferometer.frequency_array[interferometer.frequency_mask]
        band_masks = []
        for index, (lower, upper) in enumerate(
            zip(self._frequency_band_edges[:-1], self._frequency_band_edges[1:])
        ):
            if index == self.num_frequency_bands - 1:
                band_mask = (frequencies >= lower) & (frequencies <= upper)
            else:
                band_mask = (frequencies >= lower) & (frequencies < upper)
            band_masks.append(band_mask)
        return band_masks

    @staticmethod
    def _log_bessel_k(order, value):
        value = np.asarray(value, dtype=float)
        if np.any(value <= 0) or not np.all(np.isfinite(value)):
            return np.full_like(value, np.nan, dtype=float)
        result = np.log(kve(order, value)) - value
        if result.ndim == 0:
            return float(result)
        return result

    @staticmethod
    def _log_hyperbolic_density(abs2_over_scale2, alpha, delta):
        normalization_argument = alpha * delta
        if normalization_argument <= 0 or not np.isfinite(normalization_argument):
            return None

        if min(alpha, delta) >= 1e8:
            precision = alpha / delta
            return (
                np.log(precision)
                - np.log(2.0 * np.pi)
                - 0.5 * precision * abs2_over_scale2
            )

        radius = np.sqrt(delta ** 2 + abs2_over_scale2)
        argument = alpha * radius
        if not np.all(np.isfinite(argument)) or np.any(argument <= 0):
            return None

        return (
            np.log(alpha)
            - np.log(2.0 * np.pi * delta)
            - HyperbolicGravitationalWaveTransient._log_bessel_k(1, normalization_argument)
            + HyperbolicGravitationalWaveTransient._log_bessel_k(0, argument)
        )

    def _resolve_likelihood_parameters(self, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)
        if parameters is self.parameters:
            parameters = parameters.copy()
        else:
            merged_parameters = self.parameters.copy()
            merged_parameters.update(parameters)
            parameters = merged_parameters
        parameters.update(self.get_sky_frame_parameters(parameters))
        return parameters

    def _get_active_shape_parameters(self, parameters, update_state=False):
        alpha_values = self._get_alpha_values(parameters)
        delta_values = self._get_delta_values(parameters)

        if update_state:
            if self.infer_alpha:
                self._store_alpha_values(alpha_values)
            if self.infer_delta:
                self._store_delta_values(delta_values)

        if not self._valid_positive_values(alpha_values):
            return None, None
        if not self._valid_positive_values(delta_values):
            return None, None
        return alpha_values, delta_values

    def _compute_scale2(self, power_spectral_density):
        # Bilby's frequency-domain convention gives Var(Re n_k) = Var(Im n_k) = S_n(f_k) T / 4.
        return power_spectral_density * self.waveform_generator.duration / 4.0

    def _compute_detector_log_likelihood(
        self,
        interferometer,
        alpha_values,
        delta_values,
        parameters=None,
        waveform_polarizations=None,
    ):
        mask = interferometer.frequency_mask
        scale2 = self._compute_scale2(interferometer.power_spectral_density_array[mask])

        if np.any(scale2 <= 0) or not np.all(np.isfinite(scale2)):
            return -np.inf

        if waveform_polarizations is None:
            residual = interferometer.frequency_domain_strain[mask]
        else:
            h_f = interferometer.get_detector_response(waveform_polarizations, parameters)
            residual = interferometer.frequency_domain_strain[mask] - h_f[mask]

        abs2 = residual.real ** 2 + residual.imag ** 2
        band_masks = self._get_frequency_band_masks(interferometer)

        logl = 0.0
        for alpha, delta, band_mask in zip(alpha_values, delta_values, band_masks):
            if not np.any(band_mask):
                continue

            band_scale2 = scale2[band_mask]
            band_abs2_over_scale2 = abs2[band_mask] / band_scale2
            band_log_density = self._log_hyperbolic_density(
                abs2_over_scale2=band_abs2_over_scale2,
                alpha=alpha,
                delta=delta,
            )
            if band_log_density is None or not np.all(np.isfinite(band_log_density)):
                return -np.inf

            logl += np.sum(band_log_density - np.log(band_scale2))

        return float(logl)

    def log_likelihood(self, parameters=None):
        parameters = self._resolve_likelihood_parameters(parameters)
        alpha_values, delta_values = self._get_active_shape_parameters(
            parameters, update_state=True
        )
        if alpha_values is None or delta_values is None:
            return np.nan_to_num(-np.inf)

        pols = self.waveform_generator.frequency_domain_strain(parameters)
        if pols is None:
            return np.nan_to_num(-np.inf)

        logl = 0.0
        for ifo in self.interferometers:
            detector_logl = self._compute_detector_log_likelihood(
                interferometer=ifo,
                alpha_values=alpha_values,
                delta_values=delta_values,
                parameters=parameters,
                waveform_polarizations=pols,
            )
            if not np.isfinite(detector_logl):
                return np.nan_to_num(-np.inf)
            logl += detector_logl

        return float(logl)

    def noise_log_likelihood(self):
        alpha_values, delta_values = self._get_active_shape_parameters(
            self.parameters.copy(), update_state=False
        )
        if alpha_values is None or delta_values is None:
            return np.nan_to_num(-np.inf)

        logl = 0.0
        for ifo in self.interferometers:
            detector_logl = self._compute_detector_log_likelihood(
                interferometer=ifo,
                alpha_values=alpha_values,
                delta_values=delta_values,
            )
            if not np.isfinite(detector_logl):
                return np.nan_to_num(-np.inf)
            logl += detector_logl

        return float(logl)

    def log_likelihood_ratio(self, parameters=None):
        parameters = self._resolve_likelihood_parameters(parameters)
        alpha_values, delta_values = self._get_active_shape_parameters(
            parameters, update_state=True
        )
        if alpha_values is None or delta_values is None:
            return np.nan_to_num(-np.inf)

        pols = self.waveform_generator.frequency_domain_strain(parameters)
        if pols is None:
            return np.nan_to_num(-np.inf)

        signal_logl = 0.0
        noise_logl = 0.0
        for ifo in self.interferometers:
            detector_signal_logl = self._compute_detector_log_likelihood(
                interferometer=ifo,
                alpha_values=alpha_values,
                delta_values=delta_values,
                parameters=parameters,
                waveform_polarizations=pols,
            )
            detector_noise_logl = self._compute_detector_log_likelihood(
                interferometer=ifo,
                alpha_values=alpha_values,
                delta_values=delta_values,
            )
            if not np.isfinite(detector_signal_logl) or not np.isfinite(detector_noise_logl):
                return np.nan_to_num(-np.inf)
            signal_logl += detector_signal_logl
            noise_logl += detector_noise_logl

        return float(signal_logl - noise_logl)

    def compute_per_detector_log_likelihood(self, parameters=None):
        parameters = self._resolve_likelihood_parameters(parameters)
        alpha_values, delta_values = self._get_active_shape_parameters(
            parameters, update_state=True
        )
        if alpha_values is None or delta_values is None:
            for interferometer in self.interferometers:
                parameters[f"{interferometer.name}_log_likelihood"] = np.nan_to_num(-np.inf)
            return parameters.copy()

        pols = self.waveform_generator.frequency_domain_strain(parameters)
        if pols is None:
            for interferometer in self.interferometers:
                parameters[f"{interferometer.name}_log_likelihood"] = np.nan_to_num(-np.inf)
            return parameters.copy()

        for interferometer in self.interferometers:
            detector_signal_logl = self._compute_detector_log_likelihood(
                interferometer=interferometer,
                alpha_values=alpha_values,
                delta_values=delta_values,
                parameters=parameters,
                waveform_polarizations=pols,
            )
            detector_noise_logl = self._compute_detector_log_likelihood(
                interferometer=interferometer,
                alpha_values=alpha_values,
                delta_values=delta_values,
            )
            parameters[f"{interferometer.name}_log_likelihood"] = float(
                detector_signal_logl - detector_noise_logl
            )

        return parameters.copy()
