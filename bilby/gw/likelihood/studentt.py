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
        detector_dependent_nu=False,
        time_marginalization=False,
        distance_marginalization=False,
        phase_marginalization=False,
        calibration_marginalization=False,
        priors=None,
        distance_marginalization_lookup_table=None,
        calibration_lookup_table=None,
        number_of_response_curves=1000,
        starting_index=0,
        jitter_time=True,
        reference_frame="sky",
        time_reference="geocenter",
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
            'nu_1', ..., 'nu_N'; you must add priors for each sampled parameter. If
            `detector_dependent_nu=True`, sampled parameters become detector-specific:
            'nu_H1', ..., or 'nu_H1_1', ..., 'nu_L1_1', ... .
        num_frequency_bands : int
            Number of equispaced contiguous frequency bands spanning the total likelihood frequency range. Each band has
            its own Student-t degrees of freedom parameter.
        detector_dependent_nu : bool
            If True, allow a distinct `nu` value for each interferometer (and for each
            frequency band if `num_frequency_bands > 1`). If False, the same `nu`
            values are shared by all detectors.
        kwargs :
            Passed to GravitationalWaveTransient. (Note: time/distance/phase marginalization in
            the base class assumes Gaussian structure; leave those False unless you re-derive them.)
        """
        # Keep the base-likelihood kwargs explicit so tools such as bilby_pipe
        # can discover and forward them when this class is selected by dotted path.
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            time_marginalization=time_marginalization,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            calibration_marginalization=calibration_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            calibration_lookup_table=calibration_lookup_table,
            number_of_response_curves=number_of_response_curves,
            starting_index=starting_index,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference,
            **kwargs,
        )

        self.num_frequency_bands   = self._validate_num_frequency_bands(num_frequency_bands)
        self.detector_dependent_nu = bool(detector_dependent_nu)
        self._detector_names       = [ifo.name for ifo in self.interferometers]
        self._fixed_nu             = self._coerce_nu_array(nu)
        self.infer_nu              = bool(infer_nu)
        self._frequency_band_edges = self._create_frequency_band_edges()

        if not self._valid_nu_values(self._fixed_nu): raise ValueError("All nu values must be positive and finite")

        if self.infer_nu:
            if not self.detector_dependent_nu:
                for key, value in zip(self.nu_parameter_keys, self._fixed_nu):
                    self.parameters.setdefault(key, float(value))
            else:
                for detector_index, detector_name in enumerate(self._detector_names):
                    for band_index in range(self.num_frequency_bands):
                        self.parameters.setdefault(
                            self._detector_nu_parameter_key(detector_name, band_index),
                            float(self._fixed_nu[detector_index, band_index]),
                        )

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
        if not self.detector_dependent_nu:
            if self.num_frequency_bands == 1:
                return float(values[0])
            return values.copy()
        if self.num_frequency_bands == 1:
            return values[:, 0].copy()
        return values.copy()

    @property
    def nu_parameter_keys(self):

        if not self.detector_dependent_nu:
            if self.num_frequency_bands == 1:
                return ["nu"]
            return [f"nu_{index}" for index in range(1, self.num_frequency_bands + 1)]

        if self.num_frequency_bands == 1:
            return [f"nu_{detector_name}" for detector_name in self._detector_names]

        return [
            f"nu_{detector_name}_{index}"
            for detector_name in self._detector_names
            for index in range(1, self.num_frequency_bands + 1)
        ]

    def _validate_num_frequency_bands(self, num_frequency_bands):

        try                                   : num_frequency_bands = int(num_frequency_bands)
        except (TypeError, ValueError) as exc : raise ValueError("num_frequency_bands must be a positive integer") from exc

        if num_frequency_bands < 1:  raise ValueError("num_frequency_bands must be a positive integer")

        return num_frequency_bands

    def _coerce_nu_array(self, nu):

        values = np.asarray(nu, dtype=float)

        if not self.detector_dependent_nu:
            if   values.ndim == 0                                             : values = np.repeat(values[None], self.num_frequency_bands)
            elif values.ndim == 1 and len(values) == 1                        : values = np.repeat(values,       self.num_frequency_bands)
            elif values.ndim != 1 or  len(values) != self.num_frequency_bands : raise ValueError("nu must be a scalar or an array with one entry per frequency band")
            return values.astype(float, copy=False)

        num_detectors = len(self.interferometers)
        if values.ndim == 0:
            values = np.full((num_detectors, self.num_frequency_bands), float(values))
        elif values.ndim == 1:
            if len(values) == 1:
                values = np.full((num_detectors, self.num_frequency_bands), float(values[0]))
            elif len(values) == self.num_frequency_bands:
                values = np.repeat(values[None, :], num_detectors, axis=0)
            elif len(values) == num_detectors and self.num_frequency_bands == 1:
                values = values[:, None]
            else:
                raise ValueError(
                    "nu must be a scalar, an array with one entry per frequency band, "
                    "an array with one entry per detector when num_frequency_bands=1, "
                    "or a 2D array with shape (num_detectors, num_frequency_bands)"
                )
        elif values.ndim == 2:
            if values.shape != (num_detectors, self.num_frequency_bands):
                raise ValueError(
                    "nu must have shape (num_detectors, num_frequency_bands) when "
                    "detector_dependent_nu=True"
                )
        else:
            raise ValueError(
                "nu must be scalar, 1D, or 2D when detector_dependent_nu=True"
            )

        return values.astype(float, copy=False)

    @staticmethod
    def _valid_nu_values(values): return np.all(np.isfinite(values)) and np.all(values > 0)

    def _create_frequency_band_edges(self):

        frequencies = self.interferometers[0].frequency_array[self.interferometers[0].frequency_mask]

        if len(frequencies) == 0: raise ValueError("No active frequencies available to construct Student-t bands")
        
        return np.linspace(frequencies[0], frequencies[-1], self.num_frequency_bands + 1)

    def _get_nu_values(self, parameters):

        if not self.infer_nu:
            return self._fixed_nu

        if "nu" in parameters:
            return self._coerce_nu_array(parameters["nu"])

        if not self.detector_dependent_nu:
            if self.num_frequency_bands == 1:
                return self._coerce_nu_array(parameters.get("nu", self._fixed_nu[0]))

            return self._coerce_nu_array(
                [
                    parameters.get(key, default)
                    for key, default in zip(self.nu_parameter_keys, self._fixed_nu)
                ]
            )

        return self._coerce_nu_array(
            [
                [
                    parameters.get(
                        self._detector_nu_parameter_key(detector_name, band_index),
                        self._fixed_nu[detector_index, band_index],
                    )
                    for band_index in range(self.num_frequency_bands)
                ]
                for detector_index, detector_name in enumerate(self._detector_names)
            ]
        )

    def _store_nu_values(self, nu_values):

        if not self.detector_dependent_nu:
            for key, value in zip(self.nu_parameter_keys, nu_values):
                self.parameters[key] = float(value)
            return

        for detector_index, detector_name in enumerate(self._detector_names):
            for band_index in range(self.num_frequency_bands):
                self.parameters[
                    self._detector_nu_parameter_key(detector_name, band_index)
                ] = float(nu_values[detector_index, band_index])

    def _detector_nu_parameter_key(self, detector_name, band_index):

        if self.num_frequency_bands == 1:
            return f"nu_{detector_name}"
        return f"nu_{detector_name}_{band_index + 1}"

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

    def _get_active_nu_values(self, parameters, update_state=False):

        nu_values = self._get_nu_values(parameters)
        if update_state and self.infer_nu:
            self._store_nu_values(nu_values)
        if not self._valid_nu_values(nu_values):
            return None
        return nu_values

    def _get_interferometer_nu_values(self, interferometer, nu_values):

        if not self.detector_dependent_nu:
            return nu_values
        return nu_values[self._detector_names.index(interferometer.name)]

    def _compute_scale2(self, power_spectral_density):

        # Bilby's frequency-domain convention gives Var(Re n_k) = Var(Im n_k) = S_n(f_k) T / 4.
        return power_spectral_density * self.waveform_generator.duration / 4.0

    def _compute_detector_log_likelihood(
        self,
        interferometer,
        nu_values,
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
        for nu, band_mask in zip(nu_values, band_masks):
            if not np.any(band_mask):
                continue

            band_scale2 = scale2[band_mask]
            band_abs2 = abs2[band_mask]
            const = (
                gammaln((nu + 2.0) / 2.0)
                - gammaln(nu / 2.0)
                - np.log(nu * np.pi * band_scale2)
            )

            logl += np.sum(
                const - 0.5 * (nu + 2.0) * np.log1p(band_abs2 / (nu * band_scale2))
            )

        return float(logl)

    def log_likelihood(self, parameters=None):

        parameters = self._resolve_likelihood_parameters(parameters)
        nu_values = self._get_active_nu_values(parameters, update_state=True)
        if nu_values is None:
            return np.nan_to_num(-np.inf)

        # waveform polarizations (dict: 'plus','cross')
        pols = self.waveform_generator.frequency_domain_strain(parameters)
        if pols is None:
            return np.nan_to_num(-np.inf)

        logl = 0.0
        for ifo in self.interferometers:
            detector_logl = self._compute_detector_log_likelihood(
                interferometer=ifo,
                nu_values=self._get_interferometer_nu_values(ifo, nu_values),
                parameters=parameters,
                waveform_polarizations=pols,
            )
            if not np.isfinite(detector_logl):
                return np.nan_to_num(-np.inf)
            logl += detector_logl

        return float(logl)

    def noise_log_likelihood(self):

        nu_values = self._get_active_nu_values(self.parameters.copy(), update_state=False)
        if nu_values is None:
            return np.nan_to_num(-np.inf)

        logl = 0.0
        for ifo in self.interferometers:
            detector_logl = self._compute_detector_log_likelihood(
                interferometer=ifo,
                nu_values=self._get_interferometer_nu_values(ifo, nu_values),
            )
            if not np.isfinite(detector_logl):
                return np.nan_to_num(-np.inf)
            logl += detector_logl

        return float(logl)

    def log_likelihood_ratio(self, parameters=None):

        parameters = self._resolve_likelihood_parameters(parameters)
        nu_values = self._get_active_nu_values(parameters, update_state=True)
        if nu_values is None:
            return np.nan_to_num(-np.inf)

        pols = self.waveform_generator.frequency_domain_strain(parameters)
        if pols is None:
            return np.nan_to_num(-np.inf)

        signal_logl = 0.0
        noise_logl = 0.0
        for ifo in self.interferometers:
            detector_signal_logl = self._compute_detector_log_likelihood(
                interferometer=ifo,
                nu_values=self._get_interferometer_nu_values(ifo, nu_values),
                parameters=parameters,
                waveform_polarizations=pols,
            )
            detector_noise_logl = self._compute_detector_log_likelihood(
                interferometer=ifo,
                nu_values=self._get_interferometer_nu_values(ifo, nu_values),
            )
            if not np.isfinite(detector_signal_logl) or not np.isfinite(detector_noise_logl):
                return np.nan_to_num(-np.inf)
            signal_logl += detector_signal_logl
            noise_logl += detector_noise_logl

        return float(signal_logl - noise_logl)

    def compute_per_detector_log_likelihood(self, parameters=None):

        parameters = self._resolve_likelihood_parameters(parameters)
        nu_values = self._get_active_nu_values(parameters, update_state=True)
        if nu_values is None:
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
                nu_values=self._get_interferometer_nu_values(interferometer, nu_values),
                parameters=parameters,
                waveform_polarizations=pols,
            )
            detector_noise_logl = self._compute_detector_log_likelihood(
                interferometer=interferometer,
                nu_values=self._get_interferometer_nu_values(interferometer, nu_values),
            )
            parameters[f"{interferometer.name}_log_likelihood"] = float(
                detector_signal_logl - detector_noise_logl
            )

        return parameters.copy()
