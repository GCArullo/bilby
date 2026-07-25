from copy import deepcopy
from itertools import product
from math import factorial

import numpy as np
from scipy.integrate import quad

from ...core.likelihood import Likelihood, _fallback_to_parameters
from ...core.prior import DeltaFunction, PriorDict
from ...core.utils import logger
from .base import GravitationalWaveTransient


class _HyperbolicNoiseOnlyLikelihood(Likelihood):
    """Auxiliary likelihood for marginalizing the hyperbolic noise evidence."""

    def __init__(self, hyperbolic_likelihood):
        super().__init__()
        self._parameters.update(
            hyperbolic_likelihood._get_default_shape_parameter_dict()
        )
        self.hyperbolic_likelihood = hyperbolic_likelihood

    def log_likelihood(self, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)
        return self.hyperbolic_likelihood._noise_log_likelihood_from_parameters(parameters)

    def noise_log_likelihood(self):
        return 0.0


class HyperbolicGravitationalWaveTransient(GravitationalWaveTransient):
    r"""
    A heavy-tailed likelihood based on the hyperbolic distribution in Eq. 8-9 of
    arXiv:2602.22074.

    By default, each detector contributes an independent two-dimensional hyperbolic
    density per active frequency bin. If ``joint=True``, detector residuals at the
    same frequency are instead combined into one real residual vector whose dimension
    is twice the number of contributing detectors.
    """

    _NOISE_EVIDENCE_QUADRATURE_EPSABS = 0.0
    _NOISE_EVIDENCE_QUADRATURE_EPSREL = 1e-8
    _NOISE_EVIDENCE_QUADRATURE_LIMIT = 200

    def __init__(
        self,
        interferometers,
        waveform_generator,
        alpha=10.0,
        delta=1.0,
        infer_alpha=False,
        infer_delta=False,
        num_frequency_bands=1,
        detector_dependent_noise=False,
        joint=False,
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
        noise_evidence_nlive=None,
        dlogz_noise=0.1,
        dlogZ_noise=None,
        noise_evidence_method="quadrature",
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
            If `detector_dependent_noise=True`, sampled parameters become
            detector-specific: 'alpha_H1', ... or 'alpha_H1_1', ..., 'alpha_L1_1', ... .
        infer_delta : bool
            If True, treat the hyperbolic scale parameter as sampled. For a single band
            this uses the parameter name 'delta'. For multiple bands this uses
            'delta_1', ..., 'delta_N'; you must add priors for each sampled parameter.
            If `detector_dependent_noise=True`, sampled parameters become
            detector-specific: 'delta_H1', ... or 'delta_H1_1', ..., 'delta_L1_1', ... .
        num_frequency_bands : int
            Number of contiguous frequency bands spanning the active analysis range. Each
            band has its own hyperbolic `alpha` and `delta` parameters.
        detector_dependent_noise : bool
            If True, allow distinct hyperbolic `alpha` and `delta` values for each
            interferometer (and for each frequency band if `num_frequency_bands > 1`).
            If False, the same hyperbolic shape parameters are shared by all detectors.
        joint : bool
            If True, combine detector residuals into one network hyperbolic density
            per frequency bin. If False, the default, multiply independent
            per-detector densities, including when the shape parameters are shared.
            Joint densities require `detector_dependent_noise=False`.
        noise_evidence_nlive : int, optional
            Number of live points to use when the noise evidence requires an
            auxiliary nested-sampling run. If not provided, use the internal
            low-dimensional heuristic.
        dlogz_noise, dlogZ_noise : float, optional
            Stopping criterion for the auxiliary nested-sampling run used to
            evaluate the noise evidence. `dlogZ_noise` is accepted as an alias.
        noise_evidence_method : str, optional
            Method used for the auxiliary hyperbolic noise-evidence
            calculation. ``"quadrature"`` performs a direct 1D or 2D
            integral over the prior transform and is the default. If more
            than two noise parameters are sampled, this falls back to the
            nested-sampling helper run. ``"nested"`` keeps the auxiliary
            dynesty calculation.
        kwargs :
            Passed to GravitationalWaveTransient.
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

        self.num_frequency_bands = self._validate_num_frequency_bands(num_frequency_bands)
        self.detector_dependent_noise = bool(detector_dependent_noise)
        self.joint = bool(joint)
        if self.joint and self.detector_dependent_noise:
            raise ValueError(
                "joint=True requires detector_dependent_noise=False"
            )
        self._detector_names = [ifo.name for ifo in self.interferometers]
        self._fixed_alpha = self._coerce_parameter_array(alpha, "alpha")
        self._fixed_delta = self._coerce_parameter_array(delta, "delta")
        self.infer_alpha = bool(infer_alpha)
        self.infer_delta = bool(infer_delta)
        self._frequency_band_edges = self._create_frequency_band_edges()
        self.noise_evidence_nlive = self._validate_noise_evidence_nlive(
            noise_evidence_nlive
        )
        self.dlogz_noise = self._resolve_dlogz_noise(
            dlogz_noise=dlogz_noise,
            dlogZ_noise=dlogZ_noise,
        )
        self.noise_evidence_method = self._resolve_noise_evidence_method(
            noise_evidence_method
        )

        if not self._valid_positive_values(self._fixed_alpha):
            raise ValueError("All alpha values must be positive and finite")
        if not self._valid_positive_values(self._fixed_delta):
            raise ValueError("All delta values must be positive and finite")

        if self.infer_alpha:
            if not self.detector_dependent_noise:
                for key, value in zip(self.alpha_parameter_keys, self._fixed_alpha):
                    self._parameters.setdefault(key, float(value))
            else:
                for detector_index, detector_name in enumerate(self._detector_names):
                    for band_index in range(self.num_frequency_bands):
                        self._parameters.setdefault(
                            self._detector_parameter_key(
                                "alpha", detector_name, band_index
                            ),
                            float(self._fixed_alpha[detector_index, band_index]),
                        )
        if self.infer_delta:
            if not self.detector_dependent_noise:
                for key, value in zip(self.delta_parameter_keys, self._fixed_delta):
                    self._parameters.setdefault(key, float(value))
            else:
                for detector_index, detector_name in enumerate(self._detector_names):
                    for band_index in range(self.num_frequency_bands):
                        self._parameters.setdefault(
                            self._detector_parameter_key(
                                "delta", detector_name, band_index
                            ),
                            float(self._fixed_delta[detector_index, band_index]),
                        )

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
        values = self._get_alpha_values(self._parameters)
        if not self.detector_dependent_noise:
            if self.num_frequency_bands == 1:
                return float(values[0])
            return values.copy()
        if self.num_frequency_bands == 1:
            return values[:, 0].copy()
        return values.copy()

    @property
    def delta(self):
        values = self._get_delta_values(self._parameters)
        if not self.detector_dependent_noise:
            if self.num_frequency_bands == 1:
                return float(values[0])
            return values.copy()
        if self.num_frequency_bands == 1:
            return values[:, 0].copy()
        return values.copy()

    @property
    def alpha_parameter_keys(self):
        if not self.detector_dependent_noise:
            if self.num_frequency_bands == 1:
                return ["alpha"]
            return [f"alpha_{index}" for index in range(1, self.num_frequency_bands + 1)]
        if self.num_frequency_bands == 1:
            return [f"alpha_{detector_name}" for detector_name in self._detector_names]
        return [
            f"alpha_{detector_name}_{index}"
            for detector_name in self._detector_names
            for index in range(1, self.num_frequency_bands + 1)
        ]

    @property
    def delta_parameter_keys(self):
        if not self.detector_dependent_noise:
            if self.num_frequency_bands == 1:
                return ["delta"]
            return [f"delta_{index}" for index in range(1, self.num_frequency_bands + 1)]
        if self.num_frequency_bands == 1:
            return [f"delta_{detector_name}" for detector_name in self._detector_names]
        return [
            f"delta_{detector_name}_{index}"
            for detector_name in self._detector_names
            for index in range(1, self.num_frequency_bands + 1)
        ]

    @property
    def noise_parameter_keys(self):
        keys = []
        if self.infer_alpha:
            keys.extend(self.alpha_parameter_keys)
        if self.infer_delta:
            keys.extend(self.delta_parameter_keys)
        return keys

    def _gaussian_limit_parameter(self, priors, parameter_name, key, fixed_value):
        if not getattr(self, f"infer_{parameter_name}"):
            return float(fixed_value), "fixed"

        prior = priors[key]
        source = "fixed prior" if prior.is_fixed else "prior max"
        return float(prior.maximum), source

    @classmethod
    def _gaussian_limit_statistics(cls, alpha, delta, dimension):
        argument = alpha * delta
        scaled_bessel = cls._scaled_bessel_k_half_integer(dimension, argument)
        next_scaled_bessel = cls._scaled_bessel_k_half_integer(
            dimension + 2, argument
        )
        variance_scale = (delta / alpha) * next_scaled_bessel / scaled_bessel
        std_scale = np.sqrt(variance_scale)

        hyperbolic_logl = cls._compute_bin_log_terms(
            quadratic_forms=[float(dimension)],
            dimensions=[dimension],
            alpha=alpha,
            delta=delta,
        )[0]
        gaussian_logl = (
            -0.5 * dimension * np.log(2.0 * np.pi) - 0.5 * dimension
        )
        return dict(
            variance_scale=variance_scale,
            std_scale=std_scale,
            relative_variance_error=variance_scale - 1.0,
            relative_std_error=std_scale - 1.0,
            log_likelihood_ratio_at_typical_radius=hyperbolic_logl
            - gaussian_logl,
        )

    def print_gaussian_limit_diagnostic(self, priors):
        """Print the closest Gaussian limit available within the shape priors."""
        fixed_alphas = dict(zip(self.alpha_parameter_keys, np.ravel(self._fixed_alpha)))
        fixed_deltas = dict(zip(self.delta_parameter_keys, np.ravel(self._fixed_delta)))

        coordinates = []
        if self.detector_dependent_noise:
            for interferometer in self.interferometers:
                frequencies = interferometer.frequency_array[
                    interferometer.frequency_mask
                ]
                for band_index, band_mask in enumerate(
                    self._get_frequency_band_masks(frequencies)
                ):
                    if np.any(band_mask):
                        coordinates.append((interferometer.name, band_index, 2))
        elif not self.joint:
            active_frequencies = np.unique(
                np.concatenate(
                    [
                        interferometer.frequency_array[interferometer.frequency_mask]
                        for interferometer in self.interferometers
                        if np.any(interferometer.frequency_mask)
                    ]
                )
            )
            for band_index, band_mask in enumerate(
                self._get_frequency_band_masks(active_frequencies)
            ):
                if np.any(band_mask):
                    coordinates.append((None, band_index, 2))
        else:
            active_frequencies = [
                interferometer.frequency_array[interferometer.frequency_mask]
                for interferometer in self.interferometers
                if np.any(interferometer.frequency_mask)
            ]
            network_frequencies, active_counts = np.unique(
                np.concatenate(active_frequencies), return_counts=True
            )
            for band_index, band_mask in enumerate(
                self._get_frequency_band_masks(network_frequencies)
            ):
                for dimension in np.unique(2 * active_counts[band_mask]):
                    coordinates.append((None, band_index, int(dimension)))

        rows = []
        for detector_name, band_index, dimension in coordinates:
            if detector_name is None:
                alpha_key = self.alpha_parameter_keys[band_index]
                delta_key = self.delta_parameter_keys[band_index]
                coordinate = "shared"
            else:
                alpha_key = self._detector_parameter_key(
                    "alpha", detector_name, band_index
                )
                delta_key = self._detector_parameter_key(
                    "delta", detector_name, band_index
                )
                coordinate = detector_name
            if self.num_frequency_bands > 1:
                coordinate = f"{coordinate} band {band_index + 1}"

            alpha, alpha_source = self._gaussian_limit_parameter(
                priors, "alpha", alpha_key, fixed_alphas[alpha_key]
            )
            delta, delta_source = self._gaussian_limit_parameter(
                priors, "delta", delta_key, fixed_deltas[delta_key]
            )
            stats = self._gaussian_limit_statistics(alpha, delta, dimension)
            rows.append(
                (
                    coordinate,
                    dimension,
                    alpha,
                    alpha_source,
                    delta,
                    delta_source,
                    stats,
                )
            )

        closest_row = min(
            rows, key=lambda row: abs(row[6]["relative_variance_error"])
        )
        worst_row = max(
            rows, key=lambda row: abs(row[6]["relative_variance_error"])
        )
        rows_to_print = [("closest available corner", closest_row)]
        if worst_row is not closest_row:
            rows_to_print.append(("worst available corner", worst_row))

        print("\n* Hyperbolic Gaussian-limit diagnostic (per frequency bin):")
        for label, row in rows_to_print:
            coordinate, dimension, alpha, alpha_source, delta, delta_source, stats = row
            print(
                "  {} ({}, d={}): alpha = {:.6g} ({}), delta = {:.6g} ({}), "
                "variance scale = {:.6g} ({:+.3%}), std scale = {:.6g} "
                "({:+.3%}), logL_hyp-logL_gauss at q=d = {:.6g}.".format(
                    label,
                    coordinate,
                    dimension,
                    alpha,
                    alpha_source,
                    delta,
                    delta_source,
                    stats["variance_scale"],
                    stats["relative_variance_error"],
                    stats["std_scale"],
                    stats["relative_std_error"],
                    stats["log_likelihood_ratio_at_typical_radius"],
                )
            )

    @property
    def meta_data(self):
        meta_data = super().meta_data
        meta_data.update(
            likelihood_class=self.__class__,
            alpha=self._fixed_alpha.tolist(),
            delta=self._fixed_delta.tolist(),
            infer_alpha=self.infer_alpha,
            infer_delta=self.infer_delta,
            detector_dependent_noise=self.detector_dependent_noise,
            joint=self.joint,
            num_frequency_bands=self.num_frequency_bands,
            noise_evidence_method=self.noise_evidence_method,
        )
        return meta_data

    def _validate_num_frequency_bands(self, num_frequency_bands):
        try:
            num_frequency_bands = int(num_frequency_bands)
        except (TypeError, ValueError) as exc:
            raise ValueError("num_frequency_bands must be a positive integer") from exc
        if num_frequency_bands < 1:
            raise ValueError("num_frequency_bands must be a positive integer")
        return num_frequency_bands

    @staticmethod
    def _validate_noise_evidence_nlive(noise_evidence_nlive):
        if noise_evidence_nlive is None:
            return None
        try:
            noise_evidence_nlive = int(noise_evidence_nlive)
        except (TypeError, ValueError) as exc:
            raise ValueError("noise_evidence_nlive must be a positive integer") from exc
        if noise_evidence_nlive < 1:
            raise ValueError("noise_evidence_nlive must be a positive integer")
        return noise_evidence_nlive

    @staticmethod
    def _resolve_dlogz_noise(dlogz_noise, dlogZ_noise):
        if dlogZ_noise is not None:
            if dlogz_noise != 0.1 and dlogz_noise != dlogZ_noise:
                raise ValueError(
                    "Received both dlogz_noise and dlogZ_noise with different values"
                )
            dlogz_noise = dlogZ_noise
        try:
            dlogz_noise = float(dlogz_noise)
        except (TypeError, ValueError) as exc:
            raise ValueError("dlogz_noise must be a positive float") from exc
        if dlogz_noise <= 0:
            raise ValueError("dlogz_noise must be a positive float")
        return dlogz_noise

    @staticmethod
    def _resolve_noise_evidence_method(noise_evidence_method):
        try:
            noise_evidence_method = (
                str(noise_evidence_method).strip().lower().replace("-", "_")
            )
        except Exception as exc:
            raise ValueError(
                "noise_evidence_method must be 'quadrature' or 'nested'"
            ) from exc

        aliases = {
            "quad": "quadrature",
            "quadrature": "quadrature",
            "direct_quadrature": "quadrature",
            "dynesty": "nested",
            "nested": "nested",
            "nested_sampling": "nested",
            "ns": "nested",
        }
        if noise_evidence_method not in aliases:
            raise ValueError(
                "noise_evidence_method must be 'quadrature' or 'nested'"
            )
        return aliases[noise_evidence_method]

    def _coerce_parameter_array(self, values, name):
        values = np.asarray(values, dtype=float)
        if not self.detector_dependent_noise:
            if values.ndim == 0:
                values = np.repeat(values[None], self.num_frequency_bands)
            elif values.ndim == 1 and len(values) == 1:
                values = np.repeat(values, self.num_frequency_bands)
            elif values.ndim != 1 or len(values) != self.num_frequency_bands:
                raise ValueError(
                    f"{name} must be a scalar or an array with one entry per frequency band"
                )
            return values.astype(float, copy=False)

        num_detectors = len(self.interferometers)
        if values.ndim == 0:
            values = np.full((num_detectors, self.num_frequency_bands), float(values))
        elif values.ndim == 1:
            if len(values) == 1:
                values = np.full(
                    (num_detectors, self.num_frequency_bands), float(values[0])
                )
            elif len(values) == self.num_frequency_bands:
                values = np.repeat(values[None, :], num_detectors, axis=0)
            elif len(values) == num_detectors and self.num_frequency_bands == 1:
                values = values[:, None]
            else:
                raise ValueError(
                    f"{name} must be a scalar, an array with one entry per frequency band, "
                    "an array with one entry per detector when num_frequency_bands=1, "
                    "or a 2D array with shape (num_detectors, num_frequency_bands)"
                )
        elif values.ndim == 2:
            if values.shape != (num_detectors, self.num_frequency_bands):
                raise ValueError(
                    f"{name} must have shape (num_detectors, num_frequency_bands) "
                    "when detector_dependent_noise=True"
                )
        else:
            raise ValueError(
                f"{name} must be scalar, 1D, or 2D when detector_dependent_noise=True"
            )
        return values.astype(float, copy=False)

    @staticmethod
    def _valid_positive_values(values):
        return np.all(np.isfinite(values)) and np.all(values > 0)

    def _create_frequency_band_edges(self):
        active_frequencies = [
            interferometer.frequency_array[interferometer.frequency_mask]
            for interferometer in self.interferometers
            if np.any(interferometer.frequency_mask)
        ]
        if len(active_frequencies) == 0:
            raise ValueError("No active frequencies available to construct hyperbolic bands")
        frequencies = np.unique(np.concatenate(active_frequencies))
        return np.linspace(frequencies[0], frequencies[-1], self.num_frequency_bands + 1)

    def _detector_parameter_key(self, parameter_name, detector_name, band_index):
        if self.num_frequency_bands == 1:
            return f"{parameter_name}_{detector_name}"
        return f"{parameter_name}_{detector_name}_{band_index + 1}"

    def _get_alpha_values(self, parameters):
        if not self.infer_alpha:
            return self._fixed_alpha

        if "alpha" in parameters:
            return self._coerce_parameter_array(parameters["alpha"], "alpha")

        if not self.detector_dependent_noise:
            if self.num_frequency_bands == 1:
                return self._coerce_parameter_array(
                    parameters.get("alpha", self._fixed_alpha[0]), "alpha"
                )

            return self._coerce_parameter_array(
                [
                    parameters.get(key, default)
                    for key, default in zip(self.alpha_parameter_keys, self._fixed_alpha)
                ],
                "alpha",
            )

        return self._coerce_parameter_array(
            [
                [
                    parameters.get(
                        self._detector_parameter_key("alpha", detector_name, band_index),
                        self._fixed_alpha[detector_index, band_index],
                    )
                    for band_index in range(self.num_frequency_bands)
                ]
                for detector_index, detector_name in enumerate(self._detector_names)
            ],
            "alpha",
        )

    def _get_delta_values(self, parameters):
        if not self.infer_delta:
            return self._fixed_delta

        if "delta" in parameters:
            return self._coerce_parameter_array(parameters["delta"], "delta")

        if not self.detector_dependent_noise:
            if self.num_frequency_bands == 1:
                return self._coerce_parameter_array(
                    parameters.get("delta", self._fixed_delta[0]), "delta"
                )

            return self._coerce_parameter_array(
                [
                    parameters.get(key, default)
                    for key, default in zip(self.delta_parameter_keys, self._fixed_delta)
                ],
                "delta",
            )

        return self._coerce_parameter_array(
            [
                [
                    parameters.get(
                        self._detector_parameter_key("delta", detector_name, band_index),
                        self._fixed_delta[detector_index, band_index],
                    )
                    for band_index in range(self.num_frequency_bands)
                ]
                for detector_index, detector_name in enumerate(self._detector_names)
            ],
            "delta",
        )

    def _store_alpha_values(self, alpha_values):
        if not self.detector_dependent_noise:
            for key, value in zip(self.alpha_parameter_keys, alpha_values):
                self._parameters[key] = float(value)
            return

        for detector_index, detector_name in enumerate(self._detector_names):
            for band_index in range(self.num_frequency_bands):
                self._parameters[
                    self._detector_parameter_key("alpha", detector_name, band_index)
                ] = float(alpha_values[detector_index, band_index])

    def _store_delta_values(self, delta_values):
        if not self.detector_dependent_noise:
            for key, value in zip(self.delta_parameter_keys, delta_values):
                self._parameters[key] = float(value)
            return

        for detector_index, detector_name in enumerate(self._detector_names):
            for band_index in range(self.num_frequency_bands):
                self._parameters[
                    self._detector_parameter_key("delta", detector_name, band_index)
                ] = float(delta_values[detector_index, band_index])

    def _get_interferometer_parameter_values(self, interferometer, parameter_values):
        if not self.detector_dependent_noise:
            return parameter_values
        return parameter_values[self._detector_names.index(interferometer.name)]

    def _get_frequency_band_masks(self, frequencies):
        frequencies = np.asarray(frequencies, dtype=float)
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
    def _map_frequencies_to_network_grid(network_frequencies, frequencies):
        frequencies = np.asarray(frequencies, dtype=float)
        if len(frequencies) == 0:
            return np.array([], dtype=int)

        indices = np.searchsorted(network_frequencies, frequencies)
        if np.any(indices >= len(network_frequencies)):
            return None

        matched_frequencies = network_frequencies[indices]
        frequency_tolerance = 0.0
        if len(network_frequencies) > 1:
            frequency_tolerance = np.min(np.diff(network_frequencies)) * 1e-12

        if not np.all(
            np.isclose(
                matched_frequencies, frequencies, rtol=0.0, atol=frequency_tolerance
            )
        ):
            return None

        return indices

    @staticmethod
    def _scaled_bessel_k_half_integer(dimension, value):
        if dimension < 2 or dimension % 2 != 0:
            return None
        if value <= 0 or not np.isfinite(value):
            return None

        half_integer_order_index = dimension // 2
        inverse_twice_value = 0.5 / value
        polynomial = 0.0
        for index in range(half_integer_order_index + 1):
            coefficient = factorial(half_integer_order_index + index) / (
                factorial(index) * factorial(half_integer_order_index - index)
            )
            polynomial += coefficient * inverse_twice_value ** index

        return np.sqrt(np.pi / (2.0 * value)) * polynomial

    @staticmethod
    def _compute_bin_log_terms(quadratic_forms, dimensions, alpha, delta):
        quadratic_forms = np.asarray(quadratic_forms, dtype=float)
        dimensions = np.asarray(dimensions, dtype=int)

        if quadratic_forms.shape != dimensions.shape:
            return None
        if len(quadratic_forms) == 0:
            return np.array([], dtype=float)
        if alpha <= 0 or delta <= 0 or not np.isfinite(alpha) or not np.isfinite(delta):
            return None
        if np.any(quadratic_forms < 0) or not np.all(np.isfinite(quadratic_forms)):
            return None
        if np.any(dimensions < 2) or np.any(dimensions % 2 != 0):
            return None

        normalization_argument = alpha * delta
        if normalization_argument <= 0 or not np.isfinite(normalization_argument):
            return None

        radial_shift = quadratic_forms / (np.sqrt(delta ** 2 + quadratic_forms) + delta)
        if not np.all(np.isfinite(radial_shift)):
            return None

        log_terms = -alpha * radial_shift
        for dimension in np.unique(dimensions):
            order = 0.5 * (dimension + 1.0)
            scaled_bessel = (
                HyperbolicGravitationalWaveTransient._scaled_bessel_k_half_integer(
                    dimension, normalization_argument
                )
            )
            if not np.isfinite(scaled_bessel) or scaled_bessel <= 0:
                return None

            constant = (
                order * np.log(alpha / delta)
                + 0.5 * (1.0 - dimension) * np.log(2.0 * np.pi)
                - np.log(2.0 * alpha)
                - np.log(scaled_bessel)
            )
            log_terms[dimensions == dimension] += constant

        return log_terms

    def _resolve_likelihood_parameters(self, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)
        if parameters is self._parameters:
            parameters = parameters.copy()
        else:
            merged_parameters = self._parameters.copy()
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

    def _get_frequency_bin_data(
        self,
        interferometers,
        parameters=None,
        waveform_polarizations=None,
    ):
        detector_frequency_data = []
        active_frequencies = []
        for interferometer in interferometers:
            mask = interferometer.frequency_mask
            frequencies = interferometer.frequency_array[mask]
            if len(frequencies) == 0:
                continue

            scale2 = self._compute_scale2(interferometer.power_spectral_density_array[mask])
            if np.any(scale2 <= 0) or not np.all(np.isfinite(scale2)):
                return None

            if waveform_polarizations is None:
                residual = interferometer.frequency_domain_strain[mask]
            else:
                h_f = interferometer.get_detector_response(waveform_polarizations, parameters)
                residual = interferometer.frequency_domain_strain[mask] - h_f[mask]

            quadratic_contribution = (residual.real ** 2 + residual.imag ** 2) / scale2
            if not np.all(np.isfinite(quadratic_contribution)):
                return None

            detector_frequency_data.append(
                (frequencies, quadratic_contribution, np.log(scale2))
            )
            active_frequencies.append(frequencies)

        if len(active_frequencies) == 0:
            empty_float = np.array([], dtype=float)
            empty_int = np.array([], dtype=int)
            return empty_float, empty_float, empty_int, empty_float

        network_frequencies = np.unique(np.concatenate(active_frequencies))
        quadratic_forms = np.zeros(len(network_frequencies), dtype=float)
        log_scale2_terms = np.zeros(len(network_frequencies), dtype=float)
        active_counts = np.zeros(len(network_frequencies), dtype=int)

        for frequencies, quadratic_contribution, log_scale2 in detector_frequency_data:
            indices = self._map_frequencies_to_network_grid(network_frequencies, frequencies)
            if indices is None:
                return None
            quadratic_forms[indices] += quadratic_contribution
            log_scale2_terms[indices] += log_scale2
            active_counts[indices] += 1

        active_mask = active_counts > 0
        return (
            network_frequencies[active_mask],
            quadratic_forms[active_mask],
            2 * active_counts[active_mask],
            log_scale2_terms[active_mask],
        )

    def _compute_network_log_likelihood(
        self,
        interferometers,
        alpha_values,
        delta_values,
        parameters=None,
        waveform_polarizations=None,
        include_log_scale2=True,
    ):
        frequency_bin_data = self._get_frequency_bin_data(
            interferometers=interferometers,
            parameters=parameters,
            waveform_polarizations=waveform_polarizations,
        )
        if frequency_bin_data is None:
            return -np.inf

        frequencies, quadratic_forms, dimensions, log_scale2_terms = frequency_bin_data
        if len(frequencies) == 0:
            return 0.0

        band_masks = self._get_frequency_band_masks(frequencies)
        logl = 0.0
        for alpha, delta, band_mask in zip(alpha_values, delta_values, band_masks):
            if not np.any(band_mask):
                continue

            band_log_terms = self._compute_bin_log_terms(
                quadratic_forms=quadratic_forms[band_mask],
                dimensions=dimensions[band_mask],
                alpha=alpha,
                delta=delta,
            )
            if band_log_terms is None or not np.all(np.isfinite(band_log_terms)):
                return -np.inf
            if include_log_scale2:
                band_log_terms = band_log_terms - log_scale2_terms[band_mask]
            logl += np.sum(band_log_terms)

        return float(logl)

    def _sum_detector_log_likelihoods(
        self,
        *,
        interferometers,
        alpha_values,
        delta_values,
        parameters=None,
        waveform_polarizations=None,
        include_log_scale2=True,
    ):
        logl = 0.0
        for interferometer in interferometers:
            detector_logl = self._compute_network_log_likelihood(
                interferometers=[interferometer],
                alpha_values=self._get_interferometer_parameter_values(
                    interferometer, alpha_values
                ),
                delta_values=self._get_interferometer_parameter_values(
                    interferometer, delta_values
                ),
                parameters=parameters,
                waveform_polarizations=waveform_polarizations,
                include_log_scale2=include_log_scale2,
            )
            if not np.isfinite(detector_logl):
                return -np.inf
            logl += detector_logl
        return float(logl)

    def _noise_log_likelihood_from_parameters(self, parameters):
        alpha_values, delta_values = self._get_active_shape_parameters(
            parameters, update_state=False
        )
        if alpha_values is None or delta_values is None:
            return np.nan_to_num(-np.inf)

        if self.joint:
            logl = self._compute_network_log_likelihood(
                interferometers=self.interferometers,
                alpha_values=alpha_values,
                delta_values=delta_values,
            )
        else:
            logl = self._sum_detector_log_likelihoods(
                interferometers=self.interferometers,
                alpha_values=alpha_values,
                delta_values=delta_values,
            )
        if not np.isfinite(logl):
            return np.nan_to_num(-np.inf)
        return float(logl)

    def _get_default_shape_parameter_dict(self):
        if self._parameters is None:
            parameters = dict()
        else:
            parameters = self._parameters.copy()

        shape_parameters = dict()
        alpha_values = self._get_alpha_values(parameters)
        delta_values = self._get_delta_values(parameters)
        if self.infer_alpha:
            shape_parameters.update(
                {
                    key: float(value)
                    for key, value in zip(self.alpha_parameter_keys, np.ravel(alpha_values))
                }
            )
        if self.infer_delta:
            shape_parameters.update(
                {
                    key: float(value)
                    for key, value in zip(self.delta_parameter_keys, np.ravel(delta_values))
                }
            )
        return shape_parameters

    def _get_noise_evidence_priors(self, priors):
        if priors is None:
            priors = PriorDict()
        elif not isinstance(priors, PriorDict):
            priors = PriorDict(priors)

        default_shape_parameters = self._get_default_shape_parameter_dict()
        noise_priors = PriorDict()
        for key, value in default_shape_parameters.items():
            if key in priors:
                noise_priors[key] = deepcopy(priors[key])
            else:
                noise_priors[key] = DeltaFunction(peak=value, name=key)
        return noise_priors

    @staticmethod
    def _get_noise_evidence_quadrature_points():
        edge_points = np.geomspace(1e-12, 1e-1, 12)
        return np.unique(
            np.concatenate(
                ([0.0, 0.25, 0.5, 0.75, 1.0], edge_points, 1.0 - edge_points)
            )
        )

    def _noise_log_likelihood_from_rescaled_priors(
        self, keys, priors, unit_values, base_parameters
    ):
        parameters = base_parameters.copy()
        for key, prior, unit_value in zip(keys, priors, unit_values):
            parameters[key] = float(np.asarray(prior.rescale(unit_value), dtype=float))
        return self._noise_log_likelihood_from_parameters(parameters)

    def _noise_log_evidence_by_quadrature(self, noise_priors):
        keys = list(noise_priors.non_fixed_keys)
        priors = [noise_priors[key] for key in keys]
        dimension = len(keys)
        if dimension not in (1, 2):
            raise ValueError(
                "Hyperbolic noise-evidence quadrature supports one or two "
                "sampled noise parameters"
            )

        base_parameters = self._get_default_shape_parameter_dict()
        reference_points = self._get_noise_evidence_quadrature_points()

        candidate_logls = [
            self._noise_log_likelihood_from_parameters(base_parameters)
        ]
        candidate_logls.extend(
            self._noise_log_likelihood_from_rescaled_priors(
                keys=keys,
                priors=priors,
                unit_values=unit_values,
                base_parameters=base_parameters,
            )
            for unit_values in product(reference_points, repeat=dimension)
        )
        finite_logls = [value for value in candidate_logls if np.isfinite(value)]
        if not finite_logls:
            return np.nan_to_num(-np.inf)
        logl_reference = max(finite_logls)

        def scaled_integrand(unit_values):
            logl = self._noise_log_likelihood_from_rescaled_priors(
                keys=keys,
                priors=priors,
                unit_values=unit_values,
                base_parameters=base_parameters,
            )
            if not np.isfinite(logl):
                return 0.0
            return float(np.exp(logl - logl_reference))

        quadrature_kwargs = dict(
            epsabs=self._NOISE_EVIDENCE_QUADRATURE_EPSABS,
            epsrel=self._NOISE_EVIDENCE_QUADRATURE_EPSREL,
            limit=self._NOISE_EVIDENCE_QUADRATURE_LIMIT,
            points=reference_points[1:-1],
        )
        if dimension == 1:
            integral, _ = quad(
                lambda unit_value: scaled_integrand((unit_value,)),
                0.0,
                1.0,
                **quadrature_kwargs,
            )
        else:
            def outer_integrand(unit_value_2):
                inner_integral, _ = quad(
                    lambda unit_value_1: scaled_integrand(
                        (unit_value_1, unit_value_2)
                    ),
                    0.0,
                    1.0,
                    **quadrature_kwargs,
                )
                return float(inner_integral)

            integral, _ = quad(
                outer_integrand,
                0.0,
                1.0,
                **quadrature_kwargs,
            )
        if not np.isfinite(integral) or integral <= 0:
            raise RuntimeError(
                "Hyperbolic noise-evidence quadrature failed to return a "
                "positive finite integral"
            )
        return float(logl_reference + np.log(integral))

    def _noise_log_evidence_by_nested_sampling(self, noise_priors):
        from ...core.sampler import run_sampler

        return run_sampler(
            likelihood=_HyperbolicNoiseOnlyLikelihood(self),
            priors=noise_priors,
            label=f"{getattr(self, 'label', 'label')}_noise",
            outdir=getattr(self, "outdir", "outdir"),
            sampler="dynesty",
            use_ratio=False,
            plot=False,
            save=False,
            # Keep the auxiliary noise-evidence calculation single-process to
            # avoid duplicating the main analysis worker pool in memory.
            npool=1,
            nlive=(
                self.noise_evidence_nlive
                if self.noise_evidence_nlive is not None
                else max(100, 25 * len(noise_priors.non_fixed_keys))
            ),
            dlogz=self.dlogz_noise,
            print_progress=False,
            check_point=False,
            resume=False,
        )

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

        if self.joint:
            logl = self._compute_network_log_likelihood(
                interferometers=self.interferometers,
                alpha_values=alpha_values,
                delta_values=delta_values,
                parameters=parameters,
                waveform_polarizations=pols,
            )
        else:
            logl = self._sum_detector_log_likelihoods(
                interferometers=self.interferometers,
                alpha_values=alpha_values,
                delta_values=delta_values,
                parameters=parameters,
                waveform_polarizations=pols,
            )
        if not np.isfinite(logl):
            return np.nan_to_num(-np.inf)
        return float(logl)

    def noise_log_likelihood(self):
        return self._noise_log_likelihood_from_parameters(self._parameters.copy())

    def noise_log_evidence(self, priors=None, sampler=None, result=None, npool=1):
        if priors is None:
            sampled_parameters = []
        elif isinstance(priors, PriorDict):
            sampled_parameters = priors.non_fixed_keys
        else:
            sampled_parameters = PriorDict(priors).non_fixed_keys

        if not self.has_parameter_dependent_noise_likelihood(sampled_parameters):
            return self.noise_log_likelihood()

        noise_priors = self._get_noise_evidence_priors(priors)
        if len(noise_priors.non_fixed_keys) == 0:
            return self._noise_log_likelihood_from_parameters(
                self._get_default_shape_parameter_dict()
            )

        if self.noise_evidence_method == "quadrature":
            if len(noise_priors.non_fixed_keys) <= 2:
                return self._noise_log_evidence_by_quadrature(noise_priors)
            logger.info(
                "Hyperbolic noise-evidence quadrature supports at most two "
                "sampled noise parameters; falling back to nested sampling."
            )

        noise_result = self._noise_log_evidence_by_nested_sampling(noise_priors)
        return float(noise_result.log_evidence)

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

        if self.joint:
            signal_logl = self._compute_network_log_likelihood(
                interferometers=self.interferometers,
                alpha_values=alpha_values,
                delta_values=delta_values,
                parameters=parameters,
                waveform_polarizations=pols,
                include_log_scale2=False,
            )
            noise_logl = self._compute_network_log_likelihood(
                interferometers=self.interferometers,
                alpha_values=alpha_values,
                delta_values=delta_values,
                include_log_scale2=False,
            )
        else:
            signal_logl = self._sum_detector_log_likelihoods(
                interferometers=self.interferometers,
                alpha_values=alpha_values,
                delta_values=delta_values,
                parameters=parameters,
                waveform_polarizations=pols,
                include_log_scale2=False,
            )
            noise_logl = self._sum_detector_log_likelihoods(
                interferometers=self.interferometers,
                alpha_values=alpha_values,
                delta_values=delta_values,
                include_log_scale2=False,
            )
        if not np.isfinite(signal_logl) or not np.isfinite(noise_logl):
            return np.nan_to_num(-np.inf)
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
            detector_signal_logl = self._compute_network_log_likelihood(
                interferometers=[interferometer],
                alpha_values=self._get_interferometer_parameter_values(
                    interferometer, alpha_values
                ),
                delta_values=self._get_interferometer_parameter_values(
                    interferometer, delta_values
                ),
                parameters=parameters,
                waveform_polarizations=pols,
                include_log_scale2=False,
            )
            detector_noise_logl = self._compute_network_log_likelihood(
                interferometers=[interferometer],
                alpha_values=self._get_interferometer_parameter_values(
                    interferometer, alpha_values
                ),
                delta_values=self._get_interferometer_parameter_values(
                    interferometer, delta_values
                ),
                include_log_scale2=False,
            )
            parameters[f"{interferometer.name}_log_likelihood"] = float(
                detector_signal_logl - detector_noise_logl
            )

        return parameters.copy()
