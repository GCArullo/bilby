from copy import deepcopy
from itertools import product

import numpy as np
from scipy.integrate import quad

from ...core.likelihood import Likelihood, _fallback_to_parameters
from ...core.prior import DeltaFunction, PriorDict
from ...core.utils import logger
from .base import GravitationalWaveTransient


class _GaussianParametricNoiseOnlyLikelihood(Likelihood):
    """Auxiliary likelihood for marginalizing parametric PSD evidence."""

    def __init__(self, gaussian_parametric_likelihood):
        super().__init__()
        self._parameters.update(
            gaussian_parametric_likelihood._get_default_psd_parameter_dict()
        )
        self.gaussian_parametric_likelihood = gaussian_parametric_likelihood

    def log_likelihood(self, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)
        return (
            self.gaussian_parametric_likelihood._noise_log_likelihood_from_parameters(
                parameters
            )
        )

    def noise_log_likelihood(self):
        return 0.0


class GaussianParametricGravitationalWaveTransient(GravitationalWaveTransient):
    """
    Gaussian frequency-domain likelihood with sampled PSD scale parameters.

    The PSD in each frequency band is multiplied by ``10**log_psd_scale``. With
    all log scales fixed to zero this reduces to the standard Gaussian
    ``GravitationalWaveTransient``. The scales can be shared by all detectors
    or sampled independently for each detector.
    """

    _NOISE_EVIDENCE_QUADRATURE_EPSABS = 0.0
    _NOISE_EVIDENCE_QUADRATURE_EPSREL = 1e-8
    _NOISE_EVIDENCE_QUADRATURE_LIMIT = 200

    def __init__(
        self,
        interferometers,
        waveform_generator,
        log_psd_scale=0.0,
        infer_log_psd_scale=False,
        num_psd_frequency_bands=1,
        detector_dependent_noise=False,
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
        log_psd_scale : float, array-like
            Base-10 logarithm of the multiplicative PSD scale in each frequency
            band. A value of zero keeps the input PSD unchanged.
        infer_log_psd_scale : bool
            If True, treat the PSD scale as sampled. For a single band this uses
            ``log_psd_scale``; for multiple bands this uses
            ``log_psd_scale_1``, ..., ``log_psd_scale_N``. With
            ``detector_dependent_noise=True``, the names are detector-specific:
            ``log_psd_scale_H1``, or ``log_psd_scale_H1_1``, ... .
        num_psd_frequency_bands : int
            Number of contiguous frequency bands spanning the active analysis
            range. Each band has its own PSD scale.
        detector_dependent_noise : bool
            If True, use a distinct PSD scale for each detector and frequency
            band. If False, share each band's scale across all detectors.
        noise_evidence_nlive : int, optional
            Number of live points to use when the noise evidence requires an
            auxiliary nested-sampling run.
        dlogz_noise, dlogZ_noise : float, optional
            Stopping criterion for the auxiliary nested-sampling run used to
            evaluate the noise evidence. ``dlogZ_noise`` is accepted as an alias.
        noise_evidence_method : str, optional
            ``"quadrature"`` performs direct 1D or 2D integration over the prior
            transform, falling back to nested sampling for higher dimensions.
            ``"nested"`` always uses the auxiliary dynesty run.
        kwargs :
            Passed to ``GravitationalWaveTransient``.
        """
        if (
            time_marginalization
            or distance_marginalization
            or phase_marginalization
            or calibration_marginalization
        ):
            raise ValueError(
                "GaussianParametricGravitationalWaveTransient does not support "
                "time, distance, phase, or calibration marginalization"
            )

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

        self.num_psd_frequency_bands = self._validate_num_psd_frequency_bands(
            num_psd_frequency_bands
        )
        self.detector_dependent_noise = bool(detector_dependent_noise)
        self._detector_names = [ifo.name for ifo in self.interferometers]
        self._fixed_log_psd_scale = self._coerce_log_psd_scale_array(log_psd_scale)
        self.infer_log_psd_scale = bool(infer_log_psd_scale)
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

        if not self._valid_log_psd_scale_values(self._fixed_log_psd_scale):
            raise ValueError("All log_psd_scale values must give finite positive scales")

        if self.infer_log_psd_scale:
            for key, value in zip(
                self.log_psd_scale_parameter_keys,
                np.ravel(self._fixed_log_psd_scale),
            ):
                self._parameters.setdefault(key, float(value))

    @property
    def log_psd_scale(self):
        values = self._get_log_psd_scale_values(self._parameters)
        if self.detector_dependent_noise:
            if self.num_psd_frequency_bands == 1:
                return values[:, 0].copy()
            return values.copy()
        if self.num_psd_frequency_bands == 1:
            return float(values[0])
        return values.copy()

    @property
    def log_psd_scale_parameter_keys(self):
        if self.detector_dependent_noise:
            if self.num_psd_frequency_bands == 1:
                return [
                    f"log_psd_scale_{detector_name}"
                    for detector_name in self._detector_names
                ]
            return [
                f"log_psd_scale_{detector_name}_{index}"
                for detector_name in self._detector_names
                for index in range(1, self.num_psd_frequency_bands + 1)
            ]
        if self.num_psd_frequency_bands == 1:
            return ["log_psd_scale"]
        return [
            f"log_psd_scale_{index}"
            for index in range(1, self.num_psd_frequency_bands + 1)
        ]

    @property
    def noise_parameter_keys(self):
        if not self.infer_log_psd_scale:
            return []
        return list(self.log_psd_scale_parameter_keys)

    @property
    def meta_data(self):
        meta_data = super().meta_data
        meta_data.update(
            likelihood_class=self.__class__,
            log_psd_scale=self._fixed_log_psd_scale.tolist(),
            infer_log_psd_scale=self.infer_log_psd_scale,
            num_psd_frequency_bands=self.num_psd_frequency_bands,
            detector_dependent_noise=self.detector_dependent_noise,
            noise_evidence_method=self.noise_evidence_method,
        )
        return meta_data

    @staticmethod
    def _validate_num_psd_frequency_bands(num_psd_frequency_bands):
        try:
            num_psd_frequency_bands = int(num_psd_frequency_bands)
        except (TypeError, ValueError) as exc:
            raise ValueError("num_psd_frequency_bands must be a positive integer") from exc
        if num_psd_frequency_bands < 1:
            raise ValueError("num_psd_frequency_bands must be a positive integer")
        return num_psd_frequency_bands

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

    def _coerce_log_psd_scale_array(self, log_psd_scale):
        values = np.asarray(log_psd_scale, dtype=float)
        if not self.detector_dependent_noise:
            if values.ndim == 0:
                values = np.repeat(values[None], self.num_psd_frequency_bands)
            elif values.ndim == 1 and len(values) == 1:
                values = np.repeat(values, self.num_psd_frequency_bands)
            elif values.ndim != 1 or len(values) != self.num_psd_frequency_bands:
                raise ValueError(
                    "log_psd_scale must be a scalar or an array with one entry "
                    "per PSD frequency band"
                )
            return values.astype(float, copy=False)

        shape = (len(self._detector_names), self.num_psd_frequency_bands)
        if values.ndim == 0:
            values = np.full(shape, float(values))
        elif values.ndim == 1:
            if len(values) == 1:
                values = np.full(shape, float(values[0]))
            elif len(values) == self.num_psd_frequency_bands:
                values = np.tile(values, (len(self._detector_names), 1))
            elif (
                len(values) == len(self._detector_names)
                and self.num_psd_frequency_bands == 1
            ):
                values = values[:, None]
            else:
                raise ValueError(
                    "log_psd_scale must be scalar, have one entry per frequency "
                    "band, one entry per detector when num_psd_frequency_bands=1, "
                    "or have shape (num_detectors, num_psd_frequency_bands)"
                )
        elif values.ndim == 2:
            if values.shape != shape:
                raise ValueError(
                    "log_psd_scale must have shape "
                    "(num_detectors, num_psd_frequency_bands)"
                )
        else:
            raise ValueError(
                "log_psd_scale must be scalar, one-dimensional, or "
                "two-dimensional when detector_dependent_noise=True"
            )
        return values.astype(float, copy=False)

    @staticmethod
    def _valid_log_psd_scale_values(values):
        scales = np.power(10.0, values)
        return (
            np.all(np.isfinite(values))
            and np.all(np.isfinite(scales))
            and np.all(scales > 0)
        )

    def _create_frequency_band_edges(self):
        active_frequencies = [
            interferometer.frequency_array[interferometer.frequency_mask]
            for interferometer in self.interferometers
            if np.any(interferometer.frequency_mask)
        ]
        if len(active_frequencies) == 0:
            raise ValueError("No active frequencies available to construct PSD bands")
        frequencies = np.unique(np.concatenate(active_frequencies))
        return np.linspace(
            frequencies[0],
            frequencies[-1],
            self.num_psd_frequency_bands + 1,
        )

    def _get_frequency_band_masks(self, frequencies):
        frequencies = np.asarray(frequencies, dtype=float)
        band_masks = []
        for index, (lower, upper) in enumerate(
            zip(self._frequency_band_edges[:-1], self._frequency_band_edges[1:])
        ):
            if index == self.num_psd_frequency_bands - 1:
                band_mask = (frequencies >= lower) & (frequencies <= upper)
            else:
                band_mask = (frequencies >= lower) & (frequencies < upper)
            band_masks.append(band_mask)
        return band_masks

    def _get_log_psd_scale_values(self, parameters):
        if not self.infer_log_psd_scale:
            return self._fixed_log_psd_scale

        if self.detector_dependent_noise:
            if "log_psd_scale" in parameters:
                return self._coerce_log_psd_scale_array(
                    parameters["log_psd_scale"]
                )
            return self._coerce_log_psd_scale_array(
                [
                    [
                        parameters.get(
                            self._detector_log_psd_scale_parameter_key(
                                detector_name, band_index
                            ),
                            self._fixed_log_psd_scale[
                                detector_index, band_index
                            ],
                        )
                        for band_index in range(self.num_psd_frequency_bands)
                    ]
                    for detector_index, detector_name in enumerate(
                        self._detector_names
                    )
                ]
            )

        if self.num_psd_frequency_bands == 1:
            return self._coerce_log_psd_scale_array(
                parameters.get("log_psd_scale", self._fixed_log_psd_scale[0])
            )

        if "log_psd_scale" in parameters:
            return self._coerce_log_psd_scale_array(parameters["log_psd_scale"])

        return self._coerce_log_psd_scale_array(
            [
                parameters.get(key, default)
                for key, default in zip(
                    self.log_psd_scale_parameter_keys,
                    self._fixed_log_psd_scale,
                )
            ]
        )

    def _store_log_psd_scale_values(self, log_psd_scale_values):
        for key, value in zip(
            self.log_psd_scale_parameter_keys,
            np.ravel(log_psd_scale_values),
        ):
            self._parameters[key] = float(value)

    def _detector_log_psd_scale_parameter_key(self, detector_name, band_index):
        if self.num_psd_frequency_bands == 1:
            return f"log_psd_scale_{detector_name}"
        return f"log_psd_scale_{detector_name}_{band_index + 1}"

    def _get_detector_log_psd_scale_values(
        self, log_psd_scale_values, detector_name
    ):
        if not self.detector_dependent_noise:
            return log_psd_scale_values
        detector_index = self._detector_names.index(detector_name)
        return log_psd_scale_values[detector_index]

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

    def _get_active_log_psd_scale_values(self, parameters, update_state=False):
        log_psd_scale_values = self._get_log_psd_scale_values(parameters)
        if update_state and self.infer_log_psd_scale:
            self._store_log_psd_scale_values(log_psd_scale_values)
        if not self._valid_log_psd_scale_values(log_psd_scale_values):
            return None
        return log_psd_scale_values

    def _compute_scale2(self, power_spectral_density):
        # Bilby's frequency-domain convention gives Var(Re n_k) = Var(Im n_k) = S_n(f_k) T / 4.
        return power_spectral_density * self.waveform_generator.duration / 4.0

    def _compute_detector_log_likelihood(
        self,
        interferometer,
        log_psd_scale_values,
        parameters=None,
        waveform_polarizations=None,
        include_log_normalization=True,
    ):
        mask = interferometer.frequency_mask
        frequencies = interferometer.frequency_array[mask]
        scale2 = self._compute_scale2(interferometer.power_spectral_density_array[mask])
        if np.any(scale2 <= 0) or not np.all(np.isfinite(scale2)):
            return -np.inf

        if waveform_polarizations is None:
            residual = interferometer.frequency_domain_strain[mask]
        else:
            h_f = interferometer.get_detector_response(waveform_polarizations, parameters)
            residual = interferometer.frequency_domain_strain[mask] - h_f[mask]

        abs2 = residual.real ** 2 + residual.imag ** 2
        band_masks = self._get_frequency_band_masks(frequencies)

        logl = 0.0
        for log_psd_scale, band_mask in zip(log_psd_scale_values, band_masks):
            if not np.any(band_mask):
                continue

            level = 10.0 ** log_psd_scale
            band_scale2 = scale2[band_mask] * level
            if np.any(band_scale2 <= 0) or not np.all(np.isfinite(band_scale2)):
                return -np.inf

            band_logl = -abs2[band_mask] / (2.0 * band_scale2)
            if include_log_normalization:
                band_logl -= np.log(2.0 * np.pi * band_scale2)
            logl += np.sum(band_logl)

        return float(logl)

    def _noise_log_likelihood_from_parameters(self, parameters):
        log_psd_scale_values = self._get_active_log_psd_scale_values(
            parameters,
            update_state=False,
        )
        if log_psd_scale_values is None:
            return np.nan_to_num(-np.inf)

        logl = 0.0
        for ifo in self.interferometers:
            detector_logl = self._compute_detector_log_likelihood(
                interferometer=ifo,
                log_psd_scale_values=self._get_detector_log_psd_scale_values(
                    log_psd_scale_values, ifo.name
                ),
            )
            if not np.isfinite(detector_logl):
                return np.nan_to_num(-np.inf)
            logl += detector_logl
        return float(logl)

    def _get_default_psd_parameter_dict(self):
        if self._parameters is None:
            parameters = dict()
        else:
            parameters = self._parameters.copy()

        log_psd_scale_values = self._get_log_psd_scale_values(parameters)
        if not self.infer_log_psd_scale:
            return dict()
        return {
            key: float(value)
            for key, value in zip(
                self.log_psd_scale_parameter_keys,
                np.ravel(log_psd_scale_values),
            )
        }

    def _get_noise_evidence_priors(self, priors):
        if priors is None:
            priors = PriorDict()
        elif not isinstance(priors, PriorDict):
            priors = PriorDict(priors)

        default_psd_parameters = self._get_default_psd_parameter_dict()
        noise_priors = PriorDict()
        for key, value in default_psd_parameters.items():
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
        self,
        keys,
        priors,
        unit_values,
        base_parameters,
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
                "Gaussian-parametric noise-evidence quadrature supports one or two "
                "sampled PSD parameters"
            )

        base_parameters = self._get_default_psd_parameter_dict()
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
                "Gaussian-parametric noise-evidence quadrature failed to return a "
                "positive finite integral"
            )
        return float(logl_reference + np.log(integral))

    def _noise_log_evidence_by_nested_sampling(self, noise_priors):
        from ...core.sampler import run_sampler

        return run_sampler(
            likelihood=_GaussianParametricNoiseOnlyLikelihood(self),
            priors=noise_priors,
            label=f"{getattr(self, 'label', 'label')}_noise",
            outdir=getattr(self, "outdir", "outdir"),
            sampler="dynesty",
            use_ratio=False,
            plot=False,
            save=False,
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
        log_psd_scale_values = self._get_active_log_psd_scale_values(
            parameters,
            update_state=True,
        )
        if log_psd_scale_values is None:
            return np.nan_to_num(-np.inf)

        pols = self.waveform_generator.frequency_domain_strain(parameters)
        if pols is None:
            return np.nan_to_num(-np.inf)

        logl = 0.0
        for ifo in self.interferometers:
            detector_logl = self._compute_detector_log_likelihood(
                interferometer=ifo,
                log_psd_scale_values=self._get_detector_log_psd_scale_values(
                    log_psd_scale_values, ifo.name
                ),
                parameters=parameters,
                waveform_polarizations=pols,
            )
            if not np.isfinite(detector_logl):
                return np.nan_to_num(-np.inf)
            logl += detector_logl
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
                self._get_default_psd_parameter_dict()
            )

        if self.noise_evidence_method == "quadrature":
            if len(noise_priors.non_fixed_keys) <= 2:
                return self._noise_log_evidence_by_quadrature(noise_priors)
            logger.info(
                "Gaussian-parametric noise-evidence quadrature supports at most two "
                "sampled PSD parameters; falling back to nested sampling."
            )

        noise_result = self._noise_log_evidence_by_nested_sampling(noise_priors)
        return float(noise_result.log_evidence)

    def log_likelihood_ratio(self, parameters=None):
        parameters = self._resolve_likelihood_parameters(parameters)
        log_psd_scale_values = self._get_active_log_psd_scale_values(
            parameters,
            update_state=True,
        )
        if log_psd_scale_values is None:
            return np.nan_to_num(-np.inf)

        pols = self.waveform_generator.frequency_domain_strain(parameters)
        if pols is None:
            return np.nan_to_num(-np.inf)

        signal_logl = 0.0
        noise_logl = 0.0
        for ifo in self.interferometers:
            detector_log_psd_scale_values = (
                self._get_detector_log_psd_scale_values(
                    log_psd_scale_values, ifo.name
                )
            )
            detector_signal_logl = self._compute_detector_log_likelihood(
                interferometer=ifo,
                log_psd_scale_values=detector_log_psd_scale_values,
                parameters=parameters,
                waveform_polarizations=pols,
                include_log_normalization=False,
            )
            detector_noise_logl = self._compute_detector_log_likelihood(
                interferometer=ifo,
                log_psd_scale_values=detector_log_psd_scale_values,
                include_log_normalization=False,
            )
            if not np.isfinite(detector_signal_logl) or not np.isfinite(
                detector_noise_logl
            ):
                return np.nan_to_num(-np.inf)
            signal_logl += detector_signal_logl
            noise_logl += detector_noise_logl
        return float(signal_logl - noise_logl)

    def compute_per_detector_log_likelihood(self, parameters=None):
        parameters = self._resolve_likelihood_parameters(parameters)
        log_psd_scale_values = self._get_active_log_psd_scale_values(
            parameters,
            update_state=True,
        )
        if log_psd_scale_values is None:
            for interferometer in self.interferometers:
                parameters[f"{interferometer.name}_log_likelihood"] = np.nan_to_num(
                    -np.inf
                )
            return parameters.copy()

        pols = self.waveform_generator.frequency_domain_strain(parameters)
        if pols is None:
            for interferometer in self.interferometers:
                parameters[f"{interferometer.name}_log_likelihood"] = np.nan_to_num(
                    -np.inf
                )
            return parameters.copy()

        for interferometer in self.interferometers:
            parameters[f"{interferometer.name}_log_likelihood"] = (
                self._compute_detector_log_likelihood(
                    interferometer=interferometer,
                    log_psd_scale_values=self._get_detector_log_psd_scale_values(
                        log_psd_scale_values, interferometer.name
                    ),
                    parameters=parameters,
                    waveform_polarizations=pols,
                )
            )
        return parameters.copy()


class WhittleGravitationalWaveTransient(
    GaussianParametricGravitationalWaveTransient
):
    """Backward-compatible name for the Gaussian-parametric likelihood."""
