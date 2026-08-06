import numpy as np

from ...core.likelihood import _fallback_to_parameters
from .base import GravitationalWaveTransient
from .hyperbolic import HyperbolicGravitationalWaveTransient
from .studentt import StudentTGravitationalWaveTransient


def _normalise_detector_likelihood(detector_likelihood, key):
    aliases = {
        "gauss": "gaussian",
        "gaussian": "gaussian",
        "normal": "gaussian",
        "hyp": "hyperbolic",
        "hyperbolic": "hyperbolic",
        "student": "student-t",
        "student-t": "student-t",
        "student_t": "student-t",
        "studentt": "student-t",
    }
    try:
        return aliases[str(detector_likelihood).lower()]
    except Exception as exc:
        raise ValueError(
            f"Unknown {key} '{detector_likelihood}'. Available options are "
            "'gaussian', 'student-t', and 'hyperbolic'."
        ) from exc


def _resolve_detector_likelihoods(detector_likelihoods, detector_names):
    if isinstance(detector_likelihoods, dict):
        missing = [name for name in detector_names if name not in detector_likelihoods]
        extra = [name for name in detector_likelihoods if name not in detector_names]
        if missing or extra:
            message = []
            if missing:
                message.append("missing " + ", ".join(missing))
            if extra:
                message.append("unknown " + ", ".join(extra))
            raise ValueError("Detector likelihood mapping has " + " and ".join(message))
        return {
            name: _normalise_detector_likelihood(
                detector_likelihoods[name], f"detector_likelihoods[{name}]"
            )
            for name in detector_names
        }

    likelihood = _normalise_detector_likelihood(
        detector_likelihoods, "detector_likelihoods"
    )
    return {name: likelihood for name in detector_names}


class MixedGravitationalWaveTransient(GravitationalWaveTransient):
    """A frequency-domain GW likelihood with one noise family per detector.

    ``detector_likelihoods`` is either one family name or a mapping from
    interferometer name to one of ``gaussian``, ``student-t``, or ``hyperbolic``.
    Every detector is evaluated independently: hyperbolic detectors use the
    factorised per-detector density (``joint=False``), matching the default of
    :class:`HyperbolicGravitationalWaveTransient`. Joint network densities are
    not available here.

    Marginalizations are unavailable because their existing derivations assume a
    single Gaussian network likelihood.
    """

    def __init__(
        self,
        interferometers,
        waveform_generator,
        detector_likelihoods="gaussian",
        nu=8.0,
        infer_nu=False,
        alpha=10.0,
        delta=1.0,
        infer_alpha=False,
        infer_delta=False,
        num_frequency_bands=1,
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
        if any(
            [
                time_marginalization,
                distance_marginalization,
                phase_marginalization,
                calibration_marginalization,
            ]
        ):
            raise ValueError(
                "MixedGravitationalWaveTransient does not support "
                "time, distance, phase, or calibration marginalization"
            )

        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            time_marginalization=False,
            distance_marginalization=False,
            phase_marginalization=False,
            calibration_marginalization=False,
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

        self.detector_dependent_noise = bool(detector_dependent_noise)
        self._detector_names = [interferometer.name for interferometer in self.interferometers]
        self.detector_likelihoods = _resolve_detector_likelihoods(
            detector_likelihoods, self._detector_names
        )
        self._interferometers_by_name = {
            interferometer.name: interferometer for interferometer in self.interferometers
        }
        self.gaussian_detector_names = self._detectors_with_likelihood("gaussian")
        self.student_t_detector_names = self._detectors_with_likelihood("student-t")
        self.hyperbolic_detector_names = self._detectors_with_likelihood("hyperbolic")

        self._student_likelihood = self._make_student_likelihood(
            nu=nu,
            infer_nu=infer_nu,
            num_frequency_bands=num_frequency_bands,
            noise_evidence_nlive=noise_evidence_nlive,
            dlogz_noise=dlogz_noise,
            dlogZ_noise=dlogZ_noise,
            noise_evidence_method=noise_evidence_method,
        )
        self._hyperbolic_likelihood = self._make_hyperbolic_likelihood(
            alpha=alpha,
            delta=delta,
            infer_alpha=infer_alpha,
            infer_delta=infer_delta,
            num_frequency_bands=num_frequency_bands,
            noise_evidence_nlive=noise_evidence_nlive,
            dlogz_noise=dlogz_noise,
            dlogZ_noise=dlogZ_noise,
            noise_evidence_method=noise_evidence_method,
        )

        for likelihood in [self._student_likelihood, self._hyperbolic_likelihood]:
            if likelihood is not None:
                for key in likelihood.noise_parameter_keys:
                    self._parameters.setdefault(key, likelihood._parameters[key])

    def _detectors_with_likelihood(self, likelihood_name):
        return [
            name for name in self._detector_names
            if self.detector_likelihoods[name] == likelihood_name
        ]

    def _subset_interferometers(self, detector_names):
        return [self._interferometers_by_name[name] for name in detector_names]

    def _subset_detector_values(self, values, detector_names, num_frequency_bands):
        """Select fixed detector-dependent values from a full network array."""
        if not self.detector_dependent_noise:
            return values

        values = np.asarray(values)
        detector_indices = [self._detector_names.index(name) for name in detector_names]
        if values.ndim == 2 and values.shape[0] == len(self._detector_names):
            return values[detector_indices]
        if (
            num_frequency_bands == 1
            and values.ndim == 1
            and values.shape == (len(self._detector_names),)
        ):
            return values[detector_indices]
        return values

    def _make_student_likelihood(self, **kwargs):
        if len(self.student_t_detector_names) == 0:
            return None
        kwargs["nu"] = self._subset_detector_values(
            kwargs["nu"], self.student_t_detector_names, kwargs["num_frequency_bands"]
        )
        return StudentTGravitationalWaveTransient(
            interferometers=self._subset_interferometers(self.student_t_detector_names),
            waveform_generator=self.waveform_generator,
            detector_dependent_noise=self.detector_dependent_noise,
            **kwargs,
        )

    def _make_hyperbolic_likelihood(self, **kwargs):
        if len(self.hyperbolic_detector_names) == 0:
            return None
        kwargs["alpha"] = self._subset_detector_values(
            kwargs["alpha"],
            self.hyperbolic_detector_names,
            kwargs["num_frequency_bands"],
        )
        kwargs["delta"] = self._subset_detector_values(
            kwargs["delta"],
            self.hyperbolic_detector_names,
            kwargs["num_frequency_bands"],
        )
        return HyperbolicGravitationalWaveTransient(
            interferometers=self._subset_interferometers(self.hyperbolic_detector_names),
            waveform_generator=self.waveform_generator,
            detector_dependent_noise=self.detector_dependent_noise,
            **kwargs,
        )

    @property
    def noise_parameter_keys(self):
        keys = []
        for likelihood in [self._student_likelihood, self._hyperbolic_likelihood]:
            if likelihood is not None:
                keys.extend(likelihood.noise_parameter_keys)
        return keys

    @property
    def meta_data(self):
        meta_data = super().meta_data
        meta_data.update(
            likelihood_class=self.__class__,
            detector_likelihoods=self.detector_likelihoods.copy(),
            detector_dependent_noise=self.detector_dependent_noise,
        )
        if self._student_likelihood is not None:
            meta_data.update(
                nu=self._student_likelihood._fixed_nu.tolist(),
                infer_nu=self._student_likelihood.infer_nu,
            )
        if self._hyperbolic_likelihood is not None:
            meta_data.update(
                alpha=self._hyperbolic_likelihood._fixed_alpha.tolist(),
                delta=self._hyperbolic_likelihood._fixed_delta.tolist(),
                infer_alpha=self._hyperbolic_likelihood.infer_alpha,
                infer_delta=self._hyperbolic_likelihood.infer_delta,
            )
        return meta_data

    def _resolve_likelihood_parameters(self, parameters=None, include_sky=True):
        parameters = _fallback_to_parameters(self, parameters)
        if parameters is self._parameters:
            parameters = parameters.copy()
        else:
            merged_parameters = self._parameters.copy()
            merged_parameters.update(parameters)
            parameters = merged_parameters
        if include_sky:
            parameters.update(self.get_sky_frame_parameters(parameters))
        return parameters

    def _gaussian_detector_log_likelihood(
        self, interferometer, parameters=None, waveform_polarizations=None
    ):
        mask = interferometer.frequency_mask
        scale2 = (
            interferometer.power_spectral_density_array[mask]
            * self.waveform_generator.duration
            / 4.0
        )
        if np.any(scale2 <= 0) or not np.all(np.isfinite(scale2)):
            return -np.inf

        residual = interferometer.frequency_domain_strain[mask]
        if waveform_polarizations is not None:
            h_f = interferometer.get_detector_response(
                waveform_polarizations, parameters
            )
            residual = residual - h_f[mask]
        quadratic_form = (residual.real ** 2 + residual.imag ** 2) / scale2
        return float(-np.sum(np.log(2 * np.pi * scale2) + quadratic_form / 2.0))

    def _active_noise_parameters(self, parameters, update_state):
        nu_values = None
        alpha_values = None
        delta_values = None
        if self._student_likelihood is not None:
            nu_values = self._student_likelihood._get_active_nu_values(
                parameters, update_state=update_state
            )
            if nu_values is None:
                return None
        if self._hyperbolic_likelihood is not None:
            alpha_values, delta_values = self._hyperbolic_likelihood._get_active_shape_parameters(
                parameters, update_state=update_state
            )
            if alpha_values is None or delta_values is None:
                return None
        return nu_values, alpha_values, delta_values

    def _mixed_log_likelihood(
        self, parameters, waveform_polarizations, noise_parameters
    ):
        nu_values, alpha_values, delta_values = noise_parameters
        log_likelihood = 0.0

        for detector_name in self.gaussian_detector_names:
            log_likelihood += self._gaussian_detector_log_likelihood(
                self._interferometers_by_name[detector_name],
                parameters=parameters,
                waveform_polarizations=waveform_polarizations,
            )

        if self._student_likelihood is not None:
            for interferometer in self._student_likelihood.interferometers:
                log_likelihood += self._student_likelihood._compute_detector_log_likelihood(
                    interferometer=interferometer,
                    nu_values=self._student_likelihood._get_interferometer_nu_values(
                        interferometer, nu_values
                    ),
                    parameters=parameters,
                    waveform_polarizations=waveform_polarizations,
                )

        if self._hyperbolic_likelihood is not None:
            log_likelihood += self._hyperbolic_likelihood._compute_log_likelihood(
                interferometers=self._hyperbolic_likelihood.interferometers,
                alpha_values=alpha_values,
                delta_values=delta_values,
                parameters=parameters,
                waveform_polarizations=waveform_polarizations,
            )

        if not np.isfinite(log_likelihood):
            return np.nan_to_num(-np.inf)
        return float(log_likelihood)

    def _noise_log_likelihood_from_parameters(self, parameters):
        parameters = self._resolve_likelihood_parameters(parameters, include_sky=False)
        noise_parameters = self._active_noise_parameters(
            parameters, update_state=False
        )
        if noise_parameters is None:
            return np.nan_to_num(-np.inf)
        return self._mixed_log_likelihood(
            parameters=parameters,
            waveform_polarizations=None,
            noise_parameters=noise_parameters,
        )

    def log_likelihood(self, parameters=None):
        parameters = self._resolve_likelihood_parameters(parameters)
        noise_parameters = self._active_noise_parameters(
            parameters, update_state=True
        )
        if noise_parameters is None:
            return np.nan_to_num(-np.inf)
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(
            parameters
        )
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)
        return self._mixed_log_likelihood(
            parameters=parameters,
            waveform_polarizations=waveform_polarizations,
            noise_parameters=noise_parameters,
        )

    def noise_log_likelihood(self):
        return self._noise_log_likelihood_from_parameters(self._parameters.copy())

    def log_likelihood_ratio(self, parameters=None):
        parameters = self._resolve_likelihood_parameters(parameters)
        noise_parameters = self._active_noise_parameters(
            parameters, update_state=True
        )
        if noise_parameters is None:
            return np.nan_to_num(-np.inf)
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(
            parameters
        )
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)
        signal_log_likelihood = self._mixed_log_likelihood(
            parameters, waveform_polarizations, noise_parameters
        )
        noise_log_likelihood = self._mixed_log_likelihood(
            parameters, None, noise_parameters
        )
        if not np.isfinite(signal_log_likelihood) or not np.isfinite(noise_log_likelihood):
            return np.nan_to_num(-np.inf)
        return float(signal_log_likelihood - noise_log_likelihood)

    def compute_per_detector_log_likelihood(self, parameters=None):
        parameters = self._resolve_likelihood_parameters(parameters)
        noise_parameters = self._active_noise_parameters(
            parameters, update_state=True
        )
        if noise_parameters is None:
            for interferometer in self.interferometers:
                parameters[f"{interferometer.name}_log_likelihood"] = np.nan_to_num(-np.inf)
            return parameters.copy()
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(
            parameters
        )
        if waveform_polarizations is None:
            for interferometer in self.interferometers:
                parameters[f"{interferometer.name}_log_likelihood"] = np.nan_to_num(-np.inf)
            return parameters.copy()

        nu_values, alpha_values, delta_values = noise_parameters
        for interferometer in self.interferometers:
            if self.detector_likelihoods[interferometer.name] == "gaussian":
                signal_log_likelihood = self._gaussian_detector_log_likelihood(
                    interferometer, parameters, waveform_polarizations
                )
                noise_log_likelihood = self._gaussian_detector_log_likelihood(interferometer)
            elif self.detector_likelihoods[interferometer.name] == "student-t":
                student = self._student_likelihood
                nu_values_for_detector = student._get_interferometer_nu_values(
                    interferometer, nu_values
                )
                signal_log_likelihood = student._compute_detector_log_likelihood(
                    interferometer, nu_values_for_detector, parameters, waveform_polarizations
                )
                noise_log_likelihood = student._compute_detector_log_likelihood(
                    interferometer, nu_values_for_detector
                )
            else:
                hyperbolic = self._hyperbolic_likelihood
                alpha_for_detector = hyperbolic._get_interferometer_parameter_values(
                    interferometer, alpha_values
                )
                delta_for_detector = hyperbolic._get_interferometer_parameter_values(
                    interferometer, delta_values
                )
                signal_log_likelihood = hyperbolic._compute_network_log_likelihood(
                    [interferometer],
                    alpha_for_detector,
                    delta_for_detector,
                    parameters=parameters,
                    waveform_polarizations=waveform_polarizations,
                    include_log_scale2=False,
                )
                noise_log_likelihood = hyperbolic._compute_network_log_likelihood(
                    [interferometer],
                    alpha_for_detector,
                    delta_for_detector,
                    include_log_scale2=False,
                )
            parameters[f"{interferometer.name}_log_likelihood"] = float(
                signal_log_likelihood - noise_log_likelihood
            )
        return parameters.copy()

    def noise_log_evidence(self, priors=None, sampler=None, result=None, npool=1):
        total_log_evidence = sum(
            self._gaussian_detector_log_likelihood(
                self._interferometers_by_name[detector_name]
            )
            for detector_name in self.gaussian_detector_names
        )
        for likelihood in [self._student_likelihood, self._hyperbolic_likelihood]:
            if likelihood is not None:
                total_log_evidence += likelihood.noise_log_evidence(
                    priors=priors,
                    sampler=sampler,
                    result=result,
                    npool=npool,
                )
        return float(total_log_evidence)
