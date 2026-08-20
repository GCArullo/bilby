"""Targeted, normalized time-frequency parametric-noise likelihoods.

The ordinary frequency-domain Gaussian likelihood is retained everywhere except
for one pre-declared, detector-localized Slepian subspace.  The active Fourier
coefficients are standardized by the analysis PSD, and the Slepian projector has
orthonormal rows.  Replacing the standard-normal density only in that subspace is
therefore a normalized likelihood, with an exactly Gaussian complement.
"""

from copy import deepcopy
from hashlib import sha256
from itertools import product

import numpy as np
from scipy.integrate import quad
from scipy.special import kve

from ...core.prior import DeltaFunction, PriorDict
from ...core.utils import logger
from .base import GravitationalWaveTransient

__all__ = ["ProjectedTimeFrequencyGravitationalWaveTransient"]

_LOG_TWO_PI = np.log(2.0 * np.pi)


def _suffix(value):
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


class ProjectedTimeFrequencyGravitationalWaveTransient(
    GravitationalWaveTransient
):
    """Modify the noise density in one localized Slepian subspace per detector.

    Parameters
    ==========
    target_time_intervals, target_frequency_intervals: dict
        Single-detector dictionaries containing one ``(lower, upper)``
        interval. Times are seconds from the segment start. Frequency intervals
        are half open, ``[lower, upper)``. The other detectors remain exactly
        Gaussian.
    minimum_concentration: float
        Retain Slepian modes whose fraction of time-domain power inside the
        requested interval is at least this value.
    noise_model: str
        ``gaussian``, ``gaussian-parametric``, or ``hyperbolic``.

    Notes
    =====
    The hyperbolic density is one rotation-invariant radial density over the
    retained real Slepian modes.  With a very small projected dimension its two
    parameters are weakly identified; a matched Gaussian-parametric run is the
    required scale control.
    """

    _QUADRATURE_EPSREL = 1e-8
    _QUADRATURE_LIMIT = 200

    def __init__(
        self,
        interferometers,
        waveform_generator,
        target_time_intervals,
        target_frequency_intervals,
        minimum_concentration=0.5,
        noise_model="hyperbolic",
        alpha=10.0,
        delta=1.0,
        log_projected_variance=0.0,
        infer_alpha=False,
        infer_delta=False,
        infer_log_projected_variance=False,
        priors=None,
        reference_frame="sky",
        time_reference="geocenter",
        jitter_time=True,
    ):
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            priors=priors,
            reference_frame=reference_frame,
            time_reference=time_reference,
            jitter_time=jitter_time,
        )
        if self.interferometers.array_backend is not np:
            raise NotImplementedError(
                "projected time-frequency likelihood currently supports the "
                "NumPy array backend only"
            )
        if noise_model not in ("gaussian", "gaussian-parametric", "hyperbolic"):
            raise ValueError(f"Unknown noise_model '{noise_model}'")
        self.noise_model = noise_model
        self.minimum_concentration = float(minimum_concentration)
        if not 0.0 < self.minimum_concentration <= 1.0:
            raise ValueError("minimum_concentration must lie in (0, 1]")

        self.infer_alpha = bool(infer_alpha)
        self.infer_delta = bool(infer_delta)
        self.infer_log_projected_variance = bool(
            infer_log_projected_variance
        )
        self._fixed = dict(
            alpha=float(alpha),
            delta=float(delta),
            log_projected_variance=float(log_projected_variance),
        )
        self._duration = float(self.waveform_generator.duration)
        self._sampling_frequency = float(
            self.waveform_generator.sampling_frequency
        )
        self._n_time = int(round(self._duration * self._sampling_frequency))
        if self._n_time % 2:
            raise ValueError("projected targets require an even number of samples")
        self._targets = self._validate_targets(
            target_time_intervals, target_frequency_intervals
        )
        self._projectors = {
            interferometer.name: self._build_projector(
                interferometer, self._targets[interferometer.name]
            )
            for interferometer in self.interferometers
            if interferometer.name in self._targets
        }

    def _validate_targets(self, time_intervals, frequency_intervals):
        if not isinstance(time_intervals, dict) or not isinstance(
            frequency_intervals, dict
        ):
            raise ValueError("target intervals must be detector-keyed dictionaries")
        if set(time_intervals) != set(frequency_intervals):
            raise ValueError("time and frequency target detectors must match")
        detector_names = {ifo.name for ifo in self.interferometers}
        unknown = set(time_intervals) - detector_names
        if unknown:
            raise ValueError(f"unknown target detectors: {sorted(unknown)}")
        if not time_intervals:
            raise ValueError("at least one detector target is required")
        if len(time_intervals) != 1:
            raise ValueError("exactly one detector target is currently supported")

        targets = {}
        for name in time_intervals:
            times = [float(value) for value in time_intervals[name]]
            frequencies = [float(value) for value in frequency_intervals[name]]
            if len(times) != 2 or not 0.0 <= times[0] < times[1] <= self._duration:
                raise ValueError(f"invalid time target for {name}: {times}")
            if len(frequencies) != 2 or not frequencies[0] < frequencies[1]:
                raise ValueError(
                    f"invalid frequency target for {name}: {frequencies}"
                )
            targets[name] = dict(times=times, frequencies=frequencies)
        return targets

    def _build_projector(self, interferometer, target):
        time_lower, time_upper = target["times"]
        frequency_lower, frequency_upper = target["frequencies"]
        sample_lower = int(round(time_lower * self._sampling_frequency))
        sample_upper = int(round(time_upper * self._sampling_frequency))
        if sample_upper - sample_lower < 2:
            raise ValueError(f"time target for {interferometer.name} is too short")

        active = interferometer.frequency_mask
        frequencies = interferometer.frequency_array
        selected = active & (frequencies >= frequency_lower) & (
            frequencies < frequency_upper
        )
        selected_indices = np.flatnonzero(selected)
        if len(selected_indices) == 0:
            raise ValueError(
                f"frequency target for {interferometer.name} has no active bins"
            )
        if np.any(selected_indices == 0) or np.any(
            selected_indices == self._n_time // 2
        ):
            raise ValueError("projected targets cannot include DC or Nyquist")

        samples = np.arange(sample_lower, sample_upper, dtype=float)
        phases = 2.0 * np.pi * np.outer(
            samples / self._sampling_frequency, frequencies[selected_indices]
        )
        synthesis = np.empty((len(samples), 2 * len(selected_indices)))
        synthesis[:, 0::2] = np.sqrt(2.0 / self._n_time) * np.cos(phases)
        synthesis[:, 1::2] = -np.sqrt(2.0 / self._n_time) * np.sin(phases)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            concentration = synthesis.T @ synthesis
        eigenvalues, eigenvectors = np.linalg.eigh(concentration)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0.0, 1.0)
        eigenvectors = eigenvectors[:, order]
        retained = eigenvalues >= self.minimum_concentration
        if not np.any(retained):
            raise ValueError(
                f"target for {interferometer.name} has no modes above "
                f"minimum_concentration={self.minimum_concentration:g}"
            )
        projector = eigenvectors[:, retained].T
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            orthogonality_error = float(
                np.max(
                    np.abs(
                        projector @ projector.T - np.eye(projector.shape[0])
                    )
                )
            )
        if orthogonality_error > 1e-10:
            raise RuntimeError("failed to construct an orthonormal projector")

        retained_eigenvalues = eigenvalues[retained]
        logger.info(
            f"Projected time-frequency target {interferometer.name}: "
            f"t=[{sample_lower / self._sampling_frequency:g}, "
            f"{sample_upper / self._sampling_frequency:g}) s, "
            f"f=[{frequency_lower:g}, {frequency_upper:g}) Hz, "
            f"{len(retained_eigenvalues)} modes, concentration "
            f"{retained_eigenvalues.min():.3f}--{retained_eigenvalues.max():.3f}"
        )
        return dict(
            selected=selected,
            projector=projector,
            eigenvalues=retained_eigenvalues,
            sample_interval=(sample_lower, sample_upper),
            selected_frequencies=frequencies[selected].copy(),
            projector_sha256=sha256(
                np.ascontiguousarray(projector, dtype="<f8").tobytes()
            ).hexdigest(),
        )

    def _parameter_key(self, name, detector):
        target = self._targets[detector]
        values = [*target["times"], *target["frequencies"]]
        return f"{name}_{detector}_" + "_".join(_suffix(value) for value in values)

    @property
    def alpha_parameter_keys(self):
        return [self._parameter_key("alpha", name) for name in self._projectors]

    @property
    def delta_parameter_keys(self):
        return [self._parameter_key("delta", name) for name in self._projectors]

    @property
    def log_projected_variance_parameter_keys(self):
        return [
            self._parameter_key("log_projected_variance", name)
            for name in self._projectors
        ]

    @property
    def noise_parameter_keys(self):
        if self.noise_model == "hyperbolic":
            keys = self.alpha_parameter_keys if self.infer_alpha else []
            if self.infer_delta:
                keys += self.delta_parameter_keys
            return keys
        if (
            self.noise_model == "gaussian-parametric"
            and self.infer_log_projected_variance
        ):
            return self.log_projected_variance_parameter_keys
        return []

    def _parameter(self, parameters, name, detector):
        infer = {
            "alpha": self.infer_alpha,
            "delta": self.infer_delta,
            "log_projected_variance": self.infer_log_projected_variance,
        }[name]
        if not infer:
            return self._fixed[name]
        return float(
            parameters.get(
                self._parameter_key(name, detector), self._fixed[name]
            )
        )

    def _projected_vector(self, interferometer, residual):
        entry = self._projectors[interferometer.name]
        selected = entry["selected"]
        scale2 = (
            interferometer.power_spectral_density_array[selected]
            * self._duration
            / 4.0
        )
        standardized = residual[selected] / np.sqrt(scale2)
        real_vector = np.empty(2 * len(standardized))
        real_vector[0::2] = standardized.real
        real_vector[1::2] = standardized.imag
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return entry["projector"] @ real_vector

    @staticmethod
    def _hyperbolic_log_density(quadratic_form, dimension, alpha, delta):
        if alpha <= 0.0 or delta <= 0.0:
            return -np.inf
        argument = alpha * delta
        scaled_bessel = kve(0.5 * (dimension + 1.0), argument)
        if not np.isfinite(scaled_bessel) or scaled_bessel <= 0.0:
            return -np.inf
        radial_shift = quadratic_form / (
            np.sqrt(delta ** 2 + quadratic_form) + delta
        )
        return float(
            0.5 * (dimension + 1.0) * np.log(alpha / delta)
            + 0.5 * (1.0 - dimension) * _LOG_TWO_PI
            - np.log(2.0 * alpha)
            - np.log(scaled_bessel)
            - alpha * radial_shift
        )

    def _projected_correction(self, interferometer, residual, parameters):
        projected = self._projected_vector(interferometer, residual)
        quadratic_form = float(projected @ projected)
        dimension = len(projected)
        gaussian = -0.5 * quadratic_form - 0.5 * dimension * _LOG_TWO_PI
        if self.noise_model == "gaussian":
            return 0.0
        if self.noise_model == "gaussian-parametric":
            log_variance = self._parameter(
                parameters, "log_projected_variance", interferometer.name
            )
            if not np.isfinite(log_variance):
                return -np.inf
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                variance = np.power(10.0, log_variance)
            if not np.isfinite(variance) or variance <= 0.0:
                return -np.inf
            alternative = (
                -0.5 * quadratic_form / variance
                - 0.5 * dimension * (_LOG_TWO_PI + np.log(variance))
            )
        else:
            alternative = self._hyperbolic_log_density(
                quadratic_form=quadratic_form,
                dimension=dimension,
                alpha=self._parameter(parameters, "alpha", interferometer.name),
                delta=self._parameter(parameters, "delta", interferometer.name),
            )
        return float(alternative - gaussian)

    def _correction(self, parameters, waveform_polarizations=None):
        correction = 0.0
        for interferometer in self.interferometers:
            if interferometer.name not in self._projectors:
                continue
            residual = interferometer.frequency_domain_strain
            if waveform_polarizations is not None:
                residual = residual - interferometer.get_detector_response(
                    waveform_polarizations, parameters
                )
            correction += self._projected_correction(
                interferometer, residual, parameters
            )
        return float(correction)

    def _resolve_parameters(self, parameters):
        parameters = parameters.copy()
        parameters.update(self.get_sky_frame_parameters(parameters))
        return parameters

    def log_likelihood(self, parameters):
        parameters = self._resolve_parameters(parameters)
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(
            parameters
        )
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)
        gaussian_ratio = GravitationalWaveTransient.log_likelihood_ratio(
            self, parameters
        )
        return float(
            gaussian_ratio
            + GravitationalWaveTransient.noise_log_likelihood(self)
            + self._correction(parameters, waveform_polarizations)
        )

    def _noise_log_likelihood_from_parameters(self, parameters):
        return float(
            GravitationalWaveTransient.noise_log_likelihood(self)
            + self._correction(parameters)
        )

    def noise_log_likelihood(self):
        return self._noise_log_likelihood_from_parameters(
            self._default_noise_parameters()
        )

    def log_likelihood_ratio(self, parameters):
        parameters = self._resolve_parameters(parameters)
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(
            parameters
        )
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)
        gaussian_ratio = GravitationalWaveTransient.log_likelihood_ratio(
            self, parameters
        )
        return float(
            gaussian_ratio
            + self._correction(parameters, waveform_polarizations)
            - self._correction(parameters)
        )

    def _default_noise_parameters(self):
        parameters = {}
        for detector in self._projectors:
            if self.noise_model == "hyperbolic":
                if self.infer_alpha:
                    parameters[self._parameter_key("alpha", detector)] = self._fixed[
                        "alpha"
                    ]
                if self.infer_delta:
                    parameters[self._parameter_key("delta", detector)] = self._fixed[
                        "delta"
                    ]
            elif self.noise_model == "gaussian-parametric" and (
                self.infer_log_projected_variance
            ):
                parameters[
                    self._parameter_key("log_projected_variance", detector)
                ] = self._fixed["log_projected_variance"]
        return parameters

    def _noise_priors(self, priors):
        priors = PriorDict() if priors is None else PriorDict(priors)
        noise_priors = PriorDict()
        for key, value in self._default_noise_parameters().items():
            noise_priors[key] = (
                deepcopy(priors[key])
                if key in priors
                else DeltaFunction(peak=value, name=key)
            )
        return noise_priors

    def noise_log_evidence(self, priors=None, sampler=None, result=None, npool=1):
        noise_priors = self._noise_priors(priors)
        keys = list(noise_priors.non_fixed_keys)
        base = self._default_noise_parameters()
        for key in noise_priors:
            if noise_priors[key].is_fixed:
                base[key] = float(noise_priors[key].peak)
        if not keys:
            return self._noise_log_likelihood_from_parameters(base)
        if len(keys) > 2:
            raise NotImplementedError(
                "projected noise-evidence quadrature supports at most two "
                "sampled noise parameters"
            )

        sampled_priors = [noise_priors[key] for key in keys]
        reference_points = np.unique(
            np.concatenate(
                (
                    [0.0, 0.25, 0.5, 0.75, 1.0],
                    np.geomspace(1e-12, 1e-1, 12),
                    1.0 - np.geomspace(1e-12, 1e-1, 12),
                )
            )
        )

        def log_likelihood(unit_values):
            parameters = base.copy()
            for key, prior, unit_value in zip(
                keys, sampled_priors, unit_values
            ):
                parameters[key] = float(prior.rescale(unit_value))
            return self._noise_log_likelihood_from_parameters(parameters)

        candidates = [
            log_likelihood(values)
            for values in product(reference_points, repeat=len(keys))
        ]
        finite = [value for value in candidates if np.isfinite(value)]
        if not finite:
            return np.nan_to_num(-np.inf)
        reference = max(finite)

        def integrand(values):
            value = log_likelihood(values)
            return 0.0 if not np.isfinite(value) else float(np.exp(value - reference))

        kwargs = dict(
            epsabs=0.0,
            epsrel=self._QUADRATURE_EPSREL,
            limit=self._QUADRATURE_LIMIT,
            points=reference_points[1:-1],
        )
        if len(keys) == 1:
            integral, _ = quad(
                lambda value: integrand((value,)), 0.0, 1.0, **kwargs
            )
        else:
            def outer(value_2):
                inner, _ = quad(
                    lambda value_1: integrand((value_1, value_2)),
                    0.0,
                    1.0,
                    **kwargs,
                )
                return float(inner)

            integral, _ = quad(outer, 0.0, 1.0, **kwargs)
        if not np.isfinite(integral) or integral <= 0.0:
            raise RuntimeError("projected noise-evidence quadrature failed")
        return float(reference + np.log(integral))

    @property
    def projector_metadata(self):
        return {
            detector: dict(
                target=deepcopy(self._targets[detector]),
                dimension=len(entry["eigenvalues"]),
                concentrations=entry["eigenvalues"].tolist(),
                sample_interval=list(entry["sample_interval"]),
                selected_frequencies=entry["selected_frequencies"].tolist(),
                projector_sha256=entry["projector_sha256"],
            )
            for detector, entry in self._projectors.items()
        }
