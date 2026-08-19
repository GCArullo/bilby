from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from math import lgamma

import numpy as np
import scipy.fft as sf
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.linalg import inv, solve_toeplitz, solve_triangular, toeplitz
from scipy.special import kve

from ...core import utils as core_utils
from ...core.likelihood import Likelihood
from ...core.prior import DeltaFunction, JointPrior, PriorDict
from ..detector import InterferometerList
from ..waveform_generator import GWSignalWaveformGenerator
from .base import GravitationalWaveTransient


_LOG_2PI = np.log(2.0 * np.pi)


def _log_scaled_bessel_second_kind_asymptotic(order, argument):
    scaled_argument = argument / order
    eta_sqrt = np.sqrt(1.0 + scaled_argument * scaled_argument)
    eta = eta_sqrt + np.log(scaled_argument / (1.0 + eta_sqrt))
    return (
        argument
        - order * eta
        + 0.5 * np.log(np.pi / (2.0 * order))
        - 0.25 * np.log1p(scaled_argument * scaled_argument)
    )


def _log_scaled_bessel_second_kind(order, argument):
    if (
        order <= 0.0
        or not np.isfinite(order)
        or argument <= 0.0
        or not np.isfinite(argument)
    ):
        raise ValueError(
            "The scaled Bessel function arguments must be positive and finite"
        )

    if argument < 0.2 * order:
        x2_over4 = 0.25 * argument * argument
        term = 1.0
        series = 1.0
        for index in range(1, 10000):
            denominator = index * (index - order)
            if denominator == 0.0:
                break
            term *= x2_over4 / denominator
            series += term
            if (not np.isfinite(term)) or (not np.isfinite(series)):
                break
            if abs(term) <= 1e-15 * abs(series):
                break

        if series > 0.0 and np.isfinite(series):
            return (
                argument
                + np.log(0.5)
                + lgamma(order)
                + order * np.log(2.0 / argument)
                + np.log(series)
            )

    if order >= 100.0:
        return _log_scaled_bessel_second_kind_asymptotic(order, argument)

    scaled_bessel = kve(order, argument)
    if scaled_bessel > 0.0 and np.isfinite(scaled_bessel):
        return np.log(scaled_bessel)
    return _log_scaled_bessel_second_kind_asymptotic(order, argument)


def _gaussian_log_likelihood_from_inner_product(residuals_inner_product, log_normalisation):
    return -0.5 * residuals_inner_product + log_normalisation


def _student_t_log_likelihood_from_inner_product(
    residuals_inner_product, logdet, dimension, nu
):
    if dimension < 1:
        raise ValueError("The Student-t likelihood dimension must be positive")
    if nu <= 0.0 or not np.isfinite(nu):
        raise ValueError("The Student-t degrees of freedom must be positive and finite")
    if residuals_inner_product < 0.0 or not np.isfinite(residuals_inner_product):
        raise ValueError(
            "The residuals inner product must be non-negative and finite"
        )
    if not np.isfinite(logdet):
        raise ValueError("The covariance log determinant must be finite")

    return (
        lgamma(0.5 * (nu + dimension))
        - lgamma(0.5 * nu)
        - 0.5 * (dimension * np.log(nu * np.pi) + logdet)
        - 0.5 * (nu + dimension) * np.log1p(residuals_inner_product / nu)
    )


def _hyperbolic_log_likelihood_from_inner_product(
    residuals_inner_product, logdet, dimension, alpha, delta
):
    if dimension < 1:
        raise ValueError("The hyperbolic likelihood dimension must be positive")
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("The hyperbolic alpha parameter must be positive and finite")
    if delta <= 0.0 or not np.isfinite(delta):
        raise ValueError("The hyperbolic delta parameter must be positive and finite")
    if residuals_inner_product < 0.0 or not np.isfinite(residuals_inner_product):
        raise ValueError(
            "The residuals inner product must be non-negative and finite"
        )
    if not np.isfinite(logdet):
        raise ValueError("The covariance log determinant must be finite")

    order = 0.5 * (dimension + 1.0)
    log_scaled_bessel = _log_scaled_bessel_second_kind(order, alpha * delta)
    radial_shift = residuals_inner_product / (
        np.sqrt(delta * delta + residuals_inner_product) + delta
    )

    return (
        order * np.log(alpha / delta)
        + 0.5 * (1.0 - dimension) * _LOG_2PI
        - np.log(2.0 * alpha)
        - log_scaled_bessel
        - alpha * radial_shift
        - 0.5 * logdet
    )


def _as_1d_float_array(name, value, check_finite=False):
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0:
        raise ValueError(f"{name} cannot be empty")
    if check_finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _check_gohberg_semencul_x0(x0):
    if abs(x0) <= np.finfo(float).eps:
        raise ValueError(
            "The first Gohberg-Semencul generator entry is numerically zero"
        )


def _toeplitz_kernel_rfft(column, row, fft_length):
    column = np.asarray(column, dtype=float)
    row = np.asarray(row, dtype=float)
    kernel = np.zeros(fft_length, dtype=float)
    kernel[: len(column)] = column
    if len(column) > 1:
        kernel[fft_length - len(column) + 1 :] = row[:0:-1]
    return sf.rfft(kernel)


def _multiply_fft_kernel(kernel_fft, vector_fft):
    if vector_fft.ndim == 1:
        return kernel_fft * vector_fft
    return kernel_fft[:, None] * vector_fft


def _irfft_head(product_fft, fft_length, n):
    return sf.irfft(product_fft, fft_length, axis=0)[:n]


@dataclass(frozen=True)
class _GohbergSemenculToeplitzInverse:
    acf: np.ndarray
    x: np.ndarray
    x0: float
    zeros: np.ndarray
    zeros_with_x0: np.ndarray
    tail_reverse_x: np.ndarray
    fft_length: int
    lower_x_fft: np.ndarray
    upper_x_fft: np.ndarray
    lower_tail_reverse_x_fft: np.ndarray
    upper_tail_reverse_x_fft: np.ndarray

    @classmethod
    def from_acf(cls, acf, check_finite=False):
        acf = _as_1d_float_array("acf", acf, check_finite=check_finite)
        basis_vector = np.zeros_like(acf)
        basis_vector[0] = 1.0
        x = solve_toeplitz(acf, basis_vector, check_finite=check_finite)
        return cls.from_acf_and_generator(acf, x, check_finite=check_finite)

    @classmethod
    def from_acf_and_generator(cls, acf, x, check_finite=False):
        acf = _as_1d_float_array("acf", acf, check_finite=check_finite)
        x = _as_1d_float_array("x", x, check_finite=check_finite)
        if len(acf) != len(x):
            raise ValueError(
                "ACF and Gohberg-Semencul generator lengths do not agree"
            )

        _check_gohberg_semencul_x0(x[0])

        zeros = np.zeros_like(x)
        zeros_with_x0 = np.zeros_like(x)
        zeros_with_x0[0] = x[0]
        tail_reverse_x = np.zeros_like(x)
        tail_reverse_x[1:] = x[:0:-1]
        fft_length = sf.next_fast_len(2 * len(x) - 1, real=True)

        return cls(
            acf=acf,
            x=x,
            x0=float(x[0]),
            zeros=zeros,
            zeros_with_x0=zeros_with_x0,
            tail_reverse_x=tail_reverse_x,
            fft_length=fft_length,
            lower_x_fft=_toeplitz_kernel_rfft(x, zeros, fft_length),
            upper_x_fft=_toeplitz_kernel_rfft(zeros_with_x0, x, fft_length),
            lower_tail_reverse_x_fft=_toeplitz_kernel_rfft(
                tail_reverse_x, zeros, fft_length
            ),
            upper_tail_reverse_x_fft=_toeplitz_kernel_rfft(
                zeros, tail_reverse_x, fft_length
            ),
        )

    def matvec(self, vector, check_finite=False):
        vector = np.asarray(vector, dtype=float)
        if vector.ndim not in (1, 2):
            raise ValueError("vector must be one- or two-dimensional")
        if vector.shape[0] != len(self.x):
            raise ValueError(
                "Gohberg-Semencul factor and target vector lengths do not agree"
            )
        if check_finite and not np.all(np.isfinite(vector)):
            raise ValueError("vector must contain only finite values")

        if len(self.x) == 1:
            return vector * self.x0

        vector_fft = sf.rfft(vector, self.fft_length, axis=0)
        upper_x_vector = _irfft_head(
            _multiply_fft_kernel(self.upper_x_fft, vector_fft),
            self.fft_length,
            len(self.x),
        )
        upper_tail_vector = _irfft_head(
            _multiply_fft_kernel(self.upper_tail_reverse_x_fft, vector_fft),
            self.fft_length,
            len(self.x),
        )
        upper_x_fft = sf.rfft(upper_x_vector, self.fft_length, axis=0)
        upper_tail_fft = sf.rfft(upper_tail_vector, self.fft_length, axis=0)

        return _irfft_head(
            _multiply_fft_kernel(self.lower_x_fft, upper_x_fft)
            - _multiply_fft_kernel(
                self.lower_tail_reverse_x_fft, upper_tail_fft
            ),
            self.fft_length,
            len(self.x),
        ) / self.x0


def _toeplitz_slogdet(acf):
    acf = np.asarray(acf, dtype=float)
    dimension = len(acf)
    r0 = acf[0]
    normalized = np.concatenate((acf, np.array([r0], dtype=float))) / r0
    logdet = dimension * np.log(abs(r0))
    sign = np.sign(r0) ** dimension

    if dimension == 1:
        return sign, logdet

    y = np.zeros(dimension, dtype=float)
    x = np.zeros(dimension, dtype=float)
    b = -normalized[1 : dimension + 1]
    r = normalized[:dimension]
    y[0] = -r[1]
    x[0] = b[0]
    beta = 1.0
    alpha = -r[1]
    determinant_update = 1.0 + (-b[0]) * x[0]
    sign *= np.sign(determinant_update)
    logdet += np.log(abs(determinant_update))

    for index in range(0, dimension - 2):
        beta = (1.0 - alpha * alpha) * beta
        mu = (b[index + 1] - np.dot(r[1 : index + 2], x[index::-1])) / beta
        x[0 : index + 1] = x[0 : index + 1] + mu * y[index::-1]
        x[index + 1] = mu

        determinant_update = 1.0 + np.dot(-b[0 : index + 2], x[0 : index + 2])
        sign *= np.sign(determinant_update)
        logdet += np.log(abs(determinant_update))

        if index < dimension - 2:
            alpha = -(r[index + 2] + np.dot(r[1 : index + 2], y[index::-1])) / beta
            y[0 : index + 1] = y[0 : index + 1] + alpha * y[index::-1]
            y[index + 1] = alpha

    return sign, logdet


def _is_time_band_cut_list(time_bands):
    return isinstance(time_bands, (list, tuple, np.ndarray))


def _is_detector_time_band_map(time_bands):
    return isinstance(time_bands, dict)


def _coerce_time_band_boundaries(time_band_boundaries):
    if not _is_time_band_cut_list(time_band_boundaries):
        raise ValueError(
            "time_band_boundaries must be a 1D list, tuple, or array of cut times in seconds"
        )

    boundaries = np.asarray(time_band_boundaries, dtype=float)
    if boundaries.ndim != 1:
        raise ValueError(
            "time_band_boundaries must be a 1D list, tuple, or array of cut times in seconds"
        )
    return boundaries.tolist()


def _coerce_detector_time_band_boundaries(time_band_boundaries, detector_names):
    missing_detectors = [
        detector_name
        for detector_name in detector_names
        if detector_name not in time_band_boundaries
    ]
    unknown_detectors = [
        detector_name
        for detector_name in time_band_boundaries
        if detector_name not in detector_names
    ]
    if missing_detectors or unknown_detectors:
        message = []
        if missing_detectors:
            message.append("missing " + ", ".join(missing_detectors))
        if unknown_detectors:
            message.append("unknown " + ", ".join(unknown_detectors))
        raise ValueError(
            "time_band_boundaries mapping has " + " and ".join(message)
        )

    boundaries = {
        detector_name: _coerce_time_band_boundaries(
            time_band_boundaries[detector_name]
        )
        for detector_name in detector_names
    }
    band_counts = {
        detector_name: len(detector_boundaries) + 1
        for detector_name, detector_boundaries in boundaries.items()
    }
    if len(set(band_counts.values())) > 1:
        raise ValueError(
            "Every detector must define the same number of time bands; "
            f"got {band_counts}"
        )
    return boundaries


def _coerce_time_band_specification(
    time_band_boundaries, detector_names, detector_dependent_noise
):
    if _is_detector_time_band_map(time_band_boundaries):
        if not detector_dependent_noise:
            raise ValueError(
                "Per-detector time_band_boundaries requires "
                "detector_dependent_noise=True"
            )
        return _coerce_detector_time_band_boundaries(
            time_band_boundaries, detector_names
        )
    return _coerce_time_band_boundaries(time_band_boundaries)


def _resolve_time_bands(
    time_bands,
    time_band_boundaries=None,
    detector_names=None,
    detector_dependent_noise=False,
):
    detector_names = [] if detector_names is None else detector_names
    if time_band_boundaries is None:
        if _is_time_band_cut_list(time_bands) or _is_detector_time_band_map(
            time_bands
        ):
            return _coerce_time_band_specification(
                time_bands, detector_names, detector_dependent_noise
            )
        return int(time_bands)

    boundaries = _coerce_time_band_specification(
        time_band_boundaries, detector_names, detector_dependent_noise
    )
    if _is_time_band_cut_list(time_bands) or _is_detector_time_band_map(
        time_bands
    ):
        if (
            _coerce_time_band_specification(
                time_bands, detector_names, detector_dependent_noise
            )
            != boundaries
        ):
            raise ValueError(
                "time_bands and time_band_boundaries must match when both are provided"
            )
        return boundaries

    number_of_time_bands = int(time_bands)
    if number_of_time_bands not in (1, _time_band_count(boundaries)):
        raise ValueError(
            "time_bands and time_band_boundaries must describe the same number of bands"
        )
    return boundaries


def _time_band_count(time_bands):
    if _is_detector_time_band_map(time_bands):
        return len(next(iter(time_bands.values()))) + 1
    if _is_time_band_cut_list(time_bands):
        return len(time_bands) + 1
    return int(time_bands)


def _time_bands_for_detector(time_bands, detector_name):
    if _is_detector_time_band_map(time_bands):
        return time_bands[detector_name]
    return time_bands


def _time_band_sample_slices(dimension, time_bands, sampling_rate=None):
    dimension = int(dimension)
    if dimension < 1:
        raise ValueError("The time-band dimension must be positive")

    if _is_time_band_cut_list(time_bands):
        if sampling_rate is None:
            raise ValueError(
                "A sampling rate is required when time bands are specified by cut times"
            )
        sampling_rate = float(sampling_rate)
        if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
            raise ValueError("The time-band sampling rate must be positive and finite")

        cuts = np.asarray(time_bands, dtype=float)
        if cuts.size == 0:
            raise ValueError("Time-band cut-time lists must contain at least one time")
        if (not np.all(np.isfinite(cuts))) or np.any(cuts <= 0.0):
            raise ValueError("Time-band cut times must be positive and finite")
        if np.any(np.diff(cuts) <= 0.0):
            raise ValueError("Time-band cut times must be strictly increasing")

        sample_times = np.arange(dimension, dtype=float) / sampling_rate
        if cuts[-1] > sample_times[-1]:
            raise ValueError(
                "The last time-band cut must not exceed the last sample time in the segment"
            )

        edges = np.concatenate(
            ([0], np.searchsorted(sample_times, cuts, side="left"), [dimension])
        ).astype(int)
        if np.any(np.diff(edges) <= 0):
            raise ValueError("Time bands must contain at least one sample each")
        return [(int(edges[index]), int(edges[index + 1])) for index in range(len(edges) - 1)]

    number_of_bands = int(time_bands)
    if number_of_bands < 1:
        raise ValueError("The number of time bands must be positive")
    if number_of_bands > dimension:
        raise ValueError("The number of time bands cannot exceed the number of samples")

    edges = np.linspace(0, dimension, number_of_bands + 1, dtype=int)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Time bands must contain at least one sample each")

    return [(int(edges[index]), int(edges[index + 1])) for index in range(number_of_bands)]


@dataclass
class _LikelihoodCache:
    start: int
    end: int
    acf: np.ndarray
    logdet: float
    log_normalisation: float
    inverse_covariance: np.ndarray = None
    cholesky: np.ndarray = None
    gohberg_semencul_inverse: object = None


def _make_likelihood_cache(acf, likelihood_method, no_lognorm=False):
    acf = np.asarray(acf, dtype=float)
    sign, logdet = _toeplitz_slogdet(acf)
    if sign <= 0:
        raise ValueError("The Toeplitz covariance determinant must be positive")
    cache = _LikelihoodCache(
        start=0,
        end=len(acf),
        acf=acf,
        logdet=logdet,
        log_normalisation=0.0
        if no_lognorm
        else -0.5 * logdet - 0.5 * len(acf) * _LOG_2PI,
    )
    if likelihood_method == "direct-inversion":
        cache.inverse_covariance = inv(toeplitz(acf))
    elif likelihood_method == "cholesky-solve-triangular":
        cache.cholesky = np.linalg.cholesky(toeplitz(acf))
    elif likelihood_method == "gohberg-semencul":
        cache.gohberg_semencul_inverse = _GohbergSemenculToeplitzInverse.from_acf(
            acf, check_finite=False
        )
    return cache


def _make_time_band_likelihood_cache(
    acf, likelihood_method, time_bands, no_lognorm=False, sampling_rate=None
):
    band_cache = []
    for band_start, band_end in _time_band_sample_slices(
        len(acf), time_bands, sampling_rate
    ):
        band_acf = np.asarray(acf[: band_end - band_start], dtype=float)
        cache = _make_likelihood_cache(
            acf=band_acf,
            likelihood_method=likelihood_method,
            no_lognorm=no_lognorm,
        )
        cache.start = band_start
        cache.end = band_end
        band_cache.append(cache)
    return band_cache


def _residuals_inner_product_from_cache(residuals, cache, likelihood_method):
    residuals = np.asarray(residuals, dtype=float)
    if likelihood_method == "direct-inversion":
        return float(np.dot(residuals, np.dot(cache.inverse_covariance, residuals)))
    if likelihood_method == "cholesky-solve-triangular":
        whitened_residuals = solve_triangular(
            cache.cholesky, residuals, lower=True, check_finite=False
        )
        return float(np.dot(whitened_residuals, whitened_residuals))
    if likelihood_method == "toeplitz-inversion":
        return float(
            np.dot(residuals, solve_toeplitz(cache.acf, residuals, check_finite=False))
        )
    if likelihood_method == "gohberg-semencul":
        return float(
            np.dot(
                residuals,
                cache.gohberg_semencul_inverse.matvec(residuals, check_finite=False),
            )
        )
    raise ValueError("Unknown likelihood method requested")


def _resolve_likelihood_method(likelihood_method):
    aliases = {
        "direct-inversion": "direct-inversion",
        "direct": "direct-inversion",
        "cholesky-solve-triangular": "cholesky-solve-triangular",
        "cholesky": "cholesky-solve-triangular",
        "toeplitz-inversion": "toeplitz-inversion",
        "toeplitz": "toeplitz-inversion",
        "gohberg-semencul": "gohberg-semencul",
        "gohberg_semencul": "gohberg-semencul",
        "gohberg": "gohberg-semencul",
        "gs": "gohberg-semencul",
    }
    try:
        return aliases[str(likelihood_method).lower()]
    except Exception as exc:
        raise ValueError(
            "likelihood_method must be one of 'direct-inversion', "
            "'cholesky-solve-triangular', 'toeplitz-inversion', "
            "or 'gohberg-semencul'"
        ) from exc


def _normalise_likelihood_type(likelihood_type, key="likelihood_type"):
    aliases = {
        "gaussian": "gaussian",
        "normal": "gaussian",
        "student-t": "student-t",
        "student_t": "student-t",
        "studentt": "student-t",
        "hyperbolic": "hyperbolic",
    }
    try:
        normalised = aliases[str(likelihood_type).lower()]
    except Exception as exc:
        raise ValueError(
            f"Unknown {key} '{likelihood_type}'. "
            "Available options are: ['gaussian', 'student-t', 'hyperbolic']"
        ) from exc
    return normalised


def _resolve_detector_likelihood_types(likelihood_type, detector_names):
    if isinstance(likelihood_type, dict):
        missing_detectors = [
            detector_name
            for detector_name in detector_names
            if detector_name not in likelihood_type
        ]
        extra_detectors = [
            detector_name
            for detector_name in likelihood_type
            if detector_name not in detector_names
        ]
        if missing_detectors or extra_detectors:
            message = []
            if missing_detectors:
                message.append("missing " + ", ".join(missing_detectors))
            if extra_detectors:
                message.append("unknown " + ", ".join(extra_detectors))
            raise ValueError(
                "Detector-specific likelihood mapping has " + " and ".join(message)
            )
        return {
            detector_name: _normalise_likelihood_type(
                likelihood_type[detector_name],
                key=f"likelihood_type[{detector_name}]",
            )
            for detector_name in detector_names
        }
    likelihood_type = _normalise_likelihood_type(likelihood_type)
    return {detector_name: likelihood_type for detector_name in detector_names}


def _resolve_noise_evidence_method(noise_evidence_method):
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
        raise ValueError("noise_evidence_method must be 'quadrature' or 'nested'")
    return aliases[noise_evidence_method]


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


def _noise_evidence_quadrature_points():
    edge_points = np.geomspace(1e-12, 1e-1, 12)
    return np.unique(
        np.concatenate(([0.0, 0.25, 0.5, 0.75, 1.0], edge_points, 1.0 - edge_points))
    )


def _factorized_noise_log_evidence_by_quadrature(
    *,
    blocks,
    noise_priors,
    base_parameters,
    block_log_likelihood,
    epsabs,
    epsrel,
    limit,
    error_label,
):
    coupled_keys = [
        key
        for key in noise_priors.non_fixed_keys
        if isinstance(noise_priors[key], JointPrior)
        or getattr(noise_priors[key], "required_variables", [])
    ]
    if coupled_keys:
        raise ValueError(
            f"{error_label} factorized noise evidence does not support coupled "
            f"noise priors: {sorted(coupled_keys)}"
        )

    reference_points = _noise_evidence_quadrature_points()
    non_fixed_keys = set(noise_priors.non_fixed_keys)
    total_log_evidence = 0.0

    for block in blocks:
        block_keys = [key for key in block["keys"] if key in non_fixed_keys]
        if len(block_keys) == 0:
            total_log_evidence += block_log_likelihood(block, base_parameters)
            continue
        if len(block_keys) not in (1, 2):
            raise ValueError(
                f"{error_label} quadrature supports one or two sampled noise "
                "parameters per independent block"
            )

        priors = [noise_priors[key] for key in block_keys]

        def log_likelihood_from_unit_values(unit_values):
            parameters = base_parameters.copy()
            for key, prior, unit_value in zip(block_keys, priors, unit_values):
                parameters[key] = float(
                    np.asarray(prior.rescale(unit_value), dtype=float)
                )
            return block_log_likelihood(block, parameters)

        candidate_logls = []
        if all(
            key in base_parameters
            and np.all(np.isfinite(prior.ln_prob(base_parameters[key])))
            for key, prior in zip(block_keys, priors)
        ):
            candidate_logls.append(
                block_log_likelihood(block, base_parameters)
            )
        candidate_logls.extend(
            log_likelihood_from_unit_values(unit_values)
            for unit_values in product(reference_points, repeat=len(block_keys))
        )
        finite_logls = [value for value in candidate_logls if np.isfinite(value)]
        if not finite_logls:
            total_log_evidence += np.nan_to_num(-np.inf)
            continue
        logl_reference = max(finite_logls)

        def scaled_integrand(unit_values):
            logl = log_likelihood_from_unit_values(unit_values)
            if not np.isfinite(logl):
                return 0.0
            return float(np.exp(logl - logl_reference))

        quadrature_kwargs = dict(
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
            points=reference_points[1:-1],
        )
        if len(block_keys) == 1:
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
                f"{error_label} noise-evidence quadrature failed to return a "
                "positive finite integral"
            )
        total_log_evidence += float(logl_reference + np.log(integral))

    return float(total_log_evidence)


def _patch_psd_outside_active_band(psd, frequencies, active_frequencies):
    psd = np.asarray(psd, dtype=float).copy()
    frequencies = np.asarray(frequencies, dtype=float)
    active_frequencies = np.asarray(active_frequencies, dtype=float)
    if len(active_frequencies) == 0:
        raise ValueError("Cannot patch a PSD without any active frequencies")

    low_frequency = float(active_frequencies[0])
    high_frequency = float(active_frequencies[-1])
    active_band_mask = (frequencies >= low_frequency) & (frequencies <= high_frequency)

    # Use a large but finite PSD outside the active band to suppress those
    # frequencies without making the covariance numerically singular.  Both
    # patches are referred to the in-band level: above maximum_frequency the
    # incoming array holds whatever filler _build_finite_psd_array substituted
    # for the non-finite entries bilby stores there, so taking the maximum over
    # that region would scale the patch by the filler instead of by the noise.
    # When maximum_frequency < Nyquist that inflates the patch by many orders of
    # magnitude, the Toeplitz covariance becomes indefinite, and the likelihood
    # cannot be constructed at all.
    patch_value = 10.0 * float(np.max(psd[active_band_mask]))

    psd[frequencies < low_frequency] = patch_value
    psd[frequencies > high_frequency] = patch_value
    return psd


class TimeDomainGravitationalWaveTransient(GravitationalWaveTransient):
    """
    A covariance-based time-domain gravitational-wave transient likelihood.

    The detector noise model is built from a Toeplitz covariance whose first row
    is obtained from the detector PSD through the pyRing convention

    ``acf = 0.5 * irfft(psd * df) * N``.

    Outside the active frequency band, the PSD is patched to large finite
    values before the ACF is built so that out-of-band frequencies are strongly
    suppressed without making the covariance numerically singular.

    The waveform handling remains in bilby: detector responses are built using
    the usual antenna patterns and time delays, then transformed back to the
    time domain for the residual evaluation.

    Set `time_bands` to a positive integer for equi-spaced bands, or pass
    `time_band_boundaries` as a list of cut times in seconds to match pyRing's
    explicit time-band boundaries. With detector-dependent noise, a mapping
    from detector name to cut-time list is also accepted. Noise parameters for
    explicit bands are named after their time interval, with decimal points
    written as ``p`` (for example, ``nu_H1_0_0p5``).
    """

    def __init__(
        self,
        interferometers,
        waveform_generator,
        likelihood_method="cholesky-solve-triangular",
        time_bands=1,
        time_band_boundaries=None,
        detector_dependent_noise=False,
        prefer_time_domain_waveform=False,
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
        if (
            time_marginalization
            or distance_marginalization
            or phase_marginalization
            or calibration_marginalization
        ):
            raise ValueError(
                "TimeDomainGravitationalWaveTransient does not support "
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

        self.likelihood_method = _resolve_likelihood_method(likelihood_method)
        self.detector_dependent_noise = bool(detector_dependent_noise)
        self._detector_names = [
            interferometer.name for interferometer in self.interferometers
        ]
        self.time_bands = _resolve_time_bands(
            time_bands=time_bands,
            time_band_boundaries=time_band_boundaries,
            detector_names=self._detector_names,
            detector_dependent_noise=self.detector_dependent_noise,
        )
        self.time_band_boundaries = (
            self.time_bands
            if _is_time_band_cut_list(self.time_bands)
            or _is_detector_time_band_map(self.time_bands)
            else None
        )
        self.prefer_time_domain_waveform = bool(prefer_time_domain_waveform)
        self._number_of_time_bands = _time_band_count(self.time_bands)
        self._detector_likelihood_caches = self._build_detector_likelihood_caches()

    @property
    def meta_data(self):
        meta_data = super().meta_data
        meta_data.update(
            likelihood_class=self.__class__,
            likelihood_method=self.likelihood_method,
            time_bands=self.time_bands,
            time_band_boundaries=self.time_band_boundaries,
            prefer_time_domain_waveform=self.prefer_time_domain_waveform,
        )
        return meta_data

    def _build_detector_likelihood_caches(self):
        caches = dict()
        for interferometer in self.interferometers:
            full_psd = self._build_finite_psd_array(interferometer)
            acf = self._acf_from_psd(full_psd, interferometer)
            cache = _make_likelihood_cache(
                acf=acf,
                likelihood_method=self.likelihood_method,
            )
            time_band_cache = None
            if self._number_of_time_bands > 1:
                time_band_cache = _make_time_band_likelihood_cache(
                    acf=acf,
                    likelihood_method=self.likelihood_method,
                    time_bands=_time_bands_for_detector(
                        self.time_bands, interferometer.name
                    ),
                    sampling_rate=interferometer.sampling_frequency,
                )
            caches[interferometer.name] = dict(
                full=cache,
                time_bands=time_band_cache,
                psd=full_psd,
            )
        return caches

    def _time_band_suffixes(self, detector_name=None):
        if self.time_band_boundaries is None:
            return [
                str(index) for index in range(1, self._number_of_time_bands + 1)
            ]

        if detector_name is None:
            detector_name = self._detector_names[0]
        boundaries = _time_bands_for_detector(
            self.time_band_boundaries, detector_name
        )
        interferometer = next(
            interferometer
            for interferometer in self.interferometers
            if interferometer.name == detector_name
        )
        edges = [0.0, *boundaries, float(interferometer.duration)]
        return [
            f"{lower:g}_{upper:g}".replace(".", "p")
            for lower, upper in zip(edges[:-1], edges[1:])
        ]

    @staticmethod
    def _acf_from_psd(psd, interferometer):
        df = 1.0 / interferometer.duration
        number_of_samples = len(interferometer.time_array)
        return (
            0.5
            * np.real(np.fft.irfft(np.asarray(psd, dtype=float) * df, n=number_of_samples))
            * number_of_samples
        )

    @staticmethod
    def _build_finite_psd_array(interferometer):
        psd = interferometer.power_spectral_density
        source_frequencies = np.asarray(psd.frequency_array, dtype=float)
        source_psd = np.asarray(psd.psd_array, dtype=float)
        if len(source_frequencies) == 0 or len(source_psd) == 0:
            raise ValueError(
                f"Unable to construct a time-domain covariance for {interferometer.name}: "
                "empty PSD"
            )

        analysis_frequencies = np.asarray(interferometer.frequency_array, dtype=float)
        active_frequencies = analysis_frequencies[interferometer.frequency_mask]
        if len(active_frequencies) == 0:
            raise ValueError(
                f"Unable to construct a time-domain covariance for {interferometer.name}: "
                "no active frequencies"
            )

        interpolation = interp1d(
            source_frequencies,
            source_psd,
            bounds_error=False,
            fill_value=(float(source_psd[0]), float(source_psd[-1])),
        )
        finite_psd = np.asarray(interpolation(analysis_frequencies), dtype=float)
        finite_mask = np.isfinite(finite_psd) & (finite_psd > 0.0)
        if not np.all(finite_mask):
            fill_value = float(
                np.max(source_psd[np.isfinite(source_psd) & (source_psd > 0.0)])
            )
            finite_psd[~finite_mask] = fill_value
        return _patch_psd_outside_active_band(
            psd=finite_psd,
            frequencies=analysis_frequencies,
            active_frequencies=active_frequencies,
        )

    def _resolve_likelihood_parameters(self, parameters):
        return parameters.copy()

    def _resolve_signal_likelihood_parameters(self, parameters):
        parameters = self._resolve_likelihood_parameters(parameters)
        parameters.update(self.get_sky_frame_parameters(parameters))
        return parameters

    def _detector_response_frequency_domain(self, waveform_polarizations, interferometer, parameters):
        return interferometer.get_detector_response(
            waveform_polarizations,
            parameters,
            frequencies=interferometer.frequency_array,
        )

    def _waveform_polarizations_frequency_domain(self, parameters):
        use_native_time_domain = (
            self.prefer_time_domain_waveform
            and (
                self.waveform_generator.time_domain_source_model is not None
                or isinstance(self.waveform_generator, GWSignalWaveformGenerator)
            )
        )
        if not use_native_time_domain:
            return self.waveform_generator.frequency_domain_strain(parameters)

        waveform_polarizations = self.waveform_generator.time_domain_strain(parameters)
        if waveform_polarizations is None:
            return None

        frequency_domain_polarizations = dict()
        for mode, strain in waveform_polarizations.items():
            frequency_domain_polarizations[mode], _ = core_utils.nfft(
                strain, self.waveform_generator.sampling_frequency
            )
        return frequency_domain_polarizations

    def _signal_time_domain(self, interferometer, waveform_polarizations, parameters):
        signal_frequency_domain = self._detector_response_frequency_domain(
            waveform_polarizations=waveform_polarizations,
            interferometer=interferometer,
            parameters=parameters,
        )
        return np.real(
            core_utils.infft(
                signal_frequency_domain,
                self.waveform_generator.sampling_frequency,
            )
        )

    @staticmethod
    def _data_time_domain(interferometer):
        return np.asarray(interferometer.time_domain_strain, dtype=float)

    def _residual_time_domain(self, interferometer, parameters, waveform_polarizations=None):
        data = self._data_time_domain(interferometer)
        if waveform_polarizations is None:
            return data
        signal = self._signal_time_domain(
            interferometer=interferometer,
            waveform_polarizations=waveform_polarizations,
            parameters=parameters,
        )
        return data - signal

    def _use_time_band_cache(self):
        return False

    def _log_likelihood_from_inner_product(
        self, residuals_inner_product, cache, dimension, interferometer_name, band_index, parameters
    ):
        del dimension, interferometer_name, band_index, parameters
        return _gaussian_log_likelihood_from_inner_product(
            residuals_inner_product=residuals_inner_product,
            log_normalisation=cache.log_normalisation,
        )

    def _compute_detector_log_likelihood(
        self, interferometer, residuals, parameters
    ):
        detector_cache = self._detector_likelihood_caches[interferometer.name]
        if self._use_time_band_cache() and detector_cache["time_bands"] is not None:
            log_likelihood = 0.0
            for band_index, band_cache in enumerate(detector_cache["time_bands"]):
                band_residuals = residuals[band_cache.start : band_cache.end]
                residuals_inner_product = _residuals_inner_product_from_cache(
                    band_residuals,
                    band_cache,
                    self.likelihood_method,
                )
                log_likelihood += self._log_likelihood_from_inner_product(
                    residuals_inner_product=residuals_inner_product,
                    cache=band_cache,
                    dimension=band_cache.end - band_cache.start,
                    interferometer_name=interferometer.name,
                    band_index=band_index,
                    parameters=parameters,
                )
            return float(log_likelihood)

        cache = detector_cache["full"]
        residuals_inner_product = _residuals_inner_product_from_cache(
            residuals,
            cache,
            self.likelihood_method,
        )
        return float(
            self._log_likelihood_from_inner_product(
                residuals_inner_product=residuals_inner_product,
                cache=cache,
                dimension=len(residuals),
                interferometer_name=interferometer.name,
                band_index=0,
                parameters=parameters,
            )
        )

    def _network_residual_statistics(self, residuals_by_detector):
        statistics = []
        for band_index in range(self._number_of_time_bands):
            residuals_inner_product = 0.0
            logdet = 0.0
            dimension = 0
            for interferometer in self.interferometers:
                detector_cache = self._detector_likelihood_caches[
                    interferometer.name
                ]
                if self._use_time_band_cache():
                    cache = detector_cache["time_bands"][band_index]
                else:
                    cache = detector_cache["full"]
                residuals = residuals_by_detector[interferometer.name]
                band_residuals = residuals[cache.start : cache.end]
                residuals_inner_product += _residuals_inner_product_from_cache(
                    band_residuals,
                    cache,
                    self.likelihood_method,
                )
                logdet += cache.logdet
                dimension += cache.end - cache.start
            statistics.append(
                (residuals_inner_product, logdet, dimension)
            )
        return statistics

    def _noise_log_likelihood_from_parameters(self, parameters):
        parameters = self._resolve_likelihood_parameters(parameters)
        log_likelihood = 0.0
        for interferometer in self.interferometers:
            residuals = self._residual_time_domain(
                interferometer=interferometer,
                parameters=parameters,
            )
            log_likelihood += self._compute_detector_log_likelihood(
                interferometer=interferometer,
                residuals=residuals,
                parameters=parameters,
            )
        return float(log_likelihood)

    def log_likelihood(self, parameters):
        parameters = self._resolve_signal_likelihood_parameters(parameters)
        waveform_polarizations = self._waveform_polarizations_frequency_domain(parameters)
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        log_likelihood = 0.0
        for interferometer in self.interferometers:
            residuals = self._residual_time_domain(
                interferometer=interferometer,
                parameters=parameters,
                waveform_polarizations=waveform_polarizations,
            )
            log_likelihood += self._compute_detector_log_likelihood(
                interferometer=interferometer,
                residuals=residuals,
                parameters=parameters,
            )
        return float(log_likelihood)

    def noise_log_likelihood(self):
        return self._noise_log_likelihood_from_parameters(dict())

    def _validate_active_noise_parameters(self, parameters):
        return True

    def log_likelihood_ratio(self, parameters):
        parameters = self._resolve_signal_likelihood_parameters(parameters)
        if not self._validate_active_noise_parameters(parameters):
            return np.nan_to_num(-np.inf)
        return float(
            self.log_likelihood(parameters=parameters)
            - self._noise_log_likelihood_from_parameters(parameters)
        )

    def compute_per_detector_log_likelihood(self, parameters):
        parameters = self._resolve_signal_likelihood_parameters(parameters)
        if not self._validate_active_noise_parameters(parameters):
            for interferometer in self.interferometers:
                parameters[f"{interferometer.name}_log_likelihood"] = np.nan_to_num(
                    -np.inf
                )
            return parameters.copy()

        waveform_polarizations = self._waveform_polarizations_frequency_domain(parameters)
        if waveform_polarizations is None:
            for interferometer in self.interferometers:
                parameters[f"{interferometer.name}_log_likelihood"] = np.nan_to_num(-np.inf)
            return parameters.copy()

        for interferometer in self.interferometers:
            residuals = self._residual_time_domain(
                interferometer=interferometer,
                parameters=parameters,
                waveform_polarizations=waveform_polarizations,
            )
            signal_log_likelihood = self._compute_detector_log_likelihood(
                interferometer=interferometer,
                residuals=residuals,
                parameters=parameters,
            )
            noise_log_likelihood = self._compute_detector_log_likelihood(
                interferometer=interferometer,
                residuals=self._data_time_domain(interferometer),
                parameters=parameters,
            )
            parameters[f"{interferometer.name}_log_likelihood"] = float(
                signal_log_likelihood - noise_log_likelihood
            )
        return parameters.copy()


class _StudentTTimeDomainNoiseOnlyLikelihood(Likelihood):
    def __init__(self, student_likelihood):
        super().__init__()
        self.student_likelihood = student_likelihood

    def log_likelihood(self, parameters):
        return self.student_likelihood._noise_log_likelihood_from_parameters(parameters)

    def noise_log_likelihood(self):
        return 0.0


class StudentTTimeDomainGravitationalWaveTransient(TimeDomainGravitationalWaveTransient):
    """A time-domain multivariate Student-t likelihood.

    With ``joint=True``, each time band is one network density over the stacked
    detector residuals. The detector covariance remains block diagonal; the
    heavy-tail radial scale is shared. Per-detector outputs remain standalone
    diagnostics and are not an additive decomposition of the joint ratio.
    """

    _NOISE_EVIDENCE_QUADRATURE_EPSABS = 0.0
    _NOISE_EVIDENCE_QUADRATURE_EPSREL = 1e-8
    _NOISE_EVIDENCE_QUADRATURE_LIMIT = 200

    def __init__(
        self,
        interferometers,
        waveform_generator,
        nu=8.0,
        infer_nu=False,
        detector_dependent_noise=False,
        likelihood_method="cholesky-solve-triangular",
        time_bands=1,
        time_band_boundaries=None,
        prefer_time_domain_waveform=False,
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
        joint=False,
        **kwargs,
    ):
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            likelihood_method=likelihood_method,
            time_bands=time_bands,
            time_band_boundaries=time_band_boundaries,
            detector_dependent_noise=detector_dependent_noise,
            prefer_time_domain_waveform=prefer_time_domain_waveform,
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

        self.detector_dependent_noise = bool(detector_dependent_noise)
        self.joint = bool(joint)
        if self.joint and self.detector_dependent_noise:
            raise ValueError(
                "joint=True requires detector_dependent_noise=False"
            )
        self.infer_nu = bool(infer_nu)
        self._fixed_nu = self._coerce_nu_array(nu)
        self.noise_evidence_nlive = _validate_noise_evidence_nlive(noise_evidence_nlive)
        self.dlogz_noise = _resolve_dlogz_noise(dlogz_noise=dlogz_noise, dlogZ_noise=dlogZ_noise)
        self.noise_evidence_method = _resolve_noise_evidence_method(noise_evidence_method)

        if not self._valid_nu_values(self._fixed_nu):
            raise ValueError("All nu values must be positive and finite")

    @property
    def meta_data(self):
        meta_data = super().meta_data
        meta_data.update(
            likelihood_class=self.__class__,
            nu=np.asarray(self._fixed_nu).tolist(),
            infer_nu=self.infer_nu,
            detector_dependent_noise=self.detector_dependent_noise,
            joint=self.joint,
            noise_evidence_method=self.noise_evidence_method,
        )
        return meta_data

    @property
    def noise_parameter_keys(self):
        if not self.infer_nu:
            return []
        keys = list(self.nu_parameter_keys)
        if "nu" not in keys:
            keys.insert(0, "nu")
        return keys

    @property
    def nu_parameter_keys(self):
        if not self.detector_dependent_noise:
            if self._number_of_time_bands == 1:
                return ["nu"]
            return [f"nu_{suffix}" for suffix in self._time_band_suffixes()]

        if self._number_of_time_bands == 1:
            return [f"nu_{detector_name}" for detector_name in self._detector_names]

        return [
            f"nu_{detector_name}_{suffix}"
            for detector_name in self._detector_names
            for suffix in self._time_band_suffixes(detector_name)
        ]

    def _use_time_band_cache(self):
        return self._number_of_time_bands > 1

    def _coerce_nu_array(self, values):
        values = np.asarray(values, dtype=float)
        if not self.detector_dependent_noise:
            if values.ndim == 0:
                values = np.repeat(values[None], self._number_of_time_bands)
            elif values.ndim == 1 and len(values) == 1:
                values = np.repeat(values, self._number_of_time_bands)
            elif values.ndim != 1 or len(values) != self._number_of_time_bands:
                raise ValueError(
                    "nu must be a scalar or an array with one entry per time band"
                )
            return values.astype(float, copy=False)

        num_detectors = len(self.interferometers)
        if values.ndim == 0:
            values = np.full((num_detectors, self._number_of_time_bands), float(values))
        elif values.ndim == 1:
            if len(values) == 1:
                values = np.full((num_detectors, self._number_of_time_bands), float(values[0]))
            elif len(values) == self._number_of_time_bands:
                values = np.repeat(values[None, :], num_detectors, axis=0)
            elif len(values) == num_detectors and self._number_of_time_bands == 1:
                values = values[:, None]
            else:
                raise ValueError(
                    "nu must be a scalar, an array with one entry per time band, "
                    "an array with one entry per detector when there is one time band, "
                    "or a 2D array with shape (num_detectors, num_time_bands)"
                )
        elif values.ndim == 2:
            if values.shape != (num_detectors, self._number_of_time_bands):
                raise ValueError(
                    "nu must have shape (num_detectors, num_time_bands) "
                    "when detector_dependent_noise=True"
                )
        else:
            raise ValueError("nu must be scalar, 1D, or 2D")
        return values.astype(float, copy=False)

    @staticmethod
    def _valid_nu_values(values):
        return np.all(np.isfinite(values)) and np.all(values > 0)

    def _detector_nu_parameter_key(self, detector_name, band_index):
        if self._number_of_time_bands == 1:
            return f"nu_{detector_name}"
        suffix = self._time_band_suffixes(detector_name)[band_index]
        return f"nu_{detector_name}_{suffix}"

    def _get_nu_values(self, parameters):
        if not self.infer_nu:
            return self._fixed_nu

        if "nu" in parameters:
            return self._coerce_nu_array(parameters["nu"])

        if not self.detector_dependent_noise:
            if self._number_of_time_bands == 1:
                return self._coerce_nu_array(parameters.get("nu", self._fixed_nu[0]))
            return self._coerce_nu_array(
                [parameters.get(key, default) for key, default in zip(self.nu_parameter_keys, self._fixed_nu)]
            )

        return self._coerce_nu_array(
            [
                [
                    parameters.get(
                        self._detector_nu_parameter_key(detector_name, band_index),
                        self._fixed_nu[detector_index, band_index],
                    )
                    for band_index in range(self._number_of_time_bands)
                ]
                for detector_index, detector_name in enumerate(self._detector_names)
            ]
        )

    def _band_nu(self, parameters, interferometer_name, band_index):
        nu_values = self._get_nu_values(parameters)
        if not self.detector_dependent_noise:
            return float(nu_values[min(band_index, len(nu_values) - 1)])
        detector_index = self._detector_names.index(interferometer_name)
        return float(nu_values[detector_index, band_index])

    def _get_active_nu_values(self, parameters):
        nu_values = self._get_nu_values(parameters)
        if not self._valid_nu_values(nu_values):
            return None
        return nu_values

    def _validate_active_noise_parameters(self, parameters):
        return self._get_active_nu_values(parameters) is not None

    def _compute_network_log_likelihood(
        self, residuals_by_detector, nu_values
    ):
        log_likelihood = 0.0
        for band_index, statistics in enumerate(
            self._network_residual_statistics(residuals_by_detector)
        ):
            residuals_inner_product, logdet, dimension = statistics
            log_likelihood += _student_t_log_likelihood_from_inner_product(
                residuals_inner_product=residuals_inner_product,
                logdet=logdet,
                dimension=dimension,
                nu=float(nu_values[band_index]),
            )
        return float(log_likelihood)

    def _log_likelihood_from_inner_product(
        self, residuals_inner_product, cache, dimension, interferometer_name, band_index, parameters
    ):
        nu = self._band_nu(parameters, interferometer_name, band_index)
        return _student_t_log_likelihood_from_inner_product(
            residuals_inner_product=residuals_inner_product,
            logdet=cache.logdet,
            dimension=dimension,
            nu=nu,
        )

    def log_likelihood(self, parameters):
        parameters = self._resolve_signal_likelihood_parameters(parameters)
        nu_values = self._get_active_nu_values(parameters)
        if nu_values is None:
            return np.nan_to_num(-np.inf)
        if self.joint:
            waveform_polarizations = (
                self._waveform_polarizations_frequency_domain(parameters)
            )
            if waveform_polarizations is None:
                return np.nan_to_num(-np.inf)
            residuals_by_detector = {
                interferometer.name: self._residual_time_domain(
                    interferometer=interferometer,
                    parameters=parameters,
                    waveform_polarizations=waveform_polarizations,
                )
                for interferometer in self.interferometers
            }
            return self._compute_network_log_likelihood(
                residuals_by_detector=residuals_by_detector,
                nu_values=nu_values,
            )
        return super().log_likelihood(parameters=parameters)

    def _noise_log_likelihood_from_parameters(self, parameters):
        parameters = self._resolve_likelihood_parameters(parameters)
        nu_values = self._get_active_nu_values(parameters)
        if nu_values is None:
            return np.nan_to_num(-np.inf)
        if self.joint:
            return self._compute_network_log_likelihood(
                residuals_by_detector={
                    interferometer.name: self._data_time_domain(interferometer)
                    for interferometer in self.interferometers
                },
                nu_values=nu_values,
            )
        return super()._noise_log_likelihood_from_parameters(parameters)

    def _get_default_nu_parameter_dict(self):
        nu_values = self._get_nu_values(dict())
        return {
            key: float(value)
            for key, value in zip(self.nu_parameter_keys, np.ravel(nu_values))
        }

    def _get_noise_evidence_priors(self, priors):
        if priors is None:
            priors = PriorDict()
        elif not isinstance(priors, PriorDict):
            priors = PriorDict(priors)

        default_nu_parameters = self._get_default_nu_parameter_dict()
        noise_priors = PriorDict()

        if "nu" not in default_nu_parameters and "nu" in priors:
            conflicting_keys = [
                key for key in default_nu_parameters if key in priors
            ]
            if conflicting_keys:
                raise ValueError(
                    "Specify either the shared nu prior or per-band/per-detector "
                    "nu priors, not both"
                )
            noise_priors["nu"] = deepcopy(priors["nu"])
            return noise_priors

        for key, value in default_nu_parameters.items():
            if key in priors:
                noise_priors[key] = deepcopy(priors[key])
            else:
                noise_priors[key] = DeltaFunction(peak=value, name=key)
        return noise_priors

    def _get_noise_evidence_parameter_dict(self, noise_priors):
        parameters = self._get_default_nu_parameter_dict()
        for key in noise_priors.fixed_keys:
            parameters[key] = float(noise_priors[key].peak)
        if "nu" in noise_priors and "nu" not in parameters:
            parameters["nu"] = float(np.ravel(self._fixed_nu)[0])
        return parameters

    def _build_noise_parameter_blocks(self):
        blocks = dict()
        for detector_index, interferometer in enumerate(self.interferometers):
            detector_cache = self._detector_likelihood_caches[interferometer.name]
            caches = (
                detector_cache["time_bands"]
                if self._use_time_band_cache() and detector_cache["time_bands"] is not None
                else [detector_cache["full"]]
            )
            data = self._data_time_domain(interferometer)
            for band_index, cache in enumerate(caches):
                if self.detector_dependent_noise:
                    block_id = (interferometer.name, band_index)
                    keys = (
                        (self._detector_nu_parameter_key(interferometer.name, band_index),)
                        if self.infer_nu
                        else ()
                    )
                    default_nu = float(self._fixed_nu[detector_index, band_index])
                else:
                    block_id = band_index
                    keys = (
                        (self.nu_parameter_keys[band_index],)
                        if self.infer_nu
                        else ()
                    )
                    default_nu = float(self._fixed_nu[band_index])

                block = blocks.setdefault(
                    block_id,
                    dict(keys=keys, default_nu=default_nu, terms=[]),
                )
                band_data = data[cache.start : cache.end]
                block["terms"].append(
                    dict(
                        residuals_inner_product=_residuals_inner_product_from_cache(
                            band_data,
                            cache,
                            self.likelihood_method,
                        ),
                        logdet=cache.logdet,
                        dimension=cache.end - cache.start,
                    )
                )
        return list(blocks.values())

    def _noise_block_log_likelihood(self, block, parameters):
        nu = float(parameters.get(block["keys"][0], block["default_nu"]))
        if nu <= 0.0 or not np.isfinite(nu):
            return np.nan_to_num(-np.inf)

        if self.joint:
            return _student_t_log_likelihood_from_inner_product(
                residuals_inner_product=sum(
                    term["residuals_inner_product"] for term in block["terms"]
                ),
                logdet=sum(term["logdet"] for term in block["terms"]),
                dimension=sum(term["dimension"] for term in block["terms"]),
                nu=nu,
            )

        log_likelihood = 0.0
        for term in block["terms"]:
            log_likelihood += _student_t_log_likelihood_from_inner_product(
                residuals_inner_product=term["residuals_inner_product"],
                logdet=term["logdet"],
                dimension=term["dimension"],
                nu=nu,
            )
        return float(log_likelihood)

    def _noise_log_evidence_by_quadrature(self, noise_priors):
        base_parameters = self._get_noise_evidence_parameter_dict(noise_priors)
        if "nu" in noise_priors:
            return _factorized_noise_log_evidence_by_quadrature(
                blocks=[dict(keys=("nu",))],
                noise_priors=noise_priors,
                base_parameters=base_parameters,
                block_log_likelihood=lambda block, parameters: (
                    self._noise_log_likelihood_from_parameters(parameters)
                ),
                epsabs=self._NOISE_EVIDENCE_QUADRATURE_EPSABS,
                epsrel=self._NOISE_EVIDENCE_QUADRATURE_EPSREL,
                limit=self._NOISE_EVIDENCE_QUADRATURE_LIMIT,
                error_label="Student-t",
            )
        return _factorized_noise_log_evidence_by_quadrature(
            blocks=self._build_noise_parameter_blocks(),
            noise_priors=noise_priors,
            base_parameters=base_parameters,
            block_log_likelihood=self._noise_block_log_likelihood,
            epsabs=self._NOISE_EVIDENCE_QUADRATURE_EPSABS,
            epsrel=self._NOISE_EVIDENCE_QUADRATURE_EPSREL,
            limit=self._NOISE_EVIDENCE_QUADRATURE_LIMIT,
            error_label="Student-t",
        )

    def _noise_log_evidence_by_nested_sampling(self, noise_priors):
        from ...core.sampler import run_sampler

        return run_sampler(
            likelihood=_StudentTTimeDomainNoiseOnlyLikelihood(self),
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

    def noise_log_evidence(self, priors=None, sampler=None, result=None, npool=1):
        if not self.infer_nu:
            return self.noise_log_likelihood()

        noise_priors = self._get_noise_evidence_priors(priors)
        if len(noise_priors.non_fixed_keys) == 0:
            return self._noise_log_likelihood_from_parameters(
                self._get_noise_evidence_parameter_dict(noise_priors)
            )

        if self.noise_evidence_method == "quadrature":
            return self._noise_log_evidence_by_quadrature(noise_priors)

        noise_result = self._noise_log_evidence_by_nested_sampling(noise_priors)
        return float(noise_result.log_evidence)


class _HyperbolicTimeDomainNoiseOnlyLikelihood(Likelihood):
    def __init__(self, hyperbolic_likelihood):
        super().__init__()
        self.hyperbolic_likelihood = hyperbolic_likelihood

    def log_likelihood(self, parameters):
        return self.hyperbolic_likelihood._noise_log_likelihood_from_parameters(parameters)

    def noise_log_likelihood(self):
        return 0.0


class HyperbolicTimeDomainGravitationalWaveTransient(TimeDomainGravitationalWaveTransient):
    """A time-domain multivariate hyperbolic likelihood.

    With ``joint=True``, each time band is one network density over the stacked
    detector residuals. The detector covariance remains block diagonal; the
    heavy-tail radial scale is shared. Per-detector outputs remain standalone
    diagnostics and are not an additive decomposition of the joint ratio.
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
        detector_dependent_noise=False,
        likelihood_method="cholesky-solve-triangular",
        time_bands=1,
        time_band_boundaries=None,
        prefer_time_domain_waveform=False,
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
        joint=False,
        **kwargs,
    ):
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            likelihood_method=likelihood_method,
            time_bands=time_bands,
            time_band_boundaries=time_band_boundaries,
            detector_dependent_noise=detector_dependent_noise,
            prefer_time_domain_waveform=prefer_time_domain_waveform,
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

        self.detector_dependent_noise = bool(detector_dependent_noise)
        self.joint = bool(joint)
        if self.joint and self.detector_dependent_noise:
            raise ValueError(
                "joint=True requires detector_dependent_noise=False"
            )
        self.infer_alpha = bool(infer_alpha)
        self.infer_delta = bool(infer_delta)
        self._fixed_alpha = self._coerce_parameter_array(alpha, "alpha")
        self._fixed_delta = self._coerce_parameter_array(delta, "delta")
        self.noise_evidence_nlive = _validate_noise_evidence_nlive(noise_evidence_nlive)
        self.dlogz_noise = _resolve_dlogz_noise(dlogz_noise=dlogz_noise, dlogZ_noise=dlogZ_noise)
        self.noise_evidence_method = _resolve_noise_evidence_method(noise_evidence_method)

        if not self._valid_positive_values(self._fixed_alpha):
            raise ValueError("All alpha values must be positive and finite")
        if not self._valid_positive_values(self._fixed_delta):
            raise ValueError("All delta values must be positive and finite")

    @property
    def meta_data(self):
        meta_data = super().meta_data
        meta_data.update(
            likelihood_class=self.__class__,
            alpha=np.asarray(self._fixed_alpha).tolist(),
            delta=np.asarray(self._fixed_delta).tolist(),
            infer_alpha=self.infer_alpha,
            infer_delta=self.infer_delta,
            detector_dependent_noise=self.detector_dependent_noise,
            joint=self.joint,
            noise_evidence_method=self.noise_evidence_method,
        )
        return meta_data

    @property
    def noise_parameter_keys(self):
        keys = []
        if self.infer_alpha:
            keys.extend(self.alpha_parameter_keys)
        if self.infer_delta:
            keys.extend(self.delta_parameter_keys)
        return keys

    @property
    def alpha_parameter_keys(self):
        if not self.detector_dependent_noise:
            if self._number_of_time_bands == 1:
                return ["alpha"]
            return [f"alpha_{suffix}" for suffix in self._time_band_suffixes()]
        if self._number_of_time_bands == 1:
            return [f"alpha_{detector_name}" for detector_name in self._detector_names]
        return [
            f"alpha_{detector_name}_{suffix}"
            for detector_name in self._detector_names
            for suffix in self._time_band_suffixes(detector_name)
        ]

    @property
    def delta_parameter_keys(self):
        if not self.detector_dependent_noise:
            if self._number_of_time_bands == 1:
                return ["delta"]
            return [f"delta_{suffix}" for suffix in self._time_band_suffixes()]
        if self._number_of_time_bands == 1:
            return [f"delta_{detector_name}" for detector_name in self._detector_names]
        return [
            f"delta_{detector_name}_{suffix}"
            for detector_name in self._detector_names
            for suffix in self._time_band_suffixes(detector_name)
        ]

    def _use_time_band_cache(self):
        return self._number_of_time_bands > 1

    def _coerce_parameter_array(self, values, name):
        values = np.asarray(values, dtype=float)
        if not self.detector_dependent_noise:
            if values.ndim == 0:
                values = np.repeat(values[None], self._number_of_time_bands)
            elif values.ndim == 1 and len(values) == 1:
                values = np.repeat(values, self._number_of_time_bands)
            elif values.ndim != 1 or len(values) != self._number_of_time_bands:
                raise ValueError(
                    f"{name} must be a scalar or an array with one entry per time band"
                )
            return values.astype(float, copy=False)

        num_detectors = len(self.interferometers)
        if values.ndim == 0:
            values = np.full((num_detectors, self._number_of_time_bands), float(values))
        elif values.ndim == 1:
            if len(values) == 1:
                values = np.full((num_detectors, self._number_of_time_bands), float(values[0]))
            elif len(values) == self._number_of_time_bands:
                values = np.repeat(values[None, :], num_detectors, axis=0)
            elif len(values) == num_detectors and self._number_of_time_bands == 1:
                values = values[:, None]
            else:
                raise ValueError(
                    f"{name} must be a scalar, an array with one entry per time band, "
                    "an array with one entry per detector when there is one time band, "
                    "or a 2D array with shape (num_detectors, num_time_bands)"
                )
        elif values.ndim == 2:
            if values.shape != (num_detectors, self._number_of_time_bands):
                raise ValueError(
                    f"{name} must have shape (num_detectors, num_time_bands) "
                    "when detector_dependent_noise=True"
                )
        else:
            raise ValueError(f"{name} must be scalar, 1D, or 2D")
        return values.astype(float, copy=False)

    @staticmethod
    def _valid_positive_values(values):
        return np.all(np.isfinite(values)) and np.all(values > 0)

    def _detector_parameter_key(self, parameter_name, detector_name, band_index):
        if self._number_of_time_bands == 1:
            return f"{parameter_name}_{detector_name}"
        suffix = self._time_band_suffixes(detector_name)[band_index]
        return f"{parameter_name}_{detector_name}_{suffix}"

    def _get_parameter_values(self, parameters, parameter_name, infer_flag, fixed_values):
        if not infer_flag:
            return fixed_values

        if parameter_name in parameters and not self.detector_dependent_noise:
            return self._coerce_parameter_array(parameters[parameter_name], parameter_name)

        if not self.detector_dependent_noise:
            keys = getattr(self, f"{parameter_name}_parameter_keys")
            if self._number_of_time_bands == 1:
                return self._coerce_parameter_array(
                    parameters.get(parameter_name, fixed_values[0]),
                    parameter_name,
                )
            return self._coerce_parameter_array(
                [parameters.get(key, default) for key, default in zip(keys, fixed_values)],
                parameter_name,
            )

        return self._coerce_parameter_array(
            [
                [
                    parameters.get(
                        self._detector_parameter_key(parameter_name, detector_name, band_index),
                        fixed_values[detector_index, band_index],
                    )
                    for band_index in range(self._number_of_time_bands)
                ]
                for detector_index, detector_name in enumerate(self._detector_names)
            ],
            parameter_name,
        )

    def _get_active_shape_parameters(self, parameters):
        alpha_values = self._get_parameter_values(
            parameters, "alpha", self.infer_alpha, self._fixed_alpha
        )
        delta_values = self._get_parameter_values(
            parameters, "delta", self.infer_delta, self._fixed_delta
        )

        if not self._valid_positive_values(alpha_values):
            return None, None
        if not self._valid_positive_values(delta_values):
            return None, None
        return alpha_values, delta_values

    def _validate_active_noise_parameters(self, parameters):
        alpha_values, delta_values = self._get_active_shape_parameters(
            parameters
        )
        return alpha_values is not None and delta_values is not None

    def _compute_network_log_likelihood(
        self, residuals_by_detector, alpha_values, delta_values
    ):
        log_likelihood = 0.0
        for band_index, statistics in enumerate(
            self._network_residual_statistics(residuals_by_detector)
        ):
            residuals_inner_product, logdet, dimension = statistics
            log_likelihood += _hyperbolic_log_likelihood_from_inner_product(
                residuals_inner_product=residuals_inner_product,
                logdet=logdet,
                dimension=dimension,
                alpha=float(alpha_values[band_index]),
                delta=float(delta_values[band_index]),
            )
        return float(log_likelihood)

    def _band_shape_parameters(self, parameters, interferometer_name, band_index):
        alpha_values, delta_values = self._get_active_shape_parameters(
            parameters
        )
        if alpha_values is None or delta_values is None:
            return None, None
        if not self.detector_dependent_noise:
            return float(alpha_values[band_index]), float(delta_values[band_index])
        detector_index = self._detector_names.index(interferometer_name)
        return (
            float(alpha_values[detector_index, band_index]),
            float(delta_values[detector_index, band_index]),
        )

    def _log_likelihood_from_inner_product(
        self, residuals_inner_product, cache, dimension, interferometer_name, band_index, parameters
    ):
        alpha, delta = self._band_shape_parameters(
            parameters, interferometer_name, band_index
        )
        return _hyperbolic_log_likelihood_from_inner_product(
            residuals_inner_product=residuals_inner_product,
            logdet=cache.logdet,
            dimension=dimension,
            alpha=alpha,
            delta=delta,
        )

    def log_likelihood(self, parameters):
        parameters = self._resolve_signal_likelihood_parameters(parameters)
        alpha_values, delta_values = self._get_active_shape_parameters(
            parameters
        )
        if alpha_values is None or delta_values is None:
            return np.nan_to_num(-np.inf)
        if self.joint:
            waveform_polarizations = (
                self._waveform_polarizations_frequency_domain(parameters)
            )
            if waveform_polarizations is None:
                return np.nan_to_num(-np.inf)
            residuals_by_detector = {
                interferometer.name: self._residual_time_domain(
                    interferometer=interferometer,
                    parameters=parameters,
                    waveform_polarizations=waveform_polarizations,
                )
                for interferometer in self.interferometers
            }
            return self._compute_network_log_likelihood(
                residuals_by_detector=residuals_by_detector,
                alpha_values=alpha_values,
                delta_values=delta_values,
            )
        return super().log_likelihood(parameters=parameters)

    def _noise_log_likelihood_from_parameters(self, parameters):
        parameters = self._resolve_likelihood_parameters(parameters)
        alpha_values, delta_values = self._get_active_shape_parameters(
            parameters
        )
        if alpha_values is None or delta_values is None:
            return np.nan_to_num(-np.inf)
        if self.joint:
            return self._compute_network_log_likelihood(
                residuals_by_detector={
                    interferometer.name: self._data_time_domain(interferometer)
                    for interferometer in self.interferometers
                },
                alpha_values=alpha_values,
                delta_values=delta_values,
            )
        return super()._noise_log_likelihood_from_parameters(parameters)

    def _get_default_shape_parameter_dict(self):
        parameters = dict()
        shape_parameters = dict()
        alpha_values = self._get_parameter_values(
            parameters, "alpha", self.infer_alpha, self._fixed_alpha
        )
        delta_values = self._get_parameter_values(
            parameters, "delta", self.infer_delta, self._fixed_delta
        )
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

    def _get_noise_evidence_parameter_dict(self, noise_priors):
        parameters = self._get_default_shape_parameter_dict()
        for key in noise_priors.fixed_keys:
            parameters[key] = float(noise_priors[key].peak)
        return parameters

    def _build_noise_parameter_blocks(self):
        blocks = dict()
        for detector_index, interferometer in enumerate(self.interferometers):
            detector_cache = self._detector_likelihood_caches[interferometer.name]
            caches = (
                detector_cache["time_bands"]
                if self._use_time_band_cache() and detector_cache["time_bands"] is not None
                else [detector_cache["full"]]
            )
            data = self._data_time_domain(interferometer)
            for band_index, cache in enumerate(caches):
                if self.detector_dependent_noise:
                    block_id = (interferometer.name, band_index)
                    alpha_key = (
                        self._detector_parameter_key("alpha", interferometer.name, band_index)
                        if self.infer_alpha
                        else None
                    )
                    delta_key = (
                        self._detector_parameter_key("delta", interferometer.name, band_index)
                        if self.infer_delta
                        else None
                    )
                    default_alpha = float(self._fixed_alpha[detector_index, band_index])
                    default_delta = float(self._fixed_delta[detector_index, band_index])
                else:
                    block_id = band_index
                    alpha_key = self.alpha_parameter_keys[band_index] if self.infer_alpha else None
                    delta_key = self.delta_parameter_keys[band_index] if self.infer_delta else None
                    default_alpha = float(self._fixed_alpha[band_index])
                    default_delta = float(self._fixed_delta[band_index])

                keys = tuple(
                    key for key in (alpha_key, delta_key) if key is not None
                )
                block = blocks.setdefault(
                    block_id,
                    dict(
                        keys=keys,
                        alpha_key=alpha_key,
                        delta_key=delta_key,
                        default_alpha=default_alpha,
                        default_delta=default_delta,
                        terms=[],
                    ),
                )
                band_data = data[cache.start : cache.end]
                block["terms"].append(
                    dict(
                        residuals_inner_product=_residuals_inner_product_from_cache(
                            band_data,
                            cache,
                            self.likelihood_method,
                        ),
                        logdet=cache.logdet,
                        dimension=cache.end - cache.start,
                    )
                )
        return list(blocks.values())

    def _noise_block_log_likelihood(self, block, parameters):
        alpha = float(parameters.get(block["alpha_key"], block["default_alpha"]))
        delta = float(parameters.get(block["delta_key"], block["default_delta"]))
        if alpha <= 0.0 or delta <= 0.0:
            return np.nan_to_num(-np.inf)
        if not np.isfinite(alpha) or not np.isfinite(delta):
            return np.nan_to_num(-np.inf)

        if self.joint:
            return _hyperbolic_log_likelihood_from_inner_product(
                residuals_inner_product=sum(
                    term["residuals_inner_product"] for term in block["terms"]
                ),
                logdet=sum(term["logdet"] for term in block["terms"]),
                dimension=sum(term["dimension"] for term in block["terms"]),
                alpha=alpha,
                delta=delta,
            )

        log_likelihood = 0.0
        for term in block["terms"]:
            log_likelihood += _hyperbolic_log_likelihood_from_inner_product(
                residuals_inner_product=term["residuals_inner_product"],
                logdet=term["logdet"],
                dimension=term["dimension"],
                alpha=alpha,
                delta=delta,
            )
        return float(log_likelihood)

    def _noise_log_evidence_by_quadrature(self, noise_priors):
        base_parameters = self._get_noise_evidence_parameter_dict(noise_priors)
        return _factorized_noise_log_evidence_by_quadrature(
            blocks=self._build_noise_parameter_blocks(),
            noise_priors=noise_priors,
            base_parameters=base_parameters,
            block_log_likelihood=self._noise_block_log_likelihood,
            epsabs=self._NOISE_EVIDENCE_QUADRATURE_EPSABS,
            epsrel=self._NOISE_EVIDENCE_QUADRATURE_EPSREL,
            limit=self._NOISE_EVIDENCE_QUADRATURE_LIMIT,
            error_label="Hyperbolic",
        )

    def _noise_log_evidence_by_nested_sampling(self, noise_priors):
        from ...core.sampler import run_sampler

        return run_sampler(
            likelihood=_HyperbolicTimeDomainNoiseOnlyLikelihood(self),
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

    def noise_log_evidence(self, priors=None, sampler=None, result=None, npool=1):
        if not (self.infer_alpha or self.infer_delta):
            return self.noise_log_likelihood()

        noise_priors = self._get_noise_evidence_priors(priors)
        if len(noise_priors.non_fixed_keys) == 0:
            return self._noise_log_likelihood_from_parameters(
                self._get_noise_evidence_parameter_dict(noise_priors)
            )

        if self.noise_evidence_method == "quadrature":
            return self._noise_log_evidence_by_quadrature(noise_priors)

        noise_result = self._noise_log_evidence_by_nested_sampling(noise_priors)
        return float(noise_result.log_evidence)


class MixedTimeDomainGravitationalWaveTransient(TimeDomainGravitationalWaveTransient):
    """A time-domain likelihood with one noise family per detector.

    Joint network densities are unavailable because a mixed-family network does
    not have one shared radial probability law.
    """

    def __init__(
        self,
        interferometers,
        waveform_generator,
        likelihood_type="gaussian",
        nu=8.0,
        infer_nu=False,
        alpha=10.0,
        delta=1.0,
        infer_alpha=False,
        infer_delta=False,
        detector_dependent_noise=False,
        likelihood_method="cholesky-solve-triangular",
        time_bands=1,
        time_band_boundaries=None,
        prefer_time_domain_waveform=False,
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
        joint=False,
        **kwargs,
    ):
        if joint:
            raise ValueError(
                "MixedTimeDomainGravitationalWaveTransient does not support "
                "joint network densities"
            )
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            likelihood_method=likelihood_method,
            time_bands=time_bands,
            time_band_boundaries=time_band_boundaries,
            detector_dependent_noise=detector_dependent_noise,
            prefer_time_domain_waveform=prefer_time_domain_waveform,
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

        self.detector_dependent_noise = bool(detector_dependent_noise)
        self.likelihood_types = _resolve_detector_likelihood_types(
            likelihood_type, self._detector_names
        )
        self.gaussian_detector_names = [
            detector_name
            for detector_name in self._detector_names
            if self.likelihood_types[detector_name] == "gaussian"
        ]
        self.student_t_detector_names = [
            detector_name
            for detector_name in self._detector_names
            if self.likelihood_types[detector_name] == "student-t"
        ]
        self.hyperbolic_detector_names = [
            detector_name
            for detector_name in self._detector_names
            if self.likelihood_types[detector_name] == "hyperbolic"
        ]

        self._gaussian_likelihood = self._build_family_likelihood(
            detector_names=self.gaussian_detector_names,
            likelihood_class=TimeDomainGravitationalWaveTransient,
            likelihood_method=likelihood_method,
            time_bands=1,
            prefer_time_domain_waveform=prefer_time_domain_waveform,
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
        )
        self._student_t_likelihood = self._build_family_likelihood(
            detector_names=self.student_t_detector_names,
            likelihood_class=StudentTTimeDomainGravitationalWaveTransient,
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            nu=self._subset_detector_values(nu, self.student_t_detector_names),
            infer_nu=infer_nu,
            detector_dependent_noise=detector_dependent_noise,
            likelihood_method=likelihood_method,
            time_bands=self._subset_time_band_specification(
                self.time_bands, self.student_t_detector_names
            ),
            time_band_boundaries=self._subset_time_band_specification(
                self.time_band_boundaries, self.student_t_detector_names
            ),
            prefer_time_domain_waveform=prefer_time_domain_waveform,
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
            noise_evidence_nlive=noise_evidence_nlive,
            dlogz_noise=dlogz_noise,
            dlogZ_noise=dlogZ_noise,
            noise_evidence_method=noise_evidence_method,
        )
        self._hyperbolic_likelihood = self._build_family_likelihood(
            detector_names=self.hyperbolic_detector_names,
            likelihood_class=HyperbolicTimeDomainGravitationalWaveTransient,
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            alpha=self._subset_detector_values(
                alpha, self.hyperbolic_detector_names
            ),
            delta=self._subset_detector_values(
                delta, self.hyperbolic_detector_names
            ),
            infer_alpha=infer_alpha,
            infer_delta=infer_delta,
            detector_dependent_noise=detector_dependent_noise,
            likelihood_method=likelihood_method,
            time_bands=self._subset_time_band_specification(
                self.time_bands, self.hyperbolic_detector_names
            ),
            time_band_boundaries=self._subset_time_band_specification(
                self.time_band_boundaries, self.hyperbolic_detector_names
            ),
            prefer_time_domain_waveform=prefer_time_domain_waveform,
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
            noise_evidence_nlive=noise_evidence_nlive,
            dlogz_noise=dlogz_noise,
            dlogZ_noise=dlogZ_noise,
            noise_evidence_method=noise_evidence_method,
        )

        self._family_likelihoods = {
            "gaussian": self._gaussian_likelihood,
            "student-t": self._student_t_likelihood,
            "hyperbolic": self._hyperbolic_likelihood,
        }
        self._family_likelihood_for_detector = {
            detector_name: self._family_likelihoods[self.likelihood_types[detector_name]]
            for detector_name in self._detector_names
        }

    @property
    def meta_data(self):
        meta_data = super().meta_data
        meta_data.update(
            likelihood_class=self.__class__,
            likelihood_types=self.likelihood_types.copy(),
            detector_dependent_noise=self.detector_dependent_noise,
        )
        return meta_data

    @staticmethod
    def _subset_interferometers(interferometers, detector_names):
        return InterferometerList(
            [
                interferometer
                for interferometer in interferometers
                if interferometer.name in detector_names
            ]
        )

    def _subset_detector_values(self, values, detector_names):
        if not self.detector_dependent_noise:
            return values

        values = np.asarray(values)
        detector_indices = [
            self._detector_names.index(detector_name)
            for detector_name in detector_names
        ]
        if values.ndim == 2 and values.shape[0] == len(self._detector_names):
            return values[detector_indices]
        if (
            self._number_of_time_bands == 1
            and values.ndim == 1
            and values.shape == (len(self._detector_names),)
        ):
            return values[detector_indices]
        return values

    @staticmethod
    def _subset_time_band_specification(time_band_specification, detector_names):
        if _is_detector_time_band_map(time_band_specification):
            return {
                detector_name: time_band_specification[detector_name]
                for detector_name in detector_names
            }
        return time_band_specification

    def _get_default_noise_parameter_dict(self):
        parameters = dict()
        if self._student_t_likelihood is not None:
            parameters.update(
                self._student_t_likelihood._get_default_nu_parameter_dict()
            )
        if self._hyperbolic_likelihood is not None:
            parameters.update(
                self._hyperbolic_likelihood._get_default_shape_parameter_dict()
            )
        return parameters

    def _build_family_likelihood(
        self,
        detector_names,
        likelihood_class,
        interferometers=None,
        waveform_generator=None,
        **kwargs,
    ):
        if len(detector_names) == 0:
            return None
        if interferometers is None:
            interferometers = self.interferometers
        if waveform_generator is None:
            waveform_generator = self.waveform_generator
        return likelihood_class(
            interferometers=self._subset_interferometers(interferometers, detector_names),
            waveform_generator=waveform_generator,
            **kwargs,
        )

    @property
    def noise_parameter_keys(self):
        keys = []
        if self._student_t_likelihood is not None:
            keys.extend(self._student_t_likelihood.noise_parameter_keys)
        if self._hyperbolic_likelihood is not None:
            keys.extend(self._hyperbolic_likelihood.noise_parameter_keys)
        return keys

    def _validate_active_noise_parameters(self, parameters):
        if self._student_t_likelihood is not None:
            nu_values = self._student_t_likelihood._get_active_nu_values(
                parameters
            )
            if nu_values is None:
                return False
        if self._hyperbolic_likelihood is not None:
            alpha_values, delta_values = (
                self._hyperbolic_likelihood._get_active_shape_parameters(
                    parameters
                )
            )
            if alpha_values is None or delta_values is None:
                return False
        return True

    def _compute_detector_log_likelihood(self, interferometer, residuals, parameters):
        return self._family_likelihood_for_detector[
            interferometer.name
        ]._compute_detector_log_likelihood(
            interferometer=interferometer,
            residuals=residuals,
            parameters=parameters,
        )

    def _noise_log_likelihood_from_parameters(self, parameters):
        parameters = self._resolve_likelihood_parameters(parameters)
        if not self._validate_active_noise_parameters(parameters):
            return np.nan_to_num(-np.inf)
        return super()._noise_log_likelihood_from_parameters(parameters)

    def log_likelihood(self, parameters):
        parameters = self._resolve_signal_likelihood_parameters(parameters)
        if not self._validate_active_noise_parameters(parameters):
            return np.nan_to_num(-np.inf)

        waveform_polarizations = self._waveform_polarizations_frequency_domain(parameters)
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        log_likelihood = 0.0
        for interferometer in self.interferometers:
            residuals = self._residual_time_domain(
                interferometer=interferometer,
                parameters=parameters,
                waveform_polarizations=waveform_polarizations,
            )
            log_likelihood += self._compute_detector_log_likelihood(
                interferometer=interferometer,
                residuals=residuals,
                parameters=parameters,
            )
        return float(log_likelihood)

    def noise_log_likelihood(self):
        return self._noise_log_likelihood_from_parameters(
            self._get_default_noise_parameter_dict()
        )

    def compute_per_detector_log_likelihood(self, parameters):
        parameters = self._resolve_signal_likelihood_parameters(parameters)
        if not self._validate_active_noise_parameters(parameters):
            for interferometer in self.interferometers:
                parameters[f"{interferometer.name}_log_likelihood"] = np.nan_to_num(
                    -np.inf
                )
            return parameters.copy()

        waveform_polarizations = self._waveform_polarizations_frequency_domain(parameters)
        if waveform_polarizations is None:
            for interferometer in self.interferometers:
                parameters[f"{interferometer.name}_log_likelihood"] = np.nan_to_num(
                    -np.inf
                )
            return parameters.copy()

        for interferometer in self.interferometers:
            residuals = self._residual_time_domain(
                interferometer=interferometer,
                parameters=parameters,
                waveform_polarizations=waveform_polarizations,
            )
            signal_log_likelihood = self._compute_detector_log_likelihood(
                interferometer=interferometer,
                residuals=residuals,
                parameters=parameters,
            )
            noise_log_likelihood = self._compute_detector_log_likelihood(
                interferometer=interferometer,
                residuals=self._data_time_domain(interferometer),
                parameters=parameters,
            )
            parameters[f"{interferometer.name}_log_likelihood"] = float(
                signal_log_likelihood - noise_log_likelihood
            )
        return parameters.copy()

    def noise_log_evidence(self, priors=None, sampler=None, result=None, npool=1):
        total_log_evidence = 0.0
        if self._gaussian_likelihood is not None:
            total_log_evidence += self._gaussian_likelihood.noise_log_likelihood()
        if self._student_t_likelihood is not None:
            total_log_evidence += self._student_t_likelihood.noise_log_evidence(
                priors=priors,
                sampler=sampler,
                result=result,
                npool=npool,
            )
        if self._hyperbolic_likelihood is not None:
            total_log_evidence += self._hyperbolic_likelihood.noise_log_evidence(
                priors=priors,
                sampler=sampler,
                result=result,
                npool=npool,
            )
        return float(total_log_evidence)
