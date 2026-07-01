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
from ...core.likelihood import Likelihood, _fallback_to_parameters
from ...core.prior import DeltaFunction, PriorDict
from ...core.utils import logger
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


def _time_band_count(time_bands):
    if _is_time_band_cut_list(time_bands):
        return len(time_bands) + 1
    return int(time_bands)


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
        if missing_detectors:
            raise ValueError(
                "Detector-specific likelihood types are missing detectors: "
                + ", ".join(missing_detectors)
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
    # frequencies without making the covariance numerically singular.
    low_patch_value = 10.0 * float(np.max(psd[active_band_mask]))
    high_patch_value = 10.0 * float(np.max(psd[frequencies >= high_frequency]))

    psd[frequencies < low_frequency] = low_patch_value
    psd[frequencies > high_frequency] = high_patch_value
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
    """

    def __init__(
        self,
        interferometers,
        waveform_generator,
        likelihood_method="cholesky-solve-triangular",
        time_bands=1,
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
        self.time_bands = time_bands
        self.prefer_time_domain_waveform = bool(prefer_time_domain_waveform)
        self._number_of_time_bands = _time_band_count(time_bands)
        self._detector_names = [interferometer.name for interferometer in self.interferometers]
        self._detector_likelihood_caches = self._build_detector_likelihood_caches()

    @property
    def meta_data(self):
        meta_data = super().meta_data
        meta_data.update(
            likelihood_class=self.__class__,
            likelihood_method=self.likelihood_method,
            time_bands=self.time_bands,
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
                    time_bands=self.time_bands,
                    sampling_rate=interferometer.sampling_frequency,
                )
            caches[interferometer.name] = dict(
                full=cache,
                time_bands=time_band_cache,
                psd=full_psd,
            )
        return caches

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

    def _resolve_likelihood_parameters(self, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)
        if parameters is self._parameters:
            parameters = parameters.copy()
        else:
            merged_parameters = self._parameters.copy()
            merged_parameters.update(parameters)
            parameters = merged_parameters
        return parameters

    def _resolve_signal_likelihood_parameters(self, parameters=None):
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

    def log_likelihood(self, parameters=None):
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
        return self._noise_log_likelihood_from_parameters(self._parameters.copy())

    def log_likelihood_ratio(self, parameters=None):
        parameters = self._resolve_signal_likelihood_parameters(parameters)
        return float(
            self.log_likelihood(parameters=parameters)
            - self._noise_log_likelihood_from_parameters(parameters)
        )

    def compute_per_detector_log_likelihood(self, parameters=None):
        parameters = self._resolve_signal_likelihood_parameters(parameters)
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
            parameters[f"{interferometer.name}_log_likelihood"] = self._compute_detector_log_likelihood(
                interferometer=interferometer,
                residuals=residuals,
                parameters=parameters,
            )
        return parameters.copy()


class _StudentTTimeDomainNoiseOnlyLikelihood(Likelihood):
    def __init__(self, student_likelihood):
        super().__init__(parameters=student_likelihood._get_default_nu_parameter_dict())
        self.student_likelihood = student_likelihood

    def log_likelihood(self, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)
        return self.student_likelihood._noise_log_likelihood_from_parameters(parameters)

    def noise_log_likelihood(self):
        return 0.0


class StudentTTimeDomainGravitationalWaveTransient(TimeDomainGravitationalWaveTransient):
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
        **kwargs,
    ):
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            likelihood_method=likelihood_method,
            time_bands=time_bands,
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
        self.infer_nu = bool(infer_nu)
        self._fixed_nu = self._coerce_nu_array(nu)
        self.noise_evidence_nlive = _validate_noise_evidence_nlive(noise_evidence_nlive)
        self.dlogz_noise = _resolve_dlogz_noise(dlogz_noise=dlogz_noise, dlogZ_noise=dlogZ_noise)
        self.noise_evidence_method = _resolve_noise_evidence_method(noise_evidence_method)

        if not self._valid_nu_values(self._fixed_nu):
            raise ValueError("All nu values must be positive and finite")

        if self.infer_nu:
            if not self.detector_dependent_noise:
                for key, value in zip(self.nu_parameter_keys, self._fixed_nu):
                    self._parameters.setdefault(key, float(value))
            else:
                for detector_index, detector_name in enumerate(self._detector_names):
                    for band_index in range(self._number_of_time_bands):
                        self._parameters.setdefault(
                            self._detector_nu_parameter_key(detector_name, band_index),
                            float(self._fixed_nu[detector_index, band_index]),
                        )

    @property
    def meta_data(self):
        meta_data = super().meta_data
        meta_data.update(
            likelihood_class=self.__class__,
            nu=np.asarray(self._fixed_nu).tolist(),
            infer_nu=self.infer_nu,
            detector_dependent_noise=self.detector_dependent_noise,
            noise_evidence_method=self.noise_evidence_method,
        )
        return meta_data

    @property
    def noise_parameter_keys(self):
        if not self.infer_nu:
            return []
        return list(self.nu_parameter_keys)

    @property
    def nu_parameter_keys(self):
        if not self.detector_dependent_noise:
            if self._number_of_time_bands == 1:
                return ["nu"]
            return [f"nu_{index}" for index in range(1, self._number_of_time_bands + 1)]

        if self._number_of_time_bands == 1:
            return [f"nu_{detector_name}" for detector_name in self._detector_names]

        return [
            f"nu_{detector_name}_{index}"
            for detector_name in self._detector_names
            for index in range(1, self._number_of_time_bands + 1)
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
        return f"nu_{detector_name}_{band_index + 1}"

    def _get_nu_values(self, parameters):
        if not self.infer_nu:
            return self._fixed_nu

        if "nu" in parameters and not self.detector_dependent_noise:
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

    def _store_nu_values(self, nu_values):
        if not self.detector_dependent_noise:
            for key, value in zip(self.nu_parameter_keys, nu_values):
                self._parameters[key] = float(value)
            return

        for detector_index, detector_name in enumerate(self._detector_names):
            for band_index in range(self._number_of_time_bands):
                self._parameters[
                    self._detector_nu_parameter_key(detector_name, band_index)
                ] = float(nu_values[detector_index, band_index])

    def _band_nu(self, parameters, interferometer_name, band_index):
        nu_values = self._get_nu_values(parameters)
        if not self.detector_dependent_noise:
            return float(nu_values[min(band_index, len(nu_values) - 1)])
        detector_index = self._detector_names.index(interferometer_name)
        return float(nu_values[detector_index, band_index])

    def _get_active_nu_values(self, parameters, update_state=False):
        nu_values = self._get_nu_values(parameters)
        if update_state and self.infer_nu:
            self._store_nu_values(nu_values)
        if not self._valid_nu_values(nu_values):
            return None
        return nu_values

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

    def log_likelihood(self, parameters=None):
        parameters = self._resolve_signal_likelihood_parameters(parameters)
        nu_values = self._get_active_nu_values(parameters, update_state=True)
        if nu_values is None:
            return np.nan_to_num(-np.inf)
        return super().log_likelihood(parameters=parameters)

    def _noise_log_likelihood_from_parameters(self, parameters):
        parameters = self._resolve_likelihood_parameters(parameters)
        nu_values = self._get_active_nu_values(parameters, update_state=False)
        if nu_values is None:
            return np.nan_to_num(-np.inf)
        return super()._noise_log_likelihood_from_parameters(parameters)

    def _get_default_nu_parameter_dict(self):
        parameters = self._parameters.copy() if self._parameters is not None else dict()
        nu_values = self._get_nu_values(parameters)
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
        for key, value in default_nu_parameters.items():
            if key in priors:
                noise_priors[key] = deepcopy(priors[key])
            else:
                noise_priors[key] = DeltaFunction(peak=value, name=key)
        return noise_priors

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
                "Student-t noise-evidence quadrature supports one or two sampled noise parameters"
            )

        base_parameters = self._get_default_nu_parameter_dict()
        reference_points = _noise_evidence_quadrature_points()
        candidate_logls = [self._noise_log_likelihood_from_parameters(base_parameters)]
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
                    lambda unit_value_1: scaled_integrand((unit_value_1, unit_value_2)),
                    0.0,
                    1.0,
                    **quadrature_kwargs,
                )
                return float(inner_integral)

            integral, _ = quad(outer_integrand, 0.0, 1.0, **quadrature_kwargs)
        if not np.isfinite(integral) or integral <= 0:
            raise RuntimeError(
                "Student-t noise-evidence quadrature failed to return a positive finite integral"
            )
        return float(logl_reference + np.log(integral))

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
                self._get_default_nu_parameter_dict()
            )

        if self.noise_evidence_method == "quadrature":
            if len(noise_priors.non_fixed_keys) <= 2:
                return self._noise_log_evidence_by_quadrature(noise_priors)
            logger.info(
                "Student-t noise-evidence quadrature supports at most two "
                "sampled noise parameters; falling back to nested sampling."
            )

        noise_result = self._noise_log_evidence_by_nested_sampling(noise_priors)
        return float(noise_result.log_evidence)


class _HyperbolicTimeDomainNoiseOnlyLikelihood(Likelihood):
    def __init__(self, hyperbolic_likelihood):
        super().__init__()
        self._parameters.update(hyperbolic_likelihood._get_default_shape_parameter_dict())
        self.hyperbolic_likelihood = hyperbolic_likelihood

    def log_likelihood(self, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)
        return self.hyperbolic_likelihood._noise_log_likelihood_from_parameters(parameters)

    def noise_log_likelihood(self):
        return 0.0


class HyperbolicTimeDomainGravitationalWaveTransient(TimeDomainGravitationalWaveTransient):
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
        **kwargs,
    ):
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            likelihood_method=likelihood_method,
            time_bands=time_bands,
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

        if self.infer_alpha:
            if not self.detector_dependent_noise:
                for key, value in zip(self.alpha_parameter_keys, self._fixed_alpha):
                    self._parameters.setdefault(key, float(value))
            else:
                for detector_index, detector_name in enumerate(self._detector_names):
                    for band_index in range(self._number_of_time_bands):
                        self._parameters.setdefault(
                            self._detector_parameter_key("alpha", detector_name, band_index),
                            float(self._fixed_alpha[detector_index, band_index]),
                        )
        if self.infer_delta:
            if not self.detector_dependent_noise:
                for key, value in zip(self.delta_parameter_keys, self._fixed_delta):
                    self._parameters.setdefault(key, float(value))
            else:
                for detector_index, detector_name in enumerate(self._detector_names):
                    for band_index in range(self._number_of_time_bands):
                        self._parameters.setdefault(
                            self._detector_parameter_key("delta", detector_name, band_index),
                            float(self._fixed_delta[detector_index, band_index]),
                        )

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
            return [f"alpha_{index}" for index in range(1, self._number_of_time_bands + 1)]
        if self._number_of_time_bands == 1:
            return [f"alpha_{detector_name}" for detector_name in self._detector_names]
        return [
            f"alpha_{detector_name}_{index}"
            for detector_name in self._detector_names
            for index in range(1, self._number_of_time_bands + 1)
        ]

    @property
    def delta_parameter_keys(self):
        if not self.detector_dependent_noise:
            if self._number_of_time_bands == 1:
                return ["delta"]
            return [f"delta_{index}" for index in range(1, self._number_of_time_bands + 1)]
        if self._number_of_time_bands == 1:
            return [f"delta_{detector_name}" for detector_name in self._detector_names]
        return [
            f"delta_{detector_name}_{index}"
            for detector_name in self._detector_names
            for index in range(1, self._number_of_time_bands + 1)
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
        return f"{parameter_name}_{detector_name}_{band_index + 1}"

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

    def _store_parameter_values(self, parameter_name, parameter_keys, parameter_values):
        if not self.detector_dependent_noise:
            for key, value in zip(parameter_keys, parameter_values):
                self._parameters[key] = float(value)
            return

        for detector_index, detector_name in enumerate(self._detector_names):
            for band_index in range(self._number_of_time_bands):
                self._parameters[
                    self._detector_parameter_key(parameter_name, detector_name, band_index)
                ] = float(parameter_values[detector_index, band_index])

    def _get_active_shape_parameters(self, parameters, update_state=False):
        alpha_values = self._get_parameter_values(
            parameters, "alpha", self.infer_alpha, self._fixed_alpha
        )
        delta_values = self._get_parameter_values(
            parameters, "delta", self.infer_delta, self._fixed_delta
        )

        if update_state:
            if self.infer_alpha:
                self._store_parameter_values("alpha", self.alpha_parameter_keys, alpha_values)
            if self.infer_delta:
                self._store_parameter_values("delta", self.delta_parameter_keys, delta_values)

        if not self._valid_positive_values(alpha_values):
            return None, None
        if not self._valid_positive_values(delta_values):
            return None, None
        return alpha_values, delta_values

    def _band_shape_parameters(self, parameters, interferometer_name, band_index):
        alpha_values, delta_values = self._get_active_shape_parameters(
            parameters, update_state=False
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

    def log_likelihood(self, parameters=None):
        parameters = self._resolve_signal_likelihood_parameters(parameters)
        alpha_values, delta_values = self._get_active_shape_parameters(
            parameters, update_state=True
        )
        if alpha_values is None or delta_values is None:
            return np.nan_to_num(-np.inf)
        return super().log_likelihood(parameters=parameters)

    def _noise_log_likelihood_from_parameters(self, parameters):
        parameters = self._resolve_likelihood_parameters(parameters)
        alpha_values, delta_values = self._get_active_shape_parameters(
            parameters, update_state=False
        )
        if alpha_values is None or delta_values is None:
            return np.nan_to_num(-np.inf)
        return super()._noise_log_likelihood_from_parameters(parameters)

    def _get_default_shape_parameter_dict(self):
        parameters = self._parameters.copy() if self._parameters is not None else dict()
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
                "Hyperbolic noise-evidence quadrature supports one or two sampled noise parameters"
            )

        base_parameters = self._get_default_shape_parameter_dict()
        reference_points = _noise_evidence_quadrature_points()
        candidate_logls = [self._noise_log_likelihood_from_parameters(base_parameters)]
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
                    lambda unit_value_1: scaled_integrand((unit_value_1, unit_value_2)),
                    0.0,
                    1.0,
                    **quadrature_kwargs,
                )
                return float(inner_integral)

            integral, _ = quad(outer_integrand, 0.0, 1.0, **quadrature_kwargs)
        if not np.isfinite(integral) or integral <= 0:
            raise RuntimeError(
                "Hyperbolic noise-evidence quadrature failed to return a positive finite integral"
            )
        return float(logl_reference + np.log(integral))

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


class MixedTimeDomainGravitationalWaveTransient(TimeDomainGravitationalWaveTransient):
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
        **kwargs,
    ):
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            likelihood_method=likelihood_method,
            time_bands=time_bands,
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
            nu=nu,
            infer_nu=infer_nu,
            detector_dependent_noise=detector_dependent_noise,
            likelihood_method=likelihood_method,
            time_bands=time_bands,
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
            alpha=alpha,
            delta=delta,
            infer_alpha=infer_alpha,
            infer_delta=infer_delta,
            detector_dependent_noise=detector_dependent_noise,
            likelihood_method=likelihood_method,
            time_bands=time_bands,
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

        for family_likelihood in [
            self._student_t_likelihood,
            self._hyperbolic_likelihood,
        ]:
            if family_likelihood is None:
                continue
            for key in family_likelihood.noise_parameter_keys:
                self._parameters.setdefault(key, family_likelihood._parameters[key])

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
        return [
            interferometer
            for interferometer in interferometers
            if interferometer.name in detector_names
        ]

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

    def _validate_active_noise_parameters(self, parameters, update_state):
        if self._student_t_likelihood is not None:
            nu_values = self._student_t_likelihood._get_active_nu_values(
                parameters, update_state=update_state
            )
            if nu_values is None:
                return False
        if self._hyperbolic_likelihood is not None:
            alpha_values, delta_values = (
                self._hyperbolic_likelihood._get_active_shape_parameters(
                    parameters, update_state=update_state
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
        if not self._validate_active_noise_parameters(
            parameters, update_state=False
        ):
            return np.nan_to_num(-np.inf)
        return super()._noise_log_likelihood_from_parameters(parameters)

    def log_likelihood(self, parameters=None):
        parameters = self._resolve_signal_likelihood_parameters(parameters)
        if not self._validate_active_noise_parameters(
            parameters, update_state=True
        ):
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
        return self._noise_log_likelihood_from_parameters(self._parameters.copy())

    def compute_per_detector_log_likelihood(self, parameters=None):
        parameters = self._resolve_signal_likelihood_parameters(parameters)
        if not self._validate_active_noise_parameters(
            parameters, update_state=True
        ):
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
            parameters[f"{interferometer.name}_log_likelihood"] = (
                self._compute_detector_log_likelihood(
                    interferometer=interferometer,
                    residuals=residuals,
                    parameters=parameters,
                )
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
