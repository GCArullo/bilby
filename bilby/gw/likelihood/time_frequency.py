"""Joint time-frequency tiled parametric-noise likelihood.

Frequency bands integrate over the whole segment and time chunks integrate over
the whole band, so neither isolates a feature localised in both.  For GW191109
the scattering arch sits at 20-30 Hz beneath a merger whose power is at
35-120 Hz *within the same 0.2 s chunk*: a frequency band covering the arch also
covers two seconds of inspiral, and a time chunk covering the arch also covers
the merger.  Tiles in both axes separate them.

Construction
------------

The residual is whitened once, globally, with the segment PSD,

    w(f) = r(f) / sqrt(S_n(f) T / 4),

and inverse-transformed.  The result is white, so chunking it in time and taking
each chunk's DFT gives coefficients of uniform expected variance and, by
Parseval,

    sum_k weight_k |X_k|^2 / L  =  sum_t w(t)^2

exactly for any chunk boundaries.  With every tile at its Gaussian limit the
likelihood therefore reproduces the Whittle likelihood to machine precision, and
the tile parameters carry only *departures* from the segment PSD.

The alternative -- chunk the raw strain and estimate a short-time spectrum per
chunk -- was rejected: a 0.2 s chunk has 5 Hz resolution against a PSD spanning
five orders of magnitude, so leakage dominates and the Gaussian limit does not
reproduce the standard likelihood.

Each chunk's DFT bin ``k`` carries ``q_k = weight_k |X_k|^2 / L`` with mean 2
under the null, matching the frequency-domain convention of ``studentt.py``,
``hyperbolic.py`` and ``whittle.py``, so their density code is reused unchanged.
Bins 0 and Nyquist carry one degree of freedom rather than two and lie outside
any analysis band; they are held at the Gaussian limit.

Chunk boundaries must not bisect the merger.  A boundary placed on it destroys
20-30% of a deviation's log-likelihood evidence, whereas a chunk that *contains*
it costs well under a nat.
"""

import numpy as np

from ...core.utils.log import logger
from .base import GravitationalWaveTransient
from .hyperbolic import HyperbolicGravitationalWaveTransient
from .studentt import StudentTGravitationalWaveTransient

__all__ = ["TimeFrequencyTiledGravitationalWaveTransient"]

_LOG_TWO_PI = np.log(2.0 * np.pi)


def _suffix(value):
    return f"{value:g}".replace(".", "p").replace("-", "m")


class TimeFrequencyTiledGravitationalWaveTransient(GravitationalWaveTransient):
    """Parametric noise on a joint time-frequency tiling.

    Parameters
    ==========
    time_band_boundaries: list, optional
        Cut times in seconds from the segment start.  ``None`` gives one chunk.
    frequency_band_edges: list or dict, optional
        Band edges in Hz, shared or keyed by detector.  ``None`` gives one band
        spanning the analysis range.
    noise_model: str
        ``gaussian``, ``student`` or ``hyperbolic``.
    """

    def __init__(
        self,
        interferometers,
        waveform_generator,
        time_band_boundaries=None,
        frequency_band_edges=None,
        noise_model="hyperbolic",
        alpha=10.0,
        delta=1.0,
        nu=8.0,
        infer_alpha=False,
        infer_delta=False,
        infer_nu=False,
        detector_dependent_noise=False,
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
        if noise_model not in ("gaussian", "student", "hyperbolic"):
            raise ValueError(f"Unknown noise_model '{noise_model}'")
        self.noise_model = noise_model
        self.detector_dependent_noise = bool(detector_dependent_noise)
        self.infer_alpha = bool(infer_alpha)
        self.infer_delta = bool(infer_delta)
        self.infer_nu = bool(infer_nu)
        self._fixed = {
            "alpha": float(alpha), "delta": float(delta), "nu": float(nu),
        }

        self._duration = float(self.waveform_generator.duration)
        self._sampling_frequency = float(self.waveform_generator.sampling_frequency)
        self._n_time = int(round(self._duration * self._sampling_frequency))
        self._time_edges = self._resolve_time_edges(time_band_boundaries)
        # Keys are named from the boundaries as requested, not from the sample
        # indices they snap to, so a user can predict a tile's parameter name.
        self._time_labels = (
            [0.0, self._duration] if time_band_boundaries is None
            else [0.0, *[float(b) for b in np.atleast_1d(time_band_boundaries)],
                  self._duration]
        )
        self._frequency_edges = self._resolve_frequency_edges(frequency_band_edges)
        self._scale2 = {
            ifo.name: np.asarray(ifo.power_spectral_density_array, dtype=float)
            * self._duration / 4.0
            for ifo in self.interferometers
        }
        self._chunk_cache = self._build_chunk_cache()
        logger.info(
            f"Time-frequency tiling: {len(self._time_edges) - 1} time chunks x "
            f"{len(next(iter(self._frequency_edges.values()))) - 1} frequency "
            f"bands per detector"
        )

    # ------------------------------------------------------------------ tiling
    def _resolve_time_edges(self, boundaries):
        if boundaries is None:
            return [0, self._n_time]
        cuts = [float(b) for b in np.atleast_1d(boundaries)]
        if any(b <= 0.0 or b >= self._duration for b in cuts):
            raise ValueError("time_band_boundaries must lie inside the segment")
        if any(b >= c for b, c in zip(cuts[:-1], cuts[1:])):
            raise ValueError("time_band_boundaries must be strictly increasing")
        indices = (
            [0]
            + [int(round(b * self._sampling_frequency)) for b in cuts]
            + [self._n_time]
        )
        if any(hi - lo < 4 for lo, hi in zip(indices[:-1], indices[1:])):
            raise ValueError("every time chunk needs at least four samples")
        return indices

    def _resolve_frequency_edges(self, edges):
        names = [ifo.name for ifo in self.interferometers]
        if edges is None:
            return {
                ifo.name: [float(ifo.minimum_frequency), float(ifo.maximum_frequency)]
                for ifo in self.interferometers
            }
        if isinstance(edges, dict):
            if not self.detector_dependent_noise:
                raise ValueError(
                    "Per-detector frequency_band_edges requires "
                    "detector_dependent_noise=True"
                )
            missing = set(names) - set(edges)
            if missing:
                raise ValueError(f"frequency_band_edges is missing {sorted(missing)}")
            if len({len(np.atleast_1d(v)) for v in edges.values()}) != 1:
                raise ValueError("every detector needs the same number of bands")
            return {name: [float(v) for v in np.atleast_1d(edges[name])]
                    for name in names}
        shared = [float(v) for v in np.atleast_1d(edges)]
        if len(shared) < 2 or any(b >= c for b, c in zip(shared[:-1], shared[1:])):
            raise ValueError("frequency_band_edges must be increasing, length >= 2")
        return {name: list(shared) for name in names}

    def _build_chunk_cache(self):
        """Per detector, per chunk: sample slice, bin frequencies and dimensions."""
        cache = {}
        for ifo in self.interferometers:
            entries = []
            for lo, hi in zip(self._time_edges[:-1], self._time_edges[1:]):
                length = hi - lo
                frequencies = np.fft.rfftfreq(length, 1.0 / self._sampling_frequency)
                dimensions = np.full(len(frequencies), 2, dtype=int)
                dimensions[0] = 1
                if length % 2 == 0:
                    dimensions[-1] = 1
                entries.append(
                    dict(start=lo, end=hi, frequencies=frequencies,
                         dimensions=dimensions)
                )
            cache[ifo.name] = entries
        return cache

    def tile_keys(self, parameter_name):
        """``(key, detector, time index, frequency index)`` for every tile."""
        keys = []
        detectors = (
            [ifo.name for ifo in self.interferometers]
            if self.detector_dependent_noise
            else [None]
        )
        reference = next(iter(self._frequency_edges.values()))
        for detector in detectors:
            edges = self._frequency_edges[detector] if detector else reference
            for t_index in range(len(self._time_edges) - 1):
                for f_index, (f_lo, f_hi) in enumerate(zip(edges[:-1], edges[1:])):
                    label = (
                        f"{_suffix(self._time_labels[t_index])}_"
                        f"{_suffix(self._time_labels[t_index + 1])}_"
                        f"{_suffix(f_lo)}_{_suffix(f_hi)}"
                    )
                    prefix = (
                        f"{parameter_name}_{detector}_" if detector
                        else f"{parameter_name}_"
                    )
                    keys.append((f"{prefix}{label}", detector, t_index, f_index))
        return keys

    @property
    def noise_parameter_keys(self):
        keys = []
        if self.noise_model == "hyperbolic":
            if self.infer_alpha:
                keys += [key for key, *_ in self.tile_keys("alpha")]
            if self.infer_delta:
                keys += [key for key, *_ in self.tile_keys("delta")]
        elif self.noise_model == "student" and self.infer_nu:
            keys += [key for key, *_ in self.tile_keys("nu")]
        return keys

    # ------------------------------------------------------------- likelihood
    def _whitened_series(self, interferometer, residual):
        mask = interferometer.frequency_mask
        scale2 = self._scale2[interferometer.name]
        whitened = np.zeros(len(residual), dtype=complex)
        whitened[mask] = residual[mask] / np.sqrt(scale2[mask])
        series = np.fft.irfft(whitened, n=self._n_time)
        target = float(np.sum(np.abs(whitened) ** 2))
        current = float(np.sum(series ** 2))
        if current > 0.0:
            series = series * np.sqrt(target / current)
        return series

    @staticmethod
    def _chunk_quadratic_forms(chunk):
        coefficients = np.fft.rfft(chunk)
        weights = np.full(len(coefficients), 2.0)
        weights[0] = 1.0
        if len(chunk) % 2 == 0:
            weights[-1] = 1.0
        return weights * np.abs(coefficients) ** 2 / len(chunk)

    def _tile_parameter(self, parameters, name, detector, t_index, f_index):
        inferring = {"alpha": self.infer_alpha, "delta": self.infer_delta,
                     "nu": self.infer_nu}[name]
        if not inferring:
            return self._fixed[name]
        if name in parameters:          # bare key shared across every tile
            return float(parameters[name])
        prefix = f"{name}_{detector}_" if detector else f"{name}_"
        edges = (
            self._frequency_edges[detector] if detector
            else next(iter(self._frequency_edges.values()))
        )
        f_lo, f_hi = edges[f_index], edges[f_index + 1]
        key = (
            f"{prefix}{_suffix(self._time_labels[t_index])}_"
            f"{_suffix(self._time_labels[t_index + 1])}_"
            f"{_suffix(f_lo)}_{_suffix(f_hi)}"
        )
        return float(parameters.get(key, self._fixed[name]))

    def _log_likelihood_from_residuals(self, residuals, parameters):
        total = 0.0
        for interferometer in self.interferometers:
            name = interferometer.name
            series = self._whitened_series(interferometer, residuals[name])
            detector_key = name if self.detector_dependent_noise else None
            edges = self._frequency_edges[name]
            for t_index, entry in enumerate(self._chunk_cache[name]):
                q = self._chunk_quadratic_forms(
                    series[entry["start"]:entry["end"]]
                )
                frequencies = entry["frequencies"]
                dimensions = entry["dimensions"]
                assigned = np.zeros(len(q), dtype=bool)
                for f_index, (f_lo, f_hi) in enumerate(zip(edges[:-1], edges[1:])):
                    selection = (
                        (frequencies >= f_lo)
                        & (frequencies <= f_hi if f_hi == edges[-1]
                           else frequencies < f_hi)
                        & (dimensions == 2)
                    )
                    if not np.any(selection):
                        continue
                    assigned |= selection
                    total += self._tile_log_terms(
                        q[selection], dimensions[selection], parameters,
                        detector_key, t_index, f_index,
                    )
                # every unassigned bin (out of band, DC, Nyquist) stays Gaussian
                rest = ~assigned
                if np.any(rest):
                    total += float(np.sum(-0.5 * q[rest]
                                          - 0.5 * dimensions[rest] * _LOG_TWO_PI))
            # constant Jacobian of the global whitening, independent of the tiles
            mask = interferometer.frequency_mask
            total -= float(np.sum(np.log(self._scale2[name][mask])))
        return total

    def _tile_log_terms(self, q, dimensions, parameters, detector, t_index,
                        f_index):
        if self.noise_model == "gaussian":
            return float(np.sum(-0.5 * q - 0.5 * dimensions * _LOG_TWO_PI))
        if self.noise_model == "student":
            nu = self._tile_parameter(parameters, "nu", detector, t_index, f_index)
            terms = StudentTGravitationalWaveTransient._compute_joint_bin_log_terms(
                q, dimensions, nu
            )
        else:
            alpha = self._tile_parameter(
                parameters, "alpha", detector, t_index, f_index)
            delta = self._tile_parameter(
                parameters, "delta", detector, t_index, f_index)
            terms = HyperbolicGravitationalWaveTransient._compute_bin_log_terms(
                q, dimensions, alpha, delta
            )
        if terms is None:
            return -np.inf
        return float(np.sum(terms))

    def _residuals(self, parameters=None):
        residuals = {}
        if parameters is None:
            for interferometer in self.interferometers:
                residuals[interferometer.name] = (
                    interferometer.frequency_domain_strain
                )
            return residuals
        polarisations = self.waveform_generator.frequency_domain_strain(parameters)
        for interferometer in self.interferometers:
            response = interferometer.get_detector_response(polarisations, parameters)
            residuals[interferometer.name] = (
                interferometer.frequency_domain_strain - response
            )
        return residuals

    def log_likelihood(self, parameters):
        parameters = parameters.copy()
        parameters.update(self.get_sky_frame_parameters(parameters))
        return self._log_likelihood_from_residuals(
            self._residuals(parameters), parameters
        )

    def noise_log_likelihood(self):
        return self._log_likelihood_from_residuals(self._residuals(None), {})

    def log_likelihood_ratio(self, parameters):
        return self.log_likelihood(parameters) - self.noise_log_likelihood()
