import numpy as np


class ParametricNoiseFrequencyBands:
    """Frequency-band machinery shared by the parametric-noise likelihoods.

    Student-t, Hyperbolic and Gaussian-parametric all split the active analysis
    range into contiguous bands carrying their own noise parameters, and differ
    only in what those parameters are called. Edge resolution, validation, band
    naming and band masks live here so that a change to the banding applies to
    all three at once.

    A concrete class must set ``_detector_names`` and ``detector_dependent_noise``,
    must define ``_create_frequency_band_edges`` returning the default
    equal-width edges, and must call ``_setup_frequency_bands`` once its band
    count is set. Classes whose band count is not called ``num_frequency_bands``
    override ``_band_count_attribute``.
    """

    _band_count_attribute = "num_frequency_bands"

    @property
    def _num_bands(self):
        return getattr(self, self._band_count_attribute)

    @_num_bands.setter
    def _num_bands(self, value):
        setattr(self, self._band_count_attribute, value)

    def _setup_frequency_bands(self, frequency_band_edges):
        """Resolve the band edges and the per-band name suffixes.

        ``None`` keeps the equal-width bands built from the band count. Anything
        else replaces them, and the band count is then taken from the edges.
        """
        self._detector_frequency_band_edges = self._resolve_frequency_band_edges(
            frequency_band_edges
        )
        self._band_suffix_cache = (
            None
            if frequency_band_edges is None
            else {
                name: [
                    self._band_frequency_suffix(lower, upper)
                    for lower, upper in zip(edges[:-1], edges[1:])
                ]
                for name, edges in self._detector_frequency_band_edges.items()
            }
        )

    @staticmethod
    def _validate_frequency_band_edges(edges, name):
        edges = np.asarray(edges, dtype=float).ravel()
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError(f"{name} must contain at least two frequency edges")
        if not np.all(np.isfinite(edges)):
            raise ValueError(f"{name} must contain only finite frequencies")
        if np.any(np.diff(edges) <= 0):
            raise ValueError(f"{name} must be strictly increasing")
        if np.any(edges < 0):
            raise ValueError(f"{name} must contain only non-negative frequencies")
        return edges

    def _resolve_frequency_band_edges(self, frequency_band_edges):
        """Return band edges per detector and set the band count.

        ``None`` keeps the equal-width behaviour built from the band count. A
        sequence gives every detector the same explicit edges. A dict gives each
        detector its own edges, which requires detector-dependent noise
        parameters and the same band count everywhere.
        """
        if frequency_band_edges is None:
            edges = self._create_frequency_band_edges()
            return {name: edges for name in self._detector_names}

        if isinstance(frequency_band_edges, dict):
            if not self.detector_dependent_noise:
                raise ValueError(
                    "per-detector frequency_band_edges requires "
                    "detector_dependent_noise=True"
                )
            missing = set(self._detector_names) - set(frequency_band_edges)
            if missing:
                raise ValueError(
                    "frequency_band_edges must contain an entry for every detector; "
                    f"missing {sorted(missing)}"
                )
            unknown = set(frequency_band_edges) - set(self._detector_names)
            if unknown:
                raise ValueError(
                    f"frequency_band_edges contains unknown detectors {sorted(unknown)}"
                )
            edges_by_detector = {
                name: self._validate_frequency_band_edges(
                    frequency_band_edges[name], f"frequency_band_edges['{name}']"
                )
                for name in self._detector_names
            }
            band_counts = {
                name: len(edges) - 1 for name, edges in edges_by_detector.items()
            }
            if len(set(band_counts.values())) > 1:
                raise ValueError(
                    "every detector must define the same number of frequency bands; "
                    f"got {band_counts}"
                )
            self._num_bands = next(iter(band_counts.values()))
            return edges_by_detector

        edges = self._validate_frequency_band_edges(
            frequency_band_edges, "frequency_band_edges"
        )
        self._num_bands = len(edges) - 1
        return {name: edges for name in self._detector_names}

    def _band_edges(self, detector_name=None):
        """Band edges for one detector, or the common edges when detector-independent.

        Callers that combine detectors, that is the shared-parameter and joint
        paths, must not pass a detector name; those paths reject per-detector
        edges at construction time because they require detector-dependent noise.
        """
        if detector_name is not None:
            return self._detector_frequency_band_edges[detector_name]
        return self._frequency_band_edges

    @property
    def _frequency_band_edges(self):
        """The common band edges, which every detector shares unless a dict was given."""
        return self._detector_frequency_band_edges[self._detector_names[0]]

    def _band_suffixes(self, detector_name=None):
        """Per-band name suffixes: the band edges in Hz, or 1..N for equal-width bands.

        Explicit edges are named after the frequencies they cover, because a bare
        index says nothing once the widths are arbitrary and, with per-detector
        edges, band `i` covers a different range in each interferometer. Equal-width
        bands keep the index, so that names stay stable for existing runs and are
        never derived from a frequency grid the caller writing the priors would
        have to reconstruct.
        """
        if self._band_suffix_cache is None:
            return [str(index) for index in range(1, self._num_bands + 1)]
        return self._band_suffix_cache[
            detector_name if detector_name is not None else self._detector_names[0]
        ]

    @staticmethod
    def _band_frequency_suffix(lower, upper):
        return f"{lower:g}_{upper:g}".replace(".", "p")

    def _get_frequency_band_masks(self, frequencies, detector_name=None):
        frequencies = np.asarray(frequencies, dtype=float)
        edges = self._band_edges(detector_name)
        band_masks = []
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            if index == self._num_bands - 1:
                band_mask = (frequencies >= lower) & (frequencies <= upper)
            else:
                band_mask = (frequencies >= lower) & (frequencies < upper)
            band_masks.append(band_mask)
        return band_masks

    def _frequency_band_edges_meta_data(self):
        return {
            name: edges.tolist()
            for name, edges in self._detector_frequency_band_edges.items()
        }
