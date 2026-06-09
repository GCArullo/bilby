"""Runtime bilby_pipe patch for staged frequency-domain data."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


FD_DATA_FORMAT = "bilby_frequency_domain_hdf5"
_ORIGINAL_SET_INTERFEROMETERS_FROM_DATA = None


def _single_source(source, detector: str) -> Path:
    if isinstance(source, (list, tuple)):
        if len(source) != 1:
            raise ValueError(
                f"FD data for {detector} must resolve to one file, got {source}"
            )
        source = source[0]
    path = Path(source)
    if path.is_file():
        return path
    local_path = Path(path.name)
    if local_path.is_file():
        return local_path
    return path


def _read_frequency_domain_strain(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with h5py.File(path, "r") as h5_file:
        frequencies = np.asarray(h5_file["frequency_array"][()], dtype=float)
        strain = np.asarray(h5_file["frequency_domain_strain"][()], dtype=complex)
        attrs = dict(h5_file.attrs)
    if frequencies.shape != strain.shape:
        raise ValueError(
            f"FD data file {path} has mismatched frequency and strain shapes"
        )
    return frequencies, strain, attrs


def _check_close(name: str, actual: float, expected: float, *, atol: float) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0, atol=atol):
        raise ValueError(f"FD data {name}={actual} does not match ini value {expected}")


def _set_interferometers_from_frequency_domain_hdf5(self) -> None:
    import bilby
    from bilby_pipe.utils import BilbyPipeError, logger

    if self.injection:
        raise BilbyPipeError(
            "FD data files already include the injected signal; set injection=False"
        )
    if self.data_dict is None:
        raise BilbyPipeError("FD data loading requires data-dict")

    ifo_list = []
    for detector in self.detectors:
        if detector not in self.data_dict:
            raise BilbyPipeError(f"Detector {detector} not found in data-dict")

        ifo = bilby.gw.detector.get_empty_interferometer(detector)
        if self.psd_dict is not None and self.psd_dict.get(detector, None) is not None:
            self._set_psd_from_file(ifo)

        path = _single_source(self.data_dict[detector], detector)
        logger.info(f"Loading {detector} FD strain data from {path}")
        frequencies, strain, attrs = _read_frequency_domain_strain(path)

        for key in ("duration", "sampling_frequency", "start_time"):
            if key not in attrs:
                raise BilbyPipeError(f"FD data file {path} is missing attribute {key}")

        _check_close("duration", attrs["duration"], self.duration, atol=1e-12)
        _check_close(
            "sampling_frequency",
            attrs["sampling_frequency"],
            self.sampling_frequency,
            atol=1e-12,
        )
        _check_close("start_time", attrs["start_time"], self.start_time, atol=1e-6)

        ifo.set_strain_data_from_frequency_domain_strain(
            frequency_domain_strain=strain,
            frequency_array=frequencies,
            start_time=float(attrs["start_time"]),
        )
        ifo_list.append(ifo)

    self.interferometers = bilby.gw.detector.InterferometerList(ifo_list)


def patch() -> None:
    global _ORIGINAL_SET_INTERFEROMETERS_FROM_DATA

    from bilby_pipe.data_generation import DataGenerationInput

    if getattr(DataGenerationInput, "_bilby_fd_data_patch", False):
        return

    _ORIGINAL_SET_INTERFEROMETERS_FROM_DATA = (
        DataGenerationInput._set_interferometers_from_data
    )

    def patched_set_interferometers_from_data(self):
        if getattr(self, "data_format", None) == FD_DATA_FORMAT:
            return _set_interferometers_from_frequency_domain_hdf5(self)
        return _ORIGINAL_SET_INTERFEROMETERS_FROM_DATA(self)

    DataGenerationInput._set_interferometers_from_data = (
        patched_set_interferometers_from_data
    )
    DataGenerationInput._bilby_fd_data_patch = True
