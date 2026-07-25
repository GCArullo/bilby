#!/usr/bin/env python3
"""Add the omitted Gaussian density normalization to old bilby result files."""

import argparse
import ast
import pickle
import re
from pathlib import Path

import h5py
import numpy as np


ADJUSTMENT_ATTR = "gaussian_noise_log_likelihood_normalisation"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Run directories or result HDF5 files")
    parser.add_argument("--apply", action="store_true", help="Modify files in place")
    parser.add_argument("--force", action="store_true", help="Reapply even if already marked")
    return parser.parse_args()


def result_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.glob("final_result/*merge_result.hdf5"))


def run_dir_from_result(path):
    if path.parent.name in {"final_result", "result"}:
        return path.parent.parent
    return path.parent


def is_gaussian_run(run_dir):
    config_paths = sorted(run_dir.glob("*_config_complete.ini"))
    if not config_paths:
        return "gaussian" in run_dir.name and "student" not in run_dir.name.rsplit("_", 1)[-1]
    for line in config_paths[0].read_text(errors="replace").splitlines():
        if line.startswith("likelihood-type"):
            return "StudentTGravitationalWaveTransient" not in line
    return False


def read_config(run_dir):
    config_paths = sorted(run_dir.glob("*_config_complete.ini"))
    if not config_paths:
        return None
    config = {}
    for line in config_paths[0].read_text(errors="replace").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config


def parse_ini_dict(value):
    return {
        key: item.strip().strip("'\"")
        for key, item in re.findall(r"([A-Z]\d)\s*:\s*([^,}]+)", value)
    }


def parse_detectors(value):
    return [str(item) for item in ast.literal_eval(value)]


def parse_minimum_frequencies(value):
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = parse_ini_dict(value)
        return {key: float(item) for key, item in parsed.items()}


def find_data_dump(run_dir):
    candidates = sorted((run_dir / "data").glob("*generation_data_dump.pickle"))
    if not candidates:
        raise FileNotFoundError(f"No generation data dump found under {run_dir / 'data'}")
    return candidates[0]


def load_interferometers(data_dump):
    with data_dump.open("rb") as file:
        data = pickle.load(file)
    if isinstance(data, dict):
        return data["interferometers"]
    return data.interferometers


def gaussian_normalisation(interferometers):
    total = 0.0
    for interferometer in interferometers:
        mask = interferometer.frequency_mask
        scale2 = (
            interferometer.power_spectral_density_array[mask]
            * interferometer.duration
            / 4.0
        )
        total -= float(np.sum(np.log(2 * np.pi * scale2)))
    return total


def gaussian_normalisation_from_config(config):
    duration = float(config["duration"])
    sampling_frequency = float(config["sampling-frequency"])
    maximum_frequency = float(config["maximum-frequency"])
    minimum_frequencies = parse_minimum_frequencies(config["minimum-frequency"])
    psd_paths = parse_ini_dict(config["psd-dict"])
    frequencies = np.fft.rfftfreq(
        int(round(duration * sampling_frequency)),
        1.0 / sampling_frequency,
    )

    total = 0.0
    for detector in parse_detectors(config["detectors"]):
        psd_path = Path(psd_paths[detector])
        if not psd_path.exists():
            raise FileNotFoundError(psd_path)
        psd_frequencies, psd_values = np.loadtxt(psd_path, unpack=True)
        psd = np.interp(frequencies, psd_frequencies, psd_values)
        mask = (
            (frequencies >= float(minimum_frequencies[detector]))
            & (frequencies <= maximum_frequency)
        )
        scale2 = psd[mask] * duration / 4.0
        total -= float(np.sum(np.log(2 * np.pi * scale2)))
    return total


def gaussian_normalisation_for_run(run_dir):
    config = read_config(run_dir)
    if config is not None:
        try:
            return gaussian_normalisation_from_config(config)
        except FileNotFoundError:
            pass
    return gaussian_normalisation(load_interferometers(find_data_dump(run_dir)))


def read_scalar(file, key):
    value = file[key][()]
    if hasattr(value, "item"):
        value = value.item()
    return value


def write_scalar(file, key, value):
    if key in file:
        file[key][...] = value


def adjust_result(path, apply=False, force=False):
    run_dir = run_dir_from_result(path)
    if not is_gaussian_run(run_dir):
        return f"skip {path}: not a Gaussian recovery"

    norm = gaussian_normalisation_for_run(run_dir)
    mode = "r+" if apply else "r"
    with h5py.File(path, mode) as file:
        previous = file.attrs.get(ADJUSTMENT_ATTR)
        if previous is not None and not force:
            return f"skip {path}: already adjusted by {previous:.6f}"

        log_noise = read_scalar(file, "log_noise_evidence")
        log_evidence = read_scalar(file, "log_evidence")
        use_ratio = bool(read_scalar(file, "use_ratio")) if "use_ratio" in file else False

        new_noise = log_noise + norm
        new_evidence = log_evidence + norm

        if apply:
            write_scalar(file, "log_noise_evidence", new_noise)
            write_scalar(file, "log_evidence", new_evidence)
            write_scalar(file, "log_bayes_factor", new_evidence - new_noise)
            if not use_ratio and "posterior/log_likelihood" in file:
                file["posterior/log_likelihood"][...] += norm
            file.attrs[ADJUSTMENT_ATTR] = norm

    action = "updated" if apply else "would update"
    return (
        f"{action} {path}: "
        f"log_noise_evidence {log_noise:.6f} -> {new_noise:.6f}, "
        f"log_evidence {log_evidence:.6f} -> {new_evidence:.6f}"
    )


def main():
    args = parse_args()
    for input_path in args.paths:
        for path in result_files(input_path):
            print(adjust_result(path, apply=args.apply, force=args.force))


if __name__ == "__main__":
    main()
