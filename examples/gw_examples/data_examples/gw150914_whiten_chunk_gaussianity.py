#!/usr/bin/env python3
"""
Download GW150914 open data, whiten fixed-length chunks, and test Gaussianity.

The primary check compares the standard deviation of each whitened chunk against
the unit-variance expectation for Gaussian noise, including finite-sample
confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass

import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.stats import chi2


DEFAULT_TRIGGER_TIME = 1126259462.4


@dataclass
class ChunkStat:
    detector: str
    chunk_index: int
    gps_start: float
    gps_end: float
    n_samples: int
    mean: float
    std: float
    expected_std: float
    ci_low: float
    ci_high: float
    z_score: float
    in_expected_range: bool


def expected_sample_std_unit_gaussian(n_samples: int) -> float:
    """Return E[s] for sample standard deviation (ddof=1), for N(0, 1) data."""
    if n_samples < 2:
        return math.nan
    dof = n_samples - 1
    return math.sqrt(2.0 / dof) * math.exp(
        math.lgamma(0.5 * (dof + 1.0)) - math.lgamma(0.5 * dof)
    )


def std_confidence_interval_unit_gaussian(
    n_samples: int, confidence: float
) -> tuple[float, float]:
    """Two-sided CI for sample std under N(0, 1), using chi-square."""
    if n_samples < 2:
        return (math.nan, math.nan)
    alpha = 1.0 - confidence
    dof = n_samples - 1
    chi2_lo = chi2.ppf(alpha / 2.0, dof)
    chi2_hi = chi2.ppf(1.0 - alpha / 2.0, dof)
    return (
        math.sqrt(dof / chi2_hi),
        math.sqrt(dof / chi2_lo),
    )


def std_standard_error_unit_gaussian(n_samples: int) -> float:
    """Asymptotic SE for sample std around sigma=1."""
    if n_samples < 2:
        return math.nan
    return 1.0 / math.sqrt(2.0 * (n_samples - 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download GW150914 strain data, split into chunks, whiten each "
            "chunk, and compare per-chunk std to Gaussian expectation."
        )
    )
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    parser.add_argument("--trigger-time", type=float, default=DEFAULT_TRIGGER_TIME)
    parser.add_argument(
        "--total-duration",
        type=float,
        default=128.0,
        help="Total length (s) of fetched data if start/end are not given.",
    )
    parser.add_argument(
        "--post-trigger-duration",
        type=float,
        default=2.0,
        help="Offset (s) between trigger time and fetched end time.",
    )
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=4.0,
        help="Chunk duration (s) to whiten and test.",
    )
    parser.add_argument(
        "--fftlength",
        type=float,
        default=None,
        help="FFT length (s) for whitening; default uses chunk duration.",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.0, help="Overlap (s) for whitening PSD."
    )
    parser.add_argument(
        "--fduration",
        type=float,
        default=0.5,
        help="Filter duration (s) for whitening.",
    )
    parser.add_argument(
        "--trim-seconds",
        type=float,
        default=0.25,
        help="Trim this many seconds from each edge after whitening.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for expected std interval.",
    )
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--cache", dest="cache", action="store_true")
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.set_defaults(cache=True)
    return parser.parse_args()


def resolve_time_window(args: argparse.Namespace) -> tuple[float, float]:
    if args.start_time is not None and args.end_time is not None:
        if args.end_time <= args.start_time:
            raise ValueError("--end-time must be larger than --start-time.")
        return args.start_time, args.end_time

    if (args.start_time is None) != (args.end_time is None):
        raise ValueError("Provide both --start-time and --end-time, or neither.")

    end_time = args.trigger_time + args.post_trigger_duration
    start_time = end_time - args.total_duration
    return start_time, end_time


def analyze_detector(
    detector: str,
    start_time: float,
    end_time: float,
    chunk_duration: float,
    fftlength: float | None,
    overlap: float,
    fduration: float,
    trim_seconds: float,
    confidence: float,
    cache: bool,
) -> list[ChunkStat]:
    print(f"\nDownloading {detector} data: [{start_time:.3f}, {end_time:.3f}] GPS")
    data = TimeSeries.fetch_open_data(detector, start_time, end_time, cache=cache)

    sample_rate = float(data.sample_rate.value)
    chunk_samples = int(round(chunk_duration * sample_rate))
    trim_samples = int(round(trim_seconds * sample_rate))
    if chunk_samples < 2:
        raise ValueError("Chunk duration is too short for the data sample rate.")

    n_full_chunks = len(data) // chunk_samples
    if n_full_chunks == 0:
        raise ValueError(
            f"No complete chunks for {detector}. Increase --total-duration "
            "or reduce --chunk-duration."
        )

    dropped = len(data) - n_full_chunks * chunk_samples
    if dropped > 0:
        print(f"{detector}: dropping trailing {dropped} samples (partial chunk).")

    if fftlength is None:
        fftlength = chunk_duration

    stats: list[ChunkStat] = []
    for chunk_index in range(n_full_chunks):
        idx0 = chunk_index * chunk_samples
        idx1 = idx0 + chunk_samples
        chunk = data[idx0:idx1]
        chunk_start = float(chunk.times.value[0])
        chunk_end = float(chunk.times.value[-1] + 1.0 / sample_rate)

        whitened = chunk.whiten(
            fftlength=fftlength,
            overlap=overlap,
            fduration=fduration,
            method="median",
        )

        if trim_samples > 0:
            if 2 * trim_samples >= len(whitened):
                raise ValueError(
                    "Trimming removes the entire chunk. Lower --trim-seconds "
                    "or increase --chunk-duration."
                )
            whitened = whitened[trim_samples:-trim_samples]

        values = np.asarray(whitened.value)
        n = len(values)
        std = float(np.std(values, ddof=1))
        mean = float(np.mean(values))
        expected_std = expected_sample_std_unit_gaussian(n)
        ci_low, ci_high = std_confidence_interval_unit_gaussian(n, confidence)
        se = std_standard_error_unit_gaussian(n)
        z_score = (std - 1.0) / se if se > 0 else math.nan

        stats.append(
            ChunkStat(
                detector=detector,
                chunk_index=chunk_index,
                gps_start=chunk_start,
                gps_end=chunk_end,
                n_samples=n,
                mean=mean,
                std=std,
                expected_std=expected_std,
                ci_low=ci_low,
                ci_high=ci_high,
                z_score=z_score,
                in_expected_range=(ci_low <= std <= ci_high),
            )
        )
    return stats


def print_report(all_stats: list[ChunkStat], confidence: float) -> None:
    if not all_stats:
        return

    print(
        "\nPer-chunk whitened standard deviations\n"
        f"(expected sigma=1, finite-sample {confidence:.1%} interval):"
    )
    header = (
        f"{'det':>3} {'chunk':>5} {'gps_start':>14} {'gps_end':>14} "
        f"{'N':>7} {'mean':>10} {'std':>10} {'exp_std':>10} "
        f"{'ci_low':>10} {'ci_high':>10} {'z':>8} {'ok':>4}"
    )
    print(header)
    print("-" * len(header))
    for stat in all_stats:
        print(
            f"{stat.detector:>3} {stat.chunk_index:>5d} "
            f"{stat.gps_start:>14.3f} {stat.gps_end:>14.3f} "
            f"{stat.n_samples:>7d} {stat.mean:>10.4f} {stat.std:>10.4f} "
            f"{stat.expected_std:>10.4f} {stat.ci_low:>10.4f} {stat.ci_high:>10.4f} "
            f"{stat.z_score:>8.2f} {str(stat.in_expected_range):>4}"
        )

    print("\nSummary by detector:")
    detectors = sorted({s.detector for s in all_stats})
    for det in detectors:
        det_stats = [s for s in all_stats if s.detector == det]
        stds = np.array([s.std for s in det_stats])
        in_range = sum(s.in_expected_range for s in det_stats)
        print(
            f"{det}: chunks={len(det_stats)}, std(mean)={stds.mean():.4f}, "
            f"std(std)={stds.std(ddof=1) if len(stds) > 1 else 0.0:.4f}, "
            f"in-range={in_range}/{len(det_stats)}"
        )


def maybe_write_csv(output_csv: str | None, all_stats: list[ChunkStat]) -> None:
    if output_csv is None:
        return
    fieldnames = [
        "detector",
        "chunk_index",
        "gps_start",
        "gps_end",
        "n_samples",
        "mean",
        "std",
        "expected_std",
        "ci_low",
        "ci_high",
        "z_score",
        "in_expected_range",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for stat in all_stats:
            writer.writerow(
                {
                    "detector": stat.detector,
                    "chunk_index": stat.chunk_index,
                    "gps_start": stat.gps_start,
                    "gps_end": stat.gps_end,
                    "n_samples": stat.n_samples,
                    "mean": stat.mean,
                    "std": stat.std,
                    "expected_std": stat.expected_std,
                    "ci_low": stat.ci_low,
                    "ci_high": stat.ci_high,
                    "z_score": stat.z_score,
                    "in_expected_range": stat.in_expected_range,
                }
            )
    print(f"\nSaved CSV results to: {output_csv}")


def main() -> None:
    args = parse_args()

    if args.chunk_duration <= 0:
        raise ValueError("--chunk-duration must be positive.")
    if args.total_duration <= 0:
        raise ValueError("--total-duration must be positive.")
    if not (0 < args.confidence < 1):
        raise ValueError("--confidence must be in (0, 1).")
    if args.trim_seconds < 0:
        raise ValueError("--trim-seconds must be non-negative.")

    start_time, end_time = resolve_time_window(args)
    print(
        f"Using GW150914 trigger time {args.trigger_time:.3f} GPS.\n"
        f"Fetching data from {start_time:.3f} to {end_time:.3f} "
        f"({end_time - start_time:.1f} s)."
    )

    all_stats: list[ChunkStat] = []
    for detector in args.detectors:
        all_stats.extend(
            analyze_detector(
                detector=detector,
                start_time=start_time,
                end_time=end_time,
                chunk_duration=args.chunk_duration,
                fftlength=args.fftlength,
                overlap=args.overlap,
                fduration=args.fduration,
                trim_seconds=args.trim_seconds,
                confidence=args.confidence,
                cache=args.cache,
            )
        )

    print_report(all_stats, args.confidence)
    maybe_write_csv(args.output_csv, all_stats)


if __name__ == "__main__":
    main()
