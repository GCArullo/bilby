#!/usr/bin/env python
"""Suggest which extended likelihood can capture what a Gaussian analysis left.

Given a Gaussian LVK posterior and its data, this scans frequency bands, time
chunks and joint time-frequency tiles, and applies the diagnostic order of
``notes/robust_likelihood_regimes.tex``:

  (i)   does the artefact overlap the signal?  If not it is a constant in the
        likelihood and biases nothing, however loud -- ignore it.
  (ii)  is ``Lambda = n (L* - 1 - ln L*)`` of order a few in some resolvable
        region?  If not, no amplitude model will engage.
  (iii) is the contaminated region informative?  If it carries most of the Fisher
        information for the parameters, only a coherent (subtracting) model helps.

It then emits the ``reweight_posterior.py`` command line for the best candidate.

Where deglitched data are supplied alongside raw, the artefact is measured
directly as their difference; otherwise the residual is used as a proxy, which
conflates artefact with ordinary noise and makes the overlap estimate weaker.
"""

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reweight_posterior import (  # noqa: E402
    build_residual_series, detectability, load_analysis_data, load_posterior,
    region_quadratic_forms, whitened_series,
)


def sparsity(q_region):
    """Is the excess dense across the region, or carried by a few bins?

    A free scale suits a dense excess and a heavy tail a sparse one, per the
    "which model, when" table of the note.  Reported as the fraction of the
    excess power carried by the loudest 5% of bins; under the null that is about
    0.05 times the chi-square-2 tail weight, near 0.15.
    """
    excess = np.clip(q_region - 2.0, 0.0, None)
    if excess.sum() <= 0:
        return 0.0
    ordered = np.sort(excess, axis=-1)[..., ::-1]
    top = max(1, int(round(0.05 * ordered.shape[-1])))
    return float(np.median(ordered[..., :top].sum(axis=-1) / excess.sum(axis=-1)))


def analyse(arguments):
    samples = load_posterior(arguments.posterior)
    data = load_analysis_data(arguments.data)
    clean = load_analysis_data(arguments.deglitched_data) if \
        arguments.deglitched_data else None

    n_total = len(next(iter(samples.values())))
    rng = np.random.default_rng(arguments.seed)
    size = min(arguments.n_samples, n_total)
    indices = np.sort(rng.choice(n_total, size=size, replace=False))

    series = build_residual_series(arguments, data, samples, indices)
    signal = {}
    for name, entry in data.items():
        # signal power = data minus residual, in the whitened time domain
        signal[name] = whitened_series(entry) - series[name].mean(axis=0)

    artefact = None
    if clean is not None:
        artefact = {
            name: whitened_series(data[name])
            - whitened_series(clean[name])
            for name in data
        }

    duration = float(next(iter(data.values()))["duration"])
    sampling = float(next(iter(data.values()))["sampling_frequency"])
    nyquist = sampling / 2.0

    # The analysis segment is Tukey windowed, so the first and last
    # `taper_seconds` are tapered towards zero and carry far less power than the
    # PSD predicts.  That is a deficit, not an artefact, and it would otherwise
    # dominate the scan: on GW191109 the edge chunks reach Lambda ~ 80 with a
    # -60% excess purely from the window.
    taper = arguments.taper_seconds
    time_bands = [None] + [
        [lo, lo + arguments.chunk_seconds]
        for lo in np.arange(taper, duration - taper - arguments.chunk_seconds + 1e-9,
                            arguments.chunk_seconds)
    ]
    frequency_bands = [None] + [
        [lo, lo + arguments.band_width]
        for lo in np.arange(arguments.minimum_frequency,
                            min(nyquist, arguments.maximum_frequency)
                            - arguments.band_width + 1e-9,
                            arguments.band_width)
    ]

    records = []
    for name in data:
        total_signal = float(np.sum(signal[name] ** 2))
        for time_band in time_bands:
            for frequency_band in frequency_bands:
                if time_band is None and frequency_band is None:
                    continue
                q = np.array([
                    region_quadratic_forms(row, data[name], time_band,
                                           frequency_band)
                    for row in series[name]
                ])
                if q.shape[1] < 3:
                    continue
                stats = detectability(q)
                signal_q = region_quadratic_forms(
                    signal[name], data[name], time_band, frequency_band)
                fraction = float(signal_q.sum() / total_signal) if total_signal else 0.0
                record = dict(
                    detector=name, time_band=time_band,
                    frequency_band=frequency_band,
                    mode=("tile" if time_band and frequency_band
                          else "time" if time_band else "frequency"),
                    n_bins=stats["n_bins"], lambda_profile=stats["lambda_profile"],
                    fractional_excess=stats["median_fractional_excess"],
                    implied_artefact_snr=stats["implied_artefact_snr"],
                    signal_information_fraction=fraction,
                    sparsity=sparsity(q),
                )
                if artefact is not None:
                    a = region_quadratic_forms(
                        artefact[name], data[name], time_band, frequency_band)
                    record["artefact_snr_in_region"] = float(np.sqrt(a.sum()))
                records.append(record)
    return records, samples, data


def recommend(records, arguments):
    """Apply the diagnostic order and pick a model."""
    # Only a positive excess is something a robust likelihood can act on.  A
    # deficit means the PSD overestimates the noise there; Lambda is large for
    # both, so it must be gated on the sign.
    engaged = [
        r for r in records
        if r["lambda_profile"] >= arguments.lambda_threshold
        and r["fractional_excess"] > 0.0
    ]
    if not engaged:
        positive = [r for r in records if r["fractional_excess"] > 0.0]
        best = max(positive or records, key=lambda r: r["lambda_profile"]) \
            if records else None
        return {
            "verdict": "NO AMPLITUDE MODEL WILL ENGAGE",
            "reason": (
                f"the strongest region reaches Lambda = "
                f"{best['lambda_profile']:.2f} against a threshold of "
                f"{arguments.lambda_threshold}. Nothing here is loud enough "
                "relative to the bins it is shared over. If the Gaussian result "
                "still looks wrong, the cause is either an artefact that "
                "overlaps the signal too closely to be downweighted -- in which "
                "case only coherent subtraction helps -- or it is not the noise."
            ),
            "best_region": best,
            "command": None,
        }

    # prefer strong detectability at low collateral cost
    def score(record):
        return record["lambda_profile"] * (1.0 - record["signal_information_fraction"])

    best = max(engaged, key=score)
    family = "hyperbolic" if best["sparsity"] > arguments.sparsity_threshold \
        else "psd-scale"
    reason = (
        "the excess is carried by a few bins, so a heavy tail suits it"
        if family == "hyperbolic"
        else "the excess is dense across the region, so a free scale suits it "
             "and a heavy tail would add nothing"
    )

    warnings = []
    if best["signal_information_fraction"] < arguments.irrelevance_threshold:
        warnings.append(
            f"this region carries only "
            f"{100 * best['signal_information_fraction']:.1f}% of the signal "
            "information, so the artefact in it is very nearly a constant in the "
            "likelihood and biases the parameters hardly at all. Downweighting "
            "it will be visible in the evidence and almost invisible in the "
            "posterior. This is step (i) of the diagnostic order: detectability "
            "is not relevance. On GW191109 the loud H1 artefact sits here and "
            "distorts the likelihood shape by exactly zero, while the quiet L1 "
            "one under the merger carries all of the bias."
        )
    if best["signal_information_fraction"] > arguments.information_threshold:
        warnings.append(
            f"this region carries "
            f"{100 * best['signal_information_fraction']:.0f}% of the signal "
            "information, so downweighting it will inflate the parameter "
            "variance substantially. Coherent subtraction is strictly better "
            "here and may be the only thing that helps."
        )
    if best["mode"] == "tile":
        warnings.append(
            "the best region is a tile, meaning the artefact separates from the "
            "signal in time and frequency jointly but not on either axis alone. "
            "Use the TD_FD branch; a plain band or chunk will not capture it."
        )
    if best["n_bins"] < 10 and family == "hyperbolic":
        warnings.append(
            f"only {best['n_bins']} bins in this region: two hyperbolic shape "
            "parameters are not identifiable from so few, and the result will be "
            "prior-dominated. Prefer psd-scale, or widen the region."
        )

    command = [
        "python", "reweight_posterior.py",
        "--posterior", arguments.posterior,
        "--data", arguments.data,
        "--family", family,
        "--detectors", best["detector"],
        "--approximant", arguments.approximant,
        "--frequency-domain-source-model", arguments.frequency_domain_source_model,
        "--reference-frequency", str(arguments.reference_frequency),
        "--reference-frame", arguments.reference_frame,
        "--start-time", str(arguments.start_time),
    ]
    if best["frequency_band"]:
        command += ["--frequency-band",
                    f"{best['frequency_band'][0]:g},{best['frequency_band'][1]:g}"]
    if best["time_band"]:
        command += ["--time-band",
                    f"{best['time_band'][0]:g},{best['time_band'][1]:g}"]
    if arguments.parameters:
        command += ["--parameters", *arguments.parameters]

    return {
        "verdict": f"USE {family.upper()} ON THIS REGION",
        "reason": reason,
        "best_region": best,
        "warnings": warnings,
        "command": " ".join(shlex.quote(part) for part in command),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--posterior", required=True)
    parser.add_argument("--data", required=True,
                        help="the data actually analysed (raw or deglitched)")
    parser.add_argument("--deglitched-data", default=None,
                        help="optional second dataset; the difference from --data "
                             "measures the artefact directly")
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--approximant", default="IMRPhenomXPHM")
    parser.add_argument("--frequency-domain-source-model",
                        default="bilby.gw.source.lal_binary_black_hole")
    parser.add_argument("--reference-frequency", type=float, default=20.0)
    parser.add_argument("--reference-frame", default="sky")
    parser.add_argument("--parameters", nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--chunk-seconds", type=float, default=0.5)
    parser.add_argument("--taper-seconds", type=float, default=0.5,
                        help="seconds excluded at each segment edge, where the "
                             "Tukey window suppresses the data")
    parser.add_argument("--band-width", type=float, default=25.0)
    parser.add_argument("--minimum-frequency", type=float, default=20.0)
    parser.add_argument("--maximum-frequency", type=float, default=448.0)
    parser.add_argument("--lambda-threshold", type=float, default=5.0)
    parser.add_argument("--sparsity-threshold", type=float, default=0.5)
    parser.add_argument("--information-threshold", type=float, default=0.3,
                        help="above this signal-information fraction, warn that "
                             "downweighting inflates the parameter variance")
    parser.add_argument("--irrelevance-threshold", type=float, default=0.02,
                        help="below this signal-information fraction, warn that "
                             "the artefact biases nothing")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--output", default="likelihood_suggestion.json")
    arguments = parser.parse_args()

    records, _, _ = analyse(arguments)
    suggestion = recommend(records, arguments)

    positive = [r for r in records if r["fractional_excess"] > 0.0]
    ordered = sorted(positive or records,
                     key=lambda r: -r["lambda_profile"])[: arguments.top]
    print(f"\n{'det':>4} {'mode':>10} {'region':>26} {'bins':>6} "
          f"{'Lambda':>8} {'excess':>8} {'f_sig':>7} {'sparse':>7}")
    ordered = [r for r in ordered if r["fractional_excess"] > 0.0] or ordered
    for record in ordered:
        region = (
            f"{record['time_band'][0]:g}-{record['time_band'][1]:g}s "
            if record["time_band"] else ""
        ) + (
            f"{record['frequency_band'][0]:g}-{record['frequency_band'][1]:g}Hz"
            if record["frequency_band"] else "all f"
        )
        print(f"{record['detector']:>4} {record['mode']:>10} {region:>26} "
              f"{record['n_bins']:6d} {record['lambda_profile']:8.2f} "
              f"{100 * record['fractional_excess']:7.1f}% "
              f"{record['signal_information_fraction']:7.3f} "
              f"{record['sparsity']:7.3f}")

    print(f"\n=== {suggestion['verdict']} ===")
    print(f"{suggestion['reason']}")
    for warning in suggestion.get("warnings", []):
        print(f"\n  WARNING: {warning}")
    if suggestion["command"]:
        print("\nrun:\n")
        print(f"  {suggestion['command']}\n")

    with open(arguments.output, "w", encoding="utf-8") as stream:
        json.dump({"records": records, "suggestion": suggestion}, stream, indent=2)


if __name__ == "__main__":
    main()
