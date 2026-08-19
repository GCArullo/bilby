# GW191109_010717 without glitch subtraction

GWTC-3 is the only O3b analysis that required glitch mitigation in both LIGO
detectors: slow-scattering arches overlapped the inspiral track, and a
BayesWave glitch model was subtracted from Hanford and Livingston before
parameter estimation. `GWTC_catalog_configs/GWTC-3` reproduces that choice.
This profile repeats the same runs on the production frames the de-glitched
frames were built from, so that the glitch subtraction is the only difference.

| | frame type | channel |
| --- | --- | --- |
| GWTC-3 (de-glitched) | `H1_HOFT_CLEAN_SUB60HZ_C01_T1700406_v4` | `DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01_T1700406_v4` |
| this profile | `H1_HOFT_CLEAN_SUB60HZ_C01` | `DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01` |

Both frame sets cover the same GPS intervals (H1 `1257296641-3327`, L1
`1257295872-4096`) and differ only by the subtracted glitch model; around the
trigger the difference has an RMS of `1.1e-22` (H1) and `4.1e-23` (L1)
against a strain RMS of `1.6e-19` and `4.3e-19`. Released PSDs, calibration
envelopes, `minimum-frequency=20 Hz`, `maximum-frequency=448 Hz`, duration,
sampling rate, priors, and sampler settings are unchanged.

The PSDs were themselves estimated by BayesWave on the de-glitched data. They
are kept here deliberately: reusing them isolates the strain as the single
changed input, and the two conditioned data dumps do carry byte-identical
PSD arrays.

## What the glitch subtraction actually removed

Matched-filter SNR of the difference between the two conditioned data sets,
using the released PSDs over 20-448 Hz and the 4 s analysis segment:

| band | H1 | L1 |
| --- | --- | --- |
| 20-25 Hz | 0.15 | 2.56 |
| 25-30 Hz | 5.60 | 3.50 |
| 30-35 Hz | 8.36 | 0.00 |
| 35-40 Hz | 6.24 | 0.00 |
| 40-150 Hz | 0.05 | 0.01 |
| 150-448 Hz | 0.00 | 4.51 |
| total | 11.84 | 6.26 |

Three things follow, and they set what these runs can and cannot say.

- The Hanford subtraction dominates, and it sits at 25-40 Hz, on top of the
  inspiral track.
- In Livingston the subtraction is confined to the ~24 Hz arch below 30 Hz.
  The 30-40 Hz Livingston band is untouched: the 36 Hz excess power was
  deliberately left in the GWTC-3 data because it could not be separated from
  the signal, so it is present in both data sets. Udall et al.
  (PRD 111, 024046) show that this band is what drives the negative-`chi_eff`
  preference, so these runs do *not* probe that mechanism; restricting the
  Livingston minimum frequency does.
- The Livingston model also removed a broad feature centred at ~197 Hz, at
  roughly a quarter of the noise ASD there. This is a second difference
  between the two data sets that has nothing to do with the scattering arch,
  and it is worth keeping in mind when comparing posteriors.

Because these frames are only published to gwdatafind as local files, the
template sets `data-find-urltype=file`; the frames are resolved at DAG
creation time and no download step is needed.

```
BASE_DIR="$(git rev-parse --show-toplevel)"
UTILS="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils"
REAL="$UTILS/submit_runs_real_data.py"
INI="$UTILS/Special_events_configs/templates/GW191109_010717_no_glitch_subtraction_template.ini"
COMMON=(--event GW191109_010717
        --ini-template "$INI"
        --outdir-label no_glitch_subtraction
        --label-prefix GW191109_010717_IMRPhenomXPHM_no_glitch_subtraction
        --file-prefix GW191109_010717_IGWN_C01_IMRPhenomXPHM_no_glitch_subtraction)
```

The prior, working directory, and output base come from the GWTC-3 catalog
defaults for `GW191109_010717`, so runs land under
`$HOME/public_html/GWTC_parametric_noise/Runs/GW191109_010717` in directories
suffixed `_no_glitch_subtraction`. `--label-prefix` and `--file-prefix` keep
the generated ini, prior, and result labels distinct from the de-glitched
runs of the same event.

## Standard Gaussian

```
python "$REAL" "${COMMON[@]}" --likelihood gaussian
```

## Gaussian-parametric, N = 1 to 4

```
python "$REAL" "${COMMON[@]}" --likelihood gaussian-parametric \
  --range --num-frequency-bands 4 --no-add-gaussian
python "$REAL" "${COMMON[@]}" --likelihood gaussian-parametric \
  --range --num-frequency-bands 4 --no-add-gaussian --detector-dependent-noise
```

## Student-t, N = 1 to 4

```
python "$REAL" "${COMMON[@]}" --likelihood student \
  --range --num-frequency-bands 4 --no-add-gaussian
python "$REAL" "${COMMON[@]}" --likelihood student \
  --range --num-frequency-bands 4 --no-add-gaussian --detector-dependent-noise
```

## Hyperbolic, N = 1 to 4

```
python "$REAL" "${COMMON[@]}" --likelihood hyperbolic --shared-alpha \
  --range --num-frequency-bands 4 --no-add-gaussian
python "$REAL" "${COMMON[@]}" --likelihood hyperbolic --shared-alpha \
  --range --num-frequency-bands 4 --no-add-gaussian --detector-dependent-noise
```

`--no-add-gaussian` suppresses the single-band Gaussian companion that the
parametric likelihoods generate by default, since the Gaussian run above
already covers it. The seven commands produce 25 runs. Add `--dry-run` to
write the ini and prior files without submitting.

`--shared-alpha` samples one `alpha` across every band and detector and leaves
`delta` free per band, the parameterisation of arXiv:2602.22074, giving `N+1`
noise parameters instead of `2N`. The measured hyperbolic fits sit at
`alpha*delta` of order 70, deep in the regime where the per-band freedom acts
as a variance scale and the tail shape is common, so the shared parameter is
better identified. Pass `--per-band-alpha` to reproduce runs completed before
this change.

## Hyperbolic with band edges matched to the glitch

Equal-width bands cannot isolate the glitch: over 20-448 Hz even `N=4` leaves a
107 Hz first band against a 15 Hz glitch, so the tail parameters have to
describe contaminated and clean bins together. `--frequency-band-edges` places
the edges where the table above says the power actually is. Both runs use
detector-dependent parameters, so they differ only in whether the edges are
shared.

Shared edges, five bands, 11 noise parameters:

```
python "$REAL" "${COMMON[@]}" --likelihood hyperbolic --no-add-gaussian \
  --shared-alpha --detector-dependent-noise \
  --frequency-band-edges 20,30,40,180,235,448 \
  --outdir-label no_glitch_subtraction_bands_common
```

Per-detector edges, four bands each, 9 noise parameters. Hanford isolates its
25-40 Hz glitch; Livingston isolates the 20-30 Hz arch and the 180-235 Hz
feature, and its remaining edges only split clean data:

```
python "$REAL" "${COMMON[@]}" --likelihood hyperbolic --no-add-gaussian \
  --shared-alpha --detector-dependent-noise \
  --frequency-band-edges H1:20,25,40,235,448 L1:20,30,180,235,448 \
  --outdir-label no_glitch_subtraction_bands_per_detector
```

Band `i` means a different frequency range in each detector in the second run,
which is the point: the parameters are per detector as well as per band. Both
`--outdir-label` values replace the one in `COMMON`, so pass them after it.
