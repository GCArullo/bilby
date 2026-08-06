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
changed input. The unsubtracted excess power at 36 Hz in Livingston is what
these runs are meant to expose, and Udall et al. (PRD 111, 024046) show that
the Livingston band 30-40 Hz is what drives the negative-`chi_eff` preference.

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
python "$REAL" "${COMMON[@]}" --likelihood hyperbolic \
  --range --num-frequency-bands 4 --no-add-gaussian
python "$REAL" "${COMMON[@]}" --likelihood hyperbolic \
  --range --num-frequency-bands 4 --no-add-gaussian --detector-dependent-noise
```

`--no-add-gaussian` suppresses the single-band Gaussian companion that the
parametric likelihoods generate by default, since the Gaussian run above
already covers it. The seven commands produce 25 runs. Add `--dry-run` to
write the ini and prior files without submitting.
