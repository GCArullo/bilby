# GW150914 Hyperbolic and Gaussian-Parametric Runs

```
BASE_DIR="$(git rev-parse --show-toplevel)"
UTILS="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils"
EXTRACT="$UTILS/LVK_posteriors/Scripts/extract_gw150914_c01_products.py"
REAL="$UTILS/submit_runs_real_data.py"
INI="$UTILS/Special_events_configs/templates/GW150914_t_student_igwn_template.ini"
PRIOR="$UTILS/Special_events_configs/priors/GW150914_igwn_template.prior"
WORKING_DIRECTORY="$UTILS/LVK_posteriors/GW150914"
OUTPUT="$HOME/public_html/GWTC_parametric_noise/Runs/GW150914"
```

## Prepare the IGWN data products

The extractor only needs to be run once per checkout. It writes the calibration
and PSD products expected by the special-event template into
`LVK_posteriors/GW150914/Data/GW150914_C01_IMRPhenomXPHM`.

```
python "$EXTRACT"
```

The commands below explicitly select the special-event template, prior, and
working directory because GW150914 is also present in the GWTC-2.1 catalog.
Both use the H1/L1 setup and generate a single-band Gaussian companion run.
Run directories are created below
`$HOME/public_html/GW150914/Runs`.

Hyperbolic runs are factorized across detectors by default. Add `--joint` to
use one joint network Hyperbolic density instead; joint mode is incompatible
with `--detector-dependent-noise`.

## Detector-independent Hyperbolic runs

Bands 1 through 4:

```
python "$REAL" \
  --event GW150914 \
  --ini-template "$INI" \
  --prior-template "$PRIOR" \
  --working-directory "$WORKING_DIRECTORY" \
  --outdir-base "$OUTPUT" \
  --likelihood hyperbolic \
  --range \
  --num-frequency-bands 4
```

Single-band run without calibration uncertainties:

```
python "$REAL" \
  --event GW150914 \
  --ini-template "$INI" \
  --prior-template "$PRIOR" \
  --working-directory "$WORKING_DIRECTORY" \
  --outdir-base "$OUTPUT" \
  --likelihood hyperbolic \
  --num-frequency-bands 1 \
  --disable-calibration \
  --outdir-label no_calibration
```

## Detector-dependent Hyperbolic runs

Bands 1 through 4:

```
python "$REAL" \
  --event GW150914 \
  --ini-template "$INI" \
  --prior-template "$PRIOR" \
  --working-directory "$WORKING_DIRECTORY" \
  --outdir-base "$OUTPUT" \
  --likelihood hyperbolic \
  --range \
  --num-frequency-bands 4 \
  --detector-dependent-noise
```

## Joint detector-independent Hyperbolic run

```
python "$REAL" \
  --event GW150914 \
  --ini-template "$INI" \
  --prior-template "$PRIOR" \
  --working-directory "$WORKING_DIRECTORY" \
  --outdir-base "$OUTPUT" \
  --likelihood hyperbolic \
  --num-frequency-bands 1 \
  --joint
```

## Gaussian likelihood with inferred PSD corrections

The Gaussian-parametric likelihood multiplies the PSD in each frequency band
by `10**log_psd_scale`. The launcher assigns each sampled `log_psd_scale`
parameter a `Uniform(-1, 1)` prior, corresponding to a PSD scale between
`0.1` and `10`.

Shared H1/L1 corrections for bands 1 through 4:

```
python "$REAL" \
  --event GW150914 \
  --ini-template "$INI" \
  --prior-template "$PRIOR" \
  --working-directory "$WORKING_DIRECTORY" \
  --outdir-base "$OUTPUT" \
  --likelihood gaussian-parametric \
  --range \
  --num-frequency-bands 4
```

Independent corrections for each detector and band:

```
python "$REAL" \
  --event GW150914 \
  --ini-template "$INI" \
  --prior-template "$PRIOR" \
  --working-directory "$WORKING_DIRECTORY" \
  --outdir-base "$OUTPUT" \
  --likelihood gaussian-parametric \
  --range \
  --num-frequency-bands 4 \
  --detector-dependent-noise
```

The launcher submits by default. Add `--dry-run` to any command to write the
ini and prior files without submitting.
