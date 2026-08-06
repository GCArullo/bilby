# GW150914 Welch-PSD Gaussian and Gaussian-Parametric Runs

```
BASE_DIR="$(git rev-parse --show-toplevel)"
UTILS="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils"
EXTRACT="$UTILS/LVK_posteriors/Scripts/extract_gw150914_c01_products.py"
REAL="$UTILS/submit_runs_real_data.py"
INI="$UTILS/Special_events_configs/templates/GW150914_welch_template.ini"
PRIOR="$UTILS/Special_events_configs/priors/GW150914_igwn_template.prior"
WORKING_DIRECTORY="$UTILS/LVK_posteriors/GW150914"
OUTPUT="$HOME/public_html/GWTC_parametric_noise/Runs/GW150914"
```

Unlike `GW150914_t_student_igwn_template.ini`, this template does not use the
released IGWN `psd-dict` products. `psd-dict=None` and `psd-method=welch`, so
bilby_pipe estimates the PSD on the fly from off-source GWOSC data with
gwpy's Welch method (`psd-length=32`, i.e. a 128 s PSD segment for the 4 s
analysis duration). Calibration still uses the extracted IGWN envelopes, so
`extract_gw150914_c01_products.py` must still be run once per checkout.

`--outdir-label welch` distinguishes these runs from the existing
IGWN-PSD `gaussian_detector_independent_noise_N1` and
`gaussian-parametric_detector_independent_noise_N1` runs under the same
`OUTPUT` directory.

## Gaussian likelihood, Welch PSD, no parametric correction

```
python "$REAL" \
  --event GW150914 \
  --ini-template "$INI" \
  --prior-template "$PRIOR" \
  --working-directory "$WORKING_DIRECTORY" \
  --outdir-base "$OUTPUT" \
  --likelihood gaussian \
  --outdir-label welch
```

## Gaussian-parametric likelihood, Welch PSD, detector-independent, single band

```
python "$REAL" \
  --event GW150914 \
  --ini-template "$INI" \
  --prior-template "$PRIOR" \
  --working-directory "$WORKING_DIRECTORY" \
  --outdir-base "$OUTPUT" \
  --likelihood gaussian-parametric \
  --num-frequency-bands 1 \
  --outdir-label welch
```

The launcher submits by default. Add `--dry-run` to either command to write
the ini and prior files without submitting.
