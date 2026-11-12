# GW150914 Student-t and Gaussian Runs

```
BASE_DIR="$(git rev-parse --show-toplevel)"
UTILS="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils"
EXTRACT="$UTILS/LVK_posteriors/Scripts/extract_gw150914_c01_products.py"
REAL="$UTILS/submit_runs_real_data.py"
INI="$UTILS/Special_events_configs/templates/GW150914_t_student_igwn_template.ini"
PRIOR="$UTILS/Special_events_configs/priors/GW150914_igwn_template.prior"
WORKING_DIRECTORY="$UTILS/LVK_posteriors/GW150914"
OUTPUT="$HOME/public_html/GW150914/Runs"
```

## Prepare the IGWN data products

The extractor only needs to be run once per checkout. It writes the calibration
and PSD products expected by the special-event template into
`LVK_posteriors/GW150914/Data/GW150914_C01_IMRPhenomXPHM`.

```
python "$EXTRACT"
```

The commands below explicitly select the special-event template, prior, and
working directory because GW150914 may also be present in catalog data.

## Student-t runs

Bands 1 through 4, with a single-band Gaussian companion:

```
python "$REAL" \
  --event GW150914 \
  --ini-template "$INI" \
  --prior-template "$PRIOR" \
  --working-directory "$WORKING_DIRECTORY" \
  --outdir-base "$OUTPUT" \
  --likelihood student \
  --range \
  --num-frequency-bands 4
```

For detector-dependent Student-t degrees of freedom, add
`--detector-dependent-nu`.

## Gaussian run

```
python "$REAL" \
  --event GW150914 \
  --ini-template "$INI" \
  --prior-template "$PRIOR" \
  --working-directory "$WORKING_DIRECTORY" \
  --outdir-base "$OUTPUT" \
  --likelihood gaussian
```

The launcher submits by default. Add `--dry-run` to write the ini and prior
files without submitting.
