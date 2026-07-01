# IGWN Catalog Runs

```
BASE_DIR="$(git rev-parse --show-toplevel)"
EXTRACT="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/LVK_posteriors/Scripts/extract_gw150914_c01_products.py"
REAL="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_real_data.py"
```

The extractor only needs to be run once per checkout. It writes the calibration
and PSD products expected by the GW150914 IGWN template into
`LVK_posteriors/GW150914/Data/GW150914_C01_IMRPhenomXPHM`.

```
python "$EXTRACT"
```

The real-data launcher submits by default. Add `--dry-run` if you only want to
write the ini/prior files.

## GW150914 Hyperbolic Runs

Both commands below use the two-detector H1/L1 setup embedded in the event
defaults and generate the single-band Gaussian companion run as well.

Det-independent Hyperbolic, bands 1..4:

```
python "$REAL" \
  --event GW150914 \
  --likelihood hyperbolic \
  --range \
  --num-frequency-bands 4
```

Det-dependent Hyperbolic, bands 1..4:

```
python "$REAL" \
  --event GW150914 \
  --likelihood hyperbolic \
  --range \
  --num-frequency-bands 4 \
  --detector-dependent-noise
```
