# IGWN Catalog Runs

```
BASE_DIR="$(git rev-parse --show-toplevel)"
EXTRACT="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/LVK_posteriors/Scripts/extract_gw150914_c01_products.py"
REAL="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_real_data.py"
GWTC21="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/GWTC-2.1"
```

## GWTC-2.1 Catalog Runs

`$GWTC21` contains templates, priors, embedded source configs, PSDs, and
calibration envelopes for all 54 events in the GWTC-2.1 PE release. The
event-by-event data-quality audit is `$GWTC21/data_quality.html`; preparation
and glitch-frame download details are in `$GWTC21/README.md`.

For example:

```
python "$REAL" \
  --event GW190727_060333 \
  --likelihood hyperbolic \
  --num-frequency-bands 1 \
  --dry-run
```

Remove `--dry-run` to submit. Seven glitch-subtracted events first require the
public L1 frames downloaded by `python "$GWTC21/download_glitch_data.py"`.

The extractor only needs to be run once per checkout. It writes the calibration
and PSD products expected by the GW150914 IGWN template into
`LVK_posteriors/GW150914/Data/GW150914_C01_IMRPhenomXPHM`.

```
python "$EXTRACT"
```

The real-data launcher submits by default. Add `--dry-run` if you only want to
write the ini/prior files. Pass `--disable-calibration` to clear the calibration
model, uncertainty inputs, marginalization, and lookup table, and to omit the
calibration envelope from PESummary arguments in every generated run.

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

To submit only the detector-independent, single-band Hyperbolic run and its
Gaussian companion without calibration uncertainties:

```
python "$REAL" \
  --event GW150914 \
  --likelihood hyperbolic \
  --num-frequency-bands 1 \
  --disable-calibration \
  --outdir-label no_calibration
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
