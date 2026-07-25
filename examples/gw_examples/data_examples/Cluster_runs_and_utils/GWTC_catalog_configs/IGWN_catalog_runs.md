# IGWN Catalog Runs

```
BASE_DIR="$(git rev-parse --show-toplevel)"
UTILS="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils"
REAL="$UTILS/submit_runs_real_data.py"
GWTC_CONFIGS="$UTILS/GWTC_catalog_configs"
GWTC21="$GWTC_CONFIGS/GWTC-2.1"
GWTC3="$GWTC_CONFIGS/GWTC-3"
```

Catalog-derived templates and priors live under `$GWTC_CONFIGS`.

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
By default, each event is written below
`/home/gregorio.carullo/public_html/<event>/Runs`.
O3a event directories retain the time suffix from the full record name so
events occurring on the same date remain distinct.

## GWTC-3 Catalog Runs

`$GWTC3` contains templates, priors, embedded source configs, PSDs, and
calibration envelopes for all 36 event files in the GWTC-3 O3b PE release.
The preparation and de-glitched-frame download details are in
`$GWTC3/README.md`.

For example:

```
python "$REAL" \
  --event GW200129_065458 \
  --likelihood hyperbolic \
  --num-frequency-bands 1 \
  --dry-run
```

Remove `--dry-run` to submit. Seven BayesWave-subtracted events and the
linearly subtracted GW200129_065458 first require the public frames downloaded
by `python "$GWTC3/download_glitch_data.py"`.

## PESummary Pages

Exclude the spline-calibration nuisance parameters from every PESummary page,
for Gaussian, Student-t, and Hyperbolic likelihoods. The real-data launcher
sets this automatically for per-run pages:

```
ignore_parameters = ["recalib*"]
```

When constructing a standalone or comparison page from existing result files,
always pass the same exclusion explicitly:

```
summarypages \
  ... \
  --ignore_parameters 'recalib*'
```

Keep the wildcard quoted so that the shell does not expand it. This rule also
applies when every result file contains calibration samples.
