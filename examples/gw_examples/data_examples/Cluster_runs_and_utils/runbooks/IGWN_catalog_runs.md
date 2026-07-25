# IGWN Catalog Runs

```
BASE_DIR="$(git rev-parse --show-toplevel)"
EXTRACT="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/LVK_posteriors/Scripts/extract_gw150914_c01_products.py"
REAL="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_real_data.py"
GWTC_CONFIGS="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/GWTC_catalog_configs"
GWTC21="$GWTC_CONFIGS/GWTC-2.1"
GWTC3="$GWTC_CONFIGS/GWTC-3"
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
By default, each event is written below
`/home/gregorio.carullo/public_html/GWTC_parametric_noise/Runs/<event>`.
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

### GW200129 Hannam NRSur7dq4 reproduction

The dedicated `GW200129_065458_Hannam` event profile uses the linearly
subtracted Livingston frame and the GWTC-3 PSD and calibration products. It
sets `NRSur7dq4`, restricts the waveform to the multipoles with
`2 <= ell <= 3`, and applies the Hannam et al. detector-frame prior cuts:
`14.5 <= chirp_mass/M_sun <= 49`, `mass_ratio >= 1/4`, and
`total_mass/M_sun >= 68`.

The template transfers the surrogate data file from
`/home/pe.o4/GWTC4-fogg/NRSur7dq4_v1.0.h5`. Confirm that this shared path is
readable from the submit host before submission.

Generate the Gaussian reproduction configuration without submitting it:

```
python "$REAL" \
  --event GW200129_065458_Hannam \
  --likelihood gaussian \
  --dry-run
```

Remove `--dry-run` to submit. The same event profile can be used with
`--likelihood student` or `--likelihood hyperbolic` to retain the Hannam
signal model and prior while changing the noise likelihood.

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

## GW150914 Hyperbolic Runs

Both commands below use the two-detector H1/L1 setup embedded in the event
defaults and generate the single-band Gaussian companion run as well. Their
run directories are created below
`/home/gregorio.carullo/public_html/GWTC_parametric_noise/Runs/GW150914`.

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
