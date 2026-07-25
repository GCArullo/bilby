# IGWN Catalog Runs

```
BASE_DIR="$(git rev-parse --show-toplevel)"
UTILS="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils"
REAL="$UTILS/submit_runs_real_data.py"
GWTC_CONFIGS="$UTILS/GWTC_catalog_configs"
GWTC21="$GWTC_CONFIGS/GWTC-2.1"
GWTC3="$GWTC_CONFIGS/GWTC-3"
GWTC4="$GWTC_CONFIGS/GWTC-4"
GWTC5="$GWTC_CONFIGS/GWTC-5"
```

Catalog-derived templates and priors live under `$GWTC_CONFIGS`.

## High-mass catalog runs

`$GWTC_CONFIGS/MASS_CLASSIFICATION.md` records the published LVK median
source-frame total mass and high/low classification for every configured
GWTC-2.1 through GWTC-5 event. High mass means a median strictly above
`50 M_sun`; the equality case is assigned to low mass.

Use one flag to prepare the complete high-mass catalog:

```
python "$REAL" \
  --high-mass-catalog \
  --likelihood hyperbolic \
  --num-frequency-bands 1 \
  --dry-run
```

This applies every other command-line option to each high-mass event. Remove
`--dry-run` to submit them.

## Gaussian likelihood with inferred PSD corrections

Use `gaussian-parametric` to sample one PSD correction in each of `N`
frequency bands. Corrections are shared between detectors by default:

```
python "$REAL" \
  --event GW200129_065458 \
  --likelihood gaussian-parametric \
  --num-frequency-bands 4 \
  --dry-run
```

Add `--detector-dependent-noise` for separate H1/L1 corrections. Add
`--range` to prepare every band count from 1 through `N`. The generated
priors use `log_psd_scale ~ Uniform(-1, 1)`, so each PSD multiplier
`10**log_psd_scale` ranges from `0.1` to `10`. Remove `--dry-run` to submit.
A standard Gaussian companion is included by default; pass
`--no-add-gaussian` to omit it.

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

## GWTC-4 and GWTC-5 Catalog Runs

`$GWTC4` and `$GWTC5` contain the O4a and O4b templates respectively. They
select the released `C00:IMRPhenomXPHM-SpinTaylor` analysis, preserve its
waveform flags, and use the GWTC-2.1/3 nocosmo luminosity-distance prior with
the released event-specific bounds. GW230529_181500 is the single exception:
the release has no SpinTaylor run, so its template uses
`C00:IMRPhenomXPHM:LowSpin`.
GW240925_005809 retains XPHM-SpinTaylor but stores it under `C01` in the
release, so its generated template selects that group explicitly.

For example:

```
python "$REAL" \
  --event GW240413_022019 \
  --likelihood hyperbolic \
  --num-frequency-bands 1 \
  --dry-run
```

Before a real submission, `submit_runs_real_data.py` checks every local strain,
PSD, and calibration file referenced by the selected template. If managed
de-glitched frames are missing, it runs the appropriate catalog downloader for
the selected event and verifies the files before creating any DAG. Dry runs do
not download data.

## PESummary Pages

Exclude the spline-calibration nuisance parameters from every PESummary page,
for Gaussian, Gaussian-parametric, Student-t, and Hyperbolic likelihoods. The
real-data launcher sets this automatically for per-run pages:

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
