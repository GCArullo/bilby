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

### Keep the page label short

PESummary writes the label into its own output names twice: `html/` holds
`<label>_<label>_<parameter>.html` and `samples/` holds
`<label>_<result file name>`. Left to itself bilby_pipe labels the results-page
job with the full merge-result basename, which is 113 to 125 characters for
these runs, so both names run past the 255 byte limit and the job dies with

```
OSError: [Errno 36] File name too long
```

`submit_runs_real_data.py` therefore passes `labels` in
`summarypages-arguments`, naming each per-run page after its run directory.
Hand-written labels need the same discipline: the budget is roughly 110
characters, and comparison pages should use short names such as
`Student_ind_N2` anyway.

### Confirm the page was written

This failure is quiet. The page aborts partway, leaving a populated `web/`
directory that looks plausible, and the DAG records the failure only in
`submit/*.dagman.out`. Check the metafile, which PESummary writes last:

```
<run>/web/samples/posterior_samples.h5
```

Its presence means `home.html`, the per-parameter pages, and the metafile were
all produced. Across a run directory:

```
for d in */; do
  [ -f "${d}web/samples/posterior_samples.h5" ] || echo "incomplete: ${d%/}"
done
```

### Rebuilding a page

Rerun the node from its own submit file rather than resubmitting the DAG. Take
the argument string from the `VARS ..._pesummary_arg_0` line of
`submit/dag_*.submit`, shorten `--labels`, and redirect `log`, `output`, and
`error` to `*_pesummary_fixed.*` so the original logs survive. Move the partial
`web/` aside first, otherwise stale files from the failed attempt are left
alongside the new ones.

### NRSur data off site

NRSur7dq4 waveform generation reads `NRSur7dq4_v1.0.h5`, while `NRSur_fits`
reads `NRSur7dq4Remnant_v1.0.h5`. The managed container carries both under
`/opt/lalsimulation-data`, and generated jobs search the CIT copy in
`/scratch/lalsimulation` first and the container copy second. Container
validation generates an NRSur7dq4 waveform and evaluates the remnant fit, so an
image with Git LFS pointer files cannot be published.

Use the managed container outside CIT. The launchers' `--no-container`
configuration searches the CIT-only `/scratch/lalsimulation` copy instead.

### Comparison pages

Comparison pages are built by hand from the `final_result` files, which is what
makes a run eligible: `final_result/*_merge_result.hdf5` exists only once the
merge node has completed. Write to a staging directory and move it into place
afterwards, so a failed rebuild cannot destroy the live page, and set `baseurl`
to the final location rather than the staging one:

```
summarypages \
  --webdir  "$C/all_completed_new" \
  --baseurl "https://ldas-jobs.ligo.caltech.edu/~$USER/.../all_completed" \
  --labels  Student_ind_N1 Student_ind_N2 \
  --samples "$R/student_..._N1/final_result/..._merge_result.hdf5" \
            "$R/student_..._N2/final_result/..._merge_result.hdf5" \
  --gw --no_conversion --disable_interactive \
  --include_unshared_parameters --multi_process 4 --seed 123456789
```

`--no_conversion` is what keeps the comparison page off the conversion code
path that the per-run pages exercise; do not drop it without checking that the
page still builds.
