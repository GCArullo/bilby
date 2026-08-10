# GW200129_065458 Hannam NRSur7dq4 Reproduction

```
BASE_DIR="$(git rev-parse --show-toplevel)"
UTILS="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils"
REAL="$UTILS/submit_runs_real_data.py"
SPECIAL_CONFIGS="$UTILS/Special_events_configs"
GWTC3="$UTILS/GWTC_catalog_configs/GWTC-3"
```

The `GW200129_065458_Hannam` event profile uses the linearly subtracted
Livingston frame and the GWTC-3 PSD and calibration products. Its bespoke
template and prior are:

```
$SPECIAL_CONFIGS/templates/GW200129_065458_Hannam_NRSur7dq4.ini
$SPECIAL_CONFIGS/priors/GW200129_065458_Hannam_NRSur7dq4.prior
```

The profile sets `NRSur7dq4`, restricts the waveform to all multipoles with
`2 <= ell <= 3`, and applies the Hannam et al. detector-frame prior cuts:
`14.5 <= chirp_mass/M_sun <= 49`, `mass_ratio >= 1/4`, and
`total_mass/M_sun >= 68`.
All likelihoods are written below
`$HOME/public_html/GW200129_065458_Hannam/Runs`.

## Required files

Download the GWTC-3 de-glitched frames before the first run:

```
python "$GWTC3/download_glitch_data.py"
```

The managed container carries the NRSur7dq4 waveform and remnant HDF5 files.
Use it outside CIT. The launcher's `--no-container` configuration searches the
CIT-only `/scratch/lalsimulation` copy instead.

## Gaussian reproduction

Generate the configuration without submitting:

```
python "$REAL" \
  --event GW200129_065458_Hannam \
  --likelihood gaussian \
  --dry-run
```

Remove `--dry-run` to submit.

## Alternative noise likelihoods

The same event profile can be used with `--likelihood student` or
`--likelihood hyperbolic`. These retain the Hannam signal model and prior while
changing only the noise likelihood. For example:

```
python "$REAL" \
  --event GW200129_065458_Hannam \
  --likelihood hyperbolic \
  --num-frequency-bands 1 \
  --dry-run
```
