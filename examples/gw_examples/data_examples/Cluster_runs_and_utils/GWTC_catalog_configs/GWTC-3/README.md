# GWTC-3 run templates

This directory contains runnable Student-t/Hyperbolic/Gaussian templates for
the 36 events in the [GWTC-3 parameter-estimation release][pe-release]: the 35
O3b candidates with `p_astro > 0.5`, plus GW200105_162426.

- `templates/`: one `submit_runs_real_data.py` ini template per event.
- `priors/`: the corresponding prior templates.
- `source_configs/`: every non-empty C01 configuration embedded in the release.
- `data/`: PSDs and calibration envelopes extracted for the selected run.
- `manifest.json`: machine-readable provenance and event settings.

The HDF5 files are not copied here. `prepare_gwtc3.py` reads the v2
`mixed_nocosmo.h5` release files using HTTP range requests and retrieves only
configs, PSDs, calibration envelopes, and analytic priors. These unreweighted
files match the luminosity-distance priors used by the generated runs. The
script requires `h5py`, `numpy`, `fsspec`, and `aiohttp`:

```bash
python -m pip install h5py numpy fsspec aiohttp
python prepare_gwtc3.py
```

The script selects `C01:IMRPhenomXPHM`. GW191219_163120 and GW200115_042309
split that analysis into high- and low-spin variants; their templates select
`C01:IMRPhenomXPHM:LowSpin`, following the low-spin special-case policy used
for GW190425 in the GWTC-2.1 workflow.

## Running an event

From `Cluster_runs_and_utils`, generate files without submitting:

```bash
python submit_runs_real_data.py \
  --event GW200129_065458 \
  --likelihood hyperbolic \
  --num-frequency-bands 1 \
  --dry-run
```

Omit `--dry-run` to submit. Default run directories are created under
`/home/gregorio.carullo/public_html/GWTC_parametric_noise/Runs/<event>`.

## De-glitched events

Seven events use BayesWave-subtracted strain and GW200129_065458 uses linearly
subtracted L1 strain. Download the associated public frames from the
[GWTC-3 glitch-model release][glitch-release] before running any of their
templates:

```bash
python GWTC_catalog_configs/GWTC-3/download_glitch_data.py
```

This downloads and verifies ten files totaling 12.84 GB. Use one or more
`--event GW200115_042309` arguments to download only selected frames.
For normal non-dry-run launches, `submit_runs_real_data.py` performs this check
and event-filtered download automatically before creating any DAG.

[pe-release]: https://zenodo.org/records/8177023
[glitch-release]: https://zenodo.org/records/5546680
