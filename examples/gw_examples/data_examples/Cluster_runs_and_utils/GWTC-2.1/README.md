# GWTC-2.1 run templates

This directory contains runnable Student-t/Hyperbolic/Gaussian templates for
all 54 events in the [GWTC-2.1 parameter-estimation release][pe-release],
including GW150914.

- `templates/`: one `submit_runs_real_data.py` ini template per event.
- `priors/`: the corresponding prior templates.
- `source_configs/`: every non-empty configuration embedded in the release.
- `data/`: PSDs and calibration envelopes extracted for the selected
  `C01:IMRPhenomXPHM` run.
- `manifest.json`: machine-readable provenance and event settings.
- `data_quality.html`: event-by-event data-quality audit.

The HDF5 files are not copied here. `prepare_gwtc21.py` reads the
`mixed_nocosmo.h5` release files using HTTP range requests and retrieves only
configs, PSDs, calibration envelopes, and analytic priors. These unreweighted
files match the luminosity-distance priors used by the generated runs. The
script requires `h5py`, `numpy`, `fsspec`, and `aiohttp`:

```bash
python -m pip install h5py numpy fsspec aiohttp
python prepare_gwtc21.py
```

The script selects `C01:IMRPhenomXPHM` for each BBH and the low-spin
`C01:IMRPhenomPv2_NRTidal:LowSpin` result for GW190425. Nine selected runs
have no embedded analytic prior: their prior files are explicit translations
of bounds in the embedded LALInference config. GW190425 is the sole larger
gap: its release groups contain neither configs, PSDs, calibration envelopes,
nor analytic priors. Its template therefore estimates the PSD from strain,
disables calibration marginalization, and uses the low-spin BNS bounds
documented in its prior file.

## Running an event

From `Cluster_runs_and_utils`, generate files without submitting:

```bash
python submit_runs_real_data.py \
  --event GW190727_060333 \
  --likelihood hyperbolic \
  --num-frequency-bands 1 \
  --dry-run
```

Omit `--dry-run` to submit. The GWTC-1 events use their standard short names,
for example `--event GW150914`; the O3a events use the full record name.

## Glitch-subtracted events

Seven templates select the public L1
`DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01_T1700406_v4` channel. Download the
associated frames before running any of them:

```bash
python GWTC-2.1/download_glitch_data.py
```

This downloads and verifies seven files totaling 10.68 GB. Use one or more
`--event GW190413_134308` arguments to download only selected frames.

See [data_quality.html](data_quality.html) for which events were
glitch-subtracted, whether the released configs already select the subtracted
channel, and which low-frequency cutoffs are attributed to data quality.

[pe-release]: https://zenodo.org/records/6513631
