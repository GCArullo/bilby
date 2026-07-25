# GWTC-5 run templates

This directory contains runnable Student-t/Hyperbolic/Gaussian templates for
the 104 events in the two-part [GWTC-5 parameter-estimation release][part-1].

`prepare_gwtc5.py` reads both Zenodo records with HTTP range requests and
extracts the embedded configuration, PSDs, calibration envelopes, and analytic
priors. It selects `C00:IMRPhenomXPHM-SpinTaylor`, retains the released
SpinTaylor waveform flags, and replaces the released luminosity-distance prior
with the GWTC-2.1/3 nocosmo prior, `PowerLaw(alpha=2)`, using the same event
bounds.

GW240925_005809 stores the same XPHM-SpinTaylor analysis under `C01` rather
than `C00`; the preparation script selects that released group explicitly.

Regenerate the directory with:

```bash
python -m pip install h5py numpy fsspec aiohttp
python prepare_gwtc5.py
```

From `Cluster_runs_and_utils`, generate an event without submitting:

```bash
python submit_runs_real_data.py \
  --event GW240413_022019 \
  --likelihood hyperbolic \
  --num-frequency-bands 1 \
  --dry-run
```

[part-1]: https://zenodo.org/records/20348005
[part-2]: https://zenodo.org/records/20348006
