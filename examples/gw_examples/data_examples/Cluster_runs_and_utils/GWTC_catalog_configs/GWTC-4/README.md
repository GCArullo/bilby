# GWTC-4 run templates

This directory contains runnable Student-t/Hyperbolic/Gaussian templates for
the 86 events in the [GWTC-4 parameter-estimation release][pe-release].

`prepare_gwtc4.py` reads the event HDF5 files with HTTP range requests and
extracts the embedded configuration, PSDs, calibration envelopes, and analytic
priors. It selects `C00:IMRPhenomXPHM-SpinTaylor`, retains the released
SpinTaylor waveform flags, and replaces the released luminosity-distance prior
with the GWTC-2.1/3 nocosmo prior, `PowerLaw(alpha=2)`, using the same event
bounds. GW230529_181500 has no SpinTaylor run, so it uses the closest available
low-spin XPHM analysis, `C00:IMRPhenomXPHM:LowSpin`.

Regenerate the directory with:

```bash
python -m pip install h5py numpy fsspec aiohttp
python prepare_gwtc4.py
```

From `Cluster_runs_and_utils`, generate an event without submitting:

```bash
python submit_runs_real_data.py \
  --event GW231028_153006 \
  --likelihood hyperbolic \
  --shared-alpha \
  --num-frequency-bands 1 \
  --dry-run
```

`--shared-alpha` samples one `alpha` across every band and detector and leaves
`delta` free per band, the parameterisation of arXiv:2602.22074, giving `N+1`
noise parameters instead of `2N`. The measured hyperbolic fits sit at
`alpha*delta` of order 70, deep in the regime where the per-band freedom acts
as a variance scale and the tail shape is common, so the shared parameter is
better identified. Pass `--per-band-alpha` to reproduce runs completed before
this change.

[pe-release]: https://zenodo.org/records/17602505
