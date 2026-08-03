# GW231123 Sine-Gaussian Runs

Injected SG values for the injection commands are read from
`Special_events_configs/runbooks/injected_sine_gaussian_values.json`.

The three supported SG modes are:

- `coherent`: all SGs share the CBC sky position and polarization angle.
- `coherent-independent`: all SGs are coherent across detectors and share a
  separately sampled `ra`, `dec`, and `psi`.
- `incoherent`: SG parameters are detector-local and are not constrained by a
  network antenna response.

In `coherent-independent` mode, each SG retains its own `hrss`, `Q`,
`frequency`, `time_offset`, and `phase_offset`. The time offset remains relative
to the CBC geocentric time; the detector delays use the independently sampled
SG sky position.

```
BASE_DIR="$(git rev-parse --show-toplevel)"
REAL="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_real_data.py"
INJ="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_injection.py"
MONITOR="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/monitor_runs.py"
SG_JSON="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/Special_events_configs/runbooks/injected_sine_gaussian_values.json"
INJECTION_POSTERIOR="/home/gregorio.carullo/src/bilby_greg/examples/gw_examples/data_examples/Cluster_runs_and_utils/LVK_posteriors/GW231123/posterior_samples.h5"
BASE="$HOME/public_html/GW231123/Runs/injections_gw231123_sine_gaussians"
MAXMCMC=5000
```

Condor jobs use the Bilby container by default. Before the first submission,
run `make publish` on CIT, or `make publish CIT=false` elsewhere, in
`Cluster_runs_and_utils/container_creation`; the launcher selects the current
Git branch from `container_images.json`. The OSDF URL remains readable from
worldwide IGWN execution sites even though its namespace contains `/cit/`.
Use
`--container-image URL` to override it or `--no-container` to use the existing
node environment.

Generated configs enable file transfer and the worldwide IGWN pool with
`transfer-files=True`, `osg=True`, and `desired-sites=None`. Do not pass
`--require-epnfs` for worldwide execution; that option deliberately restricts
jobs to CIT nodes exposing EPNFS.

Immediately before calling `bilby_pipe`, the launchers verify that every local
frame/data file, PSD, calibration envelope, and additional transfer path in the
rendered config exists. Remote URLs are not treated as local paths.

Both launchers submit by default. Add `--dry-run` if you only want to write
the ini/prior files. The templates use `maxmcmc=5000`; the commands below
set this explicitly with `--maxmcmc "$MAXMCMC"`. Run and web outputs are written
below `$HOME/public_html/GW231123/Runs` by default, with each summary in
`<run-name>/web`.

The generated recovery configs use `nlive=2000` for the baseline CBC run,
`nlive=2500` for one recovered SG, and `nlive=3000` for two or more recovered
SGs in total. This includes `coherent-independent` runs and incoherent
`H1=1 L1=1`.

Monitor all active Bilby run roots discovered in your Condor queue:

```
python "$MONITOR"
```

Pass a run root to inspect only that workflow, for example:

```
python "$MONITOR" "$HOME/public_html/GW231123/t_Student/Runs"
```

## Real Runs

GW231123 NRsur baseline:

```
python "$REAL" --event GW231123 --likelihood gaussian --maxmcmc "$MAXMCMC"
```

GW231123 NRsur + 1 SG (coherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 1 --sine-gaussian-mode coherent --maxmcmc "$MAXMCMC"
```

GW231123 NRsur + 2 SG (coherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 2 --sine-gaussian-mode coherent --maxmcmc "$MAXMCMC"
```

GW231123 NRsur + 3 SG (coherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 3 --sine-gaussian-mode coherent --maxmcmc "$MAXMCMC"
```

GW231123 NRsur + 1 SG (coherent, independently localized):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 1 --sine-gaussian-mode coherent-independent --maxmcmc "$MAXMCMC"
```

GW231123 NRsur + 2 SG (coherent, independently localized):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 2 --sine-gaussian-mode coherent-independent --maxmcmc "$MAXMCMC"
```

GW231123 NRsur + 3 SG (coherent, independently localized):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 3 --sine-gaussian-mode coherent-independent --maxmcmc "$MAXMCMC"
```

GW231123 NRsur + 1 SG in H1 (incoherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 1 --sine-gaussian-mode incoherent --incoherent-detectors H1 --maxmcmc "$MAXMCMC"
```

GW231123 NRsur + 1 SG in L1 (incoherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 1 --sine-gaussian-mode incoherent --incoherent-detectors L1 --maxmcmc "$MAXMCMC"
```

GW231123 NRsur + 1 SG in H1 + 1 SG in L1 (incoherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 2 --sine-gaussian-mode incoherent --incoherent-sg-counts H1=1 L1=1 --maxmcmc "$MAXMCMC"
```

GW231123 NRsur + 2 SG in H1 (incoherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 2 --sine-gaussian-mode incoherent --incoherent-detectors H1 --maxmcmc "$MAXMCMC"
```

GW231123 NRsur + 2 SG in L1 (incoherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 2 --sine-gaussian-mode incoherent --incoherent-detectors L1 --maxmcmc "$MAXMCMC"
```

## Injections

For `coherent-independent` injections, the SG component values are taken from
the `coherent` count entries in `$SG_JSON`; the shared injected `ra`, `dec`, and
`psi` are taken from its `coherent-independent` object.

Inject: GW231123-maxL NRsur | Recover: NRsur | Zero Gaussian noise:

```
python "$INJ" \
  --base-dir "$BASE" \
  --label-prefix GW231123_maxl_nrsur_rec_nrsur_zero_gaussian \
  --injection-noise zero-gaussian \
  --likelihood gaussian \
  --maxmcmc "$MAXMCMC"
```

Inject: GW231123-maxL NRsur | Recover: NRsur | Gaussian noise:

```
python "$INJ" \
  --base-dir "$BASE" \
  --label-prefix GW231123_maxl_nrsur_rec_nrsur_gaussian_noise \
  --injection-noise gaussian \
  --likelihood gaussian \
  --maxmcmc "$MAXMCMC"
```

Inject: GW231123 NRsur + 2 SG (coherent) | Recover: NRsur | Zero Gaussian noise:

```
python "$INJ" \
  --base-dir "$BASE" \
  --label-prefix GW231123_inj_coherent2_rec_nrsur_zero_gaussian \
  --injection-noise zero-gaussian \
  --likelihood gaussian \
  --injection-num-sine-gaussians 2 \
  --injection-sine-gaussian-mode coherent \
  --maxmcmc "$MAXMCMC"
```

Inject: GW231123 NRsur + 1 SG (coherent) | Recover: NRsur + 1 SG (coherent) | Zero Gaussian noise:

```
python "$INJ" \
  --base-dir "$BASE" \
  --label-prefix GW231123_inj_coherent1_rec_coherent1_zero_gaussian \
  --injection-noise zero-gaussian \
  --likelihood gaussian \
  --injection-num-sine-gaussians 1 \
  --injection-sine-gaussian-mode coherent \
  --num-sine-gaussians 1 \
  --sine-gaussian-mode coherent \
  --maxmcmc "$MAXMCMC"
```

Inject: GW231123 NRsur + 2 SG (coherent, independently localized) | Recover: NRsur | Zero Gaussian noise:

```
python "$INJ" \
  --base-dir "$BASE" \
  --label-prefix GW231123_inj_coherent_independent2_rec_nrsur_zero_gaussian \
  --injection-noise zero-gaussian \
  --likelihood gaussian \
  --injection-num-sine-gaussians 2 \
  --injection-sine-gaussian-mode coherent-independent \
  --maxmcmc "$MAXMCMC"
```

Inject: GW231123 NRsur + 1 SG (coherent, independently localized) | Recover: NRsur + 1 SG (coherent, independently localized) | Zero Gaussian noise:

```
python "$INJ" \
  --base-dir "$BASE" \
  --label-prefix GW231123_inj_coherent_independent1_rec_coherent_independent1_zero_gaussian \
  --injection-noise zero-gaussian \
  --likelihood gaussian \
  --injection-num-sine-gaussians 1 \
  --injection-sine-gaussian-mode coherent-independent \
  --num-sine-gaussians 1 \
  --sine-gaussian-mode coherent-independent \
  --maxmcmc "$MAXMCMC"
```

Inject: GW231123-maxL NRsur + 1 SG in H1 (incoherent) | Recover: NRsur | Zero Gaussian noise:

```
python "$INJ" \
  --base-dir "$BASE" \
  --label-prefix GW231123_inj_h1x1_rec_nrsur_zero_gaussian \
  --injection-noise zero-gaussian \
  --likelihood gaussian \
  --injection-num-sine-gaussians 1 \
  --injection-sine-gaussian-mode incoherent \
  --injection-incoherent-detectors H1 \
  --maxmcmc "$MAXMCMC"
```

Inject: GW231123 NRsur + 1 SG in H1 + 1 SG in L1 (incoherent) | Recover: NRsur | Zero Gaussian noise:

```
python "$INJ" \
  --base-dir "$BASE" \
  --label-prefix GW231123_inj_h1x1_l1x1_rec_nrsur_zero_gaussian \
  --injection-noise zero-gaussian \
  --likelihood gaussian \
  --injection-num-sine-gaussians 2 \
  --injection-sine-gaussian-mode incoherent \
  --injection-incoherent-sg-counts H1=1 L1=1 \
  --maxmcmc "$MAXMCMC"
```
