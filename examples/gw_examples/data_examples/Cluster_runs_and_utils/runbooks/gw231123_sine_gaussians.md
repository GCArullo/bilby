# GW231123 Sine-Gaussian Runs

Injected SG values for the injection commands are read from `runbooks/injected_sine_gaussian_values.json`.

```
BASE_DIR="$(git rev-parse --show-toplevel)"
REAL="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_real_data.py"
INJ="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_injection.py"
SG_JSON="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/runbooks/injected_sine_gaussian_values.json"
INJECTION_POSTERIOR="/home/gregorio.carullo/src/bilby_greg/examples/gw_examples/data_examples/Cluster_runs_and_utils/LVK_posteriors/GW231123/posterior_samples.h5"
BASE="$HOME/GW231123/t_Student/Runs_injections_gw231123_sine_gaussians"
MAXMCMC=5000
```

Condor jobs use the Bilby container by default. Before the first submission,
run `make publish` in `Cluster_runs_and_utils/container_creation`; the launcher
reads the resulting `container_image.txt`. Use `--container-image URL` to
override it or `--no-container` to use the existing node environment.

Both launchers submit by default. Add `--dry-run` if you only want to write
the ini/prior files. The templates use `maxmcmc=5000`; the commands below
set this explicitly with `--maxmcmc "$MAXMCMC"`.

The generated recovery configs use `nlive=2000` for the baseline CBC run,
`nlive=2500` for one recovered SG, and `nlive=3000` for two or more recovered
SGs in total. This includes incoherent `H1=1 L1=1`.

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
