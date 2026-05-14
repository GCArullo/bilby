# GW231123 Sine-Gaussian Runs

Injected SG values for the injection commands are read from `runbooks/injected_sine_gaussian_values.json`.

```
BASE_DIR="$(git rev-parse --show-toplevel)"
REAL="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_real_data.py"
INJ="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_injection.py"
SG_JSON="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/runbooks/injected_sine_gaussian_values.json"
INJECTION_POSTERIOR="/home/gregorio.carullo/src/bilby_greg/examples/gw_examples/data_examples/Cluster_runs_and_utils/LVK_posteriors/GW231123/posterior_samples.h5"
BASE="$HOME/GW231123/t_Student/Runs_injections_gw231123_sine_gaussians"
```

Both launchers submit by default. Add `--dry-run` if you only want to write
the ini/prior files.

## Real Runs

GW231123 NRsur + 1 SG (coherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 1 --sine-gaussian-mode coherent
```

GW231123 NRsur + 2 SG (coherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 2 --sine-gaussian-mode coherent
```

GW231123 NRsur + 1 SG in H1 (incoherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 1 --sine-gaussian-mode incoherent --incoherent-detectors H1
```

GW231123 NRsur + 1 SG in L1 (incoherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 1 --sine-gaussian-mode incoherent --incoherent-detectors L1
```

GW231123 NRsur + 1 SG in H1 + 1 SG in L1 (incoherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 2 --sine-gaussian-mode incoherent --incoherent-sg-counts H1=1 L1=1
```

GW231123 NRsur + 2 SG in H1 (incoherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 2 --sine-gaussian-mode incoherent --incoherent-detectors H1
```

GW231123 NRsur + 2 SG in L1 (incoherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --num-sine-gaussians 2 --sine-gaussian-mode incoherent --incoherent-detectors L1
```

## Injections

Inject: GW231123-maxL NRsur | Recover: NRsur | Zero Gaussian noise:

```
python "$INJ" \
  --base-dir "$BASE" \
  --label-prefix GW231123_maxl_nrsur_rec_nrsur_zero_gaussian \
  --injection-noise zero-gaussian \
  --likelihood gaussian
```

Inject: GW231123-maxL NRsur | Recover: NRsur | Gaussian noise:

```
python "$INJ" \
  --base-dir "$BASE" \
  --label-prefix GW231123_maxl_nrsur_rec_nrsur_gaussian_noise \
  --injection-noise gaussian \
  --likelihood gaussian
```

Inject: GW231123 NRsur + 2 SG (coherent) | Recover: NRsur | Zero Gaussian noise:

```
python "$INJ" \
  --base-dir "$BASE" \
  --label-prefix GW231123_inj_coherent2_rec_nrsur_zero_gaussian \
  --injection-noise zero-gaussian \
  --likelihood gaussian \
  --injection-num-sine-gaussians 2 \
  --injection-sine-gaussian-mode coherent
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
  --sine-gaussian-mode coherent
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
  --injection-incoherent-detectors H1
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
  --injection-incoherent-sg-counts H1=1 L1=1
```
