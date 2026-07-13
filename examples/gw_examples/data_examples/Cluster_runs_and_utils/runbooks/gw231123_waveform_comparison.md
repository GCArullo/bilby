# GW231123 Waveform Comparison Runs

Runs with non-default waveform approximants. The `--waveform-approximant` flag
overrides the template value (NRSur7dq4) and appends the approximant name as a
suffix to all labels, output directories, and ini/prior filenames.

```
BASE_DIR="$(git rev-parse --show-toplevel)"
REAL="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_real_data.py"
```

Add `--dry-run` to write files without submitting.

## SEOBNRv5PHM

Gaussian baseline:

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant SEOBNRv5PHM
```

Gaussian + 1 SG (coherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant SEOBNRv5PHM \
  --num-sine-gaussians 1 --sine-gaussian-mode coherent
```


## IMRPhenomXPHM

Gaussian baseline:

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXPHM
```

Gaussian + 1 SG (coherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXPHM --num-sine-gaussians 1 --sine-gaussian-mode coherent
```

## IMRPhenomXPNR

Gaussian baseline:

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXPNR \
  --num-sine-gaussians 1 --sine-gaussian-mode coherent
```

## IMRPhenomX04a

Gaussian baseline:

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXO4a \
  --num-sine-gaussians 1 --sine-gaussian-mode coherent
```

## Notes

- No prior changes are needed; the CBC mass/spin priors are approximant-agnostic.
- SEOBNRv5PHM and IMRPhenomXPHM are both in LALSuite; no additional data files
  need to be transferred (`additional-transfer-paths` is only required for NRSur7dq4).
- SEOBNRv5PHM (and SEOBNRv5HM) sine-Gaussian runs: `bilby.gw.source.cbc_plus_sine_gaussians`
  auto-detects these approximants and evaluates the CBC baseline through gwsignal's
  `GenerateFDWaveform` instead of the standard lalsimulation path. No extra flags are
  required; `waveform-generator` must remain `bilby.gw.waveform_generator.WaveformGenerator`
  (already handled by this script) since `GWSignalWaveformGenerator` would bypass
  `frequency-domain-source-model` entirely.
- If distance marginalisation is ever enabled, the lookup table
  (`distance-marginalization-lookup-table`) must be regenerated for the new
  approximant — the NRSur7dq4 table in the template cannot be reused.
