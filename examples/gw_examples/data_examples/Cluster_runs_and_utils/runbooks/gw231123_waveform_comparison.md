# GW231123 Waveform Comparison Runs

Runs with non-default waveform approximants. The `--waveform-approximant` flag
overrides the template value (NRSur7dq4) and appends the approximant name as a
suffix to all labels, output directories, and ini/prior filenames.

```
BASE_DIR="$(git rev-parse --show-toplevel)"
REAL="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_real_data.py"
```

Condor jobs use the Bilby container by default. Before the first submission,
run `make publish` on CIT, or `make publish CIT=false` elsewhere, in
`Cluster_runs_and_utils/container_creation`; the launcher selects the current
Git branch from `container_images.json`. Publish and register a new official
image before production submissions. Use `--no-container` only for local testing.

Generated configs use the worldwide IGWN pool (`transfer-files=True`,
`osg=True`, `desired-sites=None`). Do not pass `--require-epnfs` unless the run
must be restricted to CIT.

Submission stops before `bilby_pipe` if a local frame/data file, PSD,
calibration envelope, or additional transfer path is missing.

Add `--dry-run` to write files without submitting. Run and web outputs are
written below `$HOME/public_html/GW231123` by default, with each summary in
`Runs/<run-name>/web`.

The commands below show both coherent SG choices where relevant. `coherent`
uses the CBC sky position; `coherent-independent` samples one separate SG
`ra`, `dec`, and `psi` shared by all SG components.

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

Gaussian + 1 SG (coherent, independently localized):

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant SEOBNRv5PHM \
  --num-sine-gaussians 1 --sine-gaussian-mode coherent-independent
```


## IMRPhenomXPHM

Gaussian baseline:

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXPHM
```

Gaussian + 1 SG (coherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXPHM \
  --num-sine-gaussians 1 --sine-gaussian-mode coherent
```

Gaussian + 1 SG (coherent, independently localized):

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXPHM \
  --num-sine-gaussians 1 --sine-gaussian-mode coherent-independent
```

## IMRPhenomXPHM with SpinTaylor precession

LALSuite has no standalone SpinTaylor approximant: the SpinTaylor precession
prescription is an option on IMRPhenomXPHM, selected through
`PhenomXPrecVersion`. Passing `IMRPhenomXPHM_SpinTaylor` sets
`waveform-approximant=IMRPhenomXPHM` and
`waveform-arguments-dict={'PhenomXPrecVersion': 320, 'PhenomXPFinalSpinMod': 2}`
and the waveform minimum to 10 Hz (the detector minimum remains 20 Hz),
matching the specified LVK settings. The launcher keeps the full
name in labels and directories so the run cannot be confused with a default
IMRPhenomXPHM one (which uses the MSA prescription, version 223).

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXPHM_SpinTaylor \
  --num-sine-gaussians 1 --sine-gaussian-mode coherent
```

Gaussian baseline:

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXPHM_SpinTaylor
```

For the `Runs_new_priors` comparison campaign, pass its outdir and webdir bases
and `--outdir-label LVK` to both commands. This retains the earlier SpinTaylor
run, which used a 20 Hz waveform minimum and final-spin modifier 4.

## IMRPhenomXPNR

Gaussian baseline:

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXPNR
```

Gaussian + 1 SG (coherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXPNR \
  --num-sine-gaussians 1 --sine-gaussian-mode coherent
```

Gaussian + 1 SG (coherent, independently localized):

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXPNR \
  --num-sine-gaussians 1 --sine-gaussian-mode coherent-independent
```

## IMRPhenomX04a

Gaussian + 1 SG (coherent):

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXO4a \
  --num-sine-gaussians 1 --sine-gaussian-mode coherent
```

Gaussian + 1 SG (coherent, independently localized):

```
python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant IMRPhenomXO4a \
  --num-sine-gaussians 1 --sine-gaussian-mode coherent-independent
```

## TEOBResumS-Dali: precessing and precessing + eccentric

The official image installs Dali commit
`7b9b3fd951f51412bcaeb50965c7a2c77c9b09a8`, recorded inside the image at
`/opt/teobresums_commit.txt`. Rebuild/publish the standard image before submitting.
Do not substitute the PyPI `teobresums` package for this pinned Dali checkout.

Run both CBC-only and CBC + 1 coherent SG for each variant:

```bash
for WF in TEOBResumS_Dali TEOBResumS_Dali_Ecc; do
  python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant "$WF"
  python "$REAL" --event GW231123 --likelihood gaussian --waveform-approximant "$WF" \
    --num-sine-gaussians 1 --sine-gaussian-mode coherent
done
```

`TEOBResumS_Dali` fixes eccentricity to zero. `TEOBResumS_Dali_Ecc` samples
eccentricity uniformly on [0, 0.5] and **true anomaly** uniformly on [0, 2 pi).
Both use the precessing CBC spin priors, all positive-m coprecessing modes through
ell=4 and their inertial-frame counterparts. Spins and eccentricity are defined
at the template reference frequency, 10 Hz. Dali starts at that frequency using
twice the orbit-averaged orbital frequency (`ecc_freq=3`, `ecc_ics=2`). The
likelihood retains the 20–448 Hz band, 8 s duration and 1024 Hz data sampling.

The source converts Bilby spin angles to Cartesian spins **and inclination**,
uses Dali's merger-time origin, tapers the start with LAL's standard taper and
transforms both polarizations to the requested frequency grid. Internal waveform
sampling is at least 4096 Hz; long waveforms are transformed without truncation.
The SG source calls the same Dali implementation before adding the SG.

The CBC and SG priors, calibration and sampler settings otherwise match this
runbook: 2000 live points for CBC-only, 2500 with one SG, three parallel chains,
eight CPUs per chain, `naccept=60`, `maxmcmc=5000`. For the existing waveform
campaign, pass `--outdir-base` and `--webdir-base` as
`/home/gregorio.carullo/public_html/GW231123/sine_gaussians/Runs_new_priors`.
PESummary's unsupported LAL multipole-SNR/spin-evolution diagnostics and
quasicircular remnant fits are disabled for Dali; Bilby generates the waveform
plots using the actual source model.

## Notes

- No manual prior changes are needed; the launcher adds the independent SG sky
  priors and the CBC mass/spin priors are approximant-agnostic.
- The managed container carries both NRSur7dq4 HDF5 files under
  `/opt/lalsimulation-data`; no additional waveform data need to be transferred.
  Use the container outside CIT; the launcher's `--no-container` configuration
  searches the CIT-only `/scratch/lalsimulation` copy instead.
- Plain SEOBNRv5PHM and SEOBNRv5HM runs use
  `bilby.gw.waveform_generator.GWSignalWaveformGenerator` directly.
- For sine-Gaussian runs with either SEOB model,
  `bilby.gw.source.cbc_plus_sine_gaussians` evaluates the CBC baseline through
  gwsignal's `GenerateFDWaveform`. The launcher retains the generic
  `WaveformGenerator` in this case because `GWSignalWaveformGenerator` would
  bypass the composite source model and omit the sine-Gaussian signal.
- If distance marginalisation is ever enabled, the lookup table
  (`distance-marginalization-lookup-table`) must be regenerated for the new
  approximant — the NRSur7dq4 table in the template cannot be reused.
