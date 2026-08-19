# Reweighting a Gaussian posterior onto a parametric-noise likelihood

`reweight_posterior.py` answers "would a heavy-tailed or rescaled noise model
have changed this result?" without rerunning the sampler. It importance
reweights an existing Gaussian (Whittle) posterior,

    w_i = exp( logL_new(theta_i) - logL_gaussian(theta_i) ),

marginalising the new likelihood's noise parameters over the same priors the
launcher writes. A scan that would cost days of cluster time as reruns takes
minutes, and its effective sample size tells you when a rerun is unavoidable.

## Reading the output

| `ESS/N` | meaning | action |
|---|---|---|
| `>= 0.5` | the noise model barely perturbs the likelihood | concluded, no rerun |
| `0.1 - 0.5` | reweighted quantiles are indicative, not precise | confirm any shift that matters |
| `< 0.1` | the model moves the posterior beyond what reweighting resolves | **this** is what justifies a rerun |

Reweighting can only lose information, never discover a new mode, so a low ESS
is a trigger to run, never evidence that anything has been healed.

## Detectability: is there anything to see?

Before the reweighting result means anything, ask whether the region contains
something an amplitude model can engage with at all. The tool reports, per
detector, the statistic of `notes/robust_likelihood_regimes.tex` Eq. (2): an
artefact depositing whitened power `rho^2` into `n` bins of expected power `2n`
gives fractional excess `eps = rho^2 / 2n`, and profiling a free scale yields

    Lambda = n (L* - 1 - ln L*),   L* = 1 + eps,   ~= rho^4 / 8n  for eps << 1.

Detectability grows as the **fourth power** of the artefact's SNR and falls
**linearly** with the size of the region the parameter is shared over. That is
why loud artefacts are trivially found, quiet ones are invisible, and spreading a
feature across a wide band destroys sensitivity to it -- the same law that makes
a handful of equal-width bands useless.

| `Lambda` | meaning |
|---|---|
| `< 1` | below threshold: no amplitude model will engage |
| `1 - 5` | marginal: noise parameters move a little, the posterior barely |
| `>= 5` | detectable: the model will engage |

Two warnings. A null reweighting in a region with `Lambda < 1` says only that the
region is below threshold, **not** that the data are clean -- a coherent
subtracting model may still be warranted. And detectability is not relevance: an
artefact disjoint from the signal is a constant in the likelihood and biases
nothing, however loud it is.

On GW191109 in 20-45 Hz the tool gives `Lambda = 11.9` for H1 on raw data and
`0.8` for L1, reproducing the note's 11.82 and 1.09 up to the choice of sample.
This is exactly the pathological case the note describes: the artefact the model
can see is the one that does not matter.

## Regions

One tool covers all three branches, because the residual is whitened once with
the segment PSD and then tiled; with everything at its Gaussian limit the tiled
quadratic form equals the Whittle one exactly.

| branch | region | flags |
|---|---|---|
| `hyp` | frequency band | `--frequency-band 20,45` |
| `TD` | time chunk | `--time-band 1.9,2.1` |
| `TD_FD` | time-frequency tile | both together |

Omitting a flag means "all of that axis".

## Examples

All three use the GW191109 pSEOB posterior. `--data` takes either a bilby_pipe
data dump pickle or an analysis npz.

**Frequency band, `hyp`.** Is the deviation sensitive to heavy tails where the
L1 glitch sits?

```bash
python reweight_posterior.py \
  --posterior pseob_posterior.npz --data gw191109_analysis_data.npz \
  --family hyperbolic --frequency-band 20,45 --detectors L1 \
  --approximant SEOBNRv5PHM \
  --frequency-domain-source-model bilby_tgr.pseob.source.gwsignal_binary_black_hole \
  --reference-frame H1L1 --start-time 1257296853.216458 \
  --parameters domega220 chi_eff --output reweight_L1_20_45.json
```

Gives `ESS/N = 0.90` and a `-0.20 sigma` shift in `domega220`: robust, no rerun.

**Time chunk, `TD`.** Does downweighting the pre-merger stretch matter?

```bash
python reweight_posterior.py \
  --posterior pseob_posterior.npz --data gw191109_analysis_data.npz \
  --family psd-scale --time-band 0.0,1.9 \
  --approximant SEOBNRv5PHM \
  --frequency-domain-source-model bilby_tgr.pseob.source.gwsignal_binary_black_hole \
  --reference-frame H1L1 --start-time 1257296853.216458 \
  --parameters domega220 chi_eff --output reweight_premerger.json
```

**Time-frequency tile, `TD_FD`.** The scattering arch, isolated from the signal
in both axes at once:

```bash
python reweight_posterior.py \
  --posterior pseob_posterior.npz --data gw191109_analysis_data.npz \
  --family hyperbolic --time-band 1.9,2.1 --frequency-band 20,35 --detectors L1 \
  --approximant SEOBNRv5PHM \
  --frequency-domain-source-model bilby_tgr.pseob.source.gwsignal_binary_black_hole \
  --reference-frame H1L1 --start-time 1257296853.216458 \
  --parameters domega220 chi_eff --output reweight_arch_tile.json
```

## Two things to watch

**Tight tiles have very few bins.** A 0.2 s chunk has 5 Hz frequency resolution,
so the 20--35 Hz tile above holds **three** bins. Two hyperbolic shape parameters
cannot be identified from three bins; the result is prior-dominated. Widen the
tile, or use `--family psd-scale`, whose single parameter is identifiable from
far fewer bins.

**Never put a chunk boundary on the merger.** A boundary that bisects it destroys
20--30% of a deviation's log-likelihood evidence purely numerically, while a chunk
that *contains* it costs under a nat. Choose `--time-band` so the merger sits
inside a chunk with margin.

**`--max-samples` thins randomly**, not by stride, because posterior files are
ordered by the sampler rather than randomly. Use `--seed` for reproducibility.
