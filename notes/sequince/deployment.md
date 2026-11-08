# Deployment record

## Sampler choice

Use the existing `sequince-bilby==0.1.0b1` plugin with `sequince==0.1.0`; Bilby's plugin registry discovers the existing sampling implementation; a thin native adapter adds restart handling. Select `bilby.sequince-ps`, the thin restart adapter around the plugin's `sequince-ps` (persistent sampling), NumPy, affine transform and differential evolution. No trained density model is required. Standard SMC was considered, but a 2,048-particle final population cannot provide the chosen archive weight ESS target. Persistent sampling has a gravitational-wave methodological precedent; that does not validate this implementation's precision or speed.

Production settings: `n_particles=2048`, `mutation_n_steps=40`, `target_ess=4096`, `target_total_ess=10000`, `dynamic=False`, `sampler_backend="numpy"`, `mutation="differential-evolution"`, `particle_transform="affine"`, `checkpoint=True`, `checkpoint_frequency=1`, `resume=True`; parallelize likelihood evaluations using the requested job CPUs. Final archive weight ESS is not a count of independent samples.

Sources: https://github.com/sequince-dev/sequince-bilby and https://github.com/sequince-dev/sequince. The related GW validation https://arxiv.org/abs/2506.18977 concerns pocomc, not a production validation of Sequince.

The adapter keeps the plugin's sampler implementation and adds graceful signal exits with Pipe's checkpoint code (77), plus regeneration of the raw result from a completed checkpoint. The latter avoids a collision if a job was interrupted after sampling but before Bilby finished writing its final result. Intermediate and completed checkpoint resumes were checked against uninterrupted runs.

## Relaunch scope and preservation

Initial global/default scheduler inventory: 39 active configuration groups, 111 running analyses, 40 DAGs. The 18 convergence configurations in the prior parallel-relaunch manifest overlap 18 original configurations. Relaunch once per scientific configuration, retaining three independent replicas, plus the two additional CBC-only SEOB variants: 20 configurations. This includes the coherent-1SG Dali convergence configuration, but excludes the additional CBC-only Dali configuration. Existing jobs and output remain intact.

Original inputs and hashes are recorded under `/home/gregorio.carullo/AI_tmp_files/sequince_20261108/source_configs.json`. Preserve source prior files, data dumps, calibration, PSDs, waveform options and pySEOBNR version (0.3.6 or 0.3.7). New output must be separate and must never merge with old sampler results.

## Validation and status

The constrained prior regression failed before the fix. After applying the one-line guard correction, 183 prior tests and 5,774 subtests passed. Remaining validation/publication/submission is tracked in README.md.

Additional checks: legacy interpolated priors, PSD pickles and known SEOB reference-frequency rejections passed 43 surrounding tests; restart adapter regressions passed five cases; nine Pipe checkpoint-transfer tests passed. A pre-existing emcee directory-creation failure was reproduced against unchanged Pipe source. The analytic constrained target at 2,048 particles/40 mutations gave log Z=-2.44888775 versus -2.43140864 analytically; completed-checkpoint resumption matched exactly.
