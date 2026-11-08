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

## Published image and launch checkpoint

Image URL: `osdf:///igwn/cit/staging/gregorio.carullo/bilby-2.8.0.dev299+g05d60db68-05d60db689ec-bilby_pipe-1.8.2.dev90+g1d2f4278d-1d2f4278d75d-pesummary_GC-1.7.1.dev17+g55104b95d-55104b95dfcd-2026-11-08T15_54_03.sif`

SHA256: `3b22ebaa2c9766cc71cb614cdcb9bd603e1c99fee3adb8121324d56aeadc0b02`

Verified installed code: Bilby `05d60db689ec2bcdb6f7c805c86519a1eec66d26`; Pipe `1d2f4278d75d1e788a65a62444f9a98197709465`; PESummary `55104b95dfcd12746f849152ace5663d356e5aba`. Runtime checks confirmed Sequince 0.1.0/plugin 0.1.0b1/orng 0.2.1, NumPy 2.2.6, SciPy 1.16.3, LALSuite 7.26.15, and both pySEOBNR versions. Standard make publication and validation exited zero. The consolidated local suite passed 231 tests and 5,774 subtests.

The user clarified that only original serial-Bilby convergence and SEOB sources should be relaunched; all 20 selected source paths are under `Runs_new_priors`, with no parallel-Bilby source. Existing serial and parallel jobs must continue. Three replicas per configuration use 16 CPUs, at least 32 GB memory and 64 GB disk, plus distinct seeds.

All 20 original-image reference checks completed successfully. The emitted-job audit checked 20 DAGs and 60 distinct analysis seeds, transfer paths, configuration/dump/prior hashes, sampler options and resource requests. Redundant NRSur transfers were removed from the new jobs because both files are bundled in the image. Original generation nodes are marked DONE and reuse byte-identical data dumps. No original job or output was changed.

## SciPy numerical comparison

Nineteen configurations meet the strict likelihood-comparison tolerance. One reference point in `gaussian_SEOBNRv5PHM_pyseobnr036` gives old log-likelihood ratio -3.3646019633279405 and new -3.3666628426949003, a difference of -0.002060879366959867. The strict comparison is retained as `strict_likelihood_match=False`. SciPy 1.14.0 alone restores the old likelihood within 1e-14; independent old NumBa/LLVM, quaternionic and original pySEOBNR-binary overlays do not remove the difference. SciPy 1.16.0 behaves identically to 1.16.3. Model options, priors, preserved data, derived masses and geocentric times agree.

The actual H1/L1 noise-weighted waveform difference is 0.0017769994647045917, with normalized mismatch 9.851995841358985e-8 and relative norm difference 0.00044389178616403315. The squared-error contribution to the Gaussian log likelihood is half the squared difference norm, approximately 1.58e-6. This is a small numerical dependency effect at the checked points; it does not establish global posterior/evidence agreement. The supported SciPy 1.16.3 stack is retained, with this single scoped numerical-equivalence review recorded explicitly. All remaining prior/constraint/transform checks passed.

The original failed strict result is preserved in `/home/gregorio.carullo/AI_tmp_files/sequince_20261108/seob_strict_validation_before_numeric_review.json`. Final `validation.json` records all 20 reviewed configurations and the explicit exception. Probe and waveform-difference evidence is in the same temporary-work directory.

## Submission and initial scheduling

Submitted 20 DAGs / 60 analyses under `/home/gregorio.carullo/public_html/GW231123/sine_gaussians/Runs_sequince_20261108`. Launch wrapper exit status: zero. DAG IDs: 13627660, 13627661, 13627662, 13627663, 13627664, 13627668, 13627669, 13627670, 13627671, 13627672, 13627673, 13627674, 13627675, 13627676, 13627677, 13627678, 13627679, 13627680, 13627681, 13627682.

Receipts: `/home/gregorio.carullo/AI_tmp_files/sequince_20261108/submissions.json`; prepared job/source mappings and hashes: `prepared.json`; final reviewed scientific checks: `validation.json`; generated-job audit: `prepared_audit.exit=0`. Submission is idempotent and records each receipt atomically. All source configurations are original serial-Bilby files under `Runs_new_priors`; existing serial and parallel jobs continue.

The initial queue check found 20 running DAG managers and all 60 analysis jobs idle, with no held jobs and no runtime starts yet. A scheduling analysis found eligible configurations but no currently willing free slot for the checked job; resources are busy. Runtime checkpoints, independent-replica agreement and any speedup remain to be assessed after execution starts.
