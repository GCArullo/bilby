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

All 20 original-image reference checks completed successfully. The 20 new job sets are prepared under `/home/gregorio.carullo/public_html/GW231123/sine_gaussians/Runs_sequince_20261108`; no jobs have been submitted yet. The emitted-job audit flagged redundant external NRSur data transfers, already bundled in the image. A surrogate-only repair is staged locally at `/private/tmp/sequince_repair_transfers.py`; it must be copied to the remote temporary-work directory and executed before rerunning emitted-job and physics checks. The audit remains strict.

CIT authentication stopped before the repair could be transferred; the existing local `~/.ssh/citgsi_mux` socket disappeared and a new connection returned GSSAPI permission denied. The user has been asked to renew it. Resume by staging the updated prepare/repair scripts into `/home/gregorio.carullo/AI_tmp_files/sequince_20261108`, executing the repair, and rerunning `sequince_prepare_validate.sh`. Only after `prepare_validate.exit=0` and `validation.json`/`prepared_audit.exit` pass should `sequince_submit_runs.py` submit the 20 DAGs. The submission script records receipts atomically and checks for existing matching DAGs.
