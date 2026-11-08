# Sequince GW231123 deployment

Requested branch: `sequince`, isolated from current Bilby and Pipe checkouts.

- [x] Inspect official plugin and production guidance. No validated Sequince LVK production recipe found.
- [x] Inventory ongoing CIT analyses and preserve scientific inputs.
- [x] Reproduce and fix finite-log-prior constraint bypass; prior tests pass.
- [x] Validate Pipe checkpoint transfer and constrained analytic sampling, including interrupted and completed-checkpoint restarts.
- [x] Publish both branches and the standard branch image.
- [x] Validate original likelihoods/priors against the new image; one small SciPy numerical difference is documented separately from strict matching.
- [x] Relaunch the 20 original serial-Bilby convergence and SEOB configurations in separate directories: 20 DAGs and 60 independent analyses submitted.
- [x] Confirm all 60 analysis queue entries and inspect initial scheduling/hold status. At the final local snapshot, all 60 were idle and none held; startup/checkpoints await available resources.

See [deployment.md](deployment.md) for settings, scope, provenance and checks.

Next checks after resources are assigned: verify runtime stage checkpoints, compare independent replicas, and measure convergence and runtime against the original runs. Posterior/evidence agreement and a speedup have not yet been established.
