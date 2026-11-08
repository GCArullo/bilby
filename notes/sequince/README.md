# Sequince GW231123 deployment

Requested branch: `sequince`, isolated from current Bilby and Pipe checkouts.

- [x] Inspect official plugin and production guidance. No validated Sequince LVK production recipe found.
- [x] Inventory ongoing CIT analyses and preserve scientific inputs.
- [x] Reproduce and fix finite-log-prior constraint bypass; prior tests pass.
- [x] Validate Pipe checkpoint transfer and constrained analytic sampling, including interrupted and completed-checkpoint restarts.
- [ ] Publish both branches and the standard branch image.
- [ ] Validate original likelihoods/priors against the new image.
- [ ] Relaunch unique convergence and SEOB configurations in separate directories.
- [ ] Verify all submitted replicas start and checkpoint.

See [deployment.md](deployment.md) for settings, scope, provenance and checks.
