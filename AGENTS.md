Configuration Protections
-------------------------

- `examples/gw_examples/production_configs/fogg-bilby-NRSur7dq4_cbc_plus_sine_gaussians.ini`
  is the original LVK configuration file. Do not edit it as part of runbook,
  prior, or generated-configuration updates.


Coupling instructions
---------------------

This repo will be used in concertation with another repo cloned in `/Users/g.carullo@bham.ac.uk/Repos/bilby_pipe_greg`. You can assume that when calling bilby_pipe, this bilby_pipe_greg version is used.


GWTC output locations
---------------------

- Store GWTC event results under
  `/home/gregorio.carullo/public_html/GWTC_parametric_noise/Runs/<event>`.
- When generating comparison pages between multiple runs for a GWTC event,
  store them under
  `/home/gregorio.carullo/public_html/GWTC_parametric_noise/Comparison_pages/<event>`.


PESummary comparison pages
--------------------------

- Plot all parameters, including parameters that are not shared by every run,
  unless explicitly instructed otherwise. Do not use
  `--exclude_unshared_parameters` by default.
