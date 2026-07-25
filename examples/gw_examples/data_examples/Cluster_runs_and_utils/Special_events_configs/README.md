# Special-event configurations

This directory contains hand-maintained configurations that do not come
directly from a GWTC data release:

- `templates/` contains bilby_pipe initialisation-file templates.
- `priors/` contains the matching prior templates.
- `source_configs/` contains released source configurations.
- `runbooks/` contains the commands and supporting inputs for each special
  event.

The GW200129 Hannam reproduction uses data products from
`../GWTC_catalog_configs/GWTC-3`; only its bespoke NRSur7dq4 configuration and
prior are stored here.

The GW190521 LVK NRSur7dq4 profile uses the official PSD and calibration
products released in LIGO-P2000158-v4. Run
`prepare_gw190521_030229_lvk_nrsur7dq4.py` before submitting it.
Its embedded LALInference configuration is stored in
`source_configs/GW190521_030229_LVK_NRSur7dq4.ini`; the executable Bilby
translation is in `templates/`.

The event runbooks are:

- `runbooks/gw150914_hyperbolic.md`
- `runbooks/gw190521_030229_lvk_nrsur7dq4.md`
- `runbooks/gw200129_065458_hannam_nrsur7dq4.md`
- `runbooks/gw230814_student.md`
- `runbooks/gw231123_student.md`
- `runbooks/gw231123_sine_gaussians.md`
- `runbooks/gw231123_waveform_comparison.md`
- `runbooks/gw241127_seob.md`
