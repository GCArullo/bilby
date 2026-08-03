# Special-event configurations

This directory contains the hand-maintained GW150914 and GW190521
configurations imported from the `hyp` branch:

- `templates/` contains bilby_pipe initialisation-file templates.
- `priors/` contains the matching prior templates.
- `source_configs/` contains released source configurations.
- `runbooks/` contains the commands and supporting inputs for each event.

The GW190521 LVK NRSur7dq4 profile uses the official PSD and calibration
products released in LIGO-P2000158-v4. Run
`prepare_gw190521_030229_lvk_nrsur7dq4.py` before submitting it.
Its embedded LALInference configuration is stored in
`source_configs/GW190521_030229_LVK_NRSur7dq4.ini`; the executable Bilby
translation is in `templates/`.

The event runbooks are:

- `runbooks/gw150914_student.md`
- `runbooks/gw190521_030229_lvk_nrsur7dq4.md`
