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

## PESummary pages

PESummary writes the label into its own output names twice: `html/` holds
`<label>_<label>_<parameter>.html` and `samples/` holds
`<label>_<result file name>`. Left to itself bilby_pipe labels the results-page
job with the full merge-result basename, which for the sine-Gaussian runs
reaches 123 characters, so both names run past the 255 byte limit and the job
dies with

```
OSError: [Errno 36] File name too long
```

`submit_runs_real_data.py` therefore passes `labels` in
`summarypages-arguments`, naming each per-run page after its run directory.
Hand-written labels need the same discipline: the budget is roughly 110
characters, and comparison pages should use short names anyway.

The failure is quiet. The page aborts partway, leaving a populated `web/`
directory that looks plausible, and the DAG records the failure only in
`submit/*.dagman.out`. Check the metafile, which PESummary writes last:

```
for d in */; do
  [ -f "${d}web/samples/posterior_samples.h5" ] || echo "incomplete: ${d%/}"
done
```

To rebuild a page, rerun the node from its own submit file rather than
resubmitting the DAG: take the argument string from the
`VARS ..._pesummary_arg_0` line of `submit/dag_*.submit`, shorten `--labels`,
and redirect `log`, `output`, and `error` so the original logs survive. Move
the partial `web/` aside first, otherwise stale files from the failed attempt
are left alongside the new ones.
