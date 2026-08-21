# GW231123 Student Runs

```
BASE_DIR="$(git rev-parse --show-toplevel)"
REAL="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_real_data.py"
INJ="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_injection.py"
INJECTION_POSTERIOR="/home/gregorio.carullo/src/bilby_greg/examples/gw_examples/data_examples/Cluster_runs_and_utils/LVK_posteriors/GW231123/posterior_samples.h5"
BASE="$HOME/public_html/GW231123/t_Student/Runs_injections_runbook_gw231123_student"
MAXMCMC=5000
```

Condor jobs use the Bilby container by default. Before the first submission,
run `make publish` on CIT, or `make publish CIT=false` elsewhere, in
`Cluster_runs_and_utils/container_creation`; the launcher selects the current
Git branch from `container_images.json`. Use
`--container-image URL` to override it or `--no-container` to use the existing
node environment.

Generated configs use the worldwide IGWN pool (`transfer-files=True`,
`osg=True`, `desired-sites=None`). Do not pass `--require-epnfs` unless the run
must be restricted to CIT.

Submission stops before `bilby_pipe` if a local frame/data file, PSD,
calibration envelope, or additional transfer path is missing.

Both launchers submit by default. Add `--dry-run` if you only want to write
the ini/prior files. The templates use `maxmcmc=5000`; the commands below
set this explicitly with `--maxmcmc "$MAXMCMC"`. Run and web outputs are written
below `$HOME/public_html/GW231123` by default, with each summary in
`Runs/<run-name>/web`.

## Real Runs

Det-independent `nu`, up to 4 frequency bands:

```
python "$REAL" --event GW231123 --likelihood student --range --num-frequency-bands 4 --maxmcmc "$MAXMCMC"
```

Det-dependent `nu`, up to 4 frequency bands:

```
python "$REAL" --event GW231123 --likelihood student --range --num-frequency-bands 4 --detector-dependent-nu --maxmcmc "$MAXMCMC"
```

## Injections

Gaussian injections with Student recovery, det-independent `nu`

```
python "$INJ" \
  --base-dir "$BASE/gaussian_injection_student_recovery_det_independent" \
  --label-prefix "GW231123_gaussianinj_studentrec_di_N${n}" \
  --injection-noise gaussian \
  --likelihood student \
  --maxmcmc "$MAXMCMC"
```

Gaussian injections with Student recovery, det-dependent `nu`

```
python "$INJ" \
  --base-dir "$BASE/gaussian_injection_student_recovery_det_dependent" \
  --label-prefix "GW231123_gaussianinj_studentrec_dd_N${n}" \
  --injection-noise gaussian \
  --likelihood student \
  --detector-dependent-nu \
  --maxmcmc "$MAXMCMC"
```

Student injections with Student recovery, det-independent `nu`, bands 1..4:

```
for n in 1 2 3 4; do
  python "$INJ" \
    --base-dir "$BASE/student_injection_student_recovery_det_independent" \
    --label-prefix "GW231123_studentinj_studentrec_di_N${n}" \
    --injection-noise student \
    --nu-injection 2.1 \
    --likelihood student \
    --num-frequency-bands "$n" \
    --maxmcmc "$MAXMCMC"
done
```

Student injections with Student recovery, det-dependent `nu`, bands 1..4:

```
for n in 1 2 3 4; do
  python "$INJ" \
    --base-dir "$BASE/student_injection_student_recovery_det_dependent" \
    --label-prefix "GW231123_studentinj_studentrec_dd_N${n}" \
    --injection-noise student \
    --nu-injection 2.1 \
    --likelihood student \
    --detector-dependent-nu \
    --num-frequency-bands "$n" \
    --maxmcmc "$MAXMCMC"
done
```
