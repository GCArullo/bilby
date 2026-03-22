# GW231123 Student Runs

```
BASE_DIR="$(git rev-parse --show-toplevel)"
REAL="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_real_data.py"
INJ="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_injection.py"
INJECTION_POSTERIOR="/home/gregorio.carullo/src/bilby_greg/examples/gw_examples/data_examples/Cluster_runs_and_utils/LVK_posteriors/GW231123/posterior_samples.h5"
BASE="$HOME/GW231123/t_Student/Runs_injections_runbook_gw231123_student"
```

## Real Runs

Det-independent `nu`, up to 4 frequency bands:

```
python "$REAL" --submit --event GW231123 --likelihood student --range --num-frequency-bands 4
```

Det-dependent `nu`, up to 4 frequency bands:

```
python "$REAL" --submit --event GW231123 --likelihood student --range --num-frequency-bands 4 --detector-dependent-nu
```

## Injections

Gaussian injections with Student recovery, det-independent `nu`

```
python "$INJ" --submit \
  --base-dir "$BASE/gaussian_injection_student_recovery_det_independent" \
  --label-prefix "GW231123_gaussianinj_studentrec_di_N${n}" \
  --injection-noise gaussian \
  --likelihood student \
  --detector-dependent-nu \
```

Gaussian injections with Student recovery, det-dependent `nu`

```
python "$INJ" --submit \
  --base-dir "$BASE/gaussian_injection_student_recovery_det_dependent" \
  --label-prefix "GW231123_gaussianinj_studentrec_dd_N${n}" \
  --injection-noise gaussian \
  --likelihood student \
  --detector-dependent-nu \
```

Student injections with Student recovery, det-independent `nu`, bands 1..4:

```
for n in 1 2 3 4; do
  python "$INJ" --submit \
    --base-dir "$BASE/student_injection_student_recovery_det_independent" \
    --label-prefix "GW231123_studentinj_studentrec_di_N${n}" \
    --injection-noise student \
    --nu-injection 2.1 \
    --likelihood student \
    --num-frequency-bands "$n"
done
```

Student injections with Student recovery, det-dependent `nu`, bands 1..4:

```
for n in 1 2 3 4; do
  python "$INJ" --submit \
    --base-dir "$BASE/student_injection_student_recovery_det_dependent" \
    --label-prefix "GW231123_studentinj_studentrec_dd_N${n}" \
    --injection-noise student \
    --nu-injection 2.1 \
    --likelihood student \
    --detector-dependent-nu \
    --num-frequency-bands "$n"
done
```
