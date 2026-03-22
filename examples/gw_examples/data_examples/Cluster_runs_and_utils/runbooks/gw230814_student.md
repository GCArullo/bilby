# GW230814 Student Runs

```
BASE_DIR="$(git rev-parse --show-toplevel)"
REAL="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_real_data.py"
```

## Real Runs

Det-independent `nu`, up to 4 frequency bands:

```
python "$REAL" --event GW230814 --likelihood student --range --num-frequency-bands 4 --submit
```
