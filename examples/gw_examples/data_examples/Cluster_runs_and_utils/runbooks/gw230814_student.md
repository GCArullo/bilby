# GW230814 Student Runs

```
BASE_DIR="$(git rev-parse --show-toplevel)"
REAL="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/submit_runs_real_data.py"
GR_PRIOR="$BASE_DIR/examples/gw_examples/data_examples/Cluster_runs_and_utils/Prior_templates/GW230814_gr_template.prior"
MAXMCMC=5000
```

Condor jobs use the Bilby container by default. Before the first submission,
run `make publish` in `Cluster_runs_and_utils/container_creation`; the launcher
selects the current Git branch from `container_images.json`. Use
`--container-image URL` to override it or `--no-container` to use the existing
node environment.

## Real Runs

The real-data launcher submits by default. Add `--dry-run` if you only want to
write the ini/prior files. The templates use `maxmcmc=5000`; the commands below
set this explicitly with `--maxmcmc "$MAXMCMC"`. Run and web outputs are written
below `$HOME/public_html/GW230814` by default, with each summary in
`Runs/<run-name>/web`.

Default non-GR run, det-independent `nu`, Student-t bands 1..4 plus the
single-band Gaussian companion:

```
python "$REAL" --event GW230814 --likelihood student --range --num-frequency-bands 4 --maxmcmc "$MAXMCMC"
```

Default non-GR Gaussian-only run:

```
python "$REAL" --event GW230814 --likelihood gaussian --maxmcmc "$MAXMCMC"
```

GR-baseline pSEOB run, det-independent `nu`, Student-t bands 1..4 plus the
single-band Gaussian companion:

```
python "$REAL" \
  --event GW230814 \
  --likelihood student \
  --range \
  --num-frequency-bands 4 \
  --prior-template "$GR_PRIOR" \
  --label-prefix GW230814_t_Student_pSEOB_GR \
  --file-prefix GW230814_GR \
  --outdir-label gr_baseline \
  --maxmcmc "$MAXMCMC"
```

GR-baseline pSEOB Gaussian-only run:

```
python "$REAL" \
  --event GW230814 \
  --likelihood gaussian \
  --prior-template "$GR_PRIOR" \
  --label-prefix GW230814_t_Student_pSEOB_GR \
  --file-prefix GW230814_GR \
  --outdir-label gr_baseline \
  --maxmcmc "$MAXMCMC"
```
