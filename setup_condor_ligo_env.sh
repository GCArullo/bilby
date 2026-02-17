#!/usr/bin/env bash
set -euo pipefail

# Create a Python environment suitable for running this bilby branch on Condor.
#
# Usage:
#   bash setup_condor_ligo_env.sh [ENV_DIR]
#
# Example:
#   bash setup_condor_ligo_env.sh /home/$USER/envs/bilby-tstudent

ENV_DIR="${1:-$PWD/.venv-condor-bilby}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
  echo "[ERROR] Could not find pyproject.toml in $REPO_ROOT"
  exit 1
fi

echo "[INFO] Creating virtual environment at: $ENV_DIR"
python3 -m venv "$ENV_DIR"

# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install --upgrade pip setuptools wheel

# Install baseline dependencies for this bilby branch.
python -m pip install -r "$REPO_ROOT/requirements.txt"
python -m pip install -r "$REPO_ROOT/gw_requirements.txt"
python -m pip install -r "$REPO_ROOT/sampler_requirements.txt"

# Install this checkout in editable mode so local branch changes are used.
python -m pip install --no-build-isolation -e "$REPO_ROOT[gw]"

# bilby_pipe currently requires bilby>=2.1.2 via metadata; for this branch checkout,
# install bilby_pipe without dependency resolution so it uses the local editable bilby.
python -m pip install --no-deps "bilby_pipe>=1.4,<1.5"

# Useful testing/runtime extras for this repo and typical LIGO workflows.
# configargparse is required by bilby_pipe and may be skipped when using --no-deps.
python -m pip install parameterized configargparse

# Quick sanity checks.
python - <<'PY'
import bilby
import bilby_pipe
import platform
import configargparse
print(f"[INFO] Python: {platform.python_version()}")
print(f"[INFO] bilby import OK from: {bilby.__file__}")
print(f"[INFO] bilby_pipe import OK from: {bilby_pipe.__file__}")
print(f"[INFO] configargparse import OK from: {configargparse.__file__}")
PY

echo "[INFO] Running pip check (metadata consistency):"
python -m pip check || true

echo
echo "[DONE] Environment ready."
echo "Activate with: source '$ENV_DIR/bin/activate'"
echo "Run bilby_pipe with your config, e.g.:"
echo "  bilby_pipe GW231123_t_student.ini"
