#!/bin/sh
#SBATCH -J lgfm_zip_audit
#SBATCH -p q02
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 1-00:00:00

set -eu

PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
DATA_ROOT=${DATA_ROOT:?DATA_ROOT is required}
INVENTORY=${INVENTORY:?INVENTORY is required}
RESULT_ROOT=${RESULT_ROOT:?RESULT_ROOT is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}

case "$PROJECT_ROOT:$DATA_ROOT:$INVENTORY:$RESULT_ROOT:$OUTPUT_ROOT:$PYTHON_BIN" in
  *",") printf '%s\n' "runtime paths must not contain commas" >&2; exit 2 ;;
esac

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

exec "$PYTHON_BIN" -B "$PROJECT_ROOT/scripts/audit_zip_archives.py" \
  --inventory "$INVENTORY" \
  --data-root "$DATA_ROOT" \
  --result-root "$RESULT_ROOT" \
  --output-dir "$OUTPUT_ROOT"
