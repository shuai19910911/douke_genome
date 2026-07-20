#!/bin/sh
#SBATCH -J lgfm_inventory
#SBATCH -p q02,q03,q04,q05
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 02:00:00

set -eu

PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
DATA_ROOT=${DATA_ROOT:?DATA_ROOT is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}

[ -d "${PROJECT_ROOT}" ]
[ -d "${DATA_ROOT}" ]
[ -x "${PYTHON_BIN}" ]

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" scripts/inventory_raw_data.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_ROOT}"
