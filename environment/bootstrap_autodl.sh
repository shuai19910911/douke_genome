#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
UV=${UV:-uv}
PYTHON=${PYTHON:-python3.10}
VENV=${VENV:-${ROOT}/.venv}
ARCHIVE=${ARCHIVE:-${ROOT}/environment/douke_genomemodel.tar.gz}
if [ -f "$ARCHIVE" ]; then
  [ ! -e "$VENV" ] || { echo "target environment already exists: $VENV" >&2; exit 2; }
  mkdir -p "$VENV"
  tar -xzf "$ARCHIVE" -C "$VENV"
  "$VENV/bin/conda-unpack"
  "$VENV/bin/python" -m pip install --no-deps -e "$ROOT"
elif [ -d "${ROOT}/wheelhouse" ]; then
  "$UV" venv --python "$PYTHON" "$VENV"
  "$UV" pip install --python "$VENV/bin/python" --no-index --find-links "${ROOT}/wheelhouse" -r "$ROOT/environment/requirements.lock.txt"
  "$UV" pip install --python "$VENV/bin/python" --no-deps -e "$ROOT"
else
  "$UV" venv --python "$PYTHON" "$VENV"
  "$UV" pip install --python "$VENV/bin/python" --build-constraints "$ROOT/environment/build-constraints.txt" -r "$ROOT/environment/requirements.lock.txt"
  "$UV" pip install --python "$VENV/bin/python" --no-deps -e "$ROOT"
fi
"$VENV/bin/python" -c 'import torch, numpy, sourmash, yaml; print(torch.__version__, torch.version.cuda)'
