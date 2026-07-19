#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
RELEASE_ROOT=$(CDPATH= cd -- "$ROOT/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.venv/bin/python}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
CONFIG=${CONFIG:-${ROOT}/configs/pretrain_stage1.yaml}
MODE=${MODE:-fresh}
INITIALIZE_FROM=${INITIALIZE_FROM:-}
MINIMUM_FREE_MIB=${MINIMUM_FREE_MIB:-6000}
[ -x "$PYTHON_BIN" ] || { echo "missing Python: $PYTHON_BIN" >&2; exit 2; }
[ -f "$RELEASE_ROOT/AUTODL_RELEASE_MANIFEST.json" ] || { echo "not an AutoDL release: $RELEASE_ROOT" >&2; exit 2; }
"$PYTHON_BIN" -B "$ROOT/scripts/verify_autodl_release.py" --release-root "$RELEASE_ROOT"
mkdir -p "$ROOT/runs/preflight"
PREFLIGHT_TMP="$ROOT/runs/preflight/$(basename "$CONFIG").${MODE}.$$.tmp"
PREFLIGHT_FINAL="$ROOT/runs/preflight/$(basename "$CONFIG").${MODE}.json"
set -- "$PYTHON_BIN" -B "$ROOT/scripts/preflight_training.py" --config "$CONFIG" --mode "$MODE" --nproc-per-node "$NPROC_PER_NODE" --minimum-free-mib "$MINIMUM_FREE_MIB"
if [ -n "$INITIALIZE_FROM" ]; then set -- "$@" --initialize-from "$INITIALIZE_FROM"; fi
"$@" > "$PREFLIGHT_TMP"
mv "$PREFLIGHT_TMP" "$PREFLIGHT_FINAL"
set -- "$PYTHON_BIN" -u -m torch.distributed.run --standalone --nproc-per-node="$NPROC_PER_NODE" "$ROOT/scripts/train_pretrain.py" --config "$CONFIG" --device cuda
case "$MODE" in
  fresh) ;;
  resume) set -- "$@" --resume ;;
  initialize) set -- "$@" --initialize-from "$INITIALIZE_FROM" ;;
  *) echo "unsupported MODE: $MODE" >&2; exit 2 ;;
esac
exec "$@"
