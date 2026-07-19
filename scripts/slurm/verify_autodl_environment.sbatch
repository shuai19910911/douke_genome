#!/bin/sh
#SBATCH -J lgfm_env_verify
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 04:00:00

set -eu
ARCHIVE=${ARCHIVE:?ARCHIVE is required}
PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
VERIFY_ROOT=${VERIFY_ROOT:?VERIFY_ROOT is required}
JOB_TAG=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
TARGET="$VERIFY_ROOT/env.${JOB_TAG}"
RECEIPT_TMP="$VERIFY_ROOT/environment_verify.json.${JOB_TAG}.tmp"
RECEIPT="$VERIFY_ROOT/environment_verify.json"
[ ! -e "$TARGET" ] || { echo "stale target: $TARGET" >&2; exit 2; }
mkdir -p "$TARGET" "$VERIFY_ROOT"
trap 'rm -rf "$TARGET"; rm -f "$RECEIPT_TMP"' EXIT HUP INT TERM
tar -xzf "$ARCHIVE" -C "$TARGET"
"$TARGET/bin/conda-unpack"
"$TARGET/bin/python" -c 'import importlib.metadata,torch,numpy,sourmash,yaml; assert torch.__version__ == "2.5.1+cu124"; print(torch.__version__,torch.version.cuda,numpy.__version__,importlib.metadata.version("sourmash"))'
ARCHIVE_SHA=$(sha256sum "$ARCHIVE" | awk '{print $1}')
"$TARGET/bin/python" -c 'import json,sys; from pathlib import Path; out=Path(sys.argv[1]); payload={"schema_version":"1.0","state":"PASS","archive_sha256":sys.argv[2],"torch_version":"2.5.1+cu124","cuda":"12.4","relocation":"conda-unpack"}; out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")' "$RECEIPT_TMP" "$ARCHIVE_SHA"
mv "$RECEIPT_TMP" "$RECEIPT"
rm -rf "$TARGET"
trap - EXIT HUP INT TERM
cat "$RECEIPT"
