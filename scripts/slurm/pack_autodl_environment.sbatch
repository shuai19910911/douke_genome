#!/bin/sh
#SBATCH -J lgfm_env_pack
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 04:00:00

set -eu
ENV_PREFIX=${ENV_PREFIX:?ENV_PREFIX is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
CONDA_PACK=${CONDA_PACK:?CONDA_PACK is required}
JOB_TAG=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
TMP="${OUTPUT}.${JOB_TAG}.tmp"
SHA_TMP="${OUTPUT}.sha256.${JOB_TAG}.tmp"
[ ! -e "$TMP" ] || { echo "stale temporary archive: $TMP" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"
trap 'rm -f "$TMP" "$SHA_TMP"' EXIT HUP INT TERM
"$CONDA_PACK" -p "$ENV_PREFIX" -o "$TMP" --format tar.gz --force
[ -s "$TMP" ] || { echo "empty environment archive" >&2; exit 2; }
sha256sum "$TMP" | awk '{print $1}' > "$SHA_TMP"
mv "$TMP" "$OUTPUT"
mv "$SHA_TMP" "$OUTPUT.sha256"
trap - EXIT HUP INT TERM
printf 'archive=%s\nsha256=%s\n' "$OUTPUT" "$(cat "$OUTPUT.sha256")"
