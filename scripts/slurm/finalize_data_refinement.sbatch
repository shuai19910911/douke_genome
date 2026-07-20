#!/bin/sh
#SBATCH -J soy_refine_final
#SBATCH -p q02,q03,q04,q05
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00

set -eu
PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
TASKS=${TASKS:?TASKS is required}
CONFIG=${CONFIG:?CONFIG is required}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON_BIN" -B scripts/verify_busco_lineage.py \
  --project-root "$PROJECT_ROOT" \
  --lineage data/reference/busco/lineages/eudicots_odb10 \
  --full

"$PYTHON_BIN" -B scripts/merge_busco_mode_shards.py \
  --tasks "$TASKS" \
  --combined-dir workspace/data_refinement_busco_shards \
  --protein-dir workspace/data_refinement_busco_protein_shards \
  --genome-dir workspace/data_refinement_busco_genome_shards \
  --lineage-ready data_manifests/busco_lineage_eudicots_odb10.READY

"$PYTHON_BIN" -B scripts/aggregate_busco_results.py \
  --tasks "$TASKS" \
  --shard-dir workspace/data_refinement_busco_shards \
  --config "$CONFIG" \
  --output data_manifests/data_refinement_busco.tsv

"$PYTHON_BIN" -B scripts/aggregate_record_qc.py \
  --project-root "$PROJECT_ROOT" \
  --tasks "$TASKS" \
  --candidates data_manifests/data_refinement_candidates.tsv \
  --contamination-shards workspace/data_refinement_contamination_shards \
  --config "$CONFIG" \
  --output-prefix data_manifests/data_refinement_record_qc

"$PYTHON_BIN" -B scripts/finalize_data_refinement.py \
  --project-root "$PROJECT_ROOT" \
  --candidates data_manifests/data_refinement_candidates.tsv \
  --closure data_manifests/data_refinement_annotation_closure.tsv \
  --provenance data_manifests/data_refinement_source_provenance.tsv \
  --busco data_manifests/data_refinement_busco.tsv \
  --record-qc-candidates data_manifests/data_refinement_record_qc.candidates.tsv \
  --record-qc-records data_manifests/data_refinement_record_qc.records.tsv \
  --record-qc-masks data_manifests/data_refinement_record_qc.masks.tsv \
  --config "$CONFIG" \
  --output-prefix data_manifests/data_refinement_final
