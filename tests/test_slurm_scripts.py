import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_inventory_sbatch_is_posix_portable_and_requires_runtime_roots() -> None:
    path = PROJECT_ROOT / "scripts" / "slurm" / "inventory_raw_data.sbatch"
    text = path.read_text()
    assert text.startswith("#!/bin/sh\n")
    assert "#!/bin/bash" not in text
    assert "pipefail" not in text
    assert re.search(r"/(?:home|Users)/", text) is None
    assert "PROJECT_ROOT=${PROJECT_ROOT:?" in text
    assert "DATA_ROOT=${DATA_ROOT:?" in text
    assert "OUTPUT_ROOT=${OUTPUT_ROOT:?" in text
    assert "PYTHON_BIN=${PYTHON_BIN:?" in text
    assert "scripts/inventory_raw_data.py" in text
    assert "#SBATCH -p fat" in text


def test_inventory_submitter_sets_absolute_log_paths_without_hardcoded_host() -> None:
    path = PROJECT_ROOT / "scripts" / "submit_raw_inventory.sh"
    text = path.read_text()
    assert text.startswith("#!/bin/sh\n")
    assert re.search(r"/(?:home|Users)/", text) is None
    assert re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text) is None
    assert "--output=" in text
    assert "--error=" in text
    assert '--export="ALL,PROJECT_ROOT=' in text
    assert "scripts/slurm/inventory_raw_data.sbatch" in text


def test_fasta_qc_sbatch_is_six_way_posix_array() -> None:
    path = PROJECT_ROOT / "scripts" / "slurm" / "audit_fasta_shards.sbatch"
    text = path.read_text()
    assert text.startswith("#!/bin/sh\n")
    assert "#SBATCH --array=0-5%6" in text
    assert "PROJECT_ROOT=${PROJECT_ROOT:?" in text
    assert "DATA_ROOT=${DATA_ROOT:?" in text
    assert "SHARD_ROOT=${SHARD_ROOT:?" in text
    assert "OUTPUT_ROOT=${OUTPUT_ROOT:?" in text
    assert "PYTHON_BIN=${PYTHON_BIN:?" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert re.search(r"/(?:home|Users)/", text) is None
    assert re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text) is None


def test_fasta_qc_submitter_binds_array_logs_and_runtime_roots() -> None:
    path = PROJECT_ROOT / "scripts" / "submit_fasta_qc.sh"
    text = path.read_text()
    assert text.startswith("#!/bin/sh\nset -eu\n")
    assert "--output=" in text
    assert "%A_%a" in text
    assert "--error=" in text
    assert '--export="ALL,PROJECT_ROOT=' in text
    assert "DATA_ROOT=" in text
    assert "SHARD_ROOT=" in text
    assert "OUTPUT_ROOT=" in text
    assert re.search(r"/(?:home|Users)/", text) is None
    assert re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text) is None


def test_annotation_qc_slurm_chain_is_posix_portable() -> None:
    sbatch_path = PROJECT_ROOT / "scripts" / "slurm" / "audit_annotation_shards.sbatch"
    submit_path = PROJECT_ROOT / "scripts" / "submit_annotation_qc.sh"
    sbatch_text = sbatch_path.read_text()
    submit_text = submit_path.read_text()

    assert sbatch_text.startswith("#!/bin/sh\n")
    assert "#SBATCH --array=0-3%4" in sbatch_text
    assert "PROJECT_ROOT=${PROJECT_ROOT:?" in sbatch_text
    assert "DATA_ROOT=${DATA_ROOT:?" in sbatch_text
    assert "SHARD_ROOT=${SHARD_ROOT:?" in sbatch_text
    assert "OUTPUT_ROOT=${OUTPUT_ROOT:?" in sbatch_text
    assert "SLURM_ARRAY_TASK_ID" in sbatch_text
    assert "audit_annotation_shard.py" in sbatch_text
    assert submit_text.startswith("#!/bin/sh\nset -eu\n")
    assert '--export="ALL,PROJECT_ROOT=' in submit_text
    assert "annotation_qc-%A_%a.out" in submit_text
    assert "scripts/slurm/audit_annotation_shards.sbatch" in submit_text
    combined = sbatch_text + submit_text
    assert re.search(r"/(?:home|Users)/", combined) is None
    assert re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", combined) is None


def test_zip_audit_slurm_chain_is_posix_portable() -> None:
    sbatch_path = PROJECT_ROOT / "scripts" / "slurm" / "audit_zip_archives.sbatch"
    submit_path = PROJECT_ROOT / "scripts" / "submit_zip_audit.sh"
    sbatch_text = sbatch_path.read_text()
    submit_text = submit_path.read_text()

    assert sbatch_text.startswith("#!/bin/sh\n")
    assert "PROJECT_ROOT=${PROJECT_ROOT:?" in sbatch_text
    assert "DATA_ROOT=${DATA_ROOT:?" in sbatch_text
    assert "INVENTORY=${INVENTORY:?" in sbatch_text
    assert "RESULT_ROOT=${RESULT_ROOT:?" in sbatch_text
    assert "OUTPUT_ROOT=${OUTPUT_ROOT:?" in sbatch_text
    assert "audit_zip_archives.py" in sbatch_text
    assert submit_text.startswith("#!/bin/sh\nset -eu\n")
    assert '--export="ALL,PROJECT_ROOT=' in submit_text
    assert "zip_audit-%j.out" in submit_text
    assert "scripts/slurm/audit_zip_archives.sbatch" in submit_text
    combined = sbatch_text + submit_text
    assert re.search(r"/(?:home|Users)/", combined) is None
    assert re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", combined) is None


def test_archive_genome_qc_slurm_chain_uses_ordinary_non_cu_partitions() -> None:
    sbatch_path = PROJECT_ROOT / "scripts" / "slurm" / "audit_archive_genomes.sbatch"
    submit_path = PROJECT_ROOT / "scripts" / "submit_archive_genome_qc.sh"
    sbatch_text = sbatch_path.read_text()
    submit_text = submit_path.read_text()

    assert sbatch_text.startswith("#!/bin/sh\n")
    assert "PROJECT_ROOT=${PROJECT_ROOT:?" in sbatch_text
    assert "DATA_ROOT=${DATA_ROOT:?" in sbatch_text
    assert "REGISTRY=${REGISTRY:?" in sbatch_text
    assert "OUTPUT_ROOT=${OUTPUT_ROOT:?" in sbatch_text
    assert "SLURM_ARRAY_TASK_ID" in sbatch_text
    assert "audit_archive_genome.py" in sbatch_text
    assert submit_text.startswith("#!/bin/sh\nset -eu\n")
    assert "PARTITIONS=${PARTITIONS:-q02,q03,q04,q05}" in submit_text
    assert "fat" not in submit_text
    assert "--array=" in submit_text
    assert "archive_genome_qc-%A_%a.out" in submit_text
    assert "scripts/slurm/audit_archive_genomes.sbatch" in submit_text
    combined = sbatch_text + submit_text
    assert re.search(r"/(?:home|Users)/", combined) is None
    assert re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", combined) is None
