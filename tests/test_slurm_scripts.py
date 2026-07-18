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
