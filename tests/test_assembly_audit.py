from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from legumegenomefm.assembly_audit import (
    AssemblyCandidate,
    audit_candidate,
    build_assembly_candidates,
    lpt_shards,
    scan_fasta,
    write_assembly_registry,
    write_shard_manifests,
)


def write_inventory(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["relative_path", "kind", "file_type", "size_bytes", "mtime_ns", "link_target", "mode_octal"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def record(path: str, *, kind: str = "file", size: int = 100) -> dict[str, object]:
    return {
        "relative_path": path,
        "kind": kind,
        "file_type": "fasta",
        "size_bytes": size,
        "mtime_ns": 123,
        "link_target": "" if kind == "file" else "target.fna.gz",
        "mode_octal": "-rw-r--r--" if kind == "file" else "lrwxrwxrwx",
    }


def test_build_assembly_candidates_excludes_non_genome_and_symlinks(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.tsv"
    write_inventory(
        inventory,
        [
            record("legume_family/legumeinfo/Medicago/truncatula/A17.gnm1/genome/medtr.genome_main.fna.gz", size=10),
            record("legume_family/legumeinfo/Medicago/truncatula/A17.gnm1/genome/medtr.genome_softmasked.fna.gz", size=12),
            record("legume_family/legumeinfo/Medicago/truncatula/A17.ann1/cds/medtr.cds.fna.gz", size=3),
            record("legume_family/ncbi/Glycine_max/ASM1/genome/GCF_1_genomic.fna.gz", size=20),
            record("legumeinfo/Wm82.gnm4/genome/glyma.genome_main.fna.gz", size=15),
            record("soyomics/Zhonghuang_13/genome/ZH13.v2.fasta.gz", size=11),
            record("legume_family/legumeinfo/Glycine/max/Wm82/genome/link.genome_main.fna.gz", kind="symlink"),
        ],
    )

    candidates = build_assembly_candidates(inventory)

    assert len(candidates) == 5
    assert [candidate.relative_path for candidate in candidates] == sorted(
        candidate.relative_path for candidate in candidates
    )
    by_path = {candidate.relative_path: candidate for candidate in candidates}
    legume = by_path[
        "legume_family/legumeinfo/Medicago/truncatula/A17.gnm1/genome/medtr.genome_main.fna.gz"
    ]
    assert legume.source == "legume_family_legumeinfo"
    assert legume.genus == "Medicago"
    assert legume.species == "Medicago truncatula"
    assert legume.assembly_label == "A17.gnm1"
    assert legume.genome_role == "main"
    assert by_path[
        "legume_family/legumeinfo/Medicago/truncatula/A17.gnm1/genome/medtr.genome_softmasked.fna.gz"
    ].genome_role == "softmasked"
    ncbi = by_path["legume_family/ncbi/Glycine_max/ASM1/genome/GCF_1_genomic.fna.gz"]
    assert ncbi.source == "legume_family_ncbi"
    assert ncbi.species == "Glycine max"
    assert ncbi.assembly_label == "ASM1"
    assert ncbi.genome_role == "ncbi_genomic"
    assert by_path["legumeinfo/Wm82.gnm4/genome/glyma.genome_main.fna.gz"].species == ""
    assert by_path["soyomics/Zhonghuang_13/genome/ZH13.v2.fasta.gz"].source == "soyomics"


def test_lpt_shards_are_deterministic_and_balance_largest_first() -> None:
    candidates = [
        AssemblyCandidate(str(i), f"p{i}", "s", "", "", "", "main", size, 1)
        for i, size in enumerate([9, 8, 7, 6, 5])
    ]

    first = lpt_shards(candidates, shard_count=2)
    second = lpt_shards(list(reversed(candidates)), shard_count=2)

    assert first == second
    assert [sum(item.size_bytes for item in shard) for shard in first] == [20, 15]
    assert sorted(item.candidate_id for shard in first for item in shard) == ["0", "1", "2", "3", "4"]


def test_write_assembly_registry_has_no_trailing_whitespace(tmp_path: Path) -> None:
    candidate = AssemblyCandidate("abc", "source/genome/a.fna.gz", "source", "Glycine", "Glycine max", "A", "main", 10, 5)

    result = write_assembly_registry([candidate], tmp_path)

    text = result.registry_tsv.read_text()
    assert len(text.splitlines()) == 2
    assert all(not line.endswith(("\t", " ")) for line in text.splitlines())
    assert result.candidate_count == 1
    assert len(result.registry_sha256) == 64


def test_scan_fasta_streams_plain_and_counts_symbols(tmp_path: Path) -> None:
    path = tmp_path / "sample.fna"
    path.write_text(">chr1 description\nACGTNacgt\n>chr2\nGGrr-X\n")

    stats = scan_fasta(path)

    assert stats.sequence_count == 2
    assert stats.total_symbols == 15
    assert stats.acgt_count == 10
    assert stats.n_count == 1
    assert stats.iupac_ambiguous_count == 2
    assert stats.invalid_symbol_count == 2
    assert stats.lowercase_count == 6
    assert stats.gc_count == 6
    assert stats.min_sequence_length == 6
    assert stats.max_sequence_length == 9
    assert stats.n50 == 9
    assert stats.duplicate_header_count == 0
    assert stats.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_scan_fasta_gzip_matches_canonical_sequence_digest(tmp_path: Path) -> None:
    plain = tmp_path / "plain.fna"
    compressed = tmp_path / "wrapped.fna.gz"
    plain.write_text(">a\nacgt\nNN\n>b\nTGCA\n")
    with gzip.open(compressed, "wt") as handle:
        handle.write(">different-a\nACGTNN\n>different-b\ntgca\n")

    plain_stats = scan_fasta(plain)
    gzip_stats = scan_fasta(compressed)

    assert gzip_stats.compression == "gzip"
    assert gzip_stats.file_sha256 == hashlib.sha256(compressed.read_bytes()).hexdigest()
    assert gzip_stats.canonical_sequence_sha256 == plain_stats.canonical_sequence_sha256
    assert gzip_stats.total_symbols == plain_stats.total_symbols == 10


def test_scan_fasta_rejects_sequence_before_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.fna"
    path.write_text("ACGT\n>chr1\nACGT\n")

    with pytest.raises(ValueError, match="before first header"):
        scan_fasta(path)


def test_write_shard_manifests_partitions_every_candidate_once(tmp_path: Path) -> None:
    candidates = [
        AssemblyCandidate(str(i), f"genome/p{i}.fna", "s", "", "", "", "main", size, 1)
        for i, size in enumerate([9, 8, 7, 6, 5])
    ]

    result = write_shard_manifests(candidates, tmp_path, shard_count=2)

    assert result.shard_count == 2
    assert result.candidate_count == 5
    assert len(result.shard_paths) == 2
    observed: list[str] = []
    for path in result.shard_paths:
        text = path.read_text()
        assert all(not line.endswith(("\t", " ")) for line in text.splitlines())
        with path.open(newline="") as handle:
            observed.extend(row["candidate_id"] for row in csv.DictReader(handle, delimiter="\t"))
    assert sorted(observed) == ["0", "1", "2", "3", "4"]
    summary = json.loads(result.summary_json.read_text())
    assert summary["candidate_count"] == 5
    assert sum(item["candidate_count"] for item in summary["shards"]) == 5


def test_audit_candidate_is_confined_atomic_and_resumable(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    output_dir = tmp_path / "qc"
    path = data_root / "source" / "genome" / "a.fna"
    path.parent.mkdir(parents=True)
    path.write_text(">chr1\nACGTN\n")
    info = path.stat()
    candidate = AssemblyCandidate(
        "abc123",
        "source/genome/a.fna",
        "source",
        "",
        "",
        "A",
        "main",
        info.st_size,
        info.st_mtime_ns,
    )
    implementation_sha256 = "a" * 64

    first = audit_candidate(data_root, candidate, output_dir, implementation_sha256)
    second = audit_candidate(data_root, candidate, output_dir, implementation_sha256)

    assert not first.reused
    assert second.reused
    assert first.result_json == output_dir / "abc123.json"
    payload = json.loads(first.result_json.read_text())
    assert payload["state"] == "PASS"
    assert payload["candidate_id"] == "abc123"
    assert payload["relative_path"] == "source/genome/a.fna"
    assert payload["implementation_sha256"] == implementation_sha256
    assert payload["stats"]["total_symbols"] == 5
    assert not list(output_dir.glob("*.tmp*"))


def test_audit_candidate_rejects_path_escape(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    data_root.mkdir()
    candidate = AssemblyCandidate("bad", "../escape.fna", "s", "", "", "", "main", 1, 1)

    with pytest.raises(ValueError, match="confined"):
        audit_candidate(data_root, candidate, tmp_path / "qc", "b" * 64)
