from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

from legumegenomefm.archive_sequence_audit import (
    ArchiveGenomeCandidate,
    audit_archive_genome,
    build_archive_genome_candidates,
)


def write_tsv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def test_build_archive_genome_candidates_selects_only_verified_safe_genomes(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.tsv"
    archive_qc = tmp_path / "archive_qc.tsv"
    members = tmp_path / "archive_members.tsv"
    genome_path = "soyod/M/genome/M.genome.zip"
    gff_path = "soyod/M/gff/M.gff.zip"
    write_tsv(
        inventory,
        ["relative_path", "kind", "file_type", "size_bytes", "mtime_ns"],
        [[genome_path, "file", "archive", 100, 11], [gff_path, "file", "archive", 50, 12]],
    )
    write_tsv(
        archive_qc,
        ["candidate_id", "relative_path", "size_bytes", "file_sha256", "status"],
        [["a", genome_path, 100, "1" * 64, "PASS"], ["b", gff_path, 50, "2" * 64, "PASS"]],
    )
    write_tsv(
        members,
        [
            "archive_candidate_id",
            "archive_relative_path",
            "member_name",
            "member_type",
            "uncompressed_bytes",
            "crc32_hex",
            "safe_path",
            "crc_verified",
            "status",
        ],
        [
            ["a", genome_path, "M.genome.fasta", "fasta", 12, "1234abcd", "true", "true", "PASS"],
            ["b", gff_path, "M.gff3", "annotation", 20, "2345abcd", "true", "true", "PASS"],
        ],
    )

    candidates = build_archive_genome_candidates(inventory, archive_qc, members)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.archive_relative_path == genome_path
    assert candidate.member_name == "M.genome.fasta"
    assert candidate.material == "M"
    assert candidate.member_uncompressed_bytes == 12
    assert len(candidate.candidate_id) == 16


def test_audit_archive_genome_scans_member_and_resumes(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    archive_path = data_root / "soyod" / "M" / "genome" / "M.genome.zip"
    archive_path.parent.mkdir(parents=True)
    member_name = "M.genome.fasta"
    member_payload = b">chr1\nACGTN\n>chr2\nrryy\n"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, member_payload)
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member_name)
    stat_result = archive_path.stat()
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    candidate = ArchiveGenomeCandidate(
        candidate_id="a" * 16,
        archive_relative_path="soyod/M/genome/M.genome.zip",
        member_name=member_name,
        material="M",
        archive_size_bytes=stat_result.st_size,
        archive_mtime_ns=stat_result.st_mtime_ns,
        archive_sha256=archive_sha,
        member_uncompressed_bytes=len(member_payload),
        member_crc32_hex=f"{info.CRC:08x}",
    )
    output_dir = tmp_path / "results"
    temporary_dir = tmp_path / "temporary"
    temporary_dir.mkdir()

    first = audit_archive_genome(
        data_root,
        candidate,
        output_dir,
        "f" * 64,
        temporary_dir=temporary_dir,
    )
    second = audit_archive_genome(
        data_root,
        candidate,
        output_dir,
        "f" * 64,
        temporary_dir=temporary_dir,
    )

    assert not first.reused
    assert second.reused
    payload = json.loads(first.result_json.read_text())
    assert payload["state"] == "PASS"
    assert payload["stats"]["sequence_count"] == 2
    assert payload["stats"]["total_symbols"] == 9
    assert payload["stats"]["iupac_ambiguous_count"] == 4
    assert len(payload["stats"]["canonical_sequence_sha256"]) == 64
    assert not list(temporary_dir.iterdir())
