from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from dataclasses import fields
from pathlib import Path

import pytest

from legumegenomefm.archive_annotation_audit import (
    ArchiveAnnotationCandidate,
    aggregate_archive_annotation_audit,
    audit_archive_annotation,
    build_archive_annotation_candidates,
    write_archive_annotation_registry,
)


def _tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_archive_annotation_end_to_end_and_resume(tmp_path: Path) -> None:
    data = tmp_path / "raw"
    archive_rel = "soyod/Test/gff/Test.annotation.zip"
    archive = data / archive_rel
    archive.parent.mkdir(parents=True)
    member = "Test.annotation.gff"
    content = b"##gff-version 3\nchr1\tsrc\tgene\t1\t10\t.\t+\t.\tID=g1\n"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(member, content)
    stat = archive.stat()
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as handle:
        info = handle.getinfo(member)
    inventory = tmp_path / "inventory.tsv"
    archive_qc = tmp_path / "archive_qc.tsv"
    members = tmp_path / "members.tsv"
    genomes = tmp_path / "genomes.tsv"
    _tsv(inventory, [{"relative_path": archive_rel, "kind": "file", "file_type": "archive", "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}])
    _tsv(archive_qc, [{"relative_path": archive_rel, "size_bytes": stat.st_size, "file_sha256": archive_sha, "status": "PASS"}])
    _tsv(members, [{"archive_relative_path": archive_rel, "member_name": member, "member_type": "annotation", "uncompressed_bytes": info.file_size, "crc32_hex": f"{info.CRC:08x}", "safe_path": "true", "crc_verified": "true", "status": "PASS"}])
    _tsv(genomes, [{"candidate_id": "a" * 16, "material": "Test"}])

    candidates = build_archive_annotation_candidates(inventory, archive_qc, members, genomes)
    assert len(candidates) == 1
    assert candidates[0].paired_genome_ids == ("a" * 16,)
    registry = write_archive_annotation_registry(candidates, tmp_path / "registry")
    output = tmp_path / "results"
    first = audit_archive_annotation(data, candidates[0], output, "b" * 64, temporary_dir=tmp_path / "work")
    second = audit_archive_annotation(data, candidates[0], output, "b" * 64, temporary_dir=tmp_path / "work")
    assert first.reused is False
    assert second.reused is True
    payload = json.loads(first.result_path.read_text())
    assert payload["stats"]["gene_count"] == 1

    summary = aggregate_archive_annotation_audit(registry.registry_path, output)
    assert summary.summary["pass_count"] == 1
    assert summary.summary["gene_count"] == 1


def test_archive_annotation_aggregation_rejects_identity_drift(tmp_path: Path) -> None:
    candidate = ArchiveAnnotationCandidate(
        candidate_id="c" * 16,
        archive_relative_path="soyod/A/gff/a.zip",
        member_name="a.gff",
        material="A",
        archive_size_bytes=10,
        archive_mtime_ns=20,
        archive_sha256="1" * 64,
        member_uncompressed_bytes=30,
        member_crc32_hex="12345678",
        paired_genome_ids=(),
    )
    registry = write_archive_annotation_registry([candidate], tmp_path / "registry")
    results = tmp_path / "results"
    results.mkdir()
    identity = {field.name: getattr(candidate, field.name) for field in fields(ArchiveAnnotationCandidate)}
    identity["member_name"] = "wrong.gff"
    (results / f"{candidate.candidate_id}.json").write_text(
        json.dumps({"schema_version": "1.0", "state": "PASS", "implementation_sha256": "2" * 64, "candidate": identity, "stats": {}}) + "\n"
    )
    with pytest.raises(ValueError, match="identity"):
        aggregate_archive_annotation_audit(registry.registry_path, results)
