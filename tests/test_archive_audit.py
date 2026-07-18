from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from legumegenomefm.archive_audit import classify_member, scan_zip_archive, zip_member_is_safe


def test_classify_member_recognizes_sequence_and_annotation_assets() -> None:
    assert classify_member("x/genome.fna") == "fasta"
    assert classify_member("x/genes.gff3.gz") == "annotation"
    assert classify_member("x/proteins.faa") == "protein_fasta"
    assert classify_member("x/data.tsv") == "table"
    assert classify_member("x/README.txt") == "other"


def test_zip_member_is_safe_rejects_absolute_and_parent_paths() -> None:
    assert zip_member_is_safe("folder/file.fna")
    assert not zip_member_is_safe("../escape.fna")
    assert not zip_member_is_safe("/absolute/file.fna")
    assert not zip_member_is_safe("folder/../../escape.fna")


def test_scan_zip_archive_hashes_validates_crc_and_reports_members(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("genome/sample.fna", ">chr1\nACGT\n")
        handle.writestr("gff/sample.gff3", "##gff-version 3\n")
        handle.writestr("../unsafe.txt", "x")

    result = scan_zip_archive(archive, verify_crc=True)

    assert result.file_sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert result.member_count == 3
    assert result.file_member_count == 3
    assert result.crc_verified_count == 3
    assert result.crc_failure_count == 0
    assert result.unsafe_member_count == 1
    assert result.encrypted_member_count == 0
    assert [member.member_name for member in result.members] == [
        "../unsafe.txt",
        "genome/sample.fna",
        "gff/sample.gff3",
    ]
    assert result.members[1].member_type == "fasta"
    assert result.members[1].crc_verified


def test_scan_zip_archive_rejects_truncated_zip(tmp_path: Path) -> None:
    archive = tmp_path / "truncated.zip"
    archive.write_bytes(b"PK\x03\x04broken")

    with pytest.raises(zipfile.BadZipFile):
        scan_zip_archive(archive, verify_crc=True)
