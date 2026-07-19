from __future__ import annotations

import csv
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

from sourmash import load_file_as_signatures

from legumegenomefm.genome_sketch import (
    audit_genome_sketch,
    build_sketch_candidates,
)
from legumegenomefm.sequence_store import PackedSequenceStore


def _write_catalog(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _base(candidate_id: str, kind: str, relative: str, member: str, payload: bytes, container: bytes) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_kind": kind,
        "source": "test",
        "relative_path": relative,
        "member_name": member,
        "status": "PASS",
        "exact_representative": "True",
        "payload_size_bytes": len(payload),
        "container_size_bytes": len(container),
        "container_sha256": hashlib.sha256(container).hexdigest(),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "sequence_sha256": "1" * 64,
        "species": "Glycine max",
        "material_key": candidate_id,
        "n_count": 0,
        "total_symbols": 100,
    }


def test_sketch_is_reverse_complement_invariant_across_file_and_zip(tmp_path: Path) -> None:
    data = tmp_path / "raw"
    data.mkdir()
    sequence = ("ACGTTGCAAGTC" * 1000).encode()
    rc = sequence.translate(bytes.maketrans(b"ACGT", b"TGCA"))[::-1]
    plain_payload = b">chr1\n" + sequence + b"\n"
    plain = data / "a.fa"
    plain.write_bytes(plain_payload)
    archive = data / "b.zip"
    zip_payload = b">other\n" + rc + b"\n"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("b.fa", zip_payload)
    catalog = tmp_path / "catalog.tsv"
    _write_catalog(
        catalog,
        [
            _base("a" * 16, "file", "a.fa", "", plain_payload, plain_payload),
            _base("b" * 16, "zip_member", "b.zip", "b.fa", zip_payload, archive.read_bytes()),
        ],
    )
    candidates = build_sketch_candidates(catalog)
    store_root = tmp_path / "stores"
    first = audit_genome_sketch(
        data, candidates[0], tmp_path / "out", "2" * 64, ksize=31, scaled=1, store_root=store_root
    )
    second = audit_genome_sketch(
        data, candidates[1], tmp_path / "out", "2" * 64, ksize=31, scaled=1, store_root=store_root
    )
    signatures = [next(iter(load_file_as_signatures(str(result.signature_path)))) for result in (first, second)]
    assert signatures[0].minhash.hashes == signatures[1].minhash.hashes
    assert signatures[0].minhash.similarity(signatures[1].minhash) == 1.0
    store = PackedSequenceStore(store_root / candidates[0].candidate_id)
    assert store.base_count == len(sequence)
    assert store.decode(0, 4).tolist() == [0, 1, 2, 3]
    reused = audit_genome_sketch(
        data, candidates[0], tmp_path / "out", "2" * 64, ksize=31, scaled=1, store_root=store_root
    )
    assert reused.reused is True


def test_sketch_candidate_registry_uses_only_pass_representatives(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.tsv"
    row = _base("a" * 16, "file", "a.fa", "", b"x", b"x")
    not_rep = dict(row, candidate_id="b" * 16, exact_representative="False")
    failed = dict(row, candidate_id="c" * 16, status="FAIL")
    hardmasked = dict(
        row,
        candidate_id="d" * 16,
        relative_path="legumeinfo/A/genome/a_hardmasked.fna.gz",
    )
    high_n = dict(row, candidate_id="e" * 16, n_count=21, total_symbols=100)
    _write_catalog(catalog, [row, not_rep, failed, hardmasked, high_n])
    candidates = build_sketch_candidates(catalog)
    assert [candidate.candidate_id for candidate in candidates] == ["a" * 16]
