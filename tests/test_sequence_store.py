from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from legumegenomefm.sequence_store import PackedSequenceStore, PackedSequenceStoreWriter


def test_packed_store_round_trip_and_callable_intervals(tmp_path: Path) -> None:
    writer = PackedSequenceStoreWriter(tmp_path / "store", {"candidate_id": "a" * 16, "source_sha256": "1" * 64})
    writer.start_contig("chr1")
    writer.add_bases(b"ACGTNNAC")
    writer.add_bases(b"GT")
    writer.start_contig("chr2")
    writer.add_bases(b"TTryACGT")
    result = writer.finalize()
    assert result.base_count == 18
    store = PackedSequenceStore(result.store_dir)
    assert store.decode(0, 4).tolist() == [0, 1, 2, 3]
    assert store.decode(6, 4).tolist() == [0, 1, 2, 3]
    assert store.decode(14, 4).tolist() == [0, 1, 2, 3]
    assert store.contigs == [
        {"name": "chr1", "offset": 0, "length": 10},
        {"name": "chr2", "offset": 10, "length": 8},
    ]
    assert store.callable_intervals == [
        {"contig_index": 0, "start": 0, "length": 4},
        {"contig_index": 0, "start": 6, "length": 4},
        {"contig_index": 1, "start": 10, "length": 2},
        {"contig_index": 1, "start": 14, "length": 4},
    ]
    assert (result.store_dir / "READY").read_text() == result.manifest_sha256 + "\n"


def test_store_writer_rejects_duplicate_or_missing_contig(tmp_path: Path) -> None:
    writer = PackedSequenceStoreWriter(tmp_path / "store", {"candidate_id": "b" * 16})
    try:
        writer.add_bases(b"ACGT")
    except ValueError as exc:
        assert "contig" in str(exc)
    else:
        raise AssertionError("bases before contig were accepted")
    writer.start_contig("chr1")
    try:
        writer.start_contig("chr1")
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate contig was accepted")
    writer.abort()
    assert not (tmp_path / "store").exists()
