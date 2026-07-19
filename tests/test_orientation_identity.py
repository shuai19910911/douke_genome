from __future__ import annotations

from pathlib import Path

from legumegenomefm.orientation_identity import assign_orientation_groups, orientation_invariant_store_signature
from legumegenomefm.sequence_store import PackedSequenceStoreWriter


def _write_store(path: Path, contigs: list[tuple[str, bytes]]) -> None:
    writer = PackedSequenceStoreWriter(path, {"candidate_id": path.name})
    for name, sequence in contigs:
        writer.start_contig(name)
        writer.add_bases(sequence)
    writer.finalize()


def _reverse_complement(sequence: bytes) -> bytes:
    return sequence.translate(bytes.maketrans(b"ACGTN", b"TGCAN"))[::-1]


def test_orientation_signature_ignores_contig_order_names_and_orientation(tmp_path: Path) -> None:
    first = [("chr1", b"ACGTNNACGTTT"), ("chr2", b"TTGCAACG")]
    second = [("renamed2", _reverse_complement(first[1][1])), ("renamed1", _reverse_complement(first[0][1]))]
    _write_store(tmp_path / "a", first)
    _write_store(tmp_path / "b", second)
    left = orientation_invariant_store_signature(tmp_path / "a", block_size=5)
    right = orientation_invariant_store_signature(tmp_path / "b", block_size=7)
    assert left.assembly_signature == right.assembly_signature
    assert left.total_symbols == right.total_symbols == 20
    assert left.contig_count == right.contig_count == 2


def test_orientation_signature_changes_for_real_sequence_difference(tmp_path: Path) -> None:
    _write_store(tmp_path / "a", [("chr1", b"ACGTACGT")])
    _write_store(tmp_path / "b", [("chr1", b"ACGTACGA")])
    assert (
        orientation_invariant_store_signature(tmp_path / "a").assembly_signature
        != orientation_invariant_store_signature(tmp_path / "b").assembly_signature
    )


def test_orientation_groups_choose_one_deterministic_representative() -> None:
    rows = assign_orientation_groups(
        [
            ("candidate-b", "a" * 64),
            ("candidate-a", "a" * 64),
            ("candidate-c", "b" * 64),
        ]
    )
    by_id = {row["candidate_id"]: row for row in rows}
    assert by_id["candidate-a"]["orientation_group_size"] == 2
    assert by_id["candidate-a"]["orientation_representative"] is True
    assert by_id["candidate-b"]["orientation_representative"] is False
    assert by_id["candidate-c"]["orientation_group_size"] == 1
