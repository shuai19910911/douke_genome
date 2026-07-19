from __future__ import annotations

import bisect
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from legumegenomefm.sequence_store import PackedSequenceStore


@dataclass(frozen=True)
class OrientationIdentityResult:
    assembly_signature: str
    total_symbols: int
    contig_count: int


def assign_orientation_groups(items: list[tuple[str, str]]) -> list[dict[str, object]]:
    by_signature: dict[str, list[str]] = defaultdict(list)
    for candidate_id, signature in items:
        by_signature[signature].append(candidate_id)
    rows: list[dict[str, object]] = []
    for signature, candidate_ids in sorted(by_signature.items()):
        ordered = sorted(candidate_ids)
        group_id = f"orientation-{signature[:16]}"
        for candidate_id in ordered:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "orientation_signature": signature,
                    "orientation_group_id": group_id,
                    "orientation_group_size": len(ordered),
                    "orientation_representative": candidate_id == ordered[0],
                }
            )
    return sorted(rows, key=lambda row: str(row["candidate_id"]))


def _normalized_codes(
    store: PackedSequenceStore,
    start: int,
    length: int,
    interval_starts: list[int],
    interval_ends: list[int],
) -> np.ndarray:
    codes = store.decode(start, length)
    end = start + length
    valid = np.zeros(length, dtype=bool)
    index = bisect.bisect_right(interval_ends, start)
    while index < len(interval_starts) and interval_starts[index] < end:
        overlap_start = max(start, interval_starts[index])
        overlap_end = min(end, interval_ends[index])
        if overlap_end > overlap_start:
            valid[overlap_start - start : overlap_end - start] = True
        index += 1
    codes[~valid] = 4
    return codes


def _contig_hashes(
    store: PackedSequenceStore,
    contig: dict[str, object],
    intervals: list[dict[str, object]],
    block_size: int,
) -> tuple[bytes, bytes]:
    contig_start = int(contig["offset"])
    contig_length = int(contig["length"])
    contig_end = contig_start + contig_length
    starts = [int(interval["start"]) for interval in intervals]
    ends = [int(interval["start"]) + int(interval["length"]) for interval in intervals]
    forward = hashlib.sha256()
    for start in range(contig_start, contig_end, block_size):
        length = min(block_size, contig_end - start)
        forward.update(_normalized_codes(store, start, length, starts, ends).tobytes())
    reverse = hashlib.sha256()
    block_end = contig_end
    while block_end > contig_start:
        start = max(contig_start, block_end - block_size)
        codes = _normalized_codes(store, start, block_end - start, starts, ends)
        valid = codes < 4
        codes[valid] = 3 - codes[valid]
        reverse.update(codes[::-1].tobytes())
        block_end = start
    return forward.digest(), reverse.digest()


def orientation_invariant_store_signature(
    store_dir: Path,
    *,
    block_size: int = 8 * 1024 * 1024,
) -> OrientationIdentityResult:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    store = PackedSequenceStore(store_dir)
    intervals_by_contig: dict[int, list[dict[str, object]]] = {}
    for interval in store.callable_intervals:
        intervals_by_contig.setdefault(int(interval["contig_index"]), []).append(interval)
    records: list[bytes] = []
    total_symbols = 0
    for index, contig in enumerate(store.contigs):
        length = int(contig["length"])
        forward, reverse = _contig_hashes(
            store,
            contig,
            intervals_by_contig.get(index, []),
            block_size,
        )
        canonical = min(forward, reverse)
        records.append(length.to_bytes(8, "big") + canonical)
        total_symbols += length
    digest = hashlib.sha256(b"legumegenomefm-orientation-multiset-v1\0")
    for record in sorted(records):
        digest.update(record)
    if total_symbols != store.base_count:
        raise ValueError("contig lengths do not sum to store base count")
    return OrientationIdentityResult(
        assembly_signature=digest.hexdigest(),
        total_symbols=total_symbols,
        contig_count=len(store.contigs),
    )
