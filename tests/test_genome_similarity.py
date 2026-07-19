from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

from legumegenomefm.genome_sketch import GenomeSketchCandidate, audit_genome_sketch, write_sketch_registry
from legumegenomefm.genome_similarity import aggregate_genome_sketches


def _fasta(path: Path, sequence: str) -> tuple[int, str]:
    payload = f">chr1\n{sequence}\n".encode()
    path.write_bytes(payload)
    return len(payload), hashlib.sha256(payload).hexdigest()


def test_similarity_aggregation_clusters_near_duplicates_across_taxon_labels(tmp_path: Path) -> None:
    data = tmp_path / "raw"
    data.mkdir()
    random.seed(7)
    sequence = "".join(random.choice("ACGT") for _ in range(20_000))
    paths = [data / name for name in ("a.fa", "b.fa", "c.fa")]
    identities = [_fasta(paths[0], sequence), _fasta(paths[1], sequence), _fasta(paths[2], sequence)]
    candidates = [
        GenomeSketchCandidate("a" * 16, "file", "a.fa", "", identities[0][0], identities[0][1], identities[0][1], "1" * 64, "Arachis hypogaea", "a"),
        GenomeSketchCandidate("b" * 16, "file", "b.fa", "", identities[1][0], identities[1][1], identities[1][1], "2" * 64, "Arachis hypogaea subsp. hypogaea", "b"),
        GenomeSketchCandidate("c" * 16, "file", "c.fa", "", identities[2][0], identities[2][1], identities[2][1], "3" * 64, "Glycine max", "c"),
    ]
    registry = write_sketch_registry(candidates, tmp_path / "registry")
    results = tmp_path / "results"
    for candidate in candidates:
        audit_genome_sketch(data, candidate, results, "4" * 64, ksize=31, scaled=100)
    aggregate = aggregate_genome_sketches(
        registry.registry_path,
        results,
        tmp_path / "summary",
        related_threshold=0.80,
        near_duplicate_threshold=0.95,
    )
    assert aggregate.summary["candidate_count"] == 3
    assert len(aggregate.summary["aggregator_implementation_sha256"]) == 64
    assert aggregate.summary["comparison_scope"] == "all_candidates"
    assert aggregate.summary["pairwise_comparison_count"] == 3
    assert aggregate.summary["near_duplicate_group_count"] == 1
    rows = aggregate.cluster_rows
    by_id = {row["candidate_id"]: row for row in rows}
    assert by_id["a" * 16]["near_duplicate_group_size"] == 3
    assert by_id["b" * 16]["near_duplicate_group_id"] == by_id["a" * 16]["near_duplicate_group_id"]
    assert by_id["c" * 16]["near_duplicate_group_id"] == by_id["a" * 16]["near_duplicate_group_id"]


def test_similarity_aggregation_rejects_result_identity_drift(tmp_path: Path) -> None:
    candidate = GenomeSketchCandidate("a" * 16, "file", "a.fa", "", 1, "1" * 64, "1" * 64, "2" * 64, "Glycine max", "a")
    registry = write_sketch_registry([candidate], tmp_path / "registry")
    results = tmp_path / "results"
    results.mkdir()
    signature = results / f"{candidate.candidate_id}.sig.gz"
    signature.write_bytes(b"not-read-before-identity-check")
    payload = {**asdict(candidate), "implementation_sha256": "3" * 64, "ksize": 31, "scaled": 10000}
    payload.update(
        {
            "schema_version": "1.0",
            "state": "PASS",
            "signature_sha256": hashlib.sha256(signature.read_bytes()).hexdigest(),
        }
    )
    payload["relative_path"] = "wrong.fa"
    (results / f"{candidate.candidate_id}.json").write_text(json.dumps(payload) + "\n")
    try:
        aggregate_genome_sketches(registry.registry_path, results, tmp_path / "out")
    except ValueError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("identity drift was accepted")
