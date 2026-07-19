from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from sourmash import load_file_as_signatures

from legumegenomefm.genome_sketch import GenomeSketchCandidate, read_sketch_registry


@dataclass(frozen=True)
class GenomeSimilarityResult:
    pairs_path: Path
    clusters_path: Path
    summary_path: Path
    pair_rows: list[dict[str, object]]
    cluster_rows: list[dict[str, object]]
    summary: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _binomial(species: str) -> str:
    return " ".join(species.split()[:2])


def _validated_signatures(
    candidates: list[GenomeSketchCandidate], result_dir: Path
) -> tuple[dict[str, object], str, int, int]:
    expected_ids = {candidate.candidate_id for candidate in candidates}
    observed_json = {path.stem for path in result_dir.glob("*.json")}
    extras = sorted(observed_json - expected_ids)
    if extras:
        raise ValueError(f"unexpected genome sketch results: {','.join(extras)}")
    signatures: dict[str, object] = {}
    implementations: set[str] = set()
    ksizes: set[int] = set()
    scales: set[int] = set()
    for candidate in candidates:
        result_path = result_dir / f"{candidate.candidate_id}.json"
        signature_path = result_dir / f"{candidate.candidate_id}.sig.gz"
        if not result_path.is_file() or not signature_path.is_file():
            raise ValueError(f"missing genome sketch result: {candidate.candidate_id}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("state") != "PASS":
            raise ValueError(f"non-PASS genome sketch result: {candidate.candidate_id}")
        expected = asdict(candidate)
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError(f"genome sketch candidate identity mismatch: {candidate.candidate_id}")
        if payload.get("signature_sha256") != _sha256(signature_path):
            raise ValueError(f"genome sketch signature hash mismatch: {candidate.candidate_id}")
        implementations.add(payload.get("implementation_sha256", ""))
        ksizes.add(payload.get("ksize"))
        scales.add(payload.get("scaled"))
        loaded = list(load_file_as_signatures(str(signature_path)))
        if len(loaded) != 1:
            raise ValueError(f"signature file must contain one genome: {candidate.candidate_id}")
        signatures[candidate.candidate_id] = loaded[0]
    if len(implementations) != 1 or not next(iter(implementations), ""):
        raise ValueError("mixed genome sketch implementation hashes")
    if len(ksizes) != 1 or len(scales) != 1:
        raise ValueError("mixed genome sketch parameters")
    return signatures, next(iter(implementations)), next(iter(ksizes)), next(iter(scales))


def aggregate_genome_sketches(
    registry_path: Path,
    result_dir: Path,
    output_dir: Path,
    *,
    related_threshold: float = 0.80,
    near_duplicate_threshold: float = 0.95,
) -> GenomeSimilarityResult:
    if not 0 <= related_threshold <= near_duplicate_threshold <= 1:
        raise ValueError("invalid similarity thresholds")
    candidates = read_sketch_registry(registry_path)
    result_dir = Path(result_dir)
    signatures, implementation, ksize, scaled = _validated_signatures(candidates, result_dir)
    by_binomial: dict[str, list[GenomeSketchCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_binomial[_binomial(candidate.species)].append(candidate)

    parent = {candidate.candidate_id: candidate.candidate_id for candidate in candidates}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    pair_rows: list[dict[str, object]] = []
    maximum_similarity = {candidate.candidate_id: 0.0 for candidate in candidates}
    for binomial, group in sorted(by_binomial.items()):
        ordered = sorted(group, key=lambda item: item.candidate_id)
        for left_index, left in enumerate(ordered):
            left_minhash = signatures[left.candidate_id].minhash
            for right in ordered[left_index + 1 :]:
                similarity = float(left_minhash.similarity(signatures[right.candidate_id].minhash))
                maximum_similarity[left.candidate_id] = max(maximum_similarity[left.candidate_id], similarity)
                maximum_similarity[right.candidate_id] = max(maximum_similarity[right.candidate_id], similarity)
                if similarity < related_threshold:
                    continue
                classification = "near_duplicate" if similarity >= near_duplicate_threshold else "related"
                if classification == "near_duplicate":
                    union(left.candidate_id, right.candidate_id)
                pair_rows.append(
                    {
                        "left_candidate_id": left.candidate_id,
                        "right_candidate_id": right.candidate_id,
                        "binomial": binomial,
                        "left_material_key": left.material_key,
                        "right_material_key": right.material_key,
                        "jaccard": f"{similarity:.8f}",
                        "classification": classification,
                    }
                )

    components: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        components[find(candidate.candidate_id)].append(candidate.candidate_id)
    normalized_components = {
        min(members): sorted(members) for members in components.values()
    }
    component_for = {
        candidate_id: representative
        for representative, members in normalized_components.items()
        for candidate_id in members
    }
    cluster_rows: list[dict[str, object]] = []
    candidate_index = {candidate.candidate_id: candidate for candidate in candidates}
    for candidate_id in sorted(candidate_index):
        candidate = candidate_index[candidate_id]
        representative = component_for[candidate_id]
        members = normalized_components[representative]
        cluster_rows.append(
            {
                "candidate_id": candidate_id,
                "species": candidate.species,
                "material_key": candidate.material_key,
                "near_duplicate_group_id": f"near-{representative}",
                "near_duplicate_group_size": len(members),
                "near_duplicate_representative": candidate_id == representative,
                "max_within_binomial_jaccard": f"{maximum_similarity[candidate_id]:.8f}",
            }
        )
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "candidate_count": len(candidates),
        "binomial_group_count": len(by_binomial),
        "pair_count_at_or_above_related_threshold": len(pair_rows),
        "near_duplicate_pair_count": sum(row["classification"] == "near_duplicate" for row in pair_rows),
        "near_duplicate_group_count": sum(len(members) > 1 for members in normalized_components.values()),
        "near_duplicate_member_count": sum(len(members) for members in normalized_components.values() if len(members) > 1),
        "related_threshold": related_threshold,
        "near_duplicate_threshold": near_duplicate_threshold,
        "ksize": ksize,
        "scaled": scaled,
        "implementation_sha256": implementation,
    }
    output_dir = Path(output_dir)
    pairs_path = output_dir / "genome_similarity_pairs.tsv"
    clusters_path = output_dir / "genome_near_duplicate_clusters.tsv"
    summary_path = output_dir / "genome_similarity.summary.json"
    pair_fields = [
        "left_candidate_id", "right_candidate_id", "binomial", "left_material_key",
        "right_material_key", "jaccard", "classification",
    ]
    cluster_fields = [
        "candidate_id", "species", "material_key", "near_duplicate_group_id",
        "near_duplicate_group_size", "near_duplicate_representative", "max_within_binomial_jaccard",
    ]
    for path, rows, fieldnames in (
        (pairs_path, pair_rows, pair_fields),
        (clusters_path, cluster_rows, cluster_fields),
    ):
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        _atomic(path, text.getvalue().encode("utf-8"))
    _atomic(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return GenomeSimilarityResult(pairs_path, clusters_path, summary_path, pair_rows, cluster_rows, summary)
