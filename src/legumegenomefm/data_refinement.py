from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath
import re
from typing import Iterable, Mapping
from urllib.parse import quote


_ZERO_QUALITY_FIELDS = (
    "duplicate_gene_id_count",
    "duplicate_transcript_id_count",
    "malformed_line_count",
    "invalid_coordinate_count",
    "invalid_strand_count",
    "invalid_phase_count",
)


def annotation_is_strict(row: Mapping[str, object]) -> bool:
    try:
        return (
            str(row.get("is_primary_gene_model", "")).lower() == "true"
            and row.get("status") == "PASS"
            and int(row.get("gene_count", 0)) > 0
            and all(int(row.get(field, 0)) == 0 for field in _ZERO_QUALITY_FIELDS)
        )
    except (TypeError, ValueError):
        return False


def audit_feature_coordinates(
    contig_lengths: Mapping[str, int],
    features: Iterable[tuple[str, int, int, str]],
) -> dict[str, object]:
    feature_count = 0
    gene_count = 0
    out_of_bounds = 0
    matched_seqids: set[str] = set()
    unknown_seqids: set[str] = set()
    for seqid, start, end, feature_type in features:
        feature_count += 1
        if feature_type.lower() == "gene":
            gene_count += 1
        contig_length = contig_lengths.get(seqid)
        if contig_length is None:
            unknown_seqids.add(seqid)
            continue
        matched_seqids.add(seqid)
        if start < 1 or end < start or end > int(contig_length):
            out_of_bounds += 1
    passed = feature_count > 0 and gene_count > 0 and not unknown_seqids and out_of_bounds == 0
    return {
        "status": "PASS" if passed else "FAIL",
        "feature_count": feature_count,
        "gene_count": gene_count,
        "matched_seqid_count": len(matched_seqids),
        "unknown_seqid_count": len(unknown_seqids),
        "unknown_seqids": sorted(unknown_seqids),
        "out_of_bounds_feature_count": out_of_bounds,
    }


def parse_gff_features(lines: Iterable[str]) -> Iterable[tuple[str, int, int, str]]:
    embedded_fasta = False
    for line_number, line in enumerate(lines, start=1):
        if embedded_fasta:
            continue
        stripped = line.rstrip("\r\n")
        if not stripped:
            continue
        if stripped.startswith("##FASTA"):
            embedded_fasta = True
            continue
        if stripped.startswith("#"):
            continue
        fields = stripped.split("\t")
        if len(fields) != 9:
            raise ValueError(f"malformed GFF row at line {line_number}")
        try:
            start = int(fields[3])
            end = int(fields[4])
        except ValueError as exc:
            raise ValueError(f"invalid GFF coordinates at line {line_number}") from exc
        yield fields[0], start, end, fields[2]


def legumeinfo_readme_url(relative_path: str, species: str) -> str:
    parts = PurePosixPath(relative_path).parts
    if len(parts) >= 6 and parts[:2] == ("legume_family", "legumeinfo"):
        genus, epithet, assembly = parts[2], parts[3], parts[4]
    elif len(parts) >= 4 and parts[0] == "legumeinfo":
        taxon = species.split()
        if len(taxon) < 2:
            raise ValueError(f"species is not binomial: {species}")
        genus, epithet, assembly = taxon[0], taxon[1], parts[1]
    else:
        raise ValueError(f"not a supported LegumeInfo path: {relative_path}")
    encoded = "/".join(quote(value, safe="._-") for value in (genus, epithet, assembly))
    filename = quote(f"README.{assembly}.yml", safe="._-")
    return f"https://data.legumeinfo.org/{encoded.split('/', 2)[0]}/{encoded.split('/', 2)[1]}/genomes/{encoded.split('/', 2)[2]}/{filename}"


def metadata_has_chromosome_evidence(metadata: Mapping[str, object]) -> bool:
    prefix = metadata.get("chromosome_prefix")
    if isinstance(prefix, str) and prefix.strip():
        return True
    if isinstance(prefix, list) and any(str(value).strip() for value in prefix):
        return True
    text = " ".join(
        str(metadata.get(field, ""))
        for field in ("synopsis", "description", "keywords", "publication_title")
    ).lower()
    terms = ("chromosome", "pseudomolecule", "telomere-to-telomere", "telomere to telomere", "t2t")
    return any(term in text for term in terms)


def metadata_license_allows_training(
    public_access_level: str,
    license_name: str,
    allowed_public_access_levels: Iterable[str],
    allowed_licenses: Iterable[str],
) -> bool:
    normalize = lambda value: str(value).strip().casefold()
    allowed_access = {normalize(value) for value in allowed_public_access_levels}
    allowed_license_names = {normalize(value) for value in allowed_licenses}
    return normalize(public_access_level) in allowed_access and normalize(license_name) in allowed_license_names


def taxon_name_matches(expected_species: str, observed_species: str, accepted_aliases: Iterable[str]) -> bool:
    normalize = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    observed = normalize(observed_species)
    return observed == normalize(expected_species) or observed in {normalize(alias) for alias in accepted_aliases}


def canonical_material_key(
    species: str,
    material_key: str,
    aliases_by_species: Mapping[str, Mapping[str, Iterable[str]]],
) -> str:
    aliases = aliases_by_species.get(species, {})
    owner: dict[str, str] = {}
    for canonical, values in aliases.items():
        for value in (canonical, *values):
            text = str(value)
            previous = owner.setdefault(text, str(canonical))
            if previous != canonical:
                raise ValueError(f"material alias {text!r} belongs to both {previous!r} and {canonical!r}")
    return owner.get(material_key, material_key)


def metadata_provenance_passes(
    metadata: Mapping[str, object],
    expected_species: str,
    accepted_aliases: Iterable[str] = (),
) -> bool:
    if not taxon_name_matches(
        expected_species,
        str(metadata.get("scientific_name", "")).strip(),
        accepted_aliases,
    ):
        return False
    if not metadata_has_chromosome_evidence(metadata):
        return False

    def present(fields: tuple[str, ...]) -> bool:
        return any(str(metadata.get(field, "")).strip() not in {"", ".", "None"} for field in fields)

    source_present = present(("source", "genbank_accession", "bioproject"))
    publication_present = present(("publication_doi", "publication_title", "citation"))
    return source_present or publication_present


def busco_gate_passes(
    annotation_complete_percent: float,
    genome_complete_percent: float,
    minimum_annotation_complete_percent: float,
    minimum_genome_complete_percent: float,
) -> bool:
    return (
        annotation_complete_percent >= minimum_annotation_complete_percent
        and genome_complete_percent >= minimum_genome_complete_percent
    )


def univec_hit_is_high_confidence(
    percent_identity: float,
    alignment_length: int,
    rules: Iterable[Mapping[str, object]],
) -> bool:
    return any(
        alignment_length >= int(rule["minimum_alignment_length"])
        and percent_identity >= float(rule["minimum_percent_identity"])
        for rule in rules
    )


def merge_inclusive_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((min(start, end), max(start, end)) for start, end in intervals)
    if not ordered:
        return []
    merged: list[tuple[int, int]] = []
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def merged_interval_bases(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in merge_inclusive_intervals(intervals))


def merge_half_open_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = []
    for start, end in intervals:
        if end < start:
            raise ValueError(f"invalid half-open interval: {(start, end)}")
        if end > start:
            ordered.append((start, end))
    ordered.sort()
    if not ordered:
        return []
    merged: list[tuple[int, int]] = []
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def subtract_half_open_intervals(
    intervals: Iterable[tuple[int, int]],
    masks: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    source = merge_half_open_intervals(intervals)
    blocked = merge_half_open_intervals(masks)
    result: list[tuple[int, int]] = []
    mask_index = 0
    for start, end in source:
        while mask_index < len(blocked) and blocked[mask_index][1] <= start:
            mask_index += 1
        cursor = start
        index = mask_index
        while index < len(blocked) and blocked[index][0] < end:
            mask_start, mask_end = blocked[index]
            if mask_start > cursor:
                result.append((cursor, min(mask_start, end)))
            cursor = max(cursor, mask_end)
            if cursor >= end:
                break
            index += 1
        if cursor < end:
            result.append((cursor, end))
    return result


def record_is_primary_nuclear(
    sequence_name: str,
    length: int,
    official_sequence_role: str,
    official_location_type: str,
    proxy_minimum_length: int,
) -> bool:
    lowered = sequence_name.lower()
    if any(term in lowered for term in ("mitochond", "chloroplast", "plastid")):
        return False
    if re.search(r"(^|[._-])(mt|cp|pt)([._-]|$)", lowered):
        return False
    role = official_sequence_role.strip().lower()
    location = official_location_type.strip().lower()
    if role not in {"", "."} or location not in {"", "."}:
        return role == "assembled-molecule" and location in {"chromosome", "linkage group"}
    return length >= proxy_minimum_length


def eligible_tiara_chunks(
    sequence: str,
    chunk_length: int,
    minimum_length: int,
    minimum_acgt_fraction: float,
) -> list[tuple[int, int, str]]:
    if chunk_length < 1 or minimum_length < 1:
        raise ValueError("chunk and minimum lengths must be positive")
    chunks: list[tuple[int, int, str]] = []
    for start in range(0, len(sequence), chunk_length):
        chunk = sequence[start : start + chunk_length].upper()
        if len(chunk) < minimum_length:
            continue
        acgt = sum(chunk.count(base) for base in "ACGT")
        if acgt / len(chunk) < minimum_acgt_fraction:
            continue
        chunks.append((start, start + len(chunk), chunk))
    return chunks


def classify_assembly(
    official_level: str,
    t2t_label: bool,
    n50: int,
    large_sequence_fraction: float,
    proxy_minimum_n50: int,
    proxy_minimum_large_sequence_fraction: float,
) -> tuple[str, int, bool]:
    if official_level == "Complete Genome":
        return "complete_genome", 4, False
    if official_level == "Chromosome" and t2t_label:
        return "chromosome_official_t2t", 5, False
    if official_level == "Chromosome":
        return "chromosome_official", 3, False
    if t2t_label:
        return "t2t_label", 5, True
    if n50 >= proxy_minimum_n50 and large_sequence_fraction >= proxy_minimum_large_sequence_fraction:
        return "structural_proxy", 2, True
    return "insufficient", 0, True


def nonoverlapping_capacity(intervals: Iterable[Mapping[str, object]], context_length: int) -> int:
    if context_length < 1:
        raise ValueError("context_length must be positive")
    return sum(int(interval["length"]) // context_length for interval in intervals)


def _quality_rank(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["assembly_tier"]),
        float(row["long_callable_fraction"]),
        -float(row["n_fraction"]),
        float(row["large_sequence_fraction"]),
        int(row["n50"]),
        -int(row["contig_count"]),
        int(row["base_count"]),
        str(row["candidate_id"]),
    )


def select_unique_candidates(rows: Iterable[Mapping[str, object]]) -> dict[str, str]:
    materialized = list(rows)
    status = {
        str(row["candidate_id"]): "REJECTED_HARD_GATE"
        for row in materialized
    }
    by_orientation: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in materialized:
        if bool(row["hard_gate_pass"]):
            by_orientation[str(row["orientation_group_id"])].append(row)
    orientation_winners: list[Mapping[str, object]] = []
    for members in by_orientation.values():
        winner = max(members, key=_quality_rank)
        orientation_winners.append(winner)
        for row in members:
            status[str(row["candidate_id"])] = (
                "ORIENTATION_WINNER"
                if row is winner
                else "REJECTED_ORIENTATION_ALTERNATIVE"
            )
    by_material: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in orientation_winners:
        by_material[(str(row["species"]), str(row["material_key"]))].append(row)
    for members in by_material.values():
        winner = max(members, key=_quality_rank)
        for row in members:
            status[str(row["candidate_id"])] = (
                "SELECTED"
                if row is winner
                else "REJECTED_MATERIAL_ALTERNATIVE"
            )
    return status
