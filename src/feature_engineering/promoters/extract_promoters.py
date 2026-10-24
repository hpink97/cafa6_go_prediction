"""Extract fixed-width upstream promoter sequences for proteins in anchor species.

Preferred path: local bulk reference files resolve UniProt accessions to Ensembl
genes, load gene coordinates, and slice promoter windows from local genome FASTA.
When local reference data is unavailable or incomplete, the extractor falls back
to the live Ensembl REST path. Proteins outside the anchor species list, or that
fail to resolve at any step, are simply omitted (no promoter feature) -- callers
should treat missing rows as "no promoter data" rather than an error. This script
only extracts sequences; DNA-LM embeddings are a separate, later step.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from Bio import SeqIO
from requests.adapters import HTTPAdapter

from feature_engineering.promoters.ensembl_client import (
    fetch_all_genes,
    fetch_gene_coordinates,
    fetch_promoter_sequences,
    resolve_gene_id,
)
from feature_engineering.promoters.local_reference import (
    LocalPromoterReference,
    fetch_local_sequences,
    find_local_reference,
    load_all_local_gene_coordinates,
    load_local_gene_coordinates,
    load_local_gene_ids,
)
from feature_engineering.promoters.operons import DEFAULT_MAX_OPERON_GAP, resolve_operon_leaders
from feature_engineering.promoters.species_map import ANCHOR_SPECIES, OPERON_AWARE_SPECIES

LOGGER = logging.getLogger("cafa6.promoters")


def _make_session(pool_size: int) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def load_protein_taxa(path: str | Path) -> pd.DataFrame:
    """Load protein_id -> taxon_id, from either a taxonomy TSV (no header) or a FASTA with '>id taxon' headers."""
    path = Path(path)
    if path.suffix.lower() in (".fa", ".fasta"):
        rows = []
        for record in SeqIO.parse(str(path), "fasta"):
            parts = record.description.split()
            if len(parts) < 2:
                continue
            rows.append({"protein_id": parts[0], "taxon_id": parts[1]})
        return pd.DataFrame(rows)
    return pd.read_csv(path, sep="\t", header=None, names=["protein_id", "taxon_id"], dtype={"taxon_id": str})


def _promoter_region(record: dict, window: int) -> tuple[str, int, int, int] | None:
    """(seq_region_name, start, end, strand) of the `window`bp region immediately upstream of the gene's TSS."""
    seq_region_name = record.get("seq_region_name")
    strand = record.get("strand")
    gene_start = record.get("start")
    gene_end = record.get("end")
    if seq_region_name is None or strand not in (1, -1) or gene_start is None or gene_end is None:
        return None
    if strand == 1:
        end = gene_start - 1
        start = max(1, end - window + 1)
    else:
        start = gene_end + 1
        end = start + window - 1
    if end < start:
        return None
    return seq_region_name, start, end, strand


def _region_string(seq_region_name: str, start: int, end: int, strand: int) -> str:
    return f"{seq_region_name}:{start}-{end}:{strand}"


def _load_gene_id_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    try:
        cache = pd.read_csv(cache_path, sep="\t", dtype=str)
    except pd.errors.EmptyDataError:
        return {}
    if cache.empty:
        return {}
    cols = {"protein_id", "ensembl_gene_id"}
    if not cols.issubset(cache.columns):
        return {}
    cache = cache[["protein_id", "ensembl_gene_id"]].drop_duplicates(subset=["protein_id"])
    return dict(zip(cache["protein_id"], cache["ensembl_gene_id"]))


def _append_gene_id_cache(cache_path: Path, new_entries: dict[str, str]) -> None:
    """Persist newly resolved (protein_id, gene_id) pairs and deduplicate them on disk."""
    if not new_entries:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged = _load_gene_id_cache(cache_path)
    merged.update(new_entries)
    frame = pd.DataFrame(
        [{"protein_id": protein_id, "ensembl_gene_id": gene_id} for protein_id, gene_id in merged.items()]
    ).drop_duplicates(subset=["protein_id"])
    frame = frame[["protein_id", "ensembl_gene_id"]].sort_values("protein_id")
    frame.to_csv(cache_path, sep="\t", index=False)


def _load_sequence_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    try:
        cache = pd.read_csv(cache_path, sep="\t", dtype=str)
    except pd.errors.EmptyDataError:
        return {}
    if cache.empty:
        return {}
    columns = {"region", "sequence"}
    if not columns.issubset(cache.columns):
        return {}
    cache = cache[["region", "sequence"]].drop_duplicates(subset=["region"])
    return dict(zip(cache["region"], cache["sequence"]))


def _append_sequence_cache(cache_path: Path, new_entries: dict[str, str]) -> None:
    if not new_entries:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged = _load_sequence_cache(cache_path)
    merged.update(new_entries)
    frame = pd.DataFrame(
        [{"region": region, "sequence": seq} for region, seq in merged.items()]
    ).drop_duplicates(subset=["region"])
    frame = frame[["region", "sequence"]].sort_values("region")
    frame.to_csv(cache_path, sep="\t", index=False)


def _resolve_gene_ids_cached(
    protein_ids: list[str], species: str, n_workers: int, cache_path: Path, session: requests.Session
) -> dict[str, str]:
    """Resolve protein_id -> Ensembl gene_id via xrefs/symbol, reusing/extending an on-disk cache."""
    protein_ids = list(dict.fromkeys(protein_ids))
    gene_by_protein = _load_gene_id_cache(cache_path)
    to_resolve = [protein_id for protein_id in protein_ids if protein_id not in gene_by_protein]
    LOGGER.info(
        "[%s] %d gene IDs already cached, resolving %d more (rate-limited, ~5 req/s)",
        species,
        len(protein_ids) - len(to_resolve),
        len(to_resolve),
    )
    if not to_resolve:
        return gene_by_protein

    pending_cache_writes: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
        futures = {pool.submit(resolve_gene_id, protein_id, species, session): protein_id for protein_id in to_resolve}
        for index, future in enumerate(as_completed(futures), start=1):
            protein_id = futures[future]
            try:
                gene_id = future.result()
            except Exception:
                # A single protein's lookup should never take down an hours-long run; skip and move on.
                LOGGER.exception("[%s] unexpected failure resolving %s; treating as unresolved", species, protein_id)
                gene_id = None
            if gene_id is not None:
                gene_by_protein[protein_id] = gene_id
                pending_cache_writes[protein_id] = gene_id
            if index % 250 == 0:
                LOGGER.info("[%s] resolved %d/%d proteins so far", species, index, len(to_resolve))
                _append_gene_id_cache(cache_path, pending_cache_writes)
                pending_cache_writes = {}
    _append_gene_id_cache(cache_path, pending_cache_writes)
    LOGGER.info("[%s] resolved %d/%d proteins to gene IDs", species, len(gene_by_protein), len(protein_ids))
    return gene_by_protein


def extract_for_species(
    protein_ids: list[str],
    species: str,
    window: int,
    n_workers: int,
    cache_path: Path,
    sequence_cache_path: Path,
    local_reference: LocalPromoterReference | None = None,
    allow_rest_fallback: bool = True,
) -> pd.DataFrame:
    """Resolve gene IDs, coordinates, and promoter sequences for one species' proteins."""
    columns = [
        "protein_id",
        "species",
        "ensembl_gene_id",
        "seq_region_name",
        "promoter_start",
        "promoter_end",
        "strand",
        "promoter_length",
        "promoter_sequence",
    ]
    session = _make_session(max(10, n_workers))

    if local_reference is None and not allow_rest_fallback:
        LOGGER.info("[%s] skipping species because no local reference is available", species)
        return pd.DataFrame(columns=columns)

    if local_reference is not None:
        gene_by_protein = load_local_gene_ids(protein_ids, local_reference)
        unresolved_ids = [protein_id for protein_id in protein_ids if protein_id not in gene_by_protein]
        if unresolved_ids and allow_rest_fallback:
            LOGGER.info("[%s] falling back to REST for %d proteins missing from local xref", species, len(unresolved_ids))
            gene_by_protein.update(_resolve_gene_ids_cached(unresolved_ids, species, n_workers, cache_path, session))
        elif unresolved_ids:
            LOGGER.info("[%s] skipping %d proteins missing from local xref", species, len(unresolved_ids))
    else:
        gene_by_protein = _resolve_gene_ids_cached(protein_ids, species, n_workers, cache_path, session)

    unique_gene_ids = sorted(set(gene_by_protein.values()))
    coordinates = load_local_gene_coordinates(unique_gene_ids, local_reference) if local_reference is not None else {}
    missing_gene_ids = [gene_id for gene_id in unique_gene_ids if gene_id not in coordinates]
    if missing_gene_ids and allow_rest_fallback:
        coordinates.update(fetch_gene_coordinates(missing_gene_ids, species, session))
    elif missing_gene_ids:
        LOGGER.info("[%s] skipping %d genes missing from local coordinates", species, len(missing_gene_ids))
    LOGGER.info("[%s] fetched coordinates for %d/%d genes", species, len(coordinates), len(unique_gene_ids))

    regions_by_gene = {
        gene_id: region for gene_id, record in coordinates.items() if (region := _promoter_region(record, window)) is not None
    }
    region_strings = sorted({_region_string(*region) for region in regions_by_gene.values()})
    cached_sequences = _load_sequence_cache(sequence_cache_path)
    missing_regions = [region for region in region_strings if region not in cached_sequences]
    if missing_regions:
        if local_reference is not None:
            fetched = fetch_local_sequences([regions_by_gene[gene_id] for gene_id in regions_by_gene if _region_string(*regions_by_gene[gene_id]) in missing_regions], local_reference)
            rest_missing_regions = [region for region in missing_regions if region not in fetched]
            if rest_missing_regions and allow_rest_fallback:
                LOGGER.info("[%s] falling back to REST for %d promoter regions missing from local FASTA", species, len(rest_missing_regions))
                fetched.update(fetch_promoter_sequences(rest_missing_regions, species, session))
            elif rest_missing_regions:
                LOGGER.info("[%s] skipping %d promoter regions missing from local FASTA", species, len(rest_missing_regions))
        else:
            fetched = fetch_promoter_sequences(missing_regions, species, session)
        LOGGER.info("[%s] fetched %d/%d promoter sequences", species, len(fetched), len(missing_regions))
        _append_sequence_cache(sequence_cache_path, fetched)
        cached_sequences.update(fetched)
    sequences_by_region = cached_sequences

    rows = []
    for protein_id, gene_id in gene_by_protein.items():
        region = regions_by_gene.get(gene_id)
        if region is None:
            continue
        sequence = sequences_by_region.get(_region_string(*region))
        if sequence is None:
            continue
        seq_region_name, start, end, strand = region
        rows.append(
            {
                "protein_id": protein_id,
                "species": species,
                "ensembl_gene_id": gene_id,
                "seq_region_name": seq_region_name,
                "promoter_start": start,
                "promoter_end": end,
                "strand": strand,
                "promoter_length": len(sequence),
                "promoter_sequence": sequence,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def extract_for_operon_species(
    protein_ids: list[str],
    species: str,
    window: int,
    n_workers: int,
    cache_path: Path,
    sequence_cache_path: Path,
    max_operon_gap: int = DEFAULT_MAX_OPERON_GAP,
    local_reference: LocalPromoterReference | None = None,
    allow_rest_fallback: bool = True,
) -> pd.DataFrame:
    """Like extract_for_species, but resolves each gene's operon leader first and uses the
    leader's upstream window as the shared promoter for every gene in the operon.
    """
    columns = [
        "protein_id",
        "species",
        "ensembl_gene_id",
        "operon_leader_gene_id",
        "seq_region_name",
        "promoter_start",
        "promoter_end",
        "strand",
        "promoter_length",
        "promoter_sequence",
    ]
    session = _make_session(max(10, n_workers))

    if local_reference is None and not allow_rest_fallback:
        LOGGER.info("[%s] skipping species because no local reference is available", species)
        return pd.DataFrame(columns=columns)

    if local_reference is not None:
        gene_by_protein = load_local_gene_ids(protein_ids, local_reference)
        unresolved_ids = [protein_id for protein_id in protein_ids if protein_id not in gene_by_protein]
        if unresolved_ids and allow_rest_fallback:
            LOGGER.info("[%s] falling back to REST for %d proteins missing from local xref", species, len(unresolved_ids))
            gene_by_protein.update(_resolve_gene_ids_cached(unresolved_ids, species, n_workers, cache_path, session))
        elif unresolved_ids:
            LOGGER.info("[%s] skipping %d proteins missing from local xref", species, len(unresolved_ids))
    else:
        gene_by_protein = _resolve_gene_ids_cached(protein_ids, species, n_workers, cache_path, session)

    if local_reference is not None:
        LOGGER.info("[%s] loading local whole-genome gene set for operon-aware promoter resolution", species)
        genes = load_all_local_gene_coordinates(local_reference)
    else:
        LOGGER.info("[%s] fetching whole-genome gene set for operon-aware promoter resolution", species)
        genes = fetch_all_genes(species, session)
    LOGGER.info("[%s] fetched %d genes genome-wide; resolving operon leaders (max_gap=%dbp)", species, len(genes), max_operon_gap)
    leader_records = resolve_operon_leaders(genes, max_operon_gap)

    # Only fetch sequences for the genes our proteins actually resolved to, not every
    # leader in the genome -- resolve_operon_leaders needs whole-genome adjacency, but
    # sequence/region fetching should stay scoped to what's actually needed.
    needed_gene_ids = set(gene_by_protein.values())
    regions_by_gene = {
        gene_id: region
        for gene_id in needed_gene_ids
        if (record := leader_records.get(gene_id)) is not None and (region := _promoter_region(record, window)) is not None
    }
    region_strings = sorted({_region_string(*region) for region in regions_by_gene.values()})
    cached_sequences = _load_sequence_cache(sequence_cache_path)
    missing_regions = [region for region in region_strings if region not in cached_sequences]
    if missing_regions:
        if local_reference is not None:
            fetched = fetch_local_sequences([regions_by_gene[gene_id] for gene_id in regions_by_gene if _region_string(*regions_by_gene[gene_id]) in missing_regions], local_reference)
            rest_missing_regions = [region for region in missing_regions if region not in fetched]
            if rest_missing_regions and allow_rest_fallback:
                LOGGER.info("[%s] falling back to REST for %d promoter regions missing from local FASTA", species, len(rest_missing_regions))
                fetched.update(fetch_promoter_sequences(rest_missing_regions, species, session))
            elif rest_missing_regions:
                LOGGER.info("[%s] skipping %d promoter regions missing from local FASTA", species, len(rest_missing_regions))
        else:
            fetched = fetch_promoter_sequences(missing_regions, species, session)
        LOGGER.info("[%s] fetched %d/%d promoter sequences", species, len(fetched), len(missing_regions))
        _append_sequence_cache(sequence_cache_path, fetched)
        cached_sequences.update(fetched)
    sequences_by_region = cached_sequences

    rows = []
    for protein_id, gene_id in gene_by_protein.items():
        region = regions_by_gene.get(gene_id)
        if region is None:
            continue
        sequence = sequences_by_region.get(_region_string(*region))
        if sequence is None:
            continue
        seq_region_name, start, end, strand = region
        leader_record = leader_records[gene_id]
        rows.append(
            {
                "protein_id": protein_id,
                "species": species,
                "ensembl_gene_id": gene_id,
                "operon_leader_gene_id": leader_record.get("gene_id", leader_record.get("id")),
                "seq_region_name": seq_region_name,
                "promoter_start": start,
                "promoter_end": end,
                "strand": strand,
                "promoter_length": len(sequence),
                "promoter_sequence": sequence,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def run(input_path: Path, output_dir: Path, window: int, n_workers: int, species_filter: list[str] | None, local_reference_dir: Path | None = None, allow_rest_fallback: bool = True) -> None:
    taxonomy = load_protein_taxa(input_path)
    all_species = {**ANCHOR_SPECIES, **OPERON_AWARE_SPECIES}
    anchors = {taxon_id: species for taxon_id, species in all_species.items() if species_filter is None or species in species_filter}
    taxonomy = taxonomy[taxonomy["taxon_id"].isin(anchors)]
    LOGGER.info("%d proteins across %d anchor species to process", len(taxonomy), taxonomy["taxon_id"].nunique())

    output_dir.mkdir(parents=True, exist_ok=True)
    gene_cache_dir = output_dir / "gene_id_cache"
    sequence_cache_dir = output_dir / "sequence_cache"
    gene_cache_dir.mkdir(parents=True, exist_ok=True)
    sequence_cache_dir.mkdir(parents=True, exist_ok=True)
    for taxon_id, group in taxonomy.groupby("taxon_id"):
        species = anchors[taxon_id]
        species_output = output_dir / f"{species}.tsv"
        cache_path = gene_cache_dir / f"{species}.gene_id_cache.tsv"
        sequence_cache_path = sequence_cache_dir / f"{species}.promoter_sequences.tsv"
        local_reference = find_local_reference(local_reference_dir, species)

        done_ids: set[str] = set()
        existing = None
        if species_output.exists():
            existing = pd.read_csv(species_output, sep="\t", dtype={"taxon_id": str})
            done_ids = set(existing["protein_id"])

        pending_ids = [protein_id for protein_id in dict.fromkeys(group["protein_id"].tolist()) if protein_id not in done_ids]
        if not pending_ids:
            LOGGER.info("[%s] already complete (%d proteins)", species, len(done_ids))
            continue

        LOGGER.info("[%s] %d done, %d pending", species, len(done_ids), len(pending_ids))
        if taxon_id in OPERON_AWARE_SPECIES:
            new_rows = extract_for_operon_species(pending_ids, species, window, n_workers, cache_path, sequence_cache_path, local_reference=local_reference, allow_rest_fallback=allow_rest_fallback)
        else:
            new_rows = extract_for_species(pending_ids, species, window, n_workers, cache_path, sequence_cache_path, local_reference=local_reference, allow_rest_fallback=allow_rest_fallback)
        if not new_rows.empty:
            new_rows.insert(1, "taxon_id", taxon_id)
        combined = pd.concat([existing, new_rows], ignore_index=True) if existing is not None else new_rows
        combined.to_csv(species_output, sep="\t", index=False)
        LOGGER.info("[%s] wrote %d/%d promoters to %s", species, len(combined), len(group), species_output)

    write_combined(output_dir)


def load_promoter_files(output_dir: Path, exclude: set[str] | None = None) -> list[pd.DataFrame]:
    """Read every per-species promoter TSV in `output_dir`, skipping summaries and caches."""
    skip = {"combined.tsv", *(exclude or set())}
    frames = []
    for path in sorted(output_dir.glob("*.tsv")):
        if path.name in skip:
            continue
        frame = pd.read_csv(path, sep="\t", dtype={"taxon_id": str})
        if "promoter_sequence" in frame.columns:
            frames.append(frame)
    return frames


def write_combined(output_dir: Path) -> None:
    frames = load_promoter_files(output_dir)
    if not frames:
        return
    combined_all = pd.concat(frames, ignore_index=True)
    if "promoter_source" not in combined_all.columns:
        combined_all["promoter_source"] = "xref"
    combined_all["promoter_source"] = combined_all["promoter_source"].fillna("xref")
    combined_all.to_csv(output_dir / "combined.tsv", sep="\t", index=False)
    LOGGER.info("wrote %d total promoter sequences to %s", len(combined_all), output_dir / "combined.tsv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="protein_id<TAB>taxon_id TSV (no header), or a FASTA with '>id taxon' headers")
    parser.add_argument("output_dir", type=Path, help="Directory for per-species TSVs and the combined output")
    parser.add_argument("--window", type=int, default=3000, help="Promoter window size in bp, upstream of the TSS")
    parser.add_argument("--n-workers", type=int, default=5, help="Concurrent requests for the per-protein gene-ID resolution step")
    parser.add_argument("--species", nargs="*", default=None, help="Restrict to these Ensembl species names (default: all configured anchors)")
    parser.add_argument("--local-reference-dir", type=Path, default=None, help="Directory containing {species}.protein_gene.tsv, {species}.genes.tsv, and {species}.fa/.fasta files for local-first extraction")
    parser.add_argument("--offline-local-only", action="store_true", help="When local reference data is present, skip missing local xrefs/coordinates/sequences instead of falling back to Ensembl REST")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run(args.input, args.output_dir, args.window, args.n_workers, args.species, args.local_reference_dir, not args.offline_local_only)


if __name__ == "__main__":
    main()
