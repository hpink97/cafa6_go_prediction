"""Normalize CAFA6-to-anchor orthology mappings.

The primary input is a precomputed orthology export from eggNOG-mapper,
OrthoDB, Ensembl Compara, or an equivalent pipeline. DIAMOND is supported as a
sequence-similarity fallback, but those hits are labelled as such and are not
treated as orthologs.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import pandas as pd
from Bio import SeqIO

LOGGER = logging.getLogger("cafa6.homology")
HIT_COLUMNS = [
    "query_id",
    "target_id",
    "pident",
    "alignment_length",
    "query_length",
    "target_length",
    "evalue",
    "bitscore",
    "query_coverage",
    "target_coverage",
]


def fasta_ids(path: Path) -> list[str]:
    ids = []
    for record in SeqIO.parse(path, "fasta"):
        ids.append(record.id.split("|")[1] if "|" in record.id else record.id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate query IDs found in {path}")
    return ids


def run_diamond(query: Path, database: Path, output: Path, executable: str, threads: int, max_targets: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "blastp",
        "--query",
        str(query),
        "--db",
        str(database),
        "--out",
        str(output),
        "--outfmt",
        "6",
        *HIT_COLUMNS,
        "--max-target-seqs",
        str(max_targets),
        "--threads",
        str(threads),
        "--sensitive",
    ]
    LOGGER.info("running homology search with %d threads", threads)
    subprocess.run(command, check=True)


def load_hits(path: Path) -> pd.DataFrame:
    hits = pd.read_csv(path, sep="\t", names=HIT_COLUMNS, header=None)
    if hits.empty:
        raise ValueError(f"No homology hits found in {path}")
    numeric = [column for column in HIT_COLUMNS if column not in {"query_id", "target_id"}]
    hits[numeric] = hits[numeric].apply(pd.to_numeric, errors="coerce")
    hits = hits.dropna(subset=["query_id", "target_id", "pident", "query_coverage", "bitscore", "evalue"])
    return hits


def select_best_hits(hits: pd.DataFrame, manifest: pd.DataFrame, min_identity: float, min_coverage: float) -> pd.DataFrame:
    required_manifest = {"target_id", "anchor_taxon_id", "anchor_gene_id", "anchor_species"}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise ValueError(f"Anchor manifest is missing columns: {sorted(missing)}")

    filtered = hits.query("pident >= @min_identity and query_coverage >= @min_coverage").copy()
    filtered = filtered.merge(manifest, on="target_id", how="inner", validate="many_to_one")
    filtered["mapping_confidence"] = (
        (filtered["pident"] / 100.0)
        * filtered["query_coverage"]
        * (1.0 - 1.0 / (1.0 + filtered["bitscore"].clip(lower=0)))
    )
    filtered = filtered.sort_values(
        ["query_id", "mapping_confidence", "bitscore", "evalue"],
        ascending=[True, False, False, True],
    )
    filtered["mapping_rank"] = filtered.groupby("query_id").cumcount() + 1
    columns = [
        "query_id",
        "target_id",
        "anchor_taxon_id",
        "anchor_species",
        "anchor_gene_id",
        "mapping_rank",
        "mapping_confidence",
        "pident",
        "query_coverage",
        "target_coverage",
        "alignment_length",
        "evalue",
        "bitscore",
    ]
    return filtered[columns]


def load_precomputed_orthology(path: Path, manifest: pd.DataFrame | None) -> pd.DataFrame:
    """Load a normalized orthology export using common column aliases."""
    table = pd.read_csv(path, sep="\t")
    aliases = {
        "query_id": ("query_id", "query", "protein_id", "protein"),
        "anchor_gene_id": ("anchor_gene_id", "gene_id", "target_gene_id", "ortholog_gene_id"),
        "anchor_taxon_id": ("anchor_taxon_id", "taxon_id", "target_taxon_id", "species_taxon_id"),
        "mapping_confidence": ("mapping_confidence", "orthology_confidence", "confidence", "score"),
    }
    for canonical, candidates in aliases.items():
        if canonical not in table:
            source = next((candidate for candidate in candidates if candidate in table), None)
            if source is not None:
                table[canonical] = table[source]
    required = {"query_id", "anchor_gene_id", "anchor_taxon_id"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Precomputed orthology table is missing columns: {sorted(missing)}")
    if "mapping_confidence" not in table:
        table["mapping_confidence"] = 1.0
    if "source" not in table:
        table["source"] = path.stem
    table["mapping_method"] = "precomputed_orthology"
    table["mapping_confidence"] = pd.to_numeric(table["mapping_confidence"], errors="coerce").fillna(0).clip(lower=0)
    table["anchor_taxon_id"] = table["anchor_taxon_id"].astype(str)
    if manifest is not None and "anchor_species" not in table:
        species_columns = {"anchor_taxon_id", "anchor_species"}
        if species_columns.issubset(manifest.columns):
            table = table.merge(manifest[list(species_columns)].drop_duplicates(), on="anchor_taxon_id", how="left")
    if "anchor_species" not in table:
        table["anchor_species"] = "unknown"
    table = table.sort_values(["query_id", "mapping_confidence"], ascending=[True, False])
    table["mapping_rank"] = table.groupby("query_id").cumcount() + 1
    return table[["query_id", "anchor_taxon_id", "anchor_species", "anchor_gene_id", "mapping_rank", "mapping_confidence", "source", "mapping_method"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", type=Path, help="Query protein FASTA")
    parser.add_argument("output", type=Path, help="Output homolog mapping TSV")
    parser.add_argument("--manifest", type=Path, help="Optional target_id/taxon/species manifest for DIAMOND or taxon names")
    parser.add_argument("--orthology", type=Path, help="Precomputed orthology TSV (preferred)")
    parser.add_argument("--database", type=Path, help="Combined DIAMOND database")
    parser.add_argument("--hits", type=Path, help="Existing DIAMOND tabular output")
    parser.add_argument("--diamond", default="diamond", help="DIAMOND executable")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-targets", type=int, default=10)
    parser.add_argument("--min-identity", type=float, default=25.0)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--keep-top", type=int, default=3, help="Keep this many anchors per query")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    modes = [args.orthology, args.database, args.hits]
    if sum(value is not None for value in modes) != 1:
        parser.error("provide exactly one of --orthology, --database, or --hits")
    for option_name, path in (("query", args.query), ("orthology", args.orthology), ("database", args.database), ("hits", args.hits), ("manifest", args.manifest)):
        if path is not None and not path.is_file():
            parser.error(f"--{option_name} file does not exist: {path}. Supply an exported orthology or homology file first.")
    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str) if args.manifest else None
    queries = pd.DataFrame({"query_id": fasta_ids(args.query)})
    if args.orthology:
        selected = load_precomputed_orthology(args.orthology, manifest)
    else:
        if manifest is None:
            parser.error("--manifest is required for DIAMOND modes")
        hits_path = args.hits
        if args.database:
            hits_path = args.output.with_suffix(".diamond.tsv")
            run_diamond(args.query, args.database, hits_path, args.diamond, args.threads, args.max_targets)
        selected = select_best_hits(load_hits(hits_path), manifest, args.min_identity, args.min_coverage)
        selected["source"] = "diamond"
        selected["mapping_method"] = "sequence_similarity_fallback"
    selected = selected.query("mapping_rank <= @args.keep_top")
    result = queries.merge(selected, on="query_id", how="left")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    mapped_queries = result["anchor_gene_id"].notna().groupby(result["query_id"]).any().sum()
    LOGGER.info("mapped %d/%d queries with at least one anchor", mapped_queries, len(queries))


if __name__ == "__main__":
    main()