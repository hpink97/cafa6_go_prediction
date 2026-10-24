"""Local reference-data helpers for promoter extraction.

The live Ensembl REST path is useful as a fallback, but high-throughput promoter
extraction should run mostly from local bulk files:

* protein -> Ensembl gene cross-references
* Ensembl gene coordinates
* genome FASTA

Expected filenames under a reference directory are:

* ``{species}.protein_gene.tsv`` with ``protein_id`` and ``ensembl_gene_id``
* ``{species}.genes.tsv`` with ``ensembl_gene_id``, ``seq_region_name``,
  ``start``, ``end``, and ``strand``
* ``{species}.fa`` / ``{species}.fasta`` / ``{species}.dna.fa`` for sequence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

LOGGER = logging.getLogger("cafa6.promoters.local_reference")


GENE_ID_COLUMNS = ("ensembl_gene_id", "gene_id", "id")
PROTEIN_ID_COLUMNS = ("protein_id", "uniprot_id", "accession", "uniprot_accession")
FASTA_SUFFIXES = (".fa", ".fasta", ".fna", ".dna.fa", ".dna.fasta")


@dataclass(frozen=True)
class LocalPromoterReference:
    species: str
    protein_gene_path: Path
    genes_path: Path
    fasta_path: Path


def find_local_reference(reference_dir: Path | None, species: str) -> LocalPromoterReference | None:
    """Return local reference file paths for ``species`` if all required files exist."""
    if reference_dir is None:
        return None
    protein_gene_path = reference_dir / f"{species}.protein_gene.tsv"
    genes_path = reference_dir / f"{species}.genes.tsv"
    fasta_path = next((reference_dir / f"{species}{suffix}" for suffix in FASTA_SUFFIXES if (reference_dir / f"{species}{suffix}").exists()), None)
    if protein_gene_path.exists() and genes_path.exists() and fasta_path is not None:
        return LocalPromoterReference(species, protein_gene_path, genes_path, fasta_path)
    missing = [str(path.name) for path in (protein_gene_path, genes_path) if not path.exists()]
    if fasta_path is None:
        missing.append(f"{species}.fa|.fasta|.fna|.dna.fa|.dna.fasta")
    LOGGER.info("[%s] local reference incomplete in %s (%s); using REST fallback", species, reference_dir, ", ".join(missing))
    return None


def load_local_gene_ids(protein_ids: list[str], reference: LocalPromoterReference) -> dict[str, str]:
    """Load protein -> Ensembl gene IDs for requested proteins from a bulk mapping TSV."""
    frame = pd.read_csv(reference.protein_gene_path, sep="\t", dtype=str)
    protein_column = _first_present(frame, PROTEIN_ID_COLUMNS, reference.protein_gene_path)
    gene_column = _first_present(frame, GENE_ID_COLUMNS, reference.protein_gene_path)
    wanted = set(protein_ids)
    frame = frame[[protein_column, gene_column]].dropna().drop_duplicates(subset=[protein_column])
    frame = frame[frame[protein_column].isin(wanted)]
    gene_by_protein = dict(zip(frame[protein_column], frame[gene_column]))
    LOGGER.info("[%s] local xref resolved %d/%d proteins", reference.species, len(gene_by_protein), len(wanted))
    return gene_by_protein


def load_local_gene_coordinates(gene_ids: list[str], reference: LocalPromoterReference) -> dict[str, dict]:
    """Load Ensembl gene coordinates for requested gene IDs from a bulk coordinate TSV."""
    frame = pd.read_csv(reference.genes_path, sep="\t", dtype=str)
    gene_column = _first_present(frame, GENE_ID_COLUMNS, reference.genes_path)
    required = {"seq_region_name", "start", "end", "strand"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{reference.genes_path} is missing required columns: {', '.join(sorted(missing))}")
    frame = frame[[gene_column, "seq_region_name", "start", "end", "strand"]].dropna().drop_duplicates(subset=[gene_column])
    frame = frame[frame[gene_column].isin(set(gene_ids))]
    coordinates: dict[str, dict] = {}
    for row in frame.itertuples(index=False):
        gene_id, seq_region_name, start, end, strand = row
        coordinates[gene_id] = {
            "id": gene_id,
            "gene_id": gene_id,
            "seq_region_name": seq_region_name,
            "start": int(start),
            "end": int(end),
            "strand": _parse_strand(strand),
        }
    LOGGER.info("[%s] local coordinates loaded for %d/%d genes", reference.species, len(coordinates), len(set(gene_ids)))
    return coordinates


def load_all_local_gene_coordinates(reference: LocalPromoterReference) -> dict[str, dict]:
    """Load all local gene coordinates, for operon-aware species that need whole-genome adjacency."""
    frame = pd.read_csv(reference.genes_path, sep="\t", dtype=str)
    gene_column = _first_present(frame, GENE_ID_COLUMNS, reference.genes_path)
    required = {"seq_region_name", "start", "end", "strand"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{reference.genes_path} is missing required columns: {', '.join(sorted(missing))}")
    if "feature_kind" in frame.columns:
        # Transcript/protein alias rows duplicate gene coordinates and would corrupt operon adjacency.
        frame = frame[frame["feature_kind"] == "gene"]
    coordinates: dict[str, dict] = {}
    for row in frame[[gene_column, "seq_region_name", "start", "end", "strand"]].dropna().drop_duplicates(subset=[gene_column]).itertuples(index=False):
        gene_id, seq_region_name, start, end, strand = row
        coordinates[gene_id] = {
            "id": gene_id,
            "gene_id": gene_id,
            "seq_region_name": seq_region_name,
            "start": int(start),
            "end": int(end),
            "strand": _parse_strand(strand),
        }
    return coordinates


def fetch_local_sequences(regions: list[tuple[str, int, int, int]], reference: LocalPromoterReference) -> dict[str, str]:
    """Extract promoter sequences from a local FASTA for region tuples."""
    genome = SeqIO.index(str(reference.fasta_path), "fasta")
    try:
        sequences: dict[str, str] = {}
        for seq_region_name, start, end, strand in regions:
            if seq_region_name not in genome:
                continue
            sequence = str(genome[seq_region_name].seq[start - 1 : end]).upper()
            if not sequence:
                continue
            if strand == -1:
                sequence = str(Seq(sequence).reverse_complement())
            sequences[f"{seq_region_name}:{start}-{end}:{strand}"] = sequence
        return sequences
    finally:
        genome.close()


def _first_present(frame: pd.DataFrame, candidates: tuple[str, ...], path: Path) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"{path} must include one of these columns: {', '.join(candidates)}")


def _parse_strand(value: str) -> int:
    if value in {"1", "+", "+1"}:
        return 1
    if value in {"-1", "-"}:
        return -1
    raise ValueError(f"unsupported strand value: {value!r}")