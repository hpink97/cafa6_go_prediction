"""NCBI Datasets reference assemblies (genome + GFF + proteins) for taxa Ensembl does not cover."""

from __future__ import annotations

import gzip
import io
import json
import logging
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests
from Bio import SeqIO

from feature_engineering.promoters.build_reference import request_with_retries

LOGGER = logging.getLogger("cafa6.promoters.ncbi_reference")
DATASETS_API = "https://api.ncbi.nlm.nih.gov/datasets/v2"
ASSEMBLY_LEVEL_RANK = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}
GENE_COLUMNS = ["ensembl_gene_id", "seq_region_name", "start", "end", "strand", "feature_kind", "gene_biotype"]


def reference_id(accession: str) -> str:
    return f"ncbi_{accession}"


def select_assembly(taxon_id: str, timeout: float) -> dict | None:
    """Best annotated assembly for `taxon_id`: exact taxon before descendants, reference/RefSeq/complete first."""
    for exact in ("true", "false"):
        for reference_only in ("true", "false"):
            response = request_with_retries(
                "GET",
                f"{DATASETS_API}/genome/taxon/{taxon_id}/dataset_report",
                timeout=timeout,
                params={
                    "filters.has_annotation": "true",
                    "filters.reference_only": reference_only,
                    "tax_exact_match": exact,
                    "page_size": 50,
                },
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            reports = [report for report in response.json().get("reports", []) if report.get("annotation_info")]
            if not reports:
                continue
            best = min(
                reports,
                key=lambda report: (
                    not report["accession"].startswith("GCF_"),
                    ASSEMBLY_LEVEL_RANK.get(report.get("assembly_info", {}).get("assembly_level"), 9),
                    -(report.get("annotation_info", {}).get("stats", {}).get("gene_counts", {}).get("protein_coding") or 0),
                ),
            )
            return {
                "accession": best["accession"],
                "taxon_id": str(best.get("organism", {}).get("tax_id", taxon_id)),
                "organism": best.get("organism", {}).get("organism_name", ""),
                "exact_taxon": exact == "true",
            }
    return None


def select_assemblies(taxon_ids: list[str], cache_path: Path, timeout: float) -> dict[str, dict | None]:
    """Resolve (and cache) the chosen assembly for every taxon; unresolvable taxa map to None."""
    cache: dict[str, dict | None] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    pending = [taxon for taxon in taxon_ids if taxon not in cache]
    for index, taxon_id in enumerate(pending, start=1):
        try:
            cache[taxon_id] = select_assembly(taxon_id, timeout)
        except requests.RequestException as exc:
            LOGGER.warning("[taxon %s] assembly lookup failed (will retry next run): %s", taxon_id, exc)
            continue
        if index % 25 == 0 or index == len(pending):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
            LOGGER.info("resolved NCBI assemblies for %d/%d taxa", index, len(pending))
    return {taxon: cache.get(taxon) for taxon in taxon_ids}


def download_package(accession: str, annotation_types: list[str], destination: Path, timeout: float) -> Path:
    params = [("include_annotation_type", kind) for kind in annotation_types]
    archive = destination / f"{accession}.zip"
    with requests.get(f"{DATASETS_API}/genome/accession/{accession}/download", params=params, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with archive.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                handle.write(chunk)
    return archive


def _member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if not matches:
        raise RuntimeError(f"NCBI package has no *{suffix} member")
    return matches[0]


def _attributes(text: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in text.split(";") if "=" in field)


def parse_ncbi_gff(lines) -> tuple[pd.DataFrame, dict[str, str]]:
    """Gene coordinates plus protein accession -> gene id, resolved through each CDS's Parent chain."""
    genes: dict[str, dict] = {}
    gene_by_feature: dict[str, str] = {}
    parent_by_feature: dict[str, str] = {}
    cds_rows: list[tuple[str, str]] = []
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 9:
            continue
        attributes = _attributes(parts[8])
        feature_id = attributes.get("ID", "")
        parent = attributes.get("Parent", "").split(",")[0]
        if parts[2] == "gene":
            gene_id = feature_id.removeprefix("gene-")
            gene_by_feature[feature_id] = gene_id
            genes[gene_id] = {
                "ensembl_gene_id": gene_id,
                "seq_region_name": parts[0],
                "start": int(parts[3]),
                "end": int(parts[4]),
                "strand": 1 if parts[6] == "+" else -1,
                "feature_kind": "gene",
                "gene_biotype": attributes.get("gene_biotype", ""),
            }
        elif parts[2] == "CDS":
            protein_id = attributes.get("protein_id") or attributes.get("Name") or feature_id.removeprefix("cds-")
            if protein_id and parent:
                cds_rows.append((protein_id, parent))
        elif feature_id and parent:
            parent_by_feature[feature_id] = parent

    protein_to_gene: dict[str, str] = {}
    for protein_id, parent in cds_rows:
        if protein_id in protein_to_gene:
            continue
        node = parent
        for _ in range(6):
            if node in gene_by_feature:
                protein_to_gene[protein_id] = gene_by_feature[node]
                break
            node = parent_by_feature.get(node, "")
            if not node:
                break
    return pd.DataFrame(list(genes.values()), columns=GENE_COLUMNS), protein_to_gene


def build_ncbi_proteome(accession: str, cache_dir: Path, timeout: float) -> tuple[Path, Path] | None:
    """Cache ``ncbi_<acc>|gene_id|protein_id`` longest-isoform proteome and its gene coordinate table."""
    ref = reference_id(accession)
    proteome = cache_dir / f"{ref}.pep.fa.gz"
    genes_path = cache_dir / f"{ref}.genes.tsv.gz"
    unavailable = cache_dir / f"{ref}.unavailable"
    if proteome.exists() and genes_path.exists():
        return proteome, genes_path
    if unavailable.exists():
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
            archive_path = download_package(accession, ["PROT_FASTA", "GENOME_GFF"], Path(tmp), timeout)
            with zipfile.ZipFile(archive_path) as archive:
                with archive.open(_member(archive, ".gff")) as handle:
                    genes, protein_to_gene = parse_ncbi_gff(line.decode() for line in handle)
                best: dict[str, tuple[str, str]] = {}
                with archive.open(_member(archive, ".faa")) as binary, io.TextIOWrapper(binary, encoding="utf-8") as text:
                    for record in SeqIO.parse(text, "fasta"):
                        gene_id = protein_to_gene.get(record.id)
                        sequence = str(record.seq).rstrip("*")
                        if gene_id and (gene_id not in best or len(sequence) > len(best[gene_id][1])):
                            best[gene_id] = (record.id, sequence)
            if not best:
                raise RuntimeError("no proteins could be linked to genes")
            partial_genes = Path(tmp) / genes_path.name
            genes.to_csv(partial_genes, sep="\t", index=False)
            partial_proteome = Path(tmp) / proteome.name
            with gzip.open(partial_proteome, "wt") as handle:
                for gene_id, (protein_id, sequence) in best.items():
                    handle.write(f">{ref}|{gene_id}|{protein_id}\n{sequence}\n")
            partial_genes.replace(genes_path)
            partial_proteome.replace(proteome)
    except (RuntimeError, zipfile.BadZipFile) as exc:
        LOGGER.warning("[%s] no usable NCBI proteome: %s", ref, exc)
        unavailable.write_text(str(exc))
        return None
    except requests.RequestException as exc:
        LOGGER.warning("[%s] NCBI download failed (will retry next run): %s", ref, exc)
        return None
    LOGGER.info("[%s] cached %d reference proteins", ref, len(best))
    return proteome, genes_path


def download_ncbi_genome(accession: str, destination: Path, timeout: float) -> Path:
    """Download and unpack an assembly's genome FASTA into `destination`."""
    archive_path = download_package(accession, ["GENOME_FASTA"], destination, timeout)
    fasta = destination / f"{reference_id(accession)}.fa"
    with zipfile.ZipFile(archive_path) as archive, archive.open(_member(archive, ".fna")) as source, fasta.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    archive_path.unlink()
    return fasta
