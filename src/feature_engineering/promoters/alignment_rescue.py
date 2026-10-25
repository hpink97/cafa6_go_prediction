"""Recover promoters for proteins without a usable cross-reference by aligning them to reference proteomes.

Each missing protein is aligned with DIAMOND against the longest isoform of every gene in the
configured reference species. A near-identical hit in the protein's own species identifies its
actual gene (``alignment_exact``); otherwise the best hit from the phylogenetically closest
reference species within the same NCBI rank (phylum by default) supplies a labelled proxy
promoter (``alignment_proxy``).

When only part of the missing set can be aligned (``--max-queries``), queries are sampled from
ESM-2 embedding clusters in proportion to how poorly each cluster is covered by promoters.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from feature_engineering.promoters.build_reference import (
    ENSEMBL_ORGANISMS_ROOTS,
    download_gene_coordinates,
    download_genome_fasta,
    fasta_directory_url,
    normalize_ensembl_gene_id,
)
from feature_engineering.promoters.extract_promoters import (
    _promoter_region,
    _region_string,
    load_promoter_files,
    load_protein_taxa,
    write_combined,
)
from feature_engineering.promoters.local_reference import LocalPromoterReference, fetch_local_sequences
from feature_engineering.promoters.ncbi_reference import (
    build_ncbi_proteome,
    download_ncbi_genome,
    reference_id,
    select_assemblies,
)
from feature_engineering.promoters.operons import resolve_operon_leaders
from feature_engineering.promoters.species_map import ANCHOR_SPECIES, OPERON_AWARE_SPECIES
from feature_engineering.sequence.taxonomy import DEFAULT_TAXDUMP_CACHE_DIR, _download_ncbi_nodes

LOGGER = logging.getLogger("cafa6.promoters.alignment_rescue")
HIT_FIELDS = ["qseqid", "sseqid", "pident", "length", "qlen", "slen", "evalue", "bitscore", "qcovhsp", "scovhsp"]
GENE_TOKEN = re.compile(r"\bgene:(\S+)")
PROKARYOTE_TAXA = {"2", "2157"}
RESCUE_FILENAME = "alignment_rescue.tsv"
RESCUE_COLUMNS = [
    "protein_id",
    "taxon_id",
    "species",
    "ensembl_gene_id",
    "seq_region_name",
    "promoter_start",
    "promoter_end",
    "strand",
    "promoter_length",
    "promoter_sequence",
    "promoter_source",
    "reference_pident",
    "reference_query_coverage",
    "reference_lineage_depth",
]
ASSIGNMENT_COLUMNS = [
    "protein_id",
    "taxon_id",
    "reference_species",
    "reference_gene_id",
    "promoter_source",
    "reference_pident",
    "reference_query_coverage",
    "bitscore",
    "reference_lineage_depth",
]


def canonical_protein_id(protein_id: str) -> str:
    """Use the accession portion of UniProt FASTA IDs, preserving ordinary IDs unchanged."""
    parts = protein_id.split("|")
    if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
        return parts[1]
    return protein_id


def longest_isoform_per_gene(records: Iterable) -> dict[str, tuple[str, str]]:
    """gene_id -> (protein_id, sequence), keeping each gene's longest protein product."""
    best: dict[str, tuple[str, str]] = {}
    for record in records:
        match = GENE_TOKEN.search(record.description)
        gene_id = normalize_ensembl_gene_id(match.group(1)) if match else record.id
        sequence = str(record.seq).rstrip("*")
        if gene_id not in best or len(sequence) > len(best[gene_id][1]):
            best[gene_id] = (record.id, sequence)
    return best


def proteome_source(species: str, timeout: float) -> tuple[str, str]:
    """(directory_url, filename) of the species' peptide FASTA on the Ensembl FTP sites."""
    if species in ENSEMBL_ORGANISMS_ROOTS:
        return f"{ENSEMBL_ORGANISMS_ROOTS[species]}/geneset/", "pep.fa.bgz"
    directory_url = fasta_directory_url(species, timeout).removesuffix("dna/") + "pep/"
    response = requests.get(directory_url, timeout=timeout)
    response.raise_for_status()
    names = sorted(set(re.findall(r'href="([^"]+\.pep\.all\.fa\.gz)"', response.text)))
    if not names:
        raise RuntimeError(f"No .pep.all.fa.gz under {directory_url}")
    return directory_url, names[0]


def download_proteome(species: str, cache_dir: Path, timeout: float) -> Path | None:
    """Cache a longest-isoform-per-gene proteome with ``species|gene_id|protein_id`` headers."""
    output = cache_dir / f"{species}.pep.fa.gz"
    unavailable = cache_dir / f"{species}.unavailable"
    if output.exists():
        return output
    if unavailable.exists():
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        directory_url, filename = proteome_source(species, timeout)
        with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
            raw = Path(tmp) / filename
            with requests.get(directory_url + filename, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with raw.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        handle.write(chunk)
            with gzip.open(raw, "rt") as handle:
                genes = longest_isoform_per_gene(SeqIO.parse(handle, "fasta"))
            partial = Path(tmp) / output.name
            with gzip.open(partial, "wt") as handle:
                for gene_id, (protein_id, sequence) in genes.items():
                    handle.write(f">{species}|{gene_id}|{protein_id}\n{sequence}\n")
            partial.replace(output)
    except RuntimeError as exc:
        LOGGER.warning("[%s] no proteome available: %s", species, exc)
        unavailable.write_text(str(exc))
        return None
    except requests.RequestException as exc:
        if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 404:
            LOGGER.warning("[%s] no proteome available: %s", species, exc)
            unavailable.write_text(str(exc))
            return None
        LOGGER.warning("[%s] proteome download failed (will retry next run): %s", species, exc)
        return None
    LOGGER.info("[%s] cached %d reference proteins", species, len(genes))
    return output


def build_reference_database(proteomes: dict[str, Path], database: Path, executable: str, threads: int) -> Path:
    """Build (or reuse) one DIAMOND database spanning every reference proteome."""
    manifest = database.with_suffix(".species.json")
    wanted = sorted(proteomes)
    if database.exists() and manifest.exists() and json.loads(manifest.read_text()) == wanted:
        return database
    database.parent.mkdir(parents=True, exist_ok=True)
    panel = database.with_suffix(".fa")
    with panel.open("wb") as out:
        for species in wanted:
            with gzip.open(proteomes[species], "rb") as handle:
                shutil.copyfileobj(handle, out)
    LOGGER.info("building DIAMOND database over %d reference proteomes", len(wanted))
    subprocess.run([executable, "makedb", "--in", str(panel), "--db", str(database), "--threads", str(threads)], check=True)
    panel.unlink()
    manifest.write_text(json.dumps(wanted))
    return database


def run_diamond(queries: Path, database: Path, hits: Path, executable: str, threads: int, mode: str, max_targets: int) -> None:
    partial = hits.with_suffix(".partial.tsv")
    command = [
        executable,
        "blastp",
        "--query",
        str(queries),
        "--db",
        str(database),
        "--out",
        str(partial),
        "--outfmt",
        "6",
        *HIT_FIELDS,
        "--max-target-seqs",
        str(max_targets),
        "--evalue",
        "1e-5",
        "--threads",
        str(threads),
    ]
    if mode != "default":
        command.append(f"--{mode}")
    LOGGER.info("aligning %s against %s (%s mode, %d threads)", queries, database, mode, threads)
    subprocess.run(command, check=True)
    partial.replace(hits)


def load_taxonomy_nodes(nodes_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Parse NCBI nodes.dmp into (taxon -> parent, taxon -> rank) maps."""
    parent_by_taxon: dict[str, str] = {}
    rank_by_taxon: dict[str, str] = {}
    with nodes_path.open() as handle:
        for line in handle:
            fields = line.split("|")
            taxon_id = fields[0].strip()
            parent_by_taxon[taxon_id] = fields[1].strip()
            rank_by_taxon[taxon_id] = fields[2].strip()
    return parent_by_taxon, rank_by_taxon


def lineage(taxon_id: str | None, parent_by_taxon: dict[str, str]) -> tuple[str, ...]:
    """Root-first ancestor chain of `taxon_id` (just the taxon itself if it is unknown)."""
    if not taxon_id:
        return ()
    path = [taxon_id]
    current = taxon_id
    while (parent := parent_by_taxon.get(current)) is not None and parent != current and parent not in path:
        path.append(parent)
        current = parent
    return tuple(reversed(path))


def shared_lineage_depth(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    depth = 0
    for a, b in zip(left, right):
        if a != b:
            break
        depth += 1
    return depth


def assign_reference_genes(
    hits: pd.DataFrame,
    taxon_by_protein: dict[str, str],
    species_by_taxon: dict[str, str],
    taxon_by_species: dict[str, str],
    parent_by_taxon: dict[str, str],
    rank_by_taxon: dict[str, str],
    *,
    exact_identity: float,
    exact_coverage: float,
    proxy_identity: float,
    proxy_coverage: float,
    proxy_rank: str | None,
) -> pd.DataFrame:
    """Pick one reference gene per query: own-species near-identity first, else closest-lineage proxy."""
    if hits.empty:
        return pd.DataFrame(columns=ASSIGNMENT_COLUMNS)
    hits = hits.copy()
    target = hits["sseqid"].str.split("|", n=2, expand=True)
    hits["reference_species"] = target[0]
    hits["reference_gene_id"] = target[1]
    hits["taxon_id"] = hits["qseqid"].map(taxon_by_protein)
    hits["query_species"] = hits["taxon_id"].map(species_by_taxon)
    own_reference = (hits["reference_species"] == hits["query_species"]) | (
        hits["reference_species"].str.startswith("ncbi_")
        & (hits["reference_species"].map(taxon_by_species) == hits["taxon_id"])
    )

    exact = hits[
        own_reference
        & (hits["pident"] >= exact_identity)
        & (hits["qcovhsp"] >= exact_coverage)
    ]
    exact = exact.sort_values(["qseqid", "bitscore"], ascending=[True, False]).drop_duplicates("qseqid").copy()
    exact["promoter_source"] = "alignment_exact"
    exact["reference_lineage_depth"] = np.nan
    frames = [exact]

    if proxy_rank != "none":
        # An own-species hit below the exact threshold is a paralog, not the protein's gene.
        proxy = hits[
            ~hits["qseqid"].isin(exact["qseqid"])
            & ~own_reference
            & (hits["pident"] >= proxy_identity)
            & (hits["qcovhsp"] >= proxy_coverage)
        ].copy()
        lineages: dict[str | None, tuple[str, ...]] = {}

        def cached_lineage(taxon_id: str | None) -> tuple[str, ...]:
            if taxon_id not in lineages:
                lineages[taxon_id] = lineage(taxon_id, parent_by_taxon)
            return lineages[taxon_id]

        pair_stats: dict[tuple[str, str], tuple[int, bool]] = {}
        for taxon_id, species in proxy[["taxon_id", "reference_species"]].drop_duplicates().itertuples(index=False):
            query_lineage = cached_lineage(taxon_id)
            target_lineage = cached_lineage(taxon_by_species.get(species))
            depth = shared_lineage_depth(query_lineage, target_lineage)
            if proxy_rank is None:
                allowed = True
            else:
                anchor = next((node for node in query_lineage if rank_by_taxon.get(node) == proxy_rank), None)
                allowed = anchor is not None and anchor in target_lineage
            pair_stats[(taxon_id, species)] = (depth, allowed)
        keys = list(zip(proxy["taxon_id"], proxy["reference_species"]))
        proxy["reference_lineage_depth"] = [pair_stats[key][0] for key in keys]
        proxy = proxy.loc[np.array([pair_stats[key][1] for key in keys], dtype=bool)]
        proxy = proxy.sort_values(["qseqid", "reference_lineage_depth", "bitscore"], ascending=[True, False, False])
        proxy = proxy.drop_duplicates("qseqid").copy()
        proxy["promoter_source"] = "alignment_proxy"
        frames.append(proxy)

    assigned = pd.concat(frames, ignore_index=True).rename(
        columns={"qseqid": "protein_id", "pident": "reference_pident", "qcovhsp": "reference_query_coverage"}
    )
    return assigned[ASSIGNMENT_COLUMNS].sort_values("protein_id").reset_index(drop=True)


def embedding_priority(embedding_dir: Path, covered_ids: set[str], n_components: int, n_clusters: int, seed: int) -> pd.DataFrame:
    """Cluster proteins in PCA-reduced ESM-2 space; priority = 1 - the cluster's promoter coverage."""
    metadata = json.loads((embedding_dir / "metadata.json").read_text())
    count, dim = metadata["protein_count"], metadata["embedding_dim"]
    ids = pd.read_csv(embedding_dir / "ids.tsv", sep="\t")["protein_id"].to_numpy()
    done = np.memmap(embedding_dir / "done.bool.dat", dtype="bool", mode="r", shape=(count,))
    if not bool(done.all()):
        raise ValueError(f"Embedding generation incomplete in {embedding_dir}")
    embeddings = np.memmap(embedding_dir / "embeddings.f32.dat", dtype="float32", mode="r", shape=(count, dim))
    rng = np.random.default_rng(seed)
    fit_rows = np.sort(rng.choice(count, size=min(count, 50_000), replace=False))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit(embeddings[fit_rows])
    projected = np.vstack([pca.transform(embeddings[start : start + 20_000]) for start in range(0, count, 20_000)])
    labels = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=4096, n_init=3).fit_predict(projected)
    frame = pd.DataFrame({"protein_id": ids, "cluster": labels, "pc1": projected[:, 0], "pc2": projected[:, 1]})
    frame["covered"] = frame["protein_id"].isin(covered_ids)
    stats = frame.groupby("cluster")["covered"].agg(cluster_size="size", cluster_coverage="mean")
    frame = frame.join(stats, on="cluster")
    frame["priority"] = 1.0 - frame["cluster_coverage"]
    LOGGER.info(
        "embedding clusters: %d, median cluster coverage %.1f%%, worst decile below %.1f%%",
        n_clusters,
        100 * stats["cluster_coverage"].median(),
        100 * stats["cluster_coverage"].quantile(0.1),
    )
    return frame


def choose_queries(priority: pd.DataFrame, missing_ids: list[str], max_queries: int | None, seed: int) -> list[str]:
    """All missing proteins, highest priority first; or a priority-weighted sample of `max_queries`."""
    candidates = pd.DataFrame({"protein_id": missing_ids}).merge(priority[["protein_id", "priority"]], on="protein_id", how="left")
    candidates["priority"] = candidates["priority"].fillna(1.0)
    if max_queries is not None and max_queries < len(candidates):
        weights = candidates["priority"].clip(lower=1e-9).to_numpy()
        chosen = np.random.default_rng(seed).choice(len(candidates), size=max_queries, replace=False, p=weights / weights.sum())
        candidates = candidates.iloc[chosen]
    return candidates.sort_values(["priority", "protein_id"], ascending=[False, True])["protein_id"].tolist()


def load_reference_records(genes_path: Path, operon_aware: bool) -> dict[str, dict]:
    genes = pd.read_csv(genes_path, sep="\t", dtype=str)
    if "feature_kind" not in genes.columns:
        genes["feature_kind"] = "gene"
    records = {
        row.ensembl_gene_id: {
            "id": row.ensembl_gene_id,
            "gene_id": row.ensembl_gene_id,
            "seq_region_name": row.seq_region_name,
            "start": int(row.start),
            "end": int(row.end),
            "strand": int(row.strand),
        }
        for row in genes.itertuples(index=False)
    }
    if operon_aware:
        gene_rows = {gene_id: records[gene_id] for gene_id in genes.loc[genes["feature_kind"] == "gene", "ensembl_gene_id"]}
        leaders = resolve_operon_leaders(gene_rows)
        records = {gene_id: leaders.get(gene_id, record) for gene_id, record in records.items()}
    return records


def extract_reference_promoters(
    species: str,
    assignments: pd.DataFrame,
    work_dir: Path,
    window: int,
    timeout: float,
    keep_genome: bool,
    ncbi_source: dict | None = None,
) -> pd.DataFrame:
    """Download one reference genome (+ annotation for Ensembl) and slice promoters for its assigned genes.

    `ncbi_source` (``{"accession", "genes_path", "operon_aware"}``) marks an NCBI Datasets reference whose
    gene coordinates are already cached next to its proteome.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = Path(tempfile.mkdtemp(prefix=f"{species}.", dir=work_dir))
    try:
        if ncbi_source is None:
            download_gene_coordinates(species, reference_dir, timeout)
            download_genome_fasta(species, reference_dir, timeout)
            genes_path, fasta = reference_dir / f"{species}.genes.tsv", reference_dir / f"{species}.fa"
            records = load_reference_records(genes_path, species in OPERON_AWARE_SPECIES.values())
        else:
            genes_path = Path(ncbi_source["genes_path"])
            fasta = download_ncbi_genome(ncbi_source["accession"], reference_dir, timeout)
            records = load_reference_records(genes_path, bool(ncbi_source["operon_aware"]))
        regions = {
            gene_id: region
            for gene_id in assignments["reference_gene_id"].unique()
            if (record := records.get(gene_id)) is not None and (region := _promoter_region(record, window)) is not None
        }
        reference = LocalPromoterReference(species, reference_dir / "unused", genes_path, fasta)
        sequences = fetch_local_sequences(sorted(set(regions.values())), reference)
        rows = []
        for row in assignments.itertuples(index=False):
            region = regions.get(row.reference_gene_id)
            if region is None:
                continue
            sequence = sequences.get(_region_string(*region))
            if not sequence:
                continue
            seq_region_name, start, end, strand = region
            rows.append(
                {
                    "protein_id": row.protein_id,
                    "taxon_id": row.taxon_id,
                    "species": species,
                    "ensembl_gene_id": row.reference_gene_id,
                    "seq_region_name": seq_region_name,
                    "promoter_start": start,
                    "promoter_end": end,
                    "strand": strand,
                    "promoter_length": len(sequence),
                    "promoter_sequence": sequence,
                    "promoter_source": row.promoter_source,
                    "reference_pident": row.reference_pident,
                    "reference_query_coverage": row.reference_query_coverage,
                    "reference_lineage_depth": row.reference_lineage_depth,
                }
            )
        LOGGER.info("[%s] extracted %d/%d assigned promoters", species, len(rows), len(assignments))
        return pd.DataFrame(rows, columns=RESCUE_COLUMNS)
    finally:
        if keep_genome:
            LOGGER.info("[%s] keeping reference at %s", species, reference_dir)
        else:
            shutil.rmtree(reference_dir, ignore_errors=True)


def ncbi_panel(
    missing_by_taxon: pd.Series,
    min_missing: int,
    max_taxa: int | None,
    proteome_cache: Path,
    parent_by_taxon: dict[str, str],
    rank_by_taxon: dict[str, str],
    timeout: float,
    selected_taxa: list[str] | None = None,
) -> tuple[dict[str, Path], dict[str, str], dict[str, str], dict[str, dict]]:
    """NCBI assemblies for taxa with many missing proteins, including taxa with incomplete Ensembl coverage.

    Returns (proteomes, query taxon -> reference id, reference id -> assembly taxon, reference id -> extraction source).
    """
    candidates = missing_by_taxon[missing_by_taxon.index.isin(selected_taxa)] if selected_taxa is not None else missing_by_taxon[missing_by_taxon >= min_missing]
    candidates = candidates.sort_values(ascending=False)
    if max_taxa is not None:
        candidates = candidates.head(max_taxa)
    LOGGER.info("resolving NCBI assemblies for %d taxa (>= %d missing proteins each)", len(candidates), min_missing)
    assemblies = select_assemblies(list(candidates.index), proteome_cache / "ncbi_assemblies.json", timeout)
    species_parent = {
        taxon_id: next((node for node in reversed(lineage(taxon_id, parent_by_taxon)) if rank_by_taxon.get(node) == "species"), None)
        for taxon_id, assembly in assemblies.items() if assembly is None
    }
    parent_assemblies = select_assemblies(
        sorted({node for node in species_parent.values() if node}), proteome_cache / "ncbi_assemblies.json", timeout
    ) if species_parent else {}
    for taxon_id, parent in species_parent.items():
        assemblies[taxon_id] = parent_assemblies.get(parent)

    proteomes: dict[str, Path] = {}
    reference_by_taxon: dict[str, str] = {}
    taxon_by_reference: dict[str, str] = {}
    sources: dict[str, dict] = {}
    for taxon_id, assembly in assemblies.items():
        if not assembly:
            continue
        ref = reference_id(assembly["accession"])
        if ref not in sources:
            built = build_ncbi_proteome(assembly["accession"], proteome_cache, timeout)
            if built is None:
                continue
            proteomes[ref], genes_path = built
            taxon_by_reference[ref] = assembly["taxon_id"]
            sources[ref] = {
                "accession": assembly["accession"],
                "genes_path": str(genes_path),
                "operon_aware": bool(PROKARYOTE_TAXA & set(lineage(assembly["taxon_id"], parent_by_taxon))),
            }
        if assembly["taxon_id"] == taxon_id:
            reference_by_taxon[taxon_id] = ref
    LOGGER.info("NCBI panel: %d assemblies covering %d query taxa", len(sources), len(reference_by_taxon))
    return proteomes, reference_by_taxon, taxon_by_reference, sources


def write_query_fasta(fasta_path: Path, protein_ids: list[str], output: Path) -> None:
    if fasta_path.suffix.lower() == ".tsv":
        candidates = [
            fasta_path.with_name(fasta_path.name.replace("_taxonomy.tsv", "_sequences.fasta")),
            fasta_path.with_suffix(".fasta"),
            fasta_path.with_name("testsuperset.fasta"),
        ]
        fasta_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if fasta_path is None:
            raise FileNotFoundError(
                f"Could not find a sequence FASTA next to taxonomy input; tried {', '.join(str(path) for path in candidates)}"
            )
    wanted = set(protein_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        with fasta_path.open() as fasta_handle:
            for record in SeqIO.parse(fasta_handle, "fasta"):
                protein_id = canonical_protein_id(record.id)
                if protein_id in wanted:
                    handle.write(f">{protein_id}\n{record.seq}\n")


def run(
    fasta_path: Path,
    promoter_dir: Path,
    embedding_dir: Path,
    proteome_cache: Path,
    work_dir: Path,
    *,
    max_queries: int | None,
    n_clusters: int,
    n_components: int,
    threads: int,
    diamond: str,
    diamond_mode: str,
    max_targets: int,
    exact_identity: float,
    exact_coverage: float,
    proxy_identity: float,
    proxy_coverage: float,
    proxy_rank: str | None,
    window: int,
    timeout: float,
    keep_genomes: bool,
    seed: int,
    reference_species: list[str] | None,
    ncbi_min_missing: int | None,
    ncbi_max_taxa: int | None,
    ncbi_taxa: list[str] | None,
) -> None:
    rescue_dir = promoter_dir / "alignment_rescue"
    parts_dir = rescue_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_protein_taxa(fasta_path)
    taxonomy["protein_id"] = taxonomy["protein_id"].map(canonical_protein_id)
    invalid_taxa = taxonomy["taxon_id"].astype(str).str.fullmatch(r"\d+").eq(False)
    if invalid_taxa.any():
        examples = taxonomy.loc[invalid_taxa, "taxon_id"].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{fasta_path} does not contain numeric taxon IDs; use a two-column taxonomy TSV "
            f"such as data/Train/train_taxonomy.tsv (examples: {examples})"
        )
    taxon_by_protein = dict(zip(taxonomy["protein_id"], taxonomy["taxon_id"]))
    covered = set()
    for frame in load_promoter_files(promoter_dir, exclude={RESCUE_FILENAME}):
        covered.update(frame["protein_id"].dropna().map(canonical_protein_id))
    missing = [protein_id for protein_id in taxonomy["protein_id"] if protein_id not in covered]
    LOGGER.info("%d/%d proteins have xref promoters; %d missing", len(covered & set(taxon_by_protein)), len(taxonomy), len(missing))

    priority = embedding_priority(embedding_dir, covered, n_components, n_clusters, seed)
    priority.to_csv(rescue_dir / "embedding_priority.tsv", sep="\t", index=False)
    queries = choose_queries(priority, missing, max_queries, seed)
    LOGGER.info("aligning %d missing proteins", len(queries))

    panel = reference_species or sorted(set(ANCHOR_SPECIES.values()) | set(OPERON_AWARE_SPECIES.values()))
    proteomes = {species: path for species in panel if (path := download_proteome(species, proteome_cache, timeout)) is not None}

    parent_by_taxon, rank_by_taxon = load_taxonomy_nodes(_download_ncbi_nodes(DEFAULT_TAXDUMP_CACHE_DIR))
    species_by_taxon = {
        taxon_id: species for taxon_id, species in {**ANCHOR_SPECIES, **OPERON_AWARE_SPECIES}.items() if species in proteomes
    }
    taxon_by_species: dict[str, str] = {}
    for taxon_id, species in species_by_taxon.items():
        taxon_by_species.setdefault(species, taxon_id)
    ncbi_sources: dict[str, dict] = {}
    if ncbi_min_missing is not None or ncbi_taxa is not None:
        missing_by_taxon = pd.Series([taxon_by_protein[protein_id] for protein_id in missing]).value_counts()
        ncbi_proteomes, reference_by_taxon, taxon_by_reference, ncbi_sources = ncbi_panel(
            missing_by_taxon, ncbi_min_missing or 0, ncbi_max_taxa, proteome_cache, parent_by_taxon, rank_by_taxon, timeout, ncbi_taxa
        )
        proteomes.update(ncbi_proteomes)
        for taxon_id, reference in reference_by_taxon.items():
            species_by_taxon.setdefault(taxon_id, reference)
        taxon_by_species.update(taxon_by_reference)

    database = build_reference_database(proteomes, work_dir / "reference_panel.dmnd", diamond, threads)

    query_fasta = work_dir / "queries.fa"
    hits_path = work_dir / "hits.tsv"
    query_ids_path = work_dir / "hits.queries.json"
    run_key = {"queries": sorted(queries), "panel": sorted(proteomes), "mode": diamond_mode, "max_targets": max_targets}
    if hits_path.exists() and query_ids_path.exists() and json.loads(query_ids_path.read_text()) == run_key:
        LOGGER.info("reusing cached alignments in %s", hits_path)
    else:
        write_query_fasta(fasta_path, queries, query_fasta)
        run_diamond(query_fasta, database, hits_path, diamond, threads, diamond_mode, max_targets)
        query_ids_path.write_text(json.dumps(run_key))
    hits = (
        pd.read_csv(hits_path, sep="\t", names=HIT_FIELDS, dtype={"qseqid": str, "sseqid": str})
        if hits_path.stat().st_size > 0
        else pd.DataFrame(columns=HIT_FIELDS)
    )

    assignments = assign_reference_genes(
        hits,
        taxon_by_protein,
        species_by_taxon,
        taxon_by_species,
        parent_by_taxon,
        rank_by_taxon,
        exact_identity=exact_identity,
        exact_coverage=exact_coverage,
        proxy_identity=proxy_identity,
        proxy_coverage=proxy_coverage,
        proxy_rank=proxy_rank,
    )
    assignments.to_csv(rescue_dir / "assignments.tsv", sep="\t", index=False)
    LOGGER.info("assigned reference genes: %s", assignments["promoter_source"].value_counts().to_dict())

    if assignments.empty:
        pd.DataFrame(columns=RESCUE_COLUMNS).to_csv(promoter_dir / RESCUE_FILENAME, sep="\t", index=False)
        write_combined(promoter_dir)
        LOGGER.warning("No reference genes passed the identity, coverage, and lineage filters")
        return

    extracted_species = []
    for species, group in assignments.groupby("reference_species"):
        part = parts_dir / f"{species}.tsv"
        assigned_ids = parts_dir / f"{species}.assigned.json"
        expected = sorted(group["protein_id"])
        if part.exists() and assigned_ids.exists() and json.loads(assigned_ids.read_text()) == expected:
            extracted_species.append(species)
            continue
        try:
            rows = extract_reference_promoters(species, group, work_dir, window, timeout, keep_genomes, ncbi_sources.get(species))
        except (requests.RequestException, RuntimeError, OSError, ValueError):
            LOGGER.exception("[%s] promoter extraction failed; will retry next run", species)
            continue
        rows.to_csv(part, sep="\t", index=False)
        assigned_ids.write_text(json.dumps(expected))
        extracted_species.append(species)

    current = set(zip(assignments["protein_id"], assignments["reference_species"]))
    parts = [pd.read_csv(parts_dir / f"{species}.tsv", sep="\t", dtype={"taxon_id": str}) for species in extracted_species]
    rescued = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RESCUE_COLUMNS)
    rescued = rescued[[pair in current for pair in zip(rescued["protein_id"], rescued["species"])]]
    rescued = rescued[~rescued["protein_id"].isin(covered)].drop_duplicates("protein_id")
    rescued.to_csv(promoter_dir / RESCUE_FILENAME, sep="\t", index=False)
    write_combined(promoter_dir)
    total_covered = len((covered | set(rescued["protein_id"])) & set(taxon_by_protein))
    LOGGER.info(
        "rescued %d promoters (%s); coverage now %d/%d (%.2f%%)",
        len(rescued),
        rescued["promoter_source"].value_counts().to_dict(),
        total_covered,
        len(taxonomy),
        100 * total_covered / len(taxonomy),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fasta", type=Path, help="Query FASTA with '>protein_id taxon_id' headers (e.g. testsuperset.fasta)")
    parser.add_argument("promoter_dir", type=Path, help="Existing promoter output directory (per-species TSVs + combined.tsv)")
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/features/embeddings/test_esm2_150M"))
    parser.add_argument("--proteome-cache", type=Path, default=Path("data/external/promoter_proteomes"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/external/promoter_rescue"))
    parser.add_argument("--max-queries", type=int, default=None, help="Align only a priority-weighted sample of this many missing proteins")
    parser.add_argument("--n-clusters", type=int, default=256, help="k-means clusters in PCA-reduced embedding space")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--diamond", default="diamond", help="DIAMOND executable")
    parser.add_argument("--diamond-mode", default="sensitive", choices=("fast", "default", "mid-sensitive", "sensitive", "more-sensitive"))
    parser.add_argument("--max-target-seqs", type=int, default=25)
    parser.add_argument("--exact-identity", type=float, default=95.0, help="Min %% identity for an own-species gene match")
    parser.add_argument("--exact-coverage", type=float, default=80.0, help="Min %% query coverage for an own-species gene match")
    parser.add_argument("--proxy-identity", type=float, default=40.0)
    parser.add_argument("--proxy-coverage", type=float, default=60.0)
    parser.add_argument(
        "--proxy-rank",
        default="phylum",
        help="Proxy reference must share the query's ancestor at this NCBI rank; 'any' disables the check, 'none' disables proxies",
    )
    parser.add_argument("--window", type=int, default=3000)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--keep-genomes", action="store_true", help="Keep downloaded reference genomes instead of deleting each after use")
    parser.add_argument("--reference-species", nargs="*", default=None, help="Restrict the reference panel (default: all configured species)")
    parser.add_argument(
        "--ncbi-min-missing",
        type=int,
        default=20,
        help="Add an NCBI Datasets annotated assembly for every taxon with at least this many missing proteins (0 disables)",
    )
    parser.add_argument("--ncbi-max-taxa", type=int, default=None, help="Cap the number of NCBI taxa (largest missing counts first)")
    parser.add_argument("--ncbi-taxa", nargs="+", default=None, help="Restrict NCBI assemblies to these query taxon IDs (for focused pilot runs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run(
        args.fasta,
        args.promoter_dir,
        args.embedding_dir,
        args.proteome_cache,
        args.work_dir,
        max_queries=args.max_queries,
        n_clusters=args.n_clusters,
        n_components=args.pca_components,
        threads=args.threads,
        diamond=args.diamond,
        diamond_mode=args.diamond_mode,
        max_targets=args.max_target_seqs,
        exact_identity=args.exact_identity,
        exact_coverage=args.exact_coverage,
        proxy_identity=args.proxy_identity,
        proxy_coverage=args.proxy_coverage,
        proxy_rank=None if args.proxy_rank == "any" else args.proxy_rank,
        window=args.window,
        timeout=args.timeout,
        keep_genomes=args.keep_genomes,
        seed=args.seed,
        reference_species=args.reference_species,
        ncbi_min_missing=args.ncbi_min_missing or None,
        ncbi_max_taxa=args.ncbi_max_taxa,
        ncbi_taxa=args.ncbi_taxa,
    )


if __name__ == "__main__":
    main()
