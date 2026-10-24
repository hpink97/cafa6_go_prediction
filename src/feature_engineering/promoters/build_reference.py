"""Build local promoter reference files from Ensembl bulk sources."""

from __future__ import annotations

import argparse
import gzip
import logging
import re
import shutil
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from feature_engineering.promoters.extract_promoters import load_protein_taxa
from feature_engineering.promoters.species_map import ANCHOR_SPECIES, OPERON_AWARE_SPECIES

LOGGER = logging.getLogger("cafa6.promoters.build_reference")
BIOMART_URL = "https://www.ensembl.org/biomart/martservice"
ENSEMBL_FTP_ROOT = "https://ftp.ensembl.org/pub/current_fasta"
ENSEMBL_GFF3_ROOT = "https://ftp.ensembl.org/pub/current_gff3"
ENSEMBL_GENOMES_ROOT = "https://ftp.ensemblgenomes.ebi.ac.uk/pub"
UNIPROT_ID_MAPPING_URL = "https://rest.uniprot.org/idmapping"
ENSEMBL_ORGANISMS_ROOTS: dict[str, str] = {
    "xenopus_laevis": "http://ftp.ebi.ac.uk/pub/ensemblorganisms/GCA/017/654/675/1/ensembl/2022_01",
}
ENSEMBL_GENOMES_SPECIES_DIVISION: dict[str, str] = {
    "oryza_sativa": "plants",
    "arabidopsis_thaliana": "plants",
    "zea_mays": "plants",
    "solanum_lycopersicum_gca000188115v5cm": "plants",
    "glycine_max": "plants",
    "solanum_tuberosum": "plants",
    "triticum_aestivum": "plants",
    "chlamydomonas_reinhardtii": "plants",
    "hordeum_vulgare": "plants",
    "drosophila_melanogaster": "metazoa",
    "drosophila_pseudoobscura_gca009870125v2rs": "metazoa",
    "drosophila_virilis_gca030788295v1rs": "metazoa",
    "drosophila_yakuba_gca016746365v2rs": "metazoa",
    "caenorhabditis_elegans": "metazoa",
    "caenorhabditis_briggsae": "metazoa",
    "anopheles_gambiae": "metazoa",
    "aedes_aegypti_lvpagwg": "metazoa",
    "bombyx_mori_gca030269925v1rs": "metazoa",
    "saccharomyces_cerevisiae": "fungi",
    "schizosaccharomyces_pombe": "fungi",
    "candida_albicans": "fungi",
    "aspergillus_nidulans": "fungi",
    "neurospora_crassa": "fungi",
    "aspergillus_fumigatus": "fungi",
    "yarrowia_lipolytica": "fungi",
    "aspergillus_oryzae": "fungi",
    "aspergillus_terreus": "fungi",
    "fusarium_graminearum": "fungi",
    "aspergillus_niger": "fungi",
    "pyricularia_oryzae": "fungi",
    "aspergillus_fischeri": "fungi",
    "aspergillus_clavatus": "fungi",
    "aspergillus_flavus": "fungi",
    "dictyostelium_discoideum": "protists",
    "plasmodium_falciparum": "protists",
}
SPECIES_UNIPROT_TARGETS: dict[str, tuple[str, ...]] = {
    "arabidopsis_thaliana": ("Araport", "Ensembl_Genomes"),
    "schizosaccharomyces_pombe": ("PomBase", "Ensembl_Genomes"),
    "dictyostelium_discoideum": ("dictyBase", "Ensembl_Genomes"),
    "zea_mays": ("MaizeGDB", "Ensembl_Genomes"),
    "drosophila_melanogaster": ("Ensembl_Genomes", "FlyBase"),
    "drosophila_pseudoobscura_gca009870125v2rs": ("Ensembl_Genomes", "FlyBase"),
    "drosophila_virilis_gca030788295v1rs": ("Ensembl_Genomes", "FlyBase"),
    "drosophila_yakuba_gca016746365v2rs": ("Ensembl_Genomes", "FlyBase"),
    "caenorhabditis_elegans": ("Ensembl_Genomes", "WormBase"),
    "caenorhabditis_briggsae": ("Ensembl_Genomes", "WormBase"),
    "xenopus_tropicalis": ("Ensembl", "Ensembl_Protein", "Ensembl_Transcript", "Xenbase"),
    "xenopus_laevis": ("Ensembl", "Ensembl_Protein", "Ensembl_Transcript", "Xenbase"),
    "danio_rerio": ("Ensembl", "Ensembl_Protein", "Ensembl_Transcript", "ZFIN"),
    "rattus_norvegicus": ("Ensembl", "Ensembl_Protein", "Ensembl_Transcript", "RGD"),
    "pseudomonas_aeruginosa_pao1_gca_000006765": ("PseudoCAP", "Ensembl_Genomes"),
    "mycobacterium_tuberculosis_h37rv": ("TubercuList", "Ensembl_Genomes"),
}


def build_reference(species: str, output_dir: Path, input_path: Path | None = None, timeout: float = 120) -> None:
    """Build protein-gene, gene-coordinate, and genome FASTA files for one Ensembl species."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_path is None:
        LOGGER.info("[%s] querying BioMart for UniProt xrefs and gene coordinates", species)
        biomart = fetch_biomart_table(species, timeout)
        write_reference_tables(species, biomart, output_dir)
    else:
        protein_ids = load_species_protein_ids(input_path, species)
        LOGGER.info("[%s] downloading and parsing Ensembl GFF3 gene coordinates", species)
        download_gene_coordinates(species, output_dir, timeout)
        valid_gene_ids = load_reference_gene_ids(output_dir / f"{species}.genes.tsv")
        LOGGER.info("[%s] resolving %d input proteins through UniProt bulk ID mapping", species, len(protein_ids))
        if species in ENSEMBL_ORGANISMS_ROOTS:
            protein_genes = fetch_ensembl_organisms_xref_mappings(species, protein_ids, timeout, valid_gene_ids)
        else:
            protein_genes = fetch_uniprot_mappings(protein_ids, timeout, targets=SPECIES_UNIPROT_TARGETS.get(species), valid_gene_ids=valid_gene_ids)
        protein_genes.to_csv(output_dir / f"{species}.protein_gene.tsv", sep="\t", index=False)
        LOGGER.info("[%s] wrote %d UniProt xref rows", species, len(protein_genes))
    LOGGER.info("[%s] downloading genome FASTA", species)
    download_genome_fasta(species, output_dir, timeout)


def load_reference_gene_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path, sep="\t", dtype=str, usecols=["ensembl_gene_id"])
    return set(frame["ensembl_gene_id"].dropna())


def fetch_ensembl_organisms_xref_mappings(species: str, protein_ids: list[str], timeout: float, valid_gene_ids: set[str]) -> pd.DataFrame:
    url = f"{ENSEMBL_ORGANISMS_ROOTS[species]}/geneset/xref.tsv.gz"
    response = request_with_retries("GET", url, timeout=timeout)
    response.raise_for_status()
    from io import BytesIO

    with gzip.open(BytesIO(response.content), "rt") as handle:
        xrefs = pd.read_csv(handle, sep="\t", dtype=str)
    wanted = set(protein_ids)
    xrefs = xrefs[xrefs["xref_id"].isin(wanted)]
    xrefs = xrefs[xrefs["gene_stable_id"].isin(valid_gene_ids)]
    frame = xrefs.rename(columns={"xref_id": "protein_id", "gene_stable_id": "ensembl_gene_id"})
    frame = frame[["protein_id", "ensembl_gene_id"]].dropna().drop_duplicates(subset=["protein_id"])
    LOGGER.info("[%s] Ensembl Organisms xref table resolved %d/%d proteins", species, len(frame), len(wanted))
    if frame.empty:
        raise RuntimeError(f"{species} Ensembl Organisms xref table returned no coordinate-resolvable UniProt mappings")
    return frame.sort_values("protein_id")


def load_species_protein_ids(input_path: Path, species: str) -> list[str]:
    all_species = {**ANCHOR_SPECIES, **OPERON_AWARE_SPECIES}
    taxon_ids = {taxon_id for taxon_id, mapped_species in all_species.items() if mapped_species == species}
    if not taxon_ids:
        raise ValueError(f"{species!r} is not present in the configured anchor species map")
    taxonomy = load_protein_taxa(input_path)
    protein_ids = taxonomy[taxonomy["taxon_id"].isin(taxon_ids)]["protein_id"].dropna().drop_duplicates().tolist()
    if not protein_ids:
        raise RuntimeError(f"No proteins for {species} found in {input_path}")
    return protein_ids


def fetch_uniprot_mappings(
    protein_ids: list[str],
    timeout: float,
    batch_size: int = 5000,
    targets: tuple[str, ...] | None = None,
    valid_gene_ids: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    targets = targets or ("Ensembl", "Ensembl_Protein", "Ensembl_Transcript", "Ensembl_Genomes")
    for target in targets:
        target_rows: list[dict[str, str]] = []
        for start in range(0, len(protein_ids), batch_size):
            batch = protein_ids[start : start + batch_size]
            response = request_with_retries(
                "POST",
                f"{UNIPROT_ID_MAPPING_URL}/run",
                timeout=timeout,
                data={"from": "UniProtKB_AC-ID", "to": target, "ids": ",".join(batch)},
            )
            response.raise_for_status()
            job_id = response.json()["jobId"]
            wait_for_uniprot_job(job_id, timeout)
            result = request_with_retries("GET", f"{UNIPROT_ID_MAPPING_URL}/stream/{job_id}", timeout=timeout, params={"format": "tsv"})
            result.raise_for_status()
            frame = pd.read_csv(StringIO(result.text), sep="\t", dtype=str)
            if frame.empty:
                continue
            frame = frame.rename(columns={"From": "protein_id", "To": "ensembl_gene_id"})
            frame["ensembl_gene_id"] = frame["ensembl_gene_id"].map(normalize_ensembl_gene_id)
            frame = frame[["protein_id", "ensembl_gene_id"]].dropna()
            if valid_gene_ids is not None:
                frame = frame[frame["ensembl_gene_id"].isin(valid_gene_ids)]
            target_rows.extend(frame.to_dict("records"))
        if valid_gene_ids is None:
            LOGGER.info("UniProt %s mapping returned %d rows", target, len(target_rows))
        else:
            LOGGER.info("UniProt %s mapping returned %d coordinate-resolvable rows", target, len(target_rows))
        rows.extend(target_rows)
    if not rows:
        raise RuntimeError("UniProt ID mapping returned no coordinate-resolvable gene IDs")
    return pd.DataFrame(rows).drop_duplicates(subset=["protein_id"]).sort_values("protein_id")


def wait_for_uniprot_job(job_id: str, timeout: float) -> None:
    deadline = time.monotonic() + max(timeout, 60)
    while True:
        response = request_with_retries("GET", f"{UNIPROT_ID_MAPPING_URL}/status/{job_id}", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if "results" in payload or payload.get("jobStatus") == "FINISHED":
            return
        if payload.get("jobStatus") in {"ERROR", "FAILED"}:
            raise RuntimeError(f"UniProt mapping job {job_id} failed: {payload}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for UniProt mapping job {job_id}")
        time.sleep(1)


def request_with_retries(method: str, url: str, timeout: float, max_retries: int = 5, **kwargs) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return requests.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            backoff = min(30, 2**attempt)
            LOGGER.warning("network error on %s %s (attempt %d/%d): %s; sleeping %ds", method, url, attempt + 1, max_retries, exc, backoff)
            time.sleep(backoff)
    assert last_exc is not None
    raise last_exc


def fetch_biomart_table(species: str, timeout: float) -> pd.DataFrame:
    dataset = f"{species}_gene_ensembl"
    query = f"""
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="{dataset}" interface="default">
    <Attribute name="ensembl_gene_id" />
    <Attribute name="chromosome_name" />
    <Attribute name="start_position" />
    <Attribute name="end_position" />
    <Attribute name="strand" />
    <Attribute name="uniprotswissprot" />
    <Attribute name="uniprotsptrembl" />
  </Dataset>
</Query>
""".strip()
    response = requests.get(BIOMART_URL, params={"query": query}, timeout=timeout)
    response.raise_for_status()
    if response.text.startswith("Query ERROR"):
        raise RuntimeError(response.text.strip())
    frame = pd.read_csv(StringIO(response.text), sep="\t", dtype=str)
    if frame.empty:
        raise RuntimeError(f"BioMart returned no rows for {dataset}")
    return frame


def write_reference_tables(species: str, biomart: pd.DataFrame, output_dir: Path) -> None:
    columns = list(biomart.columns)
    gene_column = _column_containing(columns, "Gene stable ID")
    seq_region_column = _column_containing(columns, "Chromosome/scaffold name")
    start_column = _column_containing(columns, "Gene start")
    end_column = _column_containing(columns, "Gene end")
    strand_column = _column_containing(columns, "Strand")
    swissprot_column = _column_containing(columns, "Swiss-Prot")
    trembl_column = _column_containing(columns, "TrEMBL")

    genes = biomart[[gene_column, seq_region_column, start_column, end_column, strand_column]].dropna(subset=[gene_column])
    genes = genes.drop_duplicates(subset=[gene_column]).rename(
        columns={
            gene_column: "ensembl_gene_id",
            seq_region_column: "seq_region_name",
            start_column: "start",
            end_column: "end",
            strand_column: "strand",
        }
    )
    genes = genes[["ensembl_gene_id", "seq_region_name", "start", "end", "strand"]]
    genes.to_csv(output_dir / f"{species}.genes.tsv", sep="\t", index=False)
    LOGGER.info("[%s] wrote %d gene coordinate rows", species, len(genes))

    xrefs = []
    for protein_column in (swissprot_column, trembl_column):
        entries = biomart[[gene_column, protein_column]].dropna()
        entries = entries[entries[protein_column].str.len() > 0]
        entries = entries.rename(columns={gene_column: "ensembl_gene_id", protein_column: "protein_id"})
        xrefs.append(entries[["protein_id", "ensembl_gene_id"]])
    protein_genes = pd.concat(xrefs, ignore_index=True).drop_duplicates(subset=["protein_id"])
    protein_genes = protein_genes.sort_values("protein_id")
    protein_genes.to_csv(output_dir / f"{species}.protein_gene.tsv", sep="\t", index=False)
    LOGGER.info("[%s] wrote %d UniProt xref rows", species, len(protein_genes))


def download_genome_fasta(species: str, output_dir: Path, timeout: float) -> None:
    fasta_path = output_dir / f"{species}.fa"
    if fasta_path.exists():
        LOGGER.info("[%s] genome FASTA already exists at %s", species, fasta_path)
        return

    directory_url = fasta_directory_url(species, timeout)
    response = requests.get(directory_url, timeout=timeout)
    response.raise_for_status()
    fasta_name = choose_fasta_name(response.text)
    fasta_url = f"{directory_url}{fasta_name}"
    compressed_path = output_dir / fasta_name
    with requests.get(fasta_url, stream=True, timeout=timeout) as download:
        download.raise_for_status()
        with compressed_path.open("wb") as handle:
            for chunk in download.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    with gzip.open(compressed_path, "rb") as source, fasta_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    compressed_path.unlink()
    LOGGER.info("[%s] wrote genome FASTA to %s", species, fasta_path)


def download_gene_coordinates(species: str, output_dir: Path, timeout: float) -> None:
    genes_path = output_dir / f"{species}.genes.tsv"
    if genes_path.exists():
        LOGGER.info("[%s] gene coordinate table already exists at %s", species, genes_path)
        return
    directory_url = gff3_directory_url(species, timeout)
    response = requests.get(directory_url, timeout=timeout)
    response.raise_for_status()
    gff3_name = choose_gff3_name(response.text)
    compressed_path = output_dir / gff3_name
    with requests.get(f"{directory_url}{gff3_name}", stream=True, timeout=timeout) as download:
        download.raise_for_status()
        with compressed_path.open("wb") as handle:
            for chunk in download.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    genes = parse_gff3_genes(compressed_path)
    genes.to_csv(genes_path, sep="\t", index=False)
    compressed_path.unlink()
    LOGGER.info("[%s] wrote %d gene coordinate rows", species, len(genes))


def choose_fasta_name(directory_html: str) -> str:
    names = sorted(set(re.findall(r'href="([^"]+\.fa\.(?:gz|bgz))"', directory_html)))
    if not names:
        raise RuntimeError("Could not find a genome FASTA .fa.gz/.fa.bgz link in Ensembl FTP directory listing")
    for preferred in ("unmasked.fa.bgz", "softmasked.fa.bgz", "hardmasked.fa.bgz"):
        matches = [name for name in names if name.endswith(preferred)]
        if matches:
            return matches[0]
    for preferred in ("dna.toplevel.fa.gz", "dna.primary_assembly.fa.gz"):
        matches = [name for name in names if name.endswith(preferred)]
        if matches:
            return matches[0]
    dna_names = [name for name in names if ".dna." in name]
    if dna_names:
        return dna_names[0]
    return names[0]


def choose_gff3_name(directory_html: str) -> str:
    names = sorted(set(re.findall(r'href="([^"]+\.gff3\.gz)"', directory_html)))
    if not names:
        raise RuntimeError("Could not find a .gff3.gz link in Ensembl GFF3 directory listing")
    matches = [name for name in names if name.endswith(".chr.gff3.gz")]
    if matches:
        return matches[0]
    non_abinitio = [name for name in names if ".abinitio." not in name and ".chromosome." not in name]
    if non_abinitio:
        return non_abinitio[0]
    return names[0]


def fasta_directory_url(species: str, timeout: float) -> str:
    if species in ENSEMBL_ORGANISMS_ROOTS:
        return f"{ENSEMBL_ORGANISMS_ROOTS[species]}/genome/"
    if species in ENSEMBL_GENOMES_SPECIES_DIVISION:
        division = ENSEMBL_GENOMES_SPECIES_DIVISION[species]
        return f"{ENSEMBL_GENOMES_ROOT}/{division}/current/fasta/{species}/dna/"
    if species in OPERON_AWARE_SPECIES.values():
        return f"{find_bacteria_species_url(species, 'fasta', timeout)}dna/"
    return f"{ENSEMBL_FTP_ROOT}/{species}/dna/"


def gff3_directory_url(species: str, timeout: float) -> str:
    if species in ENSEMBL_ORGANISMS_ROOTS:
        return f"{ENSEMBL_ORGANISMS_ROOTS[species]}/geneset/"
    if species in ENSEMBL_GENOMES_SPECIES_DIVISION:
        division = ENSEMBL_GENOMES_SPECIES_DIVISION[species]
        return f"{ENSEMBL_GENOMES_ROOT}/{division}/current/gff3/{species}/"
    if species in OPERON_AWARE_SPECIES.values():
        return find_bacteria_species_url(species, "gff3", timeout)
    return f"{ENSEMBL_GFF3_ROOT}/{species}/"


def find_bacteria_species_url(species: str, data_type: str, timeout: float) -> str:
    root = f"{ENSEMBL_GENOMES_ROOT}/bacteria/current/{data_type}/"
    response = requests.get(root, timeout=timeout)
    response.raise_for_status()
    collections = sorted(set(re.findall(r'href="(bacteria_\d+_collection/)"', response.text)))
    for collection in collections:
        url = f"{root}{collection}{species}/"
        species_response = requests.get(url, timeout=timeout)
        if species_response.status_code == 200 and "Index of" in species_response.text:
            return url
    raise RuntimeError(f"Could not find {species} under Ensembl Bacteria {data_type} collections")


def parse_gff3_genes(path: Path) -> pd.DataFrame:
    gene_records: dict[str, dict[str, str | int]] = {}
    transcript_to_gene: dict[str, str] = {}
    protein_to_gene: dict[str, str] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            feature_type = parts[2]
            attributes = parse_gff3_attributes(parts[8])
            if feature_type == "gene":
                gene_id = parse_gff3_gene_id(attributes)
                if gene_id is None:
                    continue
                gene_records[gene_id] = {
                    "ensembl_gene_id": gene_id,
                    "seq_region_name": parts[0],
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "strand": 1 if parts[6] == "+" else -1,
                    "feature_kind": "gene",
                }
            elif feature_type in {"mRNA", "transcript"}:
                transcript_id = normalize_ensembl_gene_id(attributes.get("transcript_id") or strip_gff3_prefix(attributes.get("ID", ""), "transcript"))
                parent_gene = normalize_ensembl_gene_id(strip_gff3_prefix(attributes.get("Parent", ""), "gene"))
                if transcript_id and parent_gene:
                    transcript_to_gene[transcript_id] = parent_gene
            elif feature_type == "CDS":
                protein_id = normalize_ensembl_gene_id(attributes.get("protein_id") or strip_gff3_prefix(attributes.get("ID", ""), "CDS"))
                parent_transcript = normalize_ensembl_gene_id(strip_gff3_prefix(attributes.get("Parent", ""), "transcript"))
                parent_gene = transcript_to_gene.get(parent_transcript)
                if protein_id and parent_gene:
                    protein_to_gene[protein_id] = parent_gene
    if not gene_records:
        raise RuntimeError(f"No gene rows found in {path}")
    rows = list(gene_records.values())
    for alias_by_gene in (transcript_to_gene, protein_to_gene):
        for alias, gene_id in alias_by_gene.items():
            record = gene_records.get(gene_id)
            if record is None or alias == gene_id:
                continue
            alias_record = dict(record)
            alias_record["ensembl_gene_id"] = alias
            alias_record["feature_kind"] = "alias"
            rows.append(alias_record)
    return pd.DataFrame(rows).drop_duplicates(subset=["ensembl_gene_id"])


def parse_gff3_gene_id(attributes: dict[str, str]) -> str | None:
    gene_id = attributes.get("gene_id") or strip_gff3_prefix(attributes.get("ID", ""), "gene")
    return normalize_ensembl_gene_id(gene_id) if gene_id else None


def parse_gff3_attributes(attributes: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for field in attributes.split(";"):
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        parsed[key] = value
    return parsed


def strip_gff3_prefix(value: str, prefix: str) -> str:
    return value.removeprefix(f"{prefix}:")


def normalize_ensembl_gene_id(gene_id: str) -> str:
    if gene_id.startswith("ENS"):
        prefix, dot, suffix = gene_id.rpartition(".")
        if dot and suffix.isdigit():
            return prefix
    return gene_id


def _column_containing(columns: list[str], text: str) -> str:
    matches = [column for column in columns if text in column]
    if not matches:
        raise ValueError(f"BioMart result is missing a column containing {text!r}; columns={columns}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory to write local promoter reference files")
    parser.add_argument("--species", nargs="+", required=True, help="Ensembl species names to build, or 'all' for configured anchors")
    parser.add_argument("--input", type=Path, default=None, help="Optional CAFA taxonomy TSV or FASTA; when supplied, resolve only this input's proteins through UniProt bulk ID mapping instead of BioMart")
    parser.add_argument("--timeout", type=float, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.species == ["all"]:
        species_names = sorted(set(ANCHOR_SPECIES.values()) | set(OPERON_AWARE_SPECIES.values()))
    else:
        species_names = args.species
    for species in species_names:
        build_reference(species, args.output_dir, args.input, args.timeout)


if __name__ == "__main__":
    main()