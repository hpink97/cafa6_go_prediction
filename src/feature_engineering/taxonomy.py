"""One-hot taxonomy features: top-N most frequent taxa, long tail bucketed as 'other',
plus a coarse taxonomical-kingdom one-hot shared by every protein regardless of vocabulary.
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
from pathlib import Path

import pandas as pd
import requests

LOGGER = logging.getLogger("cafa6.taxonomy")

NCBI_TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
DEFAULT_TAXDUMP_CACHE_DIR = Path("data/external/ncbi_taxdump")

# Fixed NCBI taxon IDs for coarse kingdom-level buckets (stable across NCBI taxonomy releases).
KINGDOM_TAXON_IDS: dict[str, str] = {
    "2": "bacteria",
    "2157": "archaea",
    "33208": "eukaryota_metazoa",
    "4751": "eukaryota_fungi",
    "33090": "eukaryota_viridiplantae",
}
EUKARYOTA_TAXON_ID = "2759"
KINGDOM_CATEGORIES = [*KINGDOM_TAXON_IDS.values(), "eukaryota_other", "unknown"]


def load_taxonomy(path: str | Path) -> pd.DataFrame:
    """Load a protein_id -> taxon_id mapping (no header, two tab-separated columns)."""
    return pd.read_csv(path, sep="\t", header=None, names=["protein_id", "taxon_id"], dtype={"taxon_id": str})


def build_vocabulary(taxonomy: pd.DataFrame, top_n: int) -> list[str]:
    """Most frequent `top_n` taxon IDs in the given table, most common first."""
    return taxonomy["taxon_id"].value_counts().head(top_n).index.tolist()


def encode_taxonomy(taxonomy: pd.DataFrame, vocabulary: list[str]) -> pd.DataFrame:
    """One-hot encode taxon_id against a fixed vocabulary; anything else -> taxon_other."""
    vocab_set = set(vocabulary)
    bucketed = taxonomy["taxon_id"].where(taxonomy["taxon_id"].isin(vocab_set), other="other")
    dummies = pd.get_dummies(bucketed, prefix="taxon", dtype=float)
    for taxon_id in [*vocabulary, "other"]:
        column = f"taxon_{taxon_id}"
        if column not in dummies:
            dummies[column] = 0.0
    ordered_columns = [f"taxon_{taxon_id}" for taxon_id in vocabulary] + ["taxon_other"]
    table = dummies[ordered_columns].copy()
    table.insert(0, "protein_id", taxonomy["protein_id"].to_numpy())
    return table


def _download_ncbi_nodes(cache_dir: Path) -> Path:
    """Download (once) and cache NCBI's taxdump, returning the path to nodes.dmp."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = cache_dir / "nodes.dmp"
    if nodes_path.exists():
        return nodes_path

    archive_path = cache_dir / "taxdump.tar.gz"
    LOGGER.info("downloading NCBI taxdump (one-time, ~60MB) from %s", NCBI_TAXDUMP_URL)
    response = requests.get(NCBI_TAXDUMP_URL, timeout=120, stream=True)
    response.raise_for_status()
    with open(archive_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1 << 20):
            file.write(chunk)
    with tarfile.open(archive_path) as archive:
        archive.extract("nodes.dmp", path=cache_dir, filter="data")
    archive_path.unlink()
    return nodes_path


def _load_parent_by_taxon(nodes_path: Path) -> dict[str, str]:
    """Parse nodes.dmp into a taxon_id -> parent_taxon_id map (ignores rank/other columns)."""
    parent_by_taxon: dict[str, str] = {}
    with open(nodes_path, encoding="utf-8") as file:
        for line in file:
            fields = line.split("\t|\t", 2)
            parent_by_taxon[fields[0].strip()] = fields[1].strip()
    return parent_by_taxon


def classify_kingdoms(taxon_ids: list[str], cache_dir: Path = DEFAULT_TAXDUMP_CACHE_DIR) -> dict[str, str]:
    """Map each unique taxon_id to a coarse kingdom bucket, via a local NCBI taxonomy lookup.

    Unlike the top-N taxon vocabulary, this gives every protein a meaningful (non-"other")
    signal regardless of how rare its exact species is, since almost everything falls under
    one of a handful of kingdom-level clades. Uses a one-time local download of NCBI's taxdump
    (nodes.dmp: taxon_id -> parent_taxon_id) instead of a per-taxon REST call -- that endpoint
    proved extremely flaky/slow (a 1,381-taxon run projected to take ~5 hours over the network).
    """
    nodes_path = _download_ncbi_nodes(cache_dir)
    parent_by_taxon = _load_parent_by_taxon(nodes_path)
    LOGGER.info("loaded %d taxonomy nodes from %s", len(parent_by_taxon), nodes_path)

    def _lineage_kingdom(taxon_id: str) -> str:
        current = taxon_id
        seen: set[str] = set()
        while current is not None and current not in seen:
            if current in KINGDOM_TAXON_IDS:
                return KINGDOM_TAXON_IDS[current]
            if current == EUKARYOTA_TAXON_ID:
                return "eukaryota_other"
            seen.add(current)
            parent = parent_by_taxon.get(current)
            if parent is None or parent == current:
                break
            current = parent
        return "unknown"

    kingdom_by_taxon = {taxon_id: _lineage_kingdom(taxon_id) for taxon_id in taxon_ids}
    unknown_count = sum(1 for kingdom in kingdom_by_taxon.values() if kingdom == "unknown")
    if unknown_count:
        LOGGER.warning("%d/%d taxa could not be classified (not found in NCBI taxdump)", unknown_count, len(taxon_ids))
    return kingdom_by_taxon


def encode_kingdom(taxonomy: pd.DataFrame, kingdom_by_taxon: dict[str, str]) -> pd.DataFrame:
    """One-hot encode each protein's coarse taxonomical kingdom."""
    kingdoms = taxonomy["taxon_id"].map(kingdom_by_taxon).fillna("unknown")
    dummies = pd.get_dummies(kingdoms, prefix="kingdom", dtype=float)
    for category in KINGDOM_CATEGORIES:
        column = f"kingdom_{category}"
        if column not in dummies:
            dummies[column] = 0.0
    ordered_columns = [f"kingdom_{category}" for category in KINGDOM_CATEGORIES]
    table = dummies[ordered_columns].copy()
    table.insert(0, "protein_id", taxonomy["protein_id"].to_numpy())
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="protein_id<TAB>taxon_id file (no header)")
    parser.add_argument("output", type=Path, help="Output feature table (TSV or CSV)")
    parser.add_argument("--top-n", type=int, default=50, help="Number of most frequent taxa to one-hot encode")
    parser.add_argument("--skip-taxon-vocabulary", action="store_true", help="Skip the individual per-species taxon one-hot block entirely (keep only kingdom, if not also skipped)")
    parser.add_argument("--vocabulary-in", type=Path, default=None, help="Reuse a vocabulary saved from a prior (e.g. train) run")
    parser.add_argument("--vocabulary-out", type=Path, default=None, help="Save the fitted vocabulary to this JSON path")
    parser.add_argument("--kingdom-map-in", type=Path, default=None, help="Reuse a taxon_id->kingdom map saved from a prior run")
    parser.add_argument("--kingdom-map-out", type=Path, default=None, help="Save the fetched taxon_id->kingdom map to this JSON path")
    parser.add_argument("--skip-kingdom", action="store_true", help="Skip the coarse kingdom one-hot block entirely")
    parser.add_argument(
        "--taxdump-cache-dir",
        type=Path,
        default=DEFAULT_TAXDUMP_CACHE_DIR,
        help="Local cache dir for NCBI's taxdump (downloaded once, reused across runs)",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    taxonomy = load_taxonomy(args.input)
    if args.skip_taxon_vocabulary:
        table = taxonomy[["protein_id"]].copy()
    else:
        if args.vocabulary_in is not None:
            vocabulary = json.loads(args.vocabulary_in.read_text())
        else:
            vocabulary = build_vocabulary(taxonomy, args.top_n)
            if args.vocabulary_out is not None:
                args.vocabulary_out.parent.mkdir(parents=True, exist_ok=True)
                args.vocabulary_out.write_text(json.dumps(vocabulary, indent=2) + "\n")
        table = encode_taxonomy(taxonomy, vocabulary)

    if not args.skip_kingdom:
        if args.kingdom_map_in is not None:
            kingdom_by_taxon = json.loads(args.kingdom_map_in.read_text())
        else:
            unique_taxa = taxonomy["taxon_id"].unique().tolist()
            kingdom_by_taxon = classify_kingdoms(unique_taxa, cache_dir=args.taxdump_cache_dir)
            if args.kingdom_map_out is not None:
                args.kingdom_map_out.parent.mkdir(parents=True, exist_ok=True)
                args.kingdom_map_out.write_text(json.dumps(kingdom_by_taxon, indent=2) + "\n")
        kingdom_table = encode_kingdom(taxonomy, kingdom_by_taxon)
        table = table.merge(kingdom_table, on="protein_id", how="left", validate="one_to_one")

    if args.output.suffix.lower() == ".csv":
        table.to_csv(args.output, index=False)
    else:
        table.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
