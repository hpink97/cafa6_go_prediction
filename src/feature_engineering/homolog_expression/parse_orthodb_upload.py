"""Normalize the nine-column result downloaded from an OrthoDB upload job.

OrthoDB's custom-sequence mapping result does not document a query identifier
column. This parser therefore joins rows to the original FASTA order, which is
why the original upload FASTA must be supplied unchanged.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from Bio import SeqIO

LOGGER = logging.getLogger("cafa6.orthodb")
ORTHODB_COLUMNS = [
    "orthogroup_id",
    "gene_name",
    "ortholog_type",
    "match_length",
    "match_start",
    "match_end",
    "score",
    "normalized_score",
    "evalue",
]


def query_ids(path: Path) -> list[str]:
    ids = [record.id.split("|")[1] if "|" in record.id else record.id for record in SeqIO.parse(path, "fasta")]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate protein IDs found in {path}")
    return ids


def parse_result(result_path: Path, fasta_path: Path) -> pd.DataFrame:
    rows = pd.read_csv(result_path, sep="\t", names=ORTHODB_COLUMNS, comment="#", header=None, dtype=str)
    rows = rows.dropna(how="all")
    if len(rows) != len(query_ids(fasta_path)):
        raise ValueError(
            f"OrthoDB result has {len(rows)} rows but uploaded FASTA has {len(query_ids(fasta_path))} records; "
            "the original upload FASTA is required and must produce one result row per query."
        )
    rows.insert(0, "query_id", query_ids(fasta_path))
    for column in ("ortholog_type", "match_length", "match_start", "match_end", "score", "normalized_score", "evalue"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["source"] = "orthodb_v12_upload"
    rows["mapping_method"] = "precomputed_orthology"
    rows["mapping_confidence"] = rows["normalized_score"].clip(lower=0).fillna(0)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path, help="The unchanged FASTA uploaded to OrthoDB")
    parser.add_argument("result", type=Path, help="Nine-column OrthoDB download")
    parser.add_argument("output", type=Path, help="Normalized mapping TSV")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    mapping = parse_result(args.result, args.fasta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.output, sep="\t", index=False)
    LOGGER.info("normalized %d OrthoDB upload mappings to %s", len(mapping), args.output)


if __name__ == "__main__":
    main()