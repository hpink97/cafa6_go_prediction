"""Rebuild missing metadata.json files for embedding directories from their memmaps.

The embedding loaders need metadata.json for the memmap shape. If it is lost, the shape
can be recovered exactly: rows come from ids.tsv and the dimension from the file size.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("cafa6.repair_metadata")
ESM_MODEL_SIZES = ("35M", "150M")
PROMOTER_MODELS = {
    "nt100m": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
    "agront1b": "InstaDeepAI/agro-nucleotide-transformer-1b",
}
SPLIT_INPUTS = {
    "train": {"esm2": "data/Train/train_sequences.fasta", "promoter": "data/features/promoters/train/combined.tsv"},
    "test": {"esm2": "data/Test/testsuperset.fasta", "promoter": "data/features/promoters/test/combined.tsv"},
}


def infer_metadata(embedding_dir: Path, project_root: Path) -> dict:
    """Infer metadata.json contents for an `{split}_esm2_{size}` or `{split}_promoter_{model}` directory."""
    split, _, kind = embedding_dir.name.partition("_")
    if split not in SPLIT_INPUTS:
        raise ValueError(f"Cannot infer split from {embedding_dir.name!r}")
    protein_count = len(pd.read_csv(embedding_dir / "ids.tsv", sep="\t"))
    if protein_count == 0:
        raise ValueError(f"{embedding_dir} has an empty ids.tsv")
    byte_count = (embedding_dir / "embeddings.f32.dat").stat().st_size
    embedding_dim, remainder = divmod(byte_count, 4 * protein_count)
    if remainder:
        raise ValueError(f"{embedding_dir}: {byte_count} bytes is not a float32 matrix with {protein_count} rows")
    done_bytes = (embedding_dir / "done.bool.dat").stat().st_size
    if done_bytes != protein_count:
        raise ValueError(f"{embedding_dir}: done.bool.dat has {done_bytes} entries, expected {protein_count}")

    metadata: dict = {}
    if kind.startswith("esm2_"):
        model_size = kind.removeprefix("esm2_")
        if model_size not in ESM_MODEL_SIZES:
            raise ValueError(f"Unknown ESM-2 size in {embedding_dir.name!r}")
        metadata["input"] = str((project_root / SPLIT_INPUTS[split]["esm2"]).resolve())
        metadata["model_size"] = model_size
    elif kind.startswith("promoter_"):
        model_key = kind.removeprefix("promoter_")
        if model_key not in PROMOTER_MODELS:
            raise ValueError(f"Unknown promoter model in {embedding_dir.name!r}")
        metadata["input"] = str((project_root / SPLIT_INPUTS[split]["promoter"]).resolve())
        metadata["model_name"] = PROMOTER_MODELS[model_key]
    else:
        raise ValueError(f"Unrecognised embedding directory {embedding_dir.name!r}")
    metadata.update({"embedding_dim": embedding_dim, "protein_count": protein_count, "limit": None, "dtype": "float32"})
    return metadata


def repair(embeddings_root: Path, project_root: Path, overwrite: bool = False) -> list[Path]:
    written = []
    for embedding_dir in sorted(path for path in embeddings_root.iterdir() if (path / "embeddings.f32.dat").exists()):
        metadata_path = embedding_dir / "metadata.json"
        if metadata_path.exists() and not overwrite:
            LOGGER.info("Keeping existing %s", metadata_path)
            continue
        metadata = infer_metadata(embedding_dir, project_root)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        LOGGER.info("Wrote %s (%d x %d)", metadata_path, metadata["protein_count"], metadata["embedding_dim"])
        written.append(metadata_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-root", type=Path, default=Path("data/features/embeddings"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true", help="Rewrite metadata.json even when it already exists.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repair(args.embeddings_root, args.project_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
