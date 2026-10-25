"""Generate resumable mean-pooled Nucleotide Transformer v2 embeddings for extracted promoter sequences."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

LOGGER = logging.getLogger("cafa6.promoters.embed")
DEFAULT_MODEL = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"


def load_promoter_sequences(input_path: str | Path) -> pd.DataFrame:
    """Load protein_id, species, promoter_sequence for every extracted promoter (any species).

    Per-species files are still being written while extraction is in progress, so a directory
    of whatever's completed so far is a valid (partial) input -- callers just get fewer rows.
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        paths = sorted(
            path
            for path in input_path.glob("*.tsv")
            if path.name != "combined.tsv" and not path.name.endswith(".gene_id_cache.tsv")
        )
        frames = [pd.read_csv(path, sep="\t", dtype={"taxon_id": str}) for path in paths]
        frames = [frame for frame in frames if "promoter_sequence" in frame.columns]
        if not frames:
            raise FileNotFoundError(f"No per-species promoter TSVs found in {input_path}")
        frame = pd.concat(frames, ignore_index=True)
    else:
        frame = pd.read_csv(input_path, sep="\t", dtype={"taxon_id": str})

    frame = frame.dropna(subset=["promoter_sequence"])
    frame = frame[frame["promoter_sequence"].str.len() > 0]
    frame = frame.drop_duplicates(subset="protein_id", keep="first").sort_values("protein_id").reset_index(drop=True)
    if "taxon_id" not in frame.columns:
        frame["taxon_id"] = None
    return frame[["protein_id", "taxon_id", "species", "promoter_sequence"]]


def _metadata_path(output: Path) -> Path:
    return output / "metadata.json"


def _prepare_output(
    output: Path,
    ids: list[str],
    lengths: list[int],
    model_name: str,
    embedding_dim: int,
    input_path: Path,
    limit: int | None,
    resume: bool,
    reuse_existing: bool,
) -> tuple[np.memmap, np.memmap, dict]:
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input": str(input_path.resolve()),
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "protein_count": len(ids),
        "limit": limit,
        "dtype": "float32",
    }
    metadata_path = _metadata_path(output)
    embeddings_path = output / "embeddings.f32.dat"
    done_path = output / "done.bool.dat"
    ids_path = output / "ids.tsv"

    if resume and reuse_existing:
        raise ValueError("--resume and --reuse-existing are mutually exclusive")
    if reuse_existing:
        if not metadata_path.exists() or not embeddings_path.exists() or not done_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"Cannot reuse incomplete output at {output}")
        existing_metadata = json.loads(metadata_path.read_text())
        if existing_metadata.get("model_name") != model_name or existing_metadata.get("embedding_dim") != embedding_dim:
            raise ValueError("Existing embedding model or dimension does not match")
        existing_ids = pd.read_csv(ids_path, sep="\t", dtype={"protein_id": str})["protein_id"].tolist()
        existing_embeddings = np.memmap(
            embeddings_path, dtype="float32", mode="r", shape=(len(existing_ids), embedding_dim)
        )
        existing_done = np.memmap(done_path, dtype="bool", mode="r", shape=(len(existing_ids),))
        existing_index = {protein_id: index for index, protein_id in enumerate(existing_ids)}
        temporary_embeddings = output / "embeddings.f32.dat.tmp"
        temporary_done = output / "done.bool.dat.tmp"
        temporary_metadata = output / "metadata.json.tmp"
        temporary_ids = output / "ids.tsv.tmp"
        temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary_ids.write_text("protein_id\tlength\n" + "\n".join(f"{protein_id}\t{length}" for protein_id, length in zip(ids, lengths)) + "\n")
        embeddings = np.memmap(temporary_embeddings, dtype="float32", mode="w+", shape=(len(ids), embedding_dim))
        done = np.memmap(temporary_done, dtype="bool", mode="w+", shape=(len(ids),))
        done[:] = False
        for new_index, protein_id in enumerate(ids):
            old_index = existing_index.get(protein_id)
            if old_index is not None and existing_done[old_index]:
                embeddings[new_index] = existing_embeddings[old_index]
                done[new_index] = True
        embeddings.flush()
        done.flush()
        del existing_embeddings, existing_done
        os.replace(temporary_embeddings, embeddings_path)
        os.replace(temporary_done, done_path)
        os.replace(temporary_ids, ids_path)
        os.replace(temporary_metadata, metadata_path)
        return (
            np.memmap(embeddings_path, dtype="float32", mode="r+", shape=(len(ids), embedding_dim)),
            np.memmap(done_path, dtype="bool", mode="r+", shape=(len(ids),)),
            metadata,
        )
    if resume:
        if not metadata_path.exists() or not embeddings_path.exists() or not done_path.exists():
            raise FileNotFoundError(f"Cannot resume incomplete output at {output}")
        existing = json.loads(metadata_path.read_text())
        for key in ("input", "model_name", "embedding_dim", "protein_count", "limit"):
            if existing.get(key) != metadata[key]:
                raise ValueError(f"Resume metadata mismatch for {key}: {existing.get(key)!r} != {metadata[key]!r}")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        ids_path.write_text("protein_id\tlength\n" + "\n".join(f"{protein_id}\t{length}" for protein_id, length in zip(ids, lengths)) + "\n")
        embeddings = np.memmap(embeddings_path, dtype="float32", mode="w+", shape=(len(ids), embedding_dim))
        done = np.memmap(done_path, dtype="bool", mode="w+", shape=(len(ids),))
        done[:] = False
        done.flush()
        return embeddings, done, metadata

    return (
        np.memmap(embeddings_path, dtype="float32", mode="r+", shape=(len(ids), embedding_dim)),
        np.memmap(done_path, dtype="bool", mode="r+", shape=(len(ids),)),
        metadata,
    )


def _embed_batch(model, tokenizer, sequences: list[str], device: torch.device) -> np.ndarray:
    # Fixed-length padding (not "longest") keeps every batch's tensor shape identical -- with
    # "longest", each batch's shape depends on the longest sequence in it, and MPS's caching
    # allocator accumulates a growing set of differently-shaped cached buffers over tens of
    # thousands of batches instead of reusing one, causing severe memory growth and slowdown.
    tokens = tokenizer.batch_encode_plus(
        sequences, return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer.model_max_length
    )["input_ids"].to(device)
    attention_mask = tokens != tokenizer.pad_token_id
    with torch.inference_mode():
        outputs = model(tokens, attention_mask=attention_mask, encoder_attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs["hidden_states"][-1]
    mask = attention_mask.unsqueeze(-1)
    mean_embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
    return mean_embeddings.float().cpu().numpy()


def generate_promoter_embeddings(
    input_path: str | Path,
    output: str | Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 8,
    device_name: str = "auto",
    resume: bool = False,
    reuse_existing: bool = False,
    limit: int | None = None,
    embed_species: list[str] | None = None,
    embed_kingdoms: list[str] | None = None,
) -> None:
    """Generate one mean-pooled DNA-LM embedding per extracted promoter sequence.

    If embed_species and/or embed_kingdoms is given, only proteins from those species or NCBI
    kingdoms get a real model embedding -- every other protein still gets a row, but filled with
    zeros (same convention as the zero-filled taxonomy features for proteins missing that block).
    """
    input_path = Path(input_path)
    output = Path(output)
    if batch_size < 1:
        raise ValueError("batch-size must be >= 1")

    sequences = load_promoter_sequences(input_path)
    if limit is not None:
        sequences = sequences.iloc[:limit]
    ids = sequences["protein_id"].tolist()
    species_list = sequences["species"].tolist()
    lengths = sequences["promoter_sequence"].str.len().tolist()
    if not ids:
        raise ValueError(f"No promoter sequences found in {input_path}")

    LOGGER.info("loading tokenizer/model=%s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).eval()
    embedding_dim = model.config.hidden_size
    if device_name == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = model.to(device)

    embeddings, done, _ = _prepare_output(output, ids, lengths, model_name, embedding_dim, input_path, limit, resume, reuse_existing)

    if embed_species is not None or embed_kingdoms is not None:
        selected = np.isin(species_list, list(embed_species or []))
        if embed_kingdoms:
            from feature_engineering.sequence.taxonomy import classify_kingdoms

            taxa = sequences["taxon_id"].astype(str).tolist()
            kingdom_by_taxon = classify_kingdoms(sorted(set(taxa)))
            selected |= np.isin([kingdom_by_taxon.get(taxon, "unknown") for taxon in taxa], embed_kingdoms)
        zero_filled = 0
        reopened = 0
        for start in range(0, len(ids), 10_000):
            stop = min(start + 10_000, len(ids))
            # Rows zero-filled by an earlier, narrower selection must be embedded now.
            is_zero = ~np.any(embeddings[start:stop] != 0.0, axis=1)
            for offset in np.flatnonzero(selected[start:stop] & is_zero & done[start:stop]):
                done[start + offset] = False
                reopened += 1
            for offset in np.flatnonzero(~selected[start:stop] & ~done[start:stop]):
                embeddings[start + offset] = 0.0
                done[start + offset] = True
                zero_filled += 1
        done.flush()
        LOGGER.info(
            "selected %d/%d proteins (species=%s kingdoms=%s); zero-filled %d, reopened %d previously zero rows",
            int(selected.sum()),
            len(ids),
            sorted(embed_species or []),
            sorted(embed_kingdoms or []),
            zero_filled,
            reopened,
        )

    pending = int((~np.asarray(done)).sum())
    log_every = max(1, math.ceil(len(ids) / 75))
    LOGGER.info("model=%s device=%s proteins=%d pending=%d batch_size=%d", model_name, device, len(ids), pending, batch_size)

    sequence_list = sequences["promoter_sequence"].tolist()
    processed = 0
    batch_indices: list[int] = []
    for index in range(len(ids)):
        if done[index]:
            continue
        batch_indices.append(index)
        if len(batch_indices) < batch_size:
            continue
        vectors = _embed_batch(model, tokenizer, [sequence_list[i] for i in batch_indices], device)
        for row, batch_index in enumerate(batch_indices):
            embeddings[batch_index] = vectors[row]
            done[batch_index] = True
        processed += len(batch_indices)
        if processed % log_every < batch_size or processed >= pending:
            LOGGER.info("processed=%d/%d pending=%d", processed, pending, pending - processed)
            embeddings.flush()
            done.flush()
            if device.type == "mps":
                torch.mps.empty_cache()
        batch_indices = []

    if batch_indices:
        vectors = _embed_batch(model, tokenizer, [sequence_list[i] for i in batch_indices], device)
        for row, batch_index in enumerate(batch_indices):
            embeddings[batch_index] = vectors[row]
            done[batch_index] = True
        processed += len(batch_indices)

    embeddings.flush()
    done.flush()
    LOGGER.info("processed=%d/%d complete output=%s", processed, pending, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="combined.tsv, or a directory of per-species promoter TSVs")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Continue an existing output directory (same input snapshot only)")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse completed vectors from an existing output by protein_id and embed only new proteins",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="HuggingFace model id")
    parser.add_argument("--batch-size", type=int, default=8, help="Sequences per inference batch")
    parser.add_argument("--device", dest="device_name", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N proteins (sorted by protein_id)")
    parser.add_argument(
        "--embed-species",
        dest="embed_species",
        nargs="*",
        default=None,
        help="Only compute real embeddings for these species; every other protein gets an all-zero vector (default: embed everyone)",
    )
    parser.add_argument(
        "--embed-kingdoms",
        dest="embed_kingdoms",
        nargs="*",
        default=None,
        help="Also compute real embeddings for proteins whose taxon falls in these kingdoms (e.g. eukaryota_viridiplantae)",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    arguments = vars(parser.parse_args())
    log_level = arguments.pop("log_level")
    logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    generate_promoter_embeddings(**arguments)


if __name__ == "__main__":
    main()
