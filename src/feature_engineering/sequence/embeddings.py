"""Generate resumable mean-pooled ESM-2 protein embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Callable, Iterator
from pathlib import Path

import esm
import numpy as np
import torch
from Bio import SeqIO

LOGGER = logging.getLogger("cafa6.embeddings")
MODEL_LOADERS: dict[str, tuple[str, Callable]] = {
    "35M": ("esm2_t12_35M_UR50D", esm.pretrained.esm2_t12_35M_UR50D),
    "150M": ("esm2_t30_150M_UR50D", esm.pretrained.esm2_t30_150M_UR50D),
}


def _records(path: Path, limit: int | None = None) -> Iterator[tuple[str, str]]:
    for index, record in enumerate(SeqIO.parse(path, "fasta")):
        if limit is not None and index >= limit:
            break
        protein_id = record.id.split("|")[1] if "|" in record.id else record.id
        yield protein_id, str(record.seq)


def _chunks(sequence: str, chunk_size: int, overlap: int) -> list[tuple[int, str]]:
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk size must be greater than overlap")
    if len(sequence) <= chunk_size:
        return [(0, sequence)]
    return [(start, sequence[start : start + chunk_size]) for start in range(0, len(sequence), step) if start < len(sequence)]


def _metadata_path(output: Path) -> Path:
    return output / "metadata.json"


def _prepare_output(
    output: Path,
    ids: list[str],
    lengths: list[int],
    model_size: str,
    embedding_dim: int,
    input_path: Path,
    limit: int | None,
    resume: bool,
) -> tuple[np.memmap, np.memmap, dict]:
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input": str(input_path.resolve()),
        "model_size": model_size,
        "embedding_dim": embedding_dim,
        "protein_count": len(ids),
        "limit": limit,
        "dtype": "float32",
    }
    metadata_path = _metadata_path(output)
    embeddings_path = output / "embeddings.f32.dat"
    done_path = output / "done.bool.dat"
    ids_path = output / "ids.tsv"

    if resume:
        if not metadata_path.exists() or not embeddings_path.exists() or not done_path.exists():
            raise FileNotFoundError(f"Cannot resume incomplete output at {output}")
        existing = json.loads(metadata_path.read_text())
        for key in ("input", "model_size", "embedding_dim", "protein_count", "limit"):
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


def _infer_chunks(model, converter, chunks: list[tuple[int, str]], batch_size: int, device: torch.device) -> np.ndarray:
    vectors = []
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        _, _, tokens = converter([(str(index), sequence) for index, sequence in batch])
        with torch.inference_mode():
            result = model(tokens.to(device), repr_layers=[model.num_layers], return_contacts=False)
        representations = result["representations"][model.num_layers]
        vectors.extend(
            representations[row, 1 : len(sequence) + 1].mean(0).float().cpu().numpy()
            for row, (_, sequence) in enumerate(batch)
        )
    return np.stack(vectors)


def generate_embeddings(
    input_path: str | Path,
    output: str | Path,
    model_size: str = "35M",
    batch_size: int = 8,
    chunk_size: int = 1022,
    overlap: int = 128,
    device_name: str = "auto",
    resume: bool = False,
    limit: int | None = None,
) -> None:
    """Generate one mean-pooled embedding per FASTA protein."""
    input_path = Path(input_path)
    output = Path(output)
    if model_size not in MODEL_LOADERS:
        raise ValueError(f"model-size must be one of {', '.join(MODEL_LOADERS)}")
    if batch_size < 1 or chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("require batch-size >= 1, chunk-size >= 1, and 0 <= overlap < chunk-size")

    records = list(_records(input_path, limit))
    ids = [protein_id for protein_id, _ in records]
    lengths = [len(sequence) for _, sequence in records]
    if not ids:
        raise ValueError(f"No FASTA records found in {input_path}")
    if len(ids) != len(set(ids)):
        raise ValueError("FASTA contains duplicate protein IDs")

    model_name, loader = MODEL_LOADERS[model_size]
    model, alphabet = loader()
    model = model.eval()
    embedding_dim = model.embed_dim
    if device_name == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = model.to(device)
    embeddings, done, _ = _prepare_output(output, ids, lengths, model_size, embedding_dim, input_path, limit, resume)
    converter = alphabet.get_batch_converter()
    pending = int((~np.asarray(done)).sum())
    log_every = max(1, math.ceil(len(ids) / 75))
    LOGGER.info("model=%s device=%s proteins=%d pending=%d batch_size=%d", model_name, device, len(ids), pending, batch_size)

    protein_index = 0
    processed = 0
    batch: list[tuple[int, str, list[tuple[int, str]]]] = []
    batch_chunks = 0

    def flush_batch() -> None:
        nonlocal processed, batch, batch_chunks
        if not batch:
            return
        flat_chunks = [(chunk_index, sequence) for _, _, chunks in batch for chunk_index, sequence in chunks]
        chunk_vectors = _infer_chunks(model, converter, flat_chunks, batch_size, device)
        vector_index = 0
        for index, _, chunks in batch:
            count = len(chunks)
            embeddings[index] = chunk_vectors[vector_index : vector_index + count].mean(axis=0)
            done[index] = True
            vector_index += count
            processed += 1
            if processed % log_every == 0 or processed == pending:
                LOGGER.info("processed=%d/%d pending=%d", processed, pending, pending - processed)
        if processed % max(log_every, 100) == 0 or processed == pending:
            embeddings.flush()
            done.flush()
        batch = []
        batch_chunks = 0

    for index, (_, sequence) in enumerate(_records(input_path, limit)):
        if done[index]:
            continue
        sequence_chunks = _chunks(sequence, chunk_size, overlap)
        if batch and batch_chunks + len(sequence_chunks) > batch_size:
            flush_batch()
        batch.append((index, sequence, sequence_chunks))
        batch_chunks += len(sequence_chunks)
    flush_batch()
    embeddings.flush()
    done.flush()
    LOGGER.info("complete output=%s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Input FASTA file")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Continue an existing output directory")
    parser.add_argument("--model-size", choices=tuple(MODEL_LOADERS), default="35M")
    parser.add_argument("--batch-size", type=int, default=8, help="Maximum sequence chunks per inference batch")
    parser.add_argument("--chunk-size", type=int, default=1022, help="Residues per chunk for long proteins")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between long-protein chunks")
    parser.add_argument("--device", dest="device_name", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N records")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    arguments = vars(parser.parse_args())
    log_level = arguments.pop("log_level")
    logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    generate_embeddings(**arguments)


if __name__ == "__main__":
    main()