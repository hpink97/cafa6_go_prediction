"""Benchmark ESM-2 embedding inference on a representative FASTA subset."""

from __future__ import annotations

import argparse
import gc
import platform
import resource
import time
from pathlib import Path

import esm
import torch
from Bio import SeqIO


def benchmark_model(name: str, loader, sequences: list[tuple[str, str]], batch_size: int, device: torch.device) -> None:
    load_start = time.perf_counter()
    model, alphabet = loader()
    model = model.eval().to(device)
    load_seconds = time.perf_counter() - load_start
    batch_converter = alphabet.get_batch_converter()

    warmup = sequences[:batch_size]
    _, _, tokens = batch_converter(warmup)
    with torch.inference_mode():
        model(tokens.to(device), repr_layers=[model.num_layers], return_contacts=False)
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    total_tokens = 0
    with torch.inference_mode():
        for offset in range(0, len(sequences), batch_size):
            batch = sequences[offset : offset + batch_size]
            _, _, tokens = batch_converter(batch)
            result = model(tokens.to(device), repr_layers=[model.num_layers], return_contacts=False)
            representations = result["representations"][model.num_layers]
            for row, (_, sequence) in enumerate(batch):
                representations[row, 1 : len(sequence) + 1].mean(0).float().cpu()
            total_tokens += int(tokens.numel())
    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start
    residues = sum(len(sequence) for _, sequence in sequences)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(
        f"{name}: load={load_seconds:.1f}s inference={elapsed:.1f}s "
        f"proteins_per_sec={len(sequences) / elapsed:.2f} "
        f"residues_per_sec={residues / elapsed:.0f} embedding_dim={model.embed_dim} "
        f"padded_tokens={total_tokens} peak_rss={peak_mb:.0f}MB"
    )

    del model, alphabet, batch_converter
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input FASTA file")
    parser.add_argument("--limit", type=int, default=256, help="Number of short sequences to benchmark")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    sequences = [
        (record.id, str(record.seq))
        for record in SeqIO.parse(args.input, "fasta")
        if len(record.seq) <= 1022
    ][: args.limit]
    if not sequences:
        raise ValueError("No sequences at or below the 1022-residue benchmark limit")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__} python={platform.python_version()}")
    print(f"sequences={len(sequences)} residues={sum(len(sequence) for _, sequence in sequences)}")
    benchmark_model("esm2_t12_35M_UR50D", esm.pretrained.esm2_t12_35M_UR50D, sequences, args.batch_size, device)
    benchmark_model("esm2_t30_150M_UR50D", esm.pretrained.esm2_t30_150M_UR50D, sequences, args.batch_size, device)


if __name__ == "__main__":
    main()