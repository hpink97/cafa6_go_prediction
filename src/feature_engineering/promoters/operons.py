"""Operon-leader resolution for species where genes are commonly co-transcribed.

For a polycistronic operon, only the *first* gene has its own promoter immediately
upstream; downstream genes in the same operon share that promoter and have no
independent regulatory region of their own. Blindly taking a fixed window upstream
of every gene's start would mostly capture the neighboring gene's coding sequence
for non-leader genes. This resolves, for every gene, the coordinates of its
operon's leader gene (itself, if it has no close same-strand upstream neighbor).
"""

from __future__ import annotations

import pandas as pd

DEFAULT_MAX_OPERON_GAP = 50  # bp; same-strand genes closer than this are treated as co-transcribed


def resolve_operon_leaders(genes: dict[str, dict], max_gap: int = DEFAULT_MAX_OPERON_GAP) -> dict[str, dict]:
    """Map every gene_id to the coordinate record of its operon's leader gene."""
    # Records from overlap/region already carry their own "gene_id" field, so just
    # flatten the dict values -- no need to round-trip the dict-key index.
    frame = pd.DataFrame.from_dict(genes, orient="index").reset_index(drop=True)

    leader_by_gene: dict[str, str] = {}
    for (_, strand), group in frame.groupby(["seq_region_name", "strand"]):
        # Walk in transcription order: increasing coordinate for +, decreasing for -.
        ordered = group.sort_values("start", ascending=(strand == 1)).reset_index(drop=True)
        current_leader = None
        previous_row = None
        for _, row in ordered.iterrows():
            if previous_row is not None:
                gap = row["start"] - previous_row["end"] - 1 if strand == 1 else previous_row["start"] - row["end"] - 1
            else:
                gap = None
            if gap is None or gap > max_gap:
                current_leader = row["gene_id"]
            leader_by_gene[row["gene_id"]] = current_leader
            previous_row = row

    return {gene_id: genes[leader_id] for gene_id, leader_id in leader_by_gene.items()}
