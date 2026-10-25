"""Auxiliary GO score channels derived only from training proteins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def load_taxonomy_map(path: str | Path) -> pd.Series:
    """Load ``protein_id -> taxon_id`` from the headerless CAFA taxonomy file."""
    table = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["protein_id", "taxon_id"],
        dtype=str,
    )
    if table["protein_id"].duplicated().any():
        raise ValueError(f"Duplicate protein IDs in taxonomy file {path}")
    return table.set_index("protein_id")["taxon_id"]


def make_taxon_stratified_validation_split(
    protein_ids: np.ndarray,
    taxonomy_by_id: pd.Series,
    validation_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split while preserving common exact-taxon frequencies.

    Singleton taxa are combined into one rare stratum because sklearn cannot
    place a class represented by one sample into both partitions.
    """
    ids = np.asarray(protein_ids, dtype=str)
    taxa = taxonomy_by_id.reindex(ids).fillna("__unknown__").astype(str)
    counts = taxa.value_counts()
    strata = taxa.where(taxa.map(counts) >= 2, "__rare__").to_numpy()
    train_ids, validation_ids = train_test_split(
        ids,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
        stratify=strata,
    )
    return train_ids, validation_ids


def _normalize_scores_by_aspect(scores: np.ndarray, aspects: np.ndarray) -> np.ndarray:
    normalized = scores.copy()
    for aspect in np.unique(aspects):
        columns = aspects == aspect
        block = normalized[:, columns]
        maxima = block.max(axis=1, keepdims=True)
        normalized[:, columns] = np.divide(
            block,
            maxima,
            out=np.zeros_like(block),
            where=maxima > 0,
        )
    return normalized


@dataclass(frozen=True)
class TaxonPrior:
    global_scores: np.ndarray
    scores_by_taxon: dict[str, np.ndarray]
    strength: float

    def predict(self, protein_ids: np.ndarray, taxonomy_by_id: pd.Series) -> np.ndarray:
        taxa = taxonomy_by_id.reindex(np.asarray(protein_ids, dtype=str)).fillna("__unknown__")
        return np.stack(
            [self.scores_by_taxon.get(str(taxon), self.global_scores) for taxon in taxa],
        ).astype(np.float32, copy=False)


def fit_taxon_prior(
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    taxonomy_by_id: pd.Series,
    aspects: np.ndarray,
    strength: float = 10.0,
) -> TaxonPrior:
    """Fit an empirical-Bayes exact-taxon prevalence score channel."""
    if strength < 0:
        raise ValueError("Taxon-prior strength must be non-negative")
    ids = np.asarray(train_ids, dtype=str)
    labels = np.asarray(train_labels, dtype=np.float32)
    if labels.ndim != 2 or len(ids) != len(labels) or labels.shape[1] != len(aspects):
        raise ValueError("Train IDs, labels, and aspects have incompatible shapes")

    raw_global = labels.mean(axis=0)
    normalized_global = _normalize_scores_by_aspect(raw_global[np.newaxis, :], aspects)[0]
    taxa = taxonomy_by_id.reindex(ids).fillna("__unknown__").astype(str).to_numpy()
    scores_by_taxon = {}
    for taxon in np.unique(taxa):
        rows = taxa == taxon
        prevalence = (labels[rows].sum(axis=0) + strength * raw_global) / (rows.sum() + strength)
        scores_by_taxon[taxon] = _normalize_scores_by_aspect(
            prevalence[np.newaxis, :],
            aspects,
        )[0]
    return TaxonPrior(
        global_scores=normalized_global.astype(np.float32),
        scores_by_taxon=scores_by_taxon,
        strength=strength,
    )


def save_taxon_prior(path: str | Path, prior: TaxonPrior, terms: list[str]) -> None:
    taxon_ids = sorted(prior.scores_by_taxon)
    np.savez_compressed(
        path,
        terms=np.asarray(terms, dtype=str),
        taxon_ids=np.asarray(taxon_ids, dtype=str),
        scores=np.stack([prior.scores_by_taxon[taxon] for taxon in taxon_ids]),
        global_scores=prior.global_scores,
        strength=np.asarray(prior.strength),
    )


def load_taxon_prior(path: str | Path) -> tuple[TaxonPrior, list[str]]:
    with np.load(path) as archive:
        prior = TaxonPrior(
            global_scores=archive["global_scores"],
            scores_by_taxon=dict(zip((str(taxon) for taxon in archive["taxon_ids"]), archive["scores"])),
            strength=float(archive["strength"]),
        )
        return prior, [str(term) for term in archive["terms"]]


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def exact_cosine_neighbors(
    donor_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int,
    *,
    device: str = "auto",
    query_batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact top-k cosine similarities and donor row indices."""
    if donor_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("Donor and query embeddings must be 2D")
    if donor_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("Donor and query embedding dimensions differ")
    if not 1 <= k <= len(donor_embeddings):
        raise ValueError(f"k must be in [1, {len(donor_embeddings)}]")
    if query_batch_size < 1:
        raise ValueError("query_batch_size must be positive")

    torch_device = _resolve_device(device)
    donors = torch.as_tensor(np.array(donor_embeddings, dtype=np.float32, copy=True), device=torch_device)
    donors = torch.nn.functional.normalize(donors, dim=1)
    similarities = np.empty((len(query_embeddings), k), dtype=np.float32)
    indices = np.empty((len(query_embeddings), k), dtype=np.int32)

    with torch.inference_mode():
        for start in range(0, len(query_embeddings), query_batch_size):
            stop = min(start + query_batch_size, len(query_embeddings))
            query = torch.as_tensor(
                np.array(query_embeddings[start:stop], dtype=np.float32, copy=True),
                device=torch_device,
            )
            query = torch.nn.functional.normalize(query, dim=1)
            batch_similarities, batch_indices = torch.topk(query @ donors.T, k=k, dim=1)
            similarities[start:stop] = batch_similarities.cpu().numpy()
            indices[start:stop] = batch_indices.cpu().numpy()
    return similarities, indices


def knn_annotation_scores(
    neighbor_indices: np.ndarray,
    neighbor_similarities: np.ndarray,
    donor_labels: np.ndarray,
    aspects: np.ndarray,
    *,
    k: int | None = None,
    similarity_power: float = 1.0,
    normalize_by_aspect: bool = True,
) -> np.ndarray:
    """Transfer donor annotations through non-negative cosine-weighted voting."""
    if neighbor_indices.shape != neighbor_similarities.shape:
        raise ValueError("Neighbor indices and similarities have different shapes")
    if donor_labels.ndim != 2 or donor_labels.shape[1] != len(aspects):
        raise ValueError("Donor labels and aspects have incompatible shapes")
    effective_k = neighbor_indices.shape[1] if k is None else k
    if not 1 <= effective_k <= neighbor_indices.shape[1]:
        raise ValueError(f"k must be in [1, {neighbor_indices.shape[1]}]")
    if similarity_power <= 0:
        raise ValueError("similarity_power must be positive")

    indices = neighbor_indices[:, :effective_k]
    weights = np.clip(neighbor_similarities[:, :effective_k], 0.0, None) ** similarity_power
    weight_sums = weights.sum(axis=1, keepdims=True)
    zero_weight_rows = weight_sums[:, 0] == 0
    if np.any(zero_weight_rows):
        weights[zero_weight_rows] = 1.0
        weight_sums[zero_weight_rows] = effective_k
    weights /= weight_sums

    scores = np.empty((len(indices), donor_labels.shape[1]), dtype=np.float32)
    vote_batch_size = max(1, 10_000_000 // (effective_k * donor_labels.shape[1]))
    for start in range(0, len(indices), vote_batch_size):
        stop = min(start + vote_batch_size, len(indices))
        neighbor_labels = donor_labels[indices[start:stop]]
        scores[start:stop] = np.einsum(
            "bk,bkt->bt",
            weights[start:stop],
            neighbor_labels,
            optimize=True,
        )
    if normalize_by_aspect:
        scores = _normalize_scores_by_aspect(scores, aspects)
    return scores


def blend_scores_by_aspect(
    base_scores: np.ndarray,
    channel_scores: np.ndarray,
    aspects: np.ndarray,
    weights_by_aspect: dict[str, float],
) -> np.ndarray:
    """Blend a score channel into the base using one weight per GO aspect."""
    if base_scores.shape != channel_scores.shape or base_scores.shape[1] != len(aspects):
        raise ValueError("Base scores, channel scores, and aspects have incompatible shapes")
    blended = base_scores.copy()
    for aspect, weight in weights_by_aspect.items():
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Blend weight for {aspect} must be in [0, 1]")
        columns = aspects == aspect
        blended[:, columns] = (
            (1.0 - weight) * blended[:, columns]
            + weight * channel_scores[:, columns]
        )
    return blended