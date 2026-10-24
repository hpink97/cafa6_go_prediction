"""Load and merge feature blocks (PLM embeddings + protein features) and GO term labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

PLANT_SPECIFIC_PROMOTER_PREFIX = "promoter_agront"


def load_embedding_table(embedding_dir: str | Path) -> pd.DataFrame:
    """Load mean-pooled ESM-2 embeddings into a wide DataFrame keyed by protein_id."""
    embedding_dir = Path(embedding_dir)
    metadata = json.loads((embedding_dir / "metadata.json").read_text())
    ids = pd.read_csv(embedding_dir / "ids.tsv", sep="\t")
    embedding_dim = metadata["embedding_dim"]
    protein_count = metadata["protein_count"]
    done = np.memmap(embedding_dir / "done.bool.dat", dtype="bool", mode="r", shape=(protein_count,))
    if not bool(done.all()):
        raise ValueError(f"Embedding generation incomplete in {embedding_dir}: {int((~np.asarray(done)).sum())} proteins pending")
    embeddings = np.memmap(embedding_dir / "embeddings.f32.dat", dtype="float32", mode="r", shape=(protein_count, embedding_dim))
    columns = [f"esm2_{embedding_dim}_{index}" for index in range(embedding_dim)]
    table = pd.DataFrame(np.asarray(embeddings), columns=columns)
    table.insert(0, "protein_id", ids["protein_id"].to_numpy())
    return table


def load_promoter_embedding_table(embedding_dir: str | Path, prefix: str) -> pd.DataFrame:
    """Load a promoter DNA-LM embedding table (see feature_engineering.promoters.embed_promoters).

    Unlike `load_embedding_table`, this is expected to be a PARTIAL table -- only proteins whose
    species had promoters extracted appear here at all (everyone else is imputed later).
    """
    embedding_dir = Path(embedding_dir)
    metadata = json.loads((embedding_dir / "metadata.json").read_text())
    ids = pd.read_csv(embedding_dir / "ids.tsv", sep="\t")
    embedding_dim = metadata["embedding_dim"]
    protein_count = metadata["protein_count"]
    done = np.memmap(embedding_dir / "done.bool.dat", dtype="bool", mode="r", shape=(protein_count,))
    if not bool(done.all()):
        raise ValueError(f"Embedding generation incomplete in {embedding_dir}: {int((~np.asarray(done)).sum())} proteins pending")
    embeddings = np.memmap(embedding_dir / "embeddings.f32.dat", dtype="float32", mode="r", shape=(protein_count, embedding_dim))
    columns = [f"{prefix}_{index}" for index in range(embedding_dim)]
    table = pd.DataFrame(np.asarray(embeddings), columns=columns)
    table.insert(0, "protein_id", ids["protein_id"].to_numpy())
    return table


@dataclass
class PromoterImputer:
    """Train-only donors used to fill promoter blocks for proteins without an extracted promoter.

    Most blocks are imputed as the mean of the k cosine-nearest ESM-2 donors. The plant-specific
    AgroNT block uses KNN only for plant proteins; other proteins get `fill_values` (all zeros).
    """

    k: int
    donor_esm2: dict[str, np.ndarray]
    donor_features: dict[str, np.ndarray]
    fill_values: dict[str, np.ndarray]

    def impute(self, prefix: str, esm2: np.ndarray, is_plant: np.ndarray | None = None) -> np.ndarray:
        features = self.donor_features[prefix]
        result = np.empty((len(esm2), features.shape[1]), dtype=np.float32)
        if prefix in self.fill_values:
            if is_plant is None:
                raise ValueError(f"is_plant is required to impute the plant-specific {prefix} block")
            use_knn = np.asarray(is_plant, dtype=bool)
            result[~use_knn] = self.fill_values[prefix]
        else:
            use_knn = np.ones(len(esm2), dtype=bool)
        if use_knn.any():
            donors = self.donor_esm2[prefix]
            neighbors = NearestNeighbors(n_neighbors=min(self.k, len(donors)), metric="cosine").fit(donors)
            _, neighbor_indices = neighbors.kneighbors(np.asarray(esm2, dtype=np.float32)[use_knn])
            result[use_knn] = features[neighbor_indices].mean(axis=1)
        return result

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {"k": np.asarray(self.k), "prefixes": np.asarray(sorted(self.donor_features))}
        for prefix in self.donor_features:
            arrays[f"{prefix}__esm2"] = self.donor_esm2[prefix]
            arrays[f"{prefix}__features"] = self.donor_features[prefix]
            if prefix in self.fill_values:
                arrays[f"{prefix}__fill"] = self.fill_values[prefix]
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> PromoterImputer:
        with np.load(path) as archive:
            prefixes = [str(prefix) for prefix in archive["prefixes"]]
            return cls(
                k=int(archive["k"]),
                donor_esm2={prefix: archive[f"{prefix}__esm2"] for prefix in prefixes},
                donor_features={prefix: archive[f"{prefix}__features"] for prefix in prefixes},
                fill_values={prefix: archive[f"{prefix}__fill"] for prefix in prefixes if f"{prefix}__fill" in archive},
            )


def fit_promoter_imputer(
    esm2_embeddings: pd.DataFrame,
    known_by_prefix: dict[str, pd.DataFrame],
    donor_ids: list[str],
    k: int = 5,
) -> PromoterImputer:
    """Collect donor ESM-2 and promoter vectors from `donor_ids` only (pass train IDs to avoid leakage)."""
    esm2_indexed = esm2_embeddings.set_index("protein_id")
    donor_esm2: dict[str, np.ndarray] = {}
    donor_features: dict[str, np.ndarray] = {}
    fill_values: dict[str, np.ndarray] = {}
    for prefix, known in known_by_prefix.items():
        columns = [column for column in known.columns if column != "protein_id"]
        ids = known["protein_id"].to_numpy()
        values = known[columns].to_numpy(dtype=np.float32)
        is_donor = known["protein_id"].isin(set(donor_ids)).to_numpy()
        if prefix == PLANT_SPECIFIC_PROMOTER_PREFIX:
            is_real = np.any(values != 0.0, axis=1)
            zero_donors = values[is_donor & ~is_real]
            fill_values[prefix] = zero_donors.mean(axis=0) if len(zero_donors) else np.zeros(len(columns), dtype=np.float32)
            is_donor = is_donor & is_real
        donor_features[prefix] = values[is_donor]
        donor_esm2[prefix] = esm2_indexed.loc[ids[is_donor]].to_numpy(dtype=np.float32)
    return PromoterImputer(k=k, donor_esm2=donor_esm2, donor_features=donor_features, fill_values=fill_values)


def _promoter_cache_key(
    esm2_embedding_dir: Path,
    promoter_embedding_dirs: dict[str, str | Path],
    train_ids: list[str],
    k: int,
) -> dict:
    def describe(directory: str | Path) -> dict:
        directory = Path(directory)
        return {
            "metadata": json.loads((directory / "metadata.json").read_text()),
            "embeddings_mtime": (directory / "embeddings.f32.dat").stat().st_mtime,
        }

    return {
        "esm2": describe(esm2_embedding_dir),
        "promoters": {prefix: describe(directory) for prefix, directory in sorted(promoter_embedding_dirs.items())},
        "train_ids_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
        "k": k,
    }


def build_promoter_feature_table(
    esm2_embedding_dir: str | Path,
    promoter_embedding_dirs: dict[str, str | Path],
    train_ids: list[str],
    cache_path: str | Path,
    taxonomy_features_path: str | Path | None = None,
    k: int = 5,
    imputer_path: str | Path | None = None,
    imputer_donor_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Build a complete (every protein covered) promoter feature block from one or more promoter
    DNA-LM embedding directories, each keyed by a distinct column prefix. Proteins missing from a
    given source entirely are imputed by a `PromoterImputer` whose donors are restricted to
    `train_ids`, so imputing never leans on held-out proteins.

    Cached to `cache_path` as CSV, with a sidecar key file; the cache is rebuilt whenever the
    embedding inputs, `train_ids`, or `k` change. If `imputer_path` is given, an imputer fitted on
    `imputer_donor_ids` (default `train_ids`) is also saved there for test-time use.
    """
    esm2_embedding_dir = Path(esm2_embedding_dir)
    cache_path = Path(cache_path)
    key_path = cache_path.with_name(cache_path.name + ".key.json")
    cache_key = _promoter_cache_key(esm2_embedding_dir, promoter_embedding_dirs, train_ids, k)
    cache_is_valid = cache_path.exists() and key_path.exists() and json.loads(key_path.read_text()) == cache_key
    if cache_is_valid and imputer_path is None:
        return pd.read_csv(cache_path)

    esm2_embeddings = load_embedding_table(esm2_embedding_dir)
    known_by_prefix = {
        prefix: load_promoter_embedding_table(embedding_dir, prefix) for prefix, embedding_dir in promoter_embedding_dirs.items()
    }
    if imputer_path is not None:
        fit_promoter_imputer(esm2_embeddings, known_by_prefix, imputer_donor_ids or train_ids, k=k).save(imputer_path)
    if cache_is_valid:
        return pd.read_csv(cache_path)

    is_plant_by_protein: dict[str, bool] = {}
    if PLANT_SPECIFIC_PROMOTER_PREFIX in promoter_embedding_dirs:
        if taxonomy_features_path is None:
            raise ValueError("taxonomy_features_path is required to impute the AgroNT (plant-specific) promoter block")
        kingdom = load_taxonomy_feature_table(taxonomy_features_path)
        is_plant_by_protein = dict(zip(kingdom["protein_id"], kingdom["kingdom_eukaryota_viridiplantae"] == 1.0))

    imputer = fit_promoter_imputer(esm2_embeddings, known_by_prefix, train_ids, k=k)
    esm2_indexed = esm2_embeddings.set_index("protein_id")
    blocks = []
    for prefix, known in known_by_prefix.items():
        known_ids = set(known["protein_id"])
        missing_ids = [protein_id for protein_id in esm2_indexed.index if protein_id not in known_ids]
        is_plant = np.asarray([is_plant_by_protein.get(protein_id, False) for protein_id in missing_ids], dtype=bool)
        imputed = pd.DataFrame(
            imputer.impute(prefix, esm2_indexed.loc[missing_ids].to_numpy(dtype=np.float32), is_plant),
            columns=[column for column in known.columns if column != "protein_id"],
        )
        imputed.insert(0, "protein_id", missing_ids)
        blocks.append(pd.concat([known, imputed], ignore_index=True).set_index("protein_id"))

    combined = pd.concat(blocks, axis=1).reset_index()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_path, index=False)
    key_path.write_text(json.dumps(cache_key, indent=2) + "\n")
    return combined


def load_protein_feature_table(path: str | Path) -> pd.DataFrame:
    """Load biophysical/topology/motif protein features."""
    return pd.read_csv(path, sep="\t")


def load_taxonomy_feature_table(path: str | Path) -> pd.DataFrame:
    """Load one-hot taxonomy features (see feature_engineering.taxonomy)."""
    return pd.read_csv(path, sep="\t")


def build_feature_table(
    embedding_dir: str | Path | None,
    protein_features_path: str | Path | None,
    taxonomy_features_path: str | Path | None = None,
    promoter_embedding_dirs: dict[str, str | Path] | None = None,
    promoter_feature_cache_path: str | Path | None = None,
    promoter_knn_k: int = 5,
    promoter_train_ids: list[str] | None = None,
    promoter_imputer_path: str | Path | None = None,
    promoter_imputer_donor_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Merge embeddings, protein features, and (optionally) taxonomy/promoter features on protein_id.

    With `embedding_dir=None` (no ESM-2 block) the protein features define the protein set, and
    missing promoters are zero-filled: imputing them from ESM-2 neighbours would leak ESM-2 back in.
    """
    if embedding_dir is None:
        if protein_features_path is None:
            raise ValueError("Without ESM-2 embeddings, protein features are required to define the protein set")
        merged = load_protein_feature_table(protein_features_path)
    else:
        embeddings = load_embedding_table(embedding_dir)
        merged = embeddings
        if protein_features_path is not None:
            protein_features = load_protein_feature_table(protein_features_path)
            merged = embeddings.merge(protein_features, on="protein_id", how="inner", validate="one_to_one")
            if len(merged) < min(len(embeddings), len(protein_features)):
                raise ValueError("Feature merge dropped proteins; embeddings and protein feature tables disagree on protein_id coverage")
    if taxonomy_features_path is not None:
        taxonomy_features = load_taxonomy_feature_table(taxonomy_features_path)
        merged = merged.merge(taxonomy_features, on="protein_id", how="left", validate="one_to_one")
        taxon_columns = [column for column in taxonomy_features.columns if column != "protein_id"]
        merged[taxon_columns] = merged[taxon_columns].fillna(0.0)
    if promoter_embedding_dirs is not None and embedding_dir is None:
        for prefix, promoter_dir in promoter_embedding_dirs.items():
            promoter_features = load_promoter_embedding_table(promoter_dir, prefix)
            merged = merged.merge(promoter_features, on="protein_id", how="left", validate="one_to_one")
            promoter_columns = [column for column in promoter_features.columns if column != "protein_id"]
            merged[promoter_columns] = merged[promoter_columns].fillna(0.0)
    elif promoter_embedding_dirs is not None:
        if promoter_feature_cache_path is None:
            raise ValueError("promoter_feature_cache_path is required when promoter_embedding_dirs is set")
        if promoter_train_ids is None:
            raise ValueError("promoter_train_ids is required when promoter_embedding_dirs is set (imputation donors are train-only)")
        promoter_features = build_promoter_feature_table(
            embedding_dir,
            promoter_embedding_dirs,
            promoter_train_ids,
            promoter_feature_cache_path,
            taxonomy_features_path=taxonomy_features_path,
            k=promoter_knn_k,
            imputer_path=promoter_imputer_path,
            imputer_donor_ids=promoter_imputer_donor_ids,
        )
        merged = merged.merge(promoter_features, on="protein_id", how="left", validate="one_to_one")
        promoter_columns = [column for column in promoter_features.columns if column != "protein_id"]
        merged[promoter_columns] = merged[promoter_columns].fillna(0.0)
    return merged.set_index("protein_id")


def load_term_counts(train_terms_path: str | Path) -> pd.Series:
    """Count distinct proteins annotated per GO term, descending by prevalence."""
    terms = pd.read_csv(train_terms_path, sep="\t", usecols=["EntryID", "term"])
    return terms.groupby("term")["EntryID"].nunique().sort_values(ascending=False)


def select_top_terms(term_counts: pd.Series, top_n: int = 100) -> list[str]:
    """Pick the `top_n` most prevalent GO terms."""
    return term_counts.head(top_n).index.tolist()


def build_label_matrix(train_terms_path: str | Path, protein_ids: pd.Index, terms: list[str]) -> pd.DataFrame:
    """Build a binary protein_id x term label matrix restricted to `protein_ids` and `terms`."""
    protein_id_set = set(protein_ids)
    terms_set = set(terms)
    annotations = pd.read_csv(train_terms_path, sep="\t", usecols=["EntryID", "term"])
    annotations = annotations[annotations["term"].isin(terms_set) & annotations["EntryID"].isin(protein_id_set)]
    wide = pd.crosstab(annotations["EntryID"], annotations["term"])
    wide = wide.reindex(index=protein_ids, columns=terms, fill_value=0)
    return (wide > 0).astype(int)


def make_train_test_split(protein_ids: pd.Index, test_size: float = 0.2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """One random protein-level split shared by every GO-term model, for comparability across terms."""
    ids = np.asarray(protein_ids)
    if test_size == 0:
        return ids, ids[:0]
    train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=random_state, shuffle=True)
    return train_ids, test_ids
