"""Predict GO-term scores for the CAFA test superset from a trained multi-label checkpoint.

Rebuilds the training feature layout for test proteins (ESM-2, sequence, taxonomy, and promoter
blocks, imputing missing promoters from the saved train-only donors), applies the saved median
imputation and scaling, runs the network, blends the taxon-prior and ESM cosine-KNN channels with
the validation-tuned weights, propagates to ancestors, and writes a headerless Kaggle submission.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feature_engineering.taxonomy import classify_kingdoms, encode_kingdom, encode_taxonomy, load_taxonomy
from modeling.data import PLANT_SPECIFIC_PROMOTER_PREFIX, PromoterImputer, load_protein_feature_table
from modeling.go_evaluation import NAMESPACE_TO_ASPECT, GeneOntology, limit_top_predictions
from modeling.score_channels import exact_cosine_neighbors, knn_annotation_scores, load_taxon_prior
from modeling.train_multilabel_nn import MultilabelGOModel, _apply_channel_selection, _resolve_device

LOGGER = logging.getLogger("cafa6.modeling.predict_multilabel_nn")
PROMOTER_PREFIXES = ("promoter_nt", PLANT_SPECIFIC_PROMOTER_PREFIX)


class EmbeddingRows:
    """Random access to a memmapped embedding directory by protein ID."""

    def __init__(self, directory: Path) -> None:
        metadata = json.loads((directory / "metadata.json").read_text())
        count, self.dim = metadata["protein_count"], metadata["embedding_dim"]
        done = np.memmap(directory / "done.bool.dat", dtype="bool", mode="r", shape=(count,))
        if not bool(done.all()):
            raise ValueError(f"Embedding generation incomplete in {directory}")
        self.ids = pd.read_csv(directory / "ids.tsv", sep="\t", dtype={"protein_id": str})["protein_id"].tolist()
        self.row_by_id = {protein_id: row for row, protein_id in enumerate(self.ids)}
        self.values = np.memmap(directory / "embeddings.f32.dat", dtype="float32", mode="r", shape=(count, self.dim))

    def lookup(self, protein_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        rows = np.fromiter((self.row_by_id.get(protein_id, -1) for protein_id in protein_ids), dtype=np.int64, count=len(protein_ids))
        known = rows >= 0
        values = np.zeros((len(protein_ids), self.dim), dtype=np.float32)
        if known.any():
            values[known] = self.values[rows[known]]
        return values, known


def feature_positions(feature_names: list[str], prefix: str, dim: int) -> np.ndarray:
    position = {name: index for index, name in enumerate(feature_names)}
    columns = [f"{prefix}_{index}" for index in range(dim)]
    missing = [column for column in columns if column not in position]
    if missing:
        raise ValueError(f"Checkpoint features lack {len(missing)} {prefix} columns (e.g. {missing[0]}); embedding dimensions differ")
    return np.asarray([position[column] for column in columns])


def submission_rows(protein_ids: list[str], terms: list[str], scores: np.ndarray, min_score: float) -> pd.DataFrame:
    rows, columns = np.nonzero(scores >= min_score)
    # CAFA-evaluator's -max_terms keeps the first N lines per protein unsorted, so emit best-first.
    order = np.lexsort((-scores[rows, columns], rows))
    rows, columns = rows[order], columns[order]
    return pd.DataFrame(
        {
            "protein_id": np.asarray(protein_ids)[rows],
            "term": np.asarray(terms)[columns],
            "score": np.minimum(scores[rows, columns], 1.0),
        }
    )


def run(
    model_dir: Path,
    embeddings_dir: Path,
    protein_features_path: Path,
    taxonomy_path: Path,
    promoter_dirs: dict[str, Path],
    output_path: Path,
    batch_size: int,
    device_name: str,
    knn_device_name: str,
    knn_query_batch_size: int,
    min_score: float,
    max_terms: int,
    limit: int | None,
    protein_ids_paths: list[Path] | None = None,
) -> None:
    config = json.loads((model_dir / "run_config.json").read_text())
    ensemble_config = json.loads((model_dir / "ensemble_config.json").read_text())
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    terms = list(checkpoint["terms"])
    ontology = GeneOntology.from_obo(config["go_obo_path"])
    aspects = np.asarray([NAMESPACE_TO_ASPECT[ontology.namespaces[term]] for term in terms])
    with np.load(model_dir / "preprocessing.npz") as preprocessing:
        feature_names = [str(name) for name in preprocessing["feature_names"]]
        medians = preprocessing["medians"]
        means = preprocessing["means"]
        scales = preprocessing["scales"]

    device = _resolve_device(device_name)
    groups = checkpoint["feature_groups"]
    model = MultilabelGOModel(
        groups,
        n_terms=len(terms),
        trunk_dim=checkpoint["trunk_dim"],
        bottleneck_dim=checkpoint["bottleneck_dim"],
        branch_dropout=checkpoint["branch_dropout"],
        trunk_dropout=checkpoint["trunk_dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    esm2 = EmbeddingRows(embeddings_dir)
    esm2_positions = feature_positions(feature_names, f"esm2_{esm2.dim}", esm2.dim)
    protein_ids = esm2.ids[:limit] if limit is not None else esm2.ids
    if protein_ids_paths:
        wanted = set().union(
            *(set(pd.read_csv(path, sep="\t", header=None, usecols=[0], dtype=str)[0]) for path in protein_ids_paths)
        )
        protein_ids = [protein_id for protein_id in protein_ids if protein_id in wanted]
        LOGGER.info("restricted prediction to %d proteins listed in %s", len(protein_ids), [str(path) for path in protein_ids_paths])

    taxonomy = load_taxonomy(taxonomy_path)
    taxonomy = taxonomy[taxonomy["protein_id"].isin(set(protein_ids))]
    taxon_by_protein = taxonomy.set_index("protein_id")["taxon_id"]
    kingdom_by_taxon = classify_kingdoms(sorted(taxonomy["taxon_id"].unique()))
    static = load_protein_feature_table(protein_features_path).set_index("protein_id")
    taxon_vocabulary = [name.removeprefix("taxon_") for name in feature_names if name.startswith("taxon_") and name != "taxon_other"]
    categorical_columns = [name for name in feature_names if name.startswith(("taxon_", "kingdom_"))]
    if any(name.startswith("taxon_") for name in feature_names):
        static = static.join(encode_taxonomy(taxonomy, taxon_vocabulary).set_index("protein_id"))
    if any(name.startswith("kingdom_") for name in feature_names):
        static = static.join(encode_kingdom(taxonomy, kingdom_by_taxon).set_index("protein_id"))
    static_columns = [name for name in feature_names if not name.startswith(("esm2_", "promoter_"))]
    missing_columns = sorted(set(static_columns) - set(static.columns))
    if missing_columns:
        raise ValueError(f"Test features lack {len(missing_columns)} training columns, e.g. {missing_columns[:5]}")
    static = static.reindex(protein_ids)[static_columns]
    if categorical_columns:
        static[categorical_columns] = static[categorical_columns].fillna(0.0)
    static_positions = np.asarray([feature_names.index(name) for name in static_columns], dtype=int)
    is_plant_by_protein = {
        protein_id: kingdom_by_taxon.get(taxon) == "eukaryota_viridiplantae" for protein_id, taxon in taxon_by_protein.items()
    }

    promoter_sources: dict[str, tuple[EmbeddingRows, np.ndarray]] = {}
    imputer = None
    for prefix in PROMOTER_PREFIXES:
        if not any(name.startswith(f"{prefix}_") for name in feature_names):
            continue
        if prefix not in promoter_dirs:
            raise ValueError(f"Checkpoint uses {prefix} features; pass its test embedding directory")
        source = EmbeddingRows(promoter_dirs[prefix])
        promoter_sources[prefix] = (source, feature_positions(feature_names, prefix, source.dim))
        imputer = imputer or PromoterImputer.load(model_dir / "inference" / "promoter_imputer.npz")
    agront_positions = promoter_sources[PLANT_SPECIFIC_PROMOTER_PREFIX][1] if PLANT_SPECIFIC_PROMOTER_PREFIX in promoter_sources else None

    taxon_prior = None
    if "taxon_prior" in ensemble_config:
        taxon_prior, prior_terms = load_taxon_prior(model_dir / "inference" / "taxon_prior.npz")
        if prior_terms != terms:
            raise ValueError("Taxon-prior terms do not match the checkpoint terms")
    knn_selection = ensemble_config.get("esm_cosine_knn")
    if knn_selection:
        with np.load(model_dir / "inference" / "knn_donors.npz") as donors:
            donor_embeddings = donors["embeddings"]
            donor_labels = donors["labels"]
        knn_k_values = sorted({int(str(settings["variant"])[1:]) for settings in knn_selection.values()})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")
    rows_written = 0
    proteins_with_predictions = 0
    with partial_path.open("w") as handle:
        for start in range(0, len(protein_ids), batch_size):
            batch_ids = protein_ids[start : start + batch_size]
            esm2_values, esm2_known = esm2.lookup(batch_ids)
            if not esm2_known.all():
                raise ValueError("Every test protein must have an ESM-2 embedding")
            x = np.empty((len(batch_ids), len(feature_names)), dtype=np.float32)
            x[:, esm2_positions] = esm2_values
            x[:, static_positions] = static.iloc[start : start + len(batch_ids)].to_numpy(dtype=np.float32)
            is_plant = np.asarray([is_plant_by_protein.get(protein_id, False) for protein_id in batch_ids], dtype=bool)
            for prefix, (source, positions) in promoter_sources.items():
                values, known = source.lookup(batch_ids)
                if (~known).any():
                    values[~known] = imputer.impute(prefix, esm2_values[~known], is_plant[~known])
                x[:, positions] = values

            invalid_rows, invalid_columns = np.where(~np.isfinite(x))
            x[invalid_rows, invalid_columns] = medians[invalid_columns]
            agront_is_applicable = np.any(x[:, agront_positions] != 0.0, axis=1) if agront_positions is not None else None
            x = ((x - means) / scales).astype(np.float32, copy=False)
            if agront_positions is not None:
                x[np.ix_(~agront_is_applicable, agront_positions)] = 0.0

            with torch.inference_mode():
                scores = torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy().astype(np.float32)
            if taxon_prior is not None:
                prior_scores = taxon_prior.predict(np.asarray(batch_ids), taxon_by_protein)
                scores, _ = _apply_channel_selection(scores, {"taxon_prior": prior_scores}, aspects, ensemble_config["taxon_prior"])
            if knn_selection:
                similarities, indices = exact_cosine_neighbors(
                    donor_embeddings, esm2_values, knn_k_values[-1], device=knn_device_name, query_batch_size=knn_query_batch_size
                )
                variants = {
                    f"k{k}": knn_annotation_scores(
                        indices, similarities, donor_labels, aspects, k=k, similarity_power=config["knn_similarity_power"]
                    )
                    for k in knn_k_values
                }
                scores, _ = _apply_channel_selection(scores, variants, aspects, knn_selection)

            limited = limit_top_predictions(scores, config["max_predictions_per_protein"])
            propagated = ontology.propagate_scores(limited, terms, copy=False)
            rows = submission_rows(batch_ids, terms, limit_top_predictions(propagated, max_terms), min_score)
            rows.to_csv(handle, sep="\t", header=False, index=False, float_format="%.3g")
            rows_written += len(rows)
            proteins_with_predictions += rows["protein_id"].nunique()
            LOGGER.info("predicted %d/%d proteins (%d rows written)", start + len(batch_ids), len(protein_ids), rows_written)
    partial_path.replace(output_path)
    LOGGER.info(
        "wrote %s: %d rows for %d/%d proteins (%.1f terms per protein)",
        output_path,
        rows_written,
        proteins_with_predictions,
        len(protein_ids),
        rows_written / max(1, proteins_with_predictions),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path, help="Output directory of cafa6-train-multilabel-nn")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/features/embeddings/test_esm2_150M"))
    parser.add_argument("--protein-features", type=Path, default=Path("data/features/sequence/test_features.tsv"))
    parser.add_argument("--taxonomy", type=Path, default=Path("data/Test/test_taxonomy.tsv"))
    parser.add_argument("--promoter-nt-dir", type=Path, default=Path("data/features/embeddings/test_promoter_nt100m"))
    parser.add_argument("--promoter-agront-dir", type=Path, default=Path("data/features/embeddings/test_promoter_agront1b"))
    parser.add_argument("--output", type=Path, default=None, help="Submission TSV (default: <model_dir>/submission.tsv)")
    parser.add_argument("--batch-size", type=int, default=8192, help="Test proteins scored per batch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--knn-device", default="auto")
    parser.add_argument("--knn-query-batch-size", type=int, default=256)
    parser.add_argument("--min-score", type=float, default=0.01, help="Omit pairs below this score (CAFA thresholds start at 0.01)")
    parser.add_argument("--max-terms", type=int, default=1500, help="Maximum GO terms per protein, after propagation")
    parser.add_argument("--limit", type=int, default=None, help="Debug: score only the first N test proteins")
    parser.add_argument("--protein-ids", type=Path, nargs="+", default=None, help="Only score proteins whose ID is in the first column of these files")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if not 0.0 < args.min_score <= 1.0:
        parser.error("--min-score must be in (0, 1]")
    run(
        model_dir=args.model_dir,
        embeddings_dir=args.embeddings_dir,
        protein_features_path=args.protein_features,
        taxonomy_path=args.taxonomy,
        promoter_dirs={"promoter_nt": args.promoter_nt_dir, PLANT_SPECIFIC_PROMOTER_PREFIX: args.promoter_agront_dir},
        output_path=args.output or args.model_dir / "submission.tsv",
        batch_size=args.batch_size,
        device_name=args.device,
        knn_device_name=args.knn_device,
        knn_query_batch_size=args.knn_query_batch_size,
        min_score=args.min_score,
        max_terms=args.max_terms,
        limit=args.limit,
        protein_ids_paths=args.protein_ids,
    )


if __name__ == "__main__":
    main()
