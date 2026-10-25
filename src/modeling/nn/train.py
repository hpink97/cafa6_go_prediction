"""Train one neural network that predicts probabilities for many GO terms.

The trainer reuses the feature construction and outer train/test split from the
per-term XGBoost baseline. Feature modalities are projected independently,
fused through a shared MLP, and mapped to one logit per GO term.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from modeling.data import (
    build_feature_table,
    load_embedding_table,
    load_protein_feature_table,
    load_term_counts,
    make_train_test_split,
    select_top_terms,
)
from modeling.evaluation.go import (
    NAMESPACE_TO_ASPECT,
    GeneOntology,
    build_closed_label_matrix,
    build_truth_ia_totals,
    cafa_fmax,
    limit_top_predictions,
    load_ia_weights,
    roc_auc_summary,
)
from modeling.nn.ensemble import apply_channel_selection, evaluate_cafa_predictions, tune_channel_by_aspect
from modeling.nn.model import (
    DEFAULT_PROJECTION_DIMS,
    PROMOTER_FUSION_DIM,
    MultilabelGOModel,
    feature_group_indices,
    make_loss,
    resolve_device,
)
from modeling.nn.score_channels import (
    blend_scores_by_aspect,
    exact_cosine_neighbors,
    fit_taxon_prior,
    knn_annotation_scores,
    load_taxonomy_map,
    make_taxon_stratified_validation_split,
    save_taxon_prior,
)

LOGGER = logging.getLogger("cafa6.modeling.multilabel_nn")


def _set_random_seed(random_state: int) -> None:
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def _limit_ids(ids: np.ndarray, max_samples: int | None, random_state: int) -> np.ndarray:
    if max_samples is None or max_samples >= len(ids):
        return ids
    rng = np.random.default_rng(random_state)
    selected = np.sort(rng.choice(len(ids), size=max_samples, replace=False))
    return ids[selected]


def _extract_arrays(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(features.loc[ids], dtype=np.float32, order="C", copy=True)
    y = np.array(labels.loc[ids], dtype=np.uint8, order="C", copy=True)
    return x, y


def _impute_nonfinite(arrays: list[np.ndarray]) -> np.ndarray:
    medians = np.nanmedian(np.where(np.isfinite(arrays[0]), arrays[0], np.nan), axis=0)
    medians = np.nan_to_num(medians, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    for values in arrays:
        invalid_rows, invalid_columns = np.where(~np.isfinite(values))
        if len(invalid_rows):
            values[invalid_rows, invalid_columns] = medians[invalid_columns]
    return medians


def _make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    random_state: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(random_state)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_features, batch_targets in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features)
        loss = criterion(logits, batch_targets)
        loss.backward()
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += loss.item() * len(batch_features)
        total_samples += len(batch_features)
    return total_loss / total_samples


def _predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    probability_batches = []
    target_batches = []
    with torch.inference_mode():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            float_targets = batch_targets.to(device, dtype=torch.float32, non_blocking=True)
            logits = model(batch_features)
            loss = criterion(logits, float_targets)
            total_loss += loss.item() * len(batch_features)
            total_samples += len(batch_features)
            probability_batches.append(torch.sigmoid(logits).cpu().numpy())
            target_batches.append(batch_targets.numpy())
    return (
        total_loss / total_samples,
        np.concatenate(probability_batches).astype(np.float32, copy=False),
        np.concatenate(target_batches).astype(np.uint8, copy=False),
    )


def _classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    terms: list[str],
    aspects: np.ndarray | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Per-term ROC-AUC/AP with macro means; given `aspects`, also per-aspect and micro ROC-AUC."""
    rows = []
    for index, term in enumerate(terms):
        term_targets = targets[:, index]
        positives = int(term_targets.sum())
        negatives = len(term_targets) - positives
        roc_auc = np.nan
        average_precision = np.nan
        if positives and negatives:
            roc_auc = float(roc_auc_score(term_targets, probabilities[:, index]))
        if positives:
            average_precision = float(average_precision_score(term_targets, probabilities[:, index]))
        rows.append(
            {
                "term": term,
                "n_positive": positives,
                "n_negative": negatives,
                "roc_auc": roc_auc,
                "average_precision": average_precision,
            }
        )
    per_term = pd.DataFrame(rows)
    valid_auc = per_term["roc_auc"].dropna()
    valid_ap = per_term["average_precision"].dropna()
    summary: dict[str, float | int] = {
        "macro_roc_auc": float(valid_auc.mean()) if len(valid_auc) else float("nan"),
        "macro_average_precision": float(valid_ap.mean()) if len(valid_ap) else float("nan"),
        "n_terms_with_roc_auc": len(valid_auc),
        "n_terms_with_average_precision": len(valid_ap),
    }
    if aspects is not None:
        summary.update(roc_auc_summary(targets, probabilities, aspects)[0])
    return summary, per_term


def _save_predictions(
    path: Path,
    protein_ids: np.ndarray,
    terms: list[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    propagated_probabilities: np.ndarray,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    arrays = {
        "protein_ids": np.asarray(protein_ids, dtype=str),
        "terms": np.asarray(terms, dtype=str),
        "targets": targets,
        "probabilities": probabilities,
        "propagated_probabilities": propagated_probabilities,
    }
    if extra_arrays:
        arrays.update(extra_arrays)
    np.savez_compressed(path, **arrays)


def run(
    embeddings_dir: Path | None,
    protein_features_path: Path | None,
    train_terms_path: Path,
    go_obo_path: Path,
    ia_path: Path,
    output_dir: Path,
    top_n: int,
    ancestor_closure: bool,
    train_taxonomy_path: Path,
    validation_split: str,
    test_size: float,
    validation_size: float,
    random_state: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    branch_dropout: float,
    trunk_dropout: float,
    trunk_dim: int,
    bottleneck_dim: int,
    loss_name: str,
    max_pos_weight: float,
    gamma_negative: float,
    gamma_positive: float,
    asymmetric_clip: float,
    patience: int,
    min_delta: float,
    checkpoint_metric: str,
    fmax_threshold_step: float,
    max_predictions_per_protein: int,
    use_taxon_prior: bool,
    taxon_prior_strength: float,
    checkpoint_taxon_weight: float,
    use_knn: bool,
    knn_k_values: list[int],
    knn_device_name: str,
    knn_query_batch_size: int,
    knn_similarity_power: float,
    ensemble_weight_step: float,
    ensemble_max_channel_weight: float,
    gradient_clip: float,
    device_name: str,
    num_workers: int,
    limit_terms: int | None,
    max_train_samples: int | None,
    max_validation_samples: int | None,
    max_test_samples: int | None,
    save_predictions: bool,
    overwrite: bool,
    taxonomy_features_path: Path | None,
    promoter_nt_dir: Path | None,
    promoter_agront_dir: Path | None,
    promoter_feature_cache: Path,
    promoter_knn_k: int,
    projection_dims: dict[str, int] | None = None,
    promoter_fusion_dim: int = PROMOTER_FUSION_DIM,
    trunk_layers: int = 2,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory {output_dir} is not empty; pass --overwrite to replace its artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    inference_dir = output_dir / "inference"
    inference_dir.mkdir(exist_ok=True)

    _set_random_seed(random_state)
    device = resolve_device(device_name)
    LOGGER.info("using device %s", device)
    ontology = GeneOntology.from_obo(go_obo_path)
    ia_weights = load_ia_weights(ia_path)

    promoter_embedding_dirs = None
    if promoter_nt_dir is not None or promoter_agront_dir is not None:
        promoter_embedding_dirs = {}
        if promoter_nt_dir is not None:
            promoter_embedding_dirs["promoter_nt"] = promoter_nt_dir
        if promoter_agront_dir is not None:
            promoter_embedding_dirs["promoter_agront"] = promoter_agront_dir

    if embeddings_dir is not None:
        all_protein_ids = load_embedding_table(embeddings_dir)["protein_id"]
    elif protein_features_path is not None:
        all_protein_ids = load_protein_feature_table(protein_features_path)["protein_id"]
    else:
        raise ValueError("At least one of ESM-2 embeddings or protein features is required")
    if use_knn and embeddings_dir is None:
        raise ValueError("The cosine-KNN channel needs ESM-2 embeddings; pass --no-knn with --no-esm2-features")
    taxonomy_by_id = load_taxonomy_map(train_taxonomy_path)
    outer_train_ids, test_ids = make_train_test_split(
        pd.Index(all_protein_ids),
        test_size=test_size,
        random_state=random_state,
    )
    if validation_split == "taxon_stratified":
        train_ids, validation_ids = make_taxon_stratified_validation_split(
            outer_train_ids,
            taxonomy_by_id,
            validation_size,
            random_state,
        )
    else:
        train_ids, validation_ids = train_test_split(
            outer_train_ids,
            test_size=validation_size,
            random_state=random_state,
            shuffle=True,
        )

    # Without ESM-2 there are no neighbours to impute promoters from; they are zero-filled instead.
    imputes_promoters = bool(promoter_embedding_dirs) and embeddings_dir is not None
    LOGGER.info("loading merged feature table")
    features = build_feature_table(
        embeddings_dir,
        protein_features_path,
        taxonomy_features_path,
        promoter_embedding_dirs=promoter_embedding_dirs,
        promoter_feature_cache_path=promoter_feature_cache if imputes_promoters else None,
        promoter_knn_k=promoter_knn_k,
        promoter_train_ids=outer_train_ids.tolist() if imputes_promoters else None,
        promoter_imputer_path=inference_dir / "promoter_imputer.npz" if imputes_promoters else None,
        promoter_imputer_donor_ids=all_protein_ids.tolist() if imputes_promoters else None,
    )
    term_counts = load_term_counts(train_terms_path)
    base_terms = select_top_terms(term_counts, top_n=top_n)
    if limit_terms is not None:
        base_terms = base_terms[:limit_terms]
    terms = ontology.expand_with_ancestors(base_terms) if ancestor_closure else base_terms
    labels = build_closed_label_matrix(
        train_terms_path,
        features.index,
        terms,
        ontology,
        ancestor_closure=ancestor_closure,
    )
    LOGGER.info(
        "selected %d base terms and %d ancestor-closed output terms",
        len(base_terms),
        len(terms),
    )

    train_ids = _limit_ids(train_ids, max_train_samples, random_state)
    validation_ids = _limit_ids(validation_ids, max_validation_samples, random_state + 1)
    test_ids = _limit_ids(test_ids, max_test_samples, random_state + 2)
    has_test = len(test_ids) > 0
    evaluation_ids = np.concatenate((validation_ids, test_ids))
    evaluation_truth_ia = build_truth_ia_totals(
        train_terms_path,
        evaluation_ids,
        ontology,
        ia_weights,
    )
    validation_truth_ia = {
        namespace: values[: len(validation_ids)]
        for namespace, values in evaluation_truth_ia.items()
    }
    test_truth_ia = {
        namespace: values[len(validation_ids):]
        for namespace, values in evaluation_truth_ia.items()
    }
    LOGGER.info(
        "training %d terms on %d train / %d validation / %d test proteins",
        len(terms),
        len(train_ids),
        len(validation_ids),
        len(test_ids),
    )

    feature_names = list(features.columns)
    groups = feature_group_indices(feature_names)
    LOGGER.info("feature groups: %s", {name: len(indices) for name, indices in groups.items()})
    x_train, y_train = _extract_arrays(features, labels, train_ids)
    x_validation, y_validation = _extract_arrays(features, labels, validation_ids)
    x_test, y_test = _extract_arrays(features, labels, test_ids)
    aspects = np.asarray([NAMESPACE_TO_ASPECT[ontology.namespaces[term]] for term in terms])
    # Test-time channels use every labelled training protein, not only the inner training split.
    inference_ids = np.asarray(labels.index, dtype=str)
    inference_labels = np.array(labels.to_numpy(), dtype=np.uint8, copy=True)
    if use_taxon_prior:
        save_taxon_prior(
            inference_dir / "taxon_prior.npz",
            fit_taxon_prior(inference_ids, inference_labels, taxonomy_by_id, aspects, strength=taxon_prior_strength),
            terms,
        )
    if use_knn:
        np.savez(
            inference_dir / "knn_donors.npz",
            protein_ids=inference_ids,
            embeddings=load_embedding_table(embeddings_dir).set_index("protein_id").loc[inference_ids].to_numpy(dtype=np.float32),
            labels=inference_labels,
        )
    del inference_labels
    taxon_prior = None
    validation_taxon_scores = None
    test_taxon_scores = None
    if use_taxon_prior:
        taxon_prior = fit_taxon_prior(
            train_ids,
            y_train,
            taxonomy_by_id,
            aspects,
            strength=taxon_prior_strength,
        )
        validation_taxon_scores = taxon_prior.predict(validation_ids, taxonomy_by_id)
        test_taxon_scores = taxon_prior.predict(test_ids, taxonomy_by_id) if has_test else None
    del features, labels
    gc.collect()

    feature_medians = _impute_nonfinite([x_train, x_validation, x_test])
    agront_indices = np.asarray(groups.get("promoter_agront", []), dtype=int)
    agront_is_applicable = []
    if len(agront_indices):
        agront_is_applicable = [
            np.any(values[:, agront_indices] != 0.0, axis=1)
            for values in (x_train, x_validation, x_test)
        ]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    x_validation = scaler.transform(x_validation).astype(np.float32, copy=False)
    if has_test:
        x_test = scaler.transform(x_test).astype(np.float32, copy=False)
    if len(agront_indices):
        for values, is_applicable in zip(
            (x_train, x_validation, x_test),
            agront_is_applicable,
        ):
            values[np.ix_(~is_applicable, agront_indices)] = 0.0

    pin_memory = device.type == "cuda"
    train_loader = _make_loader(
        x_train,
        y_train,
        batch_size,
        shuffle=True,
        random_state=random_state,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = _make_loader(
        x_validation,
        y_validation,
        batch_size,
        shuffle=False,
        random_state=random_state,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = _make_loader(
        x_test,
        y_test,
        batch_size,
        shuffle=False,
        random_state=random_state,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = MultilabelGOModel(
        groups,
        n_terms=len(terms),
        trunk_dim=trunk_dim,
        bottleneck_dim=bottleneck_dim,
        branch_dropout=branch_dropout,
        trunk_dropout=trunk_dropout,
        projection_dims=projection_dims,
        promoter_fusion_dim=promoter_fusion_dim,
        trunk_layers=trunk_layers,
    ).to(device)
    criterion, pos_weights = make_loss(
        loss_name,
        y_train,
        max_pos_weight,
        gamma_negative,
        gamma_positive,
        asymmetric_clip,
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    projection_dims = {**DEFAULT_PROJECTION_DIMS, **(projection_dims or {})}
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("model has %d trainable parameters", n_parameters)

    history = []
    best_score = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device, gradient_clip)
        validation_loss, validation_probabilities, validation_targets = _predict(
            model,
            validation_loader,
            criterion,
            device,
        )
        checkpoint_probabilities = validation_probabilities
        if validation_taxon_scores is not None and checkpoint_taxon_weight > 0:
            checkpoint_probabilities = blend_scores_by_aspect(
                validation_probabilities,
                validation_taxon_scores,
                aspects,
                {aspect: checkpoint_taxon_weight for aspect in ("P", "F", "C")},
            )
        limited_validation_probabilities = limit_top_predictions(
            checkpoint_probabilities,
            max_predictions_per_protein,
        )
        propagated_validation_probabilities = ontology.propagate_scores(
            limited_validation_probabilities,
            terms,
            copy=False,
        )
        validation_summary, _ = _classification_metrics(
            validation_targets,
            propagated_validation_probabilities,
            terms,
        )
        validation_cafa, _ = cafa_fmax(
            validation_targets,
            propagated_validation_probabilities,
            terms,
            ontology,
            ia_weights,
            truth_ia_totals=validation_truth_ia,
            threshold_step=fmax_threshold_step,
            propagate=False,
        )
        validation_summary.update(validation_cafa)
        validation_auc = float(validation_summary["macro_roc_auc"])
        metric_values = {
            "cafa_fmax": float(validation_summary["cafa_fmax_mean"]),
            "roc_auc": validation_auc,
            "average_precision": float(validation_summary["macro_average_precision"]),
            "loss": -validation_loss,
        }
        selection_score = metric_values[checkpoint_metric]
        if not np.isfinite(selection_score):
            selection_score = -validation_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **validation_summary,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        LOGGER.info(
            "epoch %d train_loss=%.5f val_loss=%.5f val_auc=%.5f val_ap=%.5f val_fmax=%.5f",
            epoch,
            train_loss,
            validation_loss,
            validation_auc,
            validation_summary["macro_average_precision"],
            validation_summary["cafa_fmax_mean"],
        )
        if selection_score > best_score + min_delta:
            best_score = selection_score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info("early stopping after epoch %d", epoch)
                break

    if best_state is None:
        raise RuntimeError("Training completed without producing a model state")
    model.load_state_dict(best_state)

    validation_loss, validation_probabilities, validation_targets = _predict(
        model,
        validation_loader,
        criterion,
        device,
    )
    test_loss, test_probabilities, test_targets = None, None, None
    test_nn_summary: dict[str, float | int] = {}
    if has_test:
        test_loss, test_probabilities, test_targets = _predict(model, test_loader, criterion, device)
    validation_nn_cafa, validation_nn_cafa_curve, propagated_validation_probabilities = evaluate_cafa_predictions(
        validation_targets,
        validation_probabilities,
        terms,
        ontology,
        ia_weights,
        validation_truth_ia,
        fmax_threshold_step,
        max_predictions_per_protein,
    )
    if has_test:
        test_nn_cafa, test_nn_cafa_curve, propagated_test_probabilities = evaluate_cafa_predictions(
            test_targets,
            test_probabilities,
            terms,
            ontology,
            ia_weights,
            test_truth_ia,
            fmax_threshold_step,
            max_predictions_per_protein,
        )
        test_nn_summary, test_nn_per_term = _classification_metrics(
            test_targets,
            propagated_test_probabilities,
            terms,
            aspects,
        )
        test_nn_summary.update(test_nn_cafa)
    validation_nn_summary, validation_nn_per_term = _classification_metrics(
        validation_targets,
        propagated_validation_probabilities,
        terms,
        aspects,
    )
    validation_nn_summary.update(validation_nn_cafa)

    validation_ensemble_probabilities = validation_probabilities.copy()
    test_ensemble_probabilities = test_probabilities.copy() if has_test else None
    validation_prediction_extras: dict[str, np.ndarray] = {}
    test_prediction_extras: dict[str, np.ndarray] = {}
    ensemble_config: dict[str, object] = {}

    if validation_taxon_scores is not None:
        taxon_selection = tune_channel_by_aspect(
            validation_ensemble_probabilities,
            {"taxon_prior": validation_taxon_scores},
            aspects,
            validation_targets,
            terms,
            ontology,
            ia_weights,
            validation_truth_ia,
            fmax_threshold_step,
            max_predictions_per_protein,
            ensemble_weight_step,
            ensemble_max_channel_weight,
        )
        validation_ensemble_probabilities, _ = apply_channel_selection(
            validation_ensemble_probabilities,
            {"taxon_prior": validation_taxon_scores},
            aspects,
            taxon_selection,
        )
        test_ensemble_probabilities, _ = apply_channel_selection(
            test_ensemble_probabilities,
            {"taxon_prior": test_taxon_scores},
            aspects,
            taxon_selection,
        ) if has_test else (None, None)
        validation_prediction_extras["taxon_prior_probabilities"] = validation_taxon_scores
        if has_test:
            test_prediction_extras["taxon_prior_probabilities"] = test_taxon_scores
        ensemble_config["taxon_prior"] = taxon_selection
        LOGGER.info("validation-tuned taxon-prior blend: %s", taxon_selection)

    if use_knn:
        k_values = sorted(set(knn_k_values))
        if not k_values or k_values[0] < 1:
            raise ValueError("knn_k_values must contain positive integers")
        if k_values[-1] > len(train_ids):
            raise ValueError(f"Largest KNN k={k_values[-1]} exceeds {len(train_ids)} training donors")
        LOGGER.info("finding exact cosine neighbors for validation/test with max k=%d", k_values[-1])
        embedding_table = load_embedding_table(embeddings_dir).set_index("protein_id")
        donor_embeddings = embedding_table.loc[train_ids].to_numpy(dtype=np.float32)
        query_embeddings = embedding_table.loc[np.concatenate((validation_ids, test_ids))].to_numpy(dtype=np.float32)
        del embedding_table
        neighbor_similarities, neighbor_indices = exact_cosine_neighbors(
            donor_embeddings,
            query_embeddings,
            k_values[-1],
            device=knn_device_name,
            query_batch_size=knn_query_batch_size,
        )
        del donor_embeddings, query_embeddings
        validation_neighbor_similarities = neighbor_similarities[: len(validation_ids)]
        test_neighbor_similarities = neighbor_similarities[len(validation_ids):]
        validation_neighbor_indices = neighbor_indices[: len(validation_ids)]
        test_neighbor_indices = neighbor_indices[len(validation_ids):]

        validation_knn_variants = {
            f"k{k}": knn_annotation_scores(
                validation_neighbor_indices,
                validation_neighbor_similarities,
                y_train,
                aspects,
                k=k,
                similarity_power=knn_similarity_power,
            )
            for k in k_values
        }
        knn_selection = tune_channel_by_aspect(
            validation_ensemble_probabilities,
            validation_knn_variants,
            aspects,
            validation_targets,
            terms,
            ontology,
            ia_weights,
            validation_truth_ia,
            fmax_threshold_step,
            max_predictions_per_protein,
            ensemble_weight_step,
            ensemble_max_channel_weight,
        )
        validation_ensemble_probabilities, selected_validation_knn = apply_channel_selection(
            validation_ensemble_probabilities,
            validation_knn_variants,
            aspects,
            knn_selection,
        )
        selected_k_values = sorted({int(str(settings["variant"])[1:]) for settings in knn_selection.values()})
        validation_prediction_extras["knn_probabilities"] = selected_validation_knn
        ensemble_config["esm_cosine_knn"] = knn_selection
        ensemble_config["esm_cosine_knn_max_similarity"] = {
            "validation_mean": float(validation_neighbor_similarities[:, 0].mean()),
        }
        if has_test:
            test_knn_variants = {
                f"k{k}": knn_annotation_scores(
                    test_neighbor_indices,
                    test_neighbor_similarities,
                    y_train,
                    aspects,
                    k=k,
                    similarity_power=knn_similarity_power,
                )
                for k in selected_k_values
            }
            test_ensemble_probabilities, selected_test_knn = apply_channel_selection(
                test_ensemble_probabilities,
                test_knn_variants,
                aspects,
                knn_selection,
            )
            test_prediction_extras["knn_probabilities"] = selected_test_knn
            ensemble_config["esm_cosine_knn_max_similarity"]["test_mean"] = float(test_neighbor_similarities[:, 0].mean())
            del test_knn_variants
        LOGGER.info("validation-tuned ESM cosine-KNN blend: %s", knn_selection)
        del validation_knn_variants, neighbor_similarities, neighbor_indices
        gc.collect()

    validation_cafa, validation_cafa_curve, propagated_validation_ensemble = evaluate_cafa_predictions(
        validation_targets,
        validation_ensemble_probabilities,
        terms,
        ontology,
        ia_weights,
        validation_truth_ia,
        fmax_threshold_step,
        max_predictions_per_protein,
    )
    test_summary: dict[str, float | int] = {}
    if has_test:
        test_cafa, test_cafa_curve, propagated_test_ensemble = evaluate_cafa_predictions(
            test_targets,
            test_ensemble_probabilities,
            terms,
            ontology,
            ia_weights,
            test_truth_ia,
            fmax_threshold_step,
            max_predictions_per_protein,
        )
        test_summary, test_per_term = _classification_metrics(
            test_targets,
            propagated_test_ensemble,
            terms,
            aspects,
        )
        test_summary.update(test_cafa)
        test_prediction_extras.update(
            {
                "ensemble_probabilities": test_ensemble_probabilities,
                "propagated_ensemble_probabilities": propagated_test_ensemble,
            }
        )
    validation_summary, validation_per_term = _classification_metrics(
        validation_targets,
        propagated_validation_ensemble,
        terms,
        aspects,
    )
    validation_summary.update(validation_cafa)
    validation_prediction_extras.update(
        {
            "ensemble_probabilities": validation_ensemble_probabilities,
            "propagated_ensemble_probabilities": propagated_validation_ensemble,
        }
    )

    pd.Series(train_ids, name="protein_id").to_csv(splits_dir / "train_protein_ids.tsv", sep="\t", index=False)
    pd.Series(validation_ids, name="protein_id").to_csv(
        splits_dir / "validation_protein_ids.tsv", sep="\t", index=False
    )
    pd.Series(test_ids, name="protein_id").to_csv(splits_dir / "test_protein_ids.tsv", sep="\t", index=False)
    base_term_set = set(base_terms)
    pd.DataFrame(
        {
            "term": terms,
            "namespace": [ontology.namespaces[term] for term in terms],
            "aspect": [NAMESPACE_TO_ASPECT[ontology.namespaces[term]] for term in terms],
            "is_base_term": [term in base_term_set for term in terms],
            "direct_annotation_count": term_counts.reindex(terms, fill_value=0).to_numpy(),
            "train_positive": y_train.sum(axis=0),
            "validation_positive": y_validation.sum(axis=0),
            "test_positive": y_test.sum(axis=0),
            "pos_weight": pos_weights,
        }
    ).to_csv(output_dir / "terms.tsv", sep="\t", index=False)
    pd.DataFrame(history).to_csv(output_dir / "history.tsv", sep="\t", index=False)
    validation_per_term.to_csv(output_dir / "validation_metrics.tsv", sep="\t", index=False)
    validation_nn_per_term.to_csv(output_dir / "validation_nn_metrics.tsv", sep="\t", index=False)
    validation_cafa_curve.to_csv(output_dir / "validation_cafa_curve.tsv", sep="\t", index=False)
    validation_nn_cafa_curve.to_csv(output_dir / "validation_nn_cafa_curve.tsv", sep="\t", index=False)
    if has_test:
        test_per_term.to_csv(output_dir / "test_metrics.tsv", sep="\t", index=False)
        test_nn_per_term.to_csv(output_dir / "test_nn_metrics.tsv", sep="\t", index=False)
        test_cafa_curve.to_csv(output_dir / "test_cafa_curve.tsv", sep="\t", index=False)
        test_nn_cafa_curve.to_csv(output_dir / "test_nn_cafa_curve.tsv", sep="\t", index=False)
    np.savez(
        output_dir / "preprocessing.npz",
        feature_names=np.asarray(feature_names),
        medians=feature_medians,
        means=scaler.mean_.astype(np.float32),
        scales=scaler.scale_.astype(np.float32),
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "feature_groups": groups,
            "projection_dims": projection_dims,
            "promoter_fusion_dim": promoter_fusion_dim,
            "trunk_layers": trunk_layers,
            "terms": terms,
            "base_terms": base_terms,
            "ancestor_closure": ancestor_closure,
            "trunk_dim": trunk_dim,
            "bottleneck_dim": bottleneck_dim,
            "branch_dropout": branch_dropout,
            "trunk_dropout": trunk_dropout,
        },
        output_dir / "best_model.pt",
    )
    if save_predictions:
        _save_predictions(
            output_dir / "validation_predictions.npz",
            validation_ids,
            terms,
            validation_targets,
            validation_probabilities,
            propagated_validation_probabilities,
            validation_prediction_extras,
        )
        if has_test:
            _save_predictions(
                output_dir / "test_predictions.npz",
                test_ids,
                terms,
                test_targets,
                test_probabilities,
                propagated_test_probabilities,
                test_prediction_extras,
            )

    if taxon_prior is not None:
        save_taxon_prior(output_dir / "taxon_prior.npz", taxon_prior, terms)
    (output_dir / "ensemble_config.json").write_text(json.dumps(ensemble_config, indent=2) + "\n")

    config = {
        "embeddings_dir": str(embeddings_dir) if embeddings_dir else None,
        "protein_features_path": str(protein_features_path) if protein_features_path else None,
        "taxonomy_features_path": str(taxonomy_features_path) if taxonomy_features_path else None,
        "promoter_nt_dir": str(promoter_nt_dir) if promoter_nt_dir else None,
        "promoter_agront_dir": str(promoter_agront_dir) if promoter_agront_dir else None,
        "promoter_feature_cache": str(promoter_feature_cache),
        "promoter_knn_k": promoter_knn_k,
        "train_terms_path": str(train_terms_path),
        "train_taxonomy_path": str(train_taxonomy_path),
        "go_obo_path": str(go_obo_path),
        "ia_path": str(ia_path),
        "top_n": top_n,
        "n_base_terms_run": len(base_terms),
        "n_terms_run": len(terms),
        "ancestor_closure": ancestor_closure,
        "validation_split": validation_split,
        "test_size": test_size,
        "validation_size_within_train": validation_size,
        "random_state": random_state,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "branch_dropout": branch_dropout,
        "trunk_dropout": trunk_dropout,
        "trunk_dim": trunk_dim,
        "bottleneck_dim": bottleneck_dim,
        "loss": loss_name,
        "max_pos_weight": max_pos_weight,
        "gamma_negative": gamma_negative,
        "gamma_positive": gamma_positive,
        "asymmetric_clip": asymmetric_clip,
        "patience": patience,
        "min_delta": min_delta,
        "checkpoint_metric": checkpoint_metric,
        "fmax_threshold_step": fmax_threshold_step,
        "max_predictions_per_protein": max_predictions_per_protein,
        "use_taxon_prior": use_taxon_prior,
        "taxon_prior_strength": taxon_prior_strength,
        "checkpoint_taxon_weight": checkpoint_taxon_weight,
        "use_knn": use_knn,
        "knn_k_values": knn_k_values,
        "knn_device": knn_device_name,
        "knn_query_batch_size": knn_query_batch_size,
        "knn_similarity_power": knn_similarity_power,
        "ensemble_weight_step": ensemble_weight_step,
        "ensemble_max_channel_weight": ensemble_max_channel_weight,
        "gradient_clip": gradient_clip,
        "device": str(device),
        "n_parameters": n_parameters,
        "projection_dims": projection_dims,
        "promoter_fusion_dim": promoter_fusion_dim,
        "trunk_layers": trunk_layers,
        "agront_nonapplicable_rows_zeroed_after_scaling": True,
        "max_train_samples": max_train_samples,
        "max_validation_samples": max_validation_samples,
        "max_test_samples": max_test_samples,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary = {
        "best_epoch": best_epoch,
        "validation_loss": validation_loss,
        **{f"validation_nn_{key}": value for key, value in validation_nn_summary.items()},
        **{f"validation_{key}": value for key, value in validation_summary.items()},
        "test_loss": test_loss,
        **{f"test_nn_{key}": value for key, value in test_nn_summary.items()},
        **{f"test_{key}": value for key, value in test_summary.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info(
        "best epoch %d: validation ensemble Fmax %.5f (NN %.5f); test ensemble Fmax %s (NN %s)",
        best_epoch,
        validation_summary["cafa_fmax_mean"],
        validation_nn_summary["cafa_fmax_mean"],
        f"{test_summary['cafa_fmax_mean']:.5f}" if has_test else "n/a",
        f"{test_nn_summary['cafa_fmax_mean']:.5f}" if has_test else "n/a",
    )


def resolve_feature_inputs(paths: dict[str, Path], disabled: set[str]) -> dict[str, Path | None]:
    """Input path per feature block, or None when disabled.

    A block that is enabled but missing on disk is an error rather than being silently dropped, so
    an ablation can never quietly train a different feature set than the one it names.
    """
    embedding_blocks = {"esm2", "promoter_nt", "promoter_agront"}
    resolved: dict[str, Path | None] = {}
    for block, path in paths.items():
        if block in disabled:
            resolved[block] = None
            continue
        marker = path / "metadata.json" if block in embedding_blocks else path
        if not marker.exists():
            flag = f"--no-{block.replace('_', '-')}-features"
            raise FileNotFoundError(f"{block} features not found at {marker}; build them or pass {flag}")
        resolved[block] = path
    if not any(resolved.get(block) for block in ("esm2", "protein")):
        raise ValueError("Keep at least one of the ESM-2 or protein feature blocks; they define the protein set")
    if resolved.get("promoter_agront") and resolved.get("esm2") and not resolved.get("taxonomy"):
        raise ValueError("AgroNT promoter imputation needs taxonomy features to identify plants; drop --no-taxonomy-features")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/features/embeddings/train_esm2_150M"))
    parser.add_argument("--protein-features", type=Path, default=Path("data/features/sequence/train_features.tsv"))
    parser.add_argument("--taxonomy-features", type=Path, default=Path("data/features/taxonomy/train_taxonomy_features.tsv"))
    parser.add_argument("--promoter-nt-dir", type=Path, default=Path("data/features/embeddings/train_promoter_nt100m"))
    parser.add_argument("--promoter-agront-dir", type=Path, default=Path("data/features/embeddings/train_promoter_agront1b"))
    parser.add_argument(
        "--promoter-feature-cache",
        type=Path,
        default=Path("data/features/promoters/imputed_promoter_features_esm2_150M.csv"),
    )
    parser.add_argument("--promoter-knn-k", type=int, default=5)
    parser.add_argument("--train-terms", type=Path, default=Path("data/Train/train_terms.tsv"))
    parser.add_argument("--train-taxonomy", dest="train_taxonomy_path", type=Path, default=Path("data/Train/train_taxonomy.tsv"))
    parser.add_argument("--go-obo", dest="go_obo_path", type=Path, default=Path("data/Train/go-basic.obo"))
    parser.add_argument("--ia", dest="ia_path", type=Path, default=Path("data/IA.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/models/go_term_multilabel_nn"))
    parser.add_argument("--top-n", type=int, default=1000, help="Frequent direct terms selected before adding ancestors")
    parser.add_argument(
        "--no-ancestor-closure",
        action="store_true",
        help="Ablation: do not append ancestors or close labels (predictions are still propagated for CAFA scoring)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.15, help="Fraction of the outer training set used for validation")
    parser.add_argument(
        "--validation-split",
        choices=("random", "taxon_stratified"),
        default="taxon_stratified",
        help="Inner validation split; rare taxa are pooled for stratification",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--branch-dropout", type=float, default=0.1)
    parser.add_argument("--trunk-dropout", type=float, default=0.2)
    parser.add_argument("--trunk-dim", type=int, default=512)
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--trunk-layers", type=int, default=2, help="Linear layers in the shared trunk, including the bottleneck")
    parser.add_argument(
        "--projection-dims",
        nargs="+",
        default=[],
        metavar="GROUP=DIM",
        help=f"Override branch projection widths, e.g. promoter_nt=64 (defaults: {DEFAULT_PROJECTION_DIMS})",
    )
    parser.add_argument("--promoter-fusion-dim", type=int, default=PROMOTER_FUSION_DIM, help="Width of the fused promoter vector entering the trunk")
    parser.add_argument("--loss", choices=("bce", "weighted_bce", "asymmetric"), default="weighted_bce")
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--gamma-negative", type=float, default=4.0)
    parser.add_argument("--gamma-positive", type=float, default=0.0)
    parser.add_argument("--asymmetric-clip", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-metric",
        choices=("cafa_fmax", "roc_auc", "average_precision", "loss"),
        default="cafa_fmax",
    )
    parser.add_argument(
        "--fmax-threshold-step",
        type=float,
        default=0.01,
        help="CAFA confidence threshold step in (0, 1); official evaluator default is 0.01",
    )
    parser.add_argument(
        "--max-predictions-per-protein",
        type=int,
        default=1500,
        help="Explicit term limit applied before ancestor propagation",
    )
    parser.add_argument("--no-taxon-prior", action="store_true", help="Disable the exact-taxon prevalence score channel")
    parser.add_argument("--taxon-prior-strength", type=float, default=10.0, help="Global-prior pseudo-count for exact-taxon prevalence")
    parser.add_argument(
        "--checkpoint-taxon-weight",
        type=float,
        default=0.45,
        help="Fixed taxon-prior weight used only for epoch checkpoint selection",
    )
    parser.add_argument("--no-knn", action="store_true", help="Disable the ESM cosine-KNN score channel")
    parser.add_argument("--knn-k-values", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--knn-device", dest="knn_device_name", default="auto")
    parser.add_argument("--knn-query-batch-size", type=int, default=256)
    parser.add_argument("--knn-similarity-power", type=float, default=1.0)
    parser.add_argument("--ensemble-weight-step", type=float, default=0.05)
    parser.add_argument("--ensemble-max-channel-weight", type=float, default=0.8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or another torch device")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-terms", type=int, default=None, help="Debug: keep only the first N base terms before ancestor closure")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Debug: cap training rows after splitting")
    parser.add_argument("--max-validation-samples", type=int, default=None, help="Debug: cap validation rows after splitting")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Debug: cap test rows after splitting")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--no-esm2-features", action="store_true", help="Ablation: drop ESM-2 (requires --no-knn; promoters are zero-filled)")
    parser.add_argument("--no-protein-features", action="store_true", help="Ablation: drop the sequence-derived protein features")
    parser.add_argument("--no-taxonomy-features", action="store_true", help="Ablation: drop the taxonomy one-hot features")
    parser.add_argument("--no-promoter-nt-features", action="store_true", help="Ablation: drop the NT promoter embeddings")
    parser.add_argument("--no-promoter-agront-features", action="store_true", help="Ablation: drop the AgroNT promoter embeddings")
    parser.add_argument("--no-promoter-features", action="store_true", help="Ablation: drop the NT and AgroNT promoter embeddings")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    arguments = vars(parser.parse_args())
    log_level = arguments.pop("log_level")
    arguments["save_predictions"] = not arguments.pop("no_save_predictions")
    arguments["ancestor_closure"] = not arguments.pop("no_ancestor_closure")
    arguments["use_taxon_prior"] = not arguments.pop("no_taxon_prior")
    arguments["use_knn"] = not arguments.pop("no_knn")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        arguments["projection_dims"] = {
            group: int(dim) for group, dim in (item.split("=", 1) for item in arguments.pop("projection_dims"))
        }
    except ValueError:
        parser.error("--projection-dims takes GROUP=DIM pairs")
    unknown = set(arguments["projection_dims"]) - set(DEFAULT_PROJECTION_DIMS)
    if unknown:
        parser.error(f"unknown projection groups {sorted(unknown)}; choose from {sorted(DEFAULT_PROJECTION_DIMS)}")
    arguments["loss_name"] = arguments.pop("loss")
    arguments["device_name"] = arguments.pop("device")
    no_promoters = arguments.pop("no_promoter_features")
    disabled = {
        block
        for block, flag in (
            ("esm2", arguments.pop("no_esm2_features")),
            ("protein", arguments.pop("no_protein_features")),
            ("taxonomy", arguments.pop("no_taxonomy_features")),
            ("promoter_nt", arguments.pop("no_promoter_nt_features") or no_promoters),
            ("promoter_agront", arguments.pop("no_promoter_agront_features") or no_promoters),
        )
        if flag
    }
    paths = {
        "esm2": arguments.pop("embeddings_dir"),
        "protein": arguments.pop("protein_features"),
        "taxonomy": arguments.pop("taxonomy_features"),
        "promoter_nt": arguments.pop("promoter_nt_dir"),
        "promoter_agront": arguments.pop("promoter_agront_dir"),
    }
    try:
        inputs = resolve_feature_inputs(paths, disabled)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    run(
        embeddings_dir=inputs["esm2"],
        protein_features_path=inputs["protein"],
        train_terms_path=arguments.pop("train_terms"),
        taxonomy_features_path=inputs["taxonomy"],
        promoter_nt_dir=inputs["promoter_nt"],
        promoter_agront_dir=inputs["promoter_agront"],
        **arguments,
    )


if __name__ == "__main__":
    main()