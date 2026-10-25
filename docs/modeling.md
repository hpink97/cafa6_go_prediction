# Models and evaluation

## Multi-label neural network (main model)

`cafa6-train-multilabel-nn` fits one PyTorch network over all selected GO terms.

**Inputs.** Each feature group gets its own projection, and the projections are concatenated
into a shared MLP trunk (512 → 256 bottleneck) that emits one logit per term:

| Group | Projection |
|---|---|
| ESM-2 150M | 640 → 256 |
| Promoter NT-100M | 512 → 128 |
| Promoter AgroNT-1B | 1500 → 128, gated to zero for non-plants |
| Handcrafted + kingdom | 63 → 32 |

The two promoter projections are fused to 128. Missing promoters are imputed as the mean promoter
vector of the 5 nearest training proteins by ESM-2 cosine similarity. AgroNT is imputed only
for plants.

**Labels.** `--top-n` picks the most frequent directly annotated terms. Every `is_a` and
`part_of` ancestor is then added, and labels are closed over the ontology. `--top-n 1000` gives
2,180 outputs.

**Training.**
- Class-weighted BCE with clipped square-root positive weights (`--loss asymmetric` is the
  alternative).
- AdamW, dropout and gradient clipping.
- Early stopping on validation IA-weighted CAFA Fmax.
- An outer random protein split (`--test-size`), plus an inner validation split that is
  taxon-stratified by default.

**Score channels.** Two non-neural channels are blended in, with one weight per aspect chosen
on validation:
- an exact-taxon term prior, shrunk toward global prevalence;
- cosine KNN over ESM-2 embeddings, transferring ancestor-closed labels from training proteins.

Turn them off with `--no-taxon-prior` / `--no-knn`.

**Feature-group switches.** `--no-protein-features`, `--no-taxonomy-features` and
`--no-promoter-features`.

```bash
uv run cafa6-train-multilabel-nn --top-n 1000 --output-dir data/models/nn_1000
# smoke test
uv run cafa6-train-multilabel-nn --limit-terms 32 --max-train-samples 2048 --max-validation-samples 512 \
    --max-test-samples 512 --epochs 1 --output-dir /tmp/cafa6_nn_smoke
```

**Outputs.** The run directory holds:
- the best checkpoint;
- preprocessing statistics;
- split IDs;
- per-epoch history;
- per-term metrics;
- CAFA curves;
- predictions;
- `inference/` artifacts fitted on all labelled proteins (promoter imputer, KNN donors, taxon
  prior).

### Prediction

```bash
uv run cafa6-predict-multilabel-nn data/models/nn_1000 --output data/submissions/nn_1000.tsv
```

This rebuilds the training feature layout for the test superset and applies the saved scaling
and channel weights. It keeps the top 1,500 terms per protein, propagates to ancestors, drops
scores below `--min-score` (0.01), and writes headerless `protein<TAB>term<TAB>score` rows.

## Per-term XGBoost baseline

`cafa6-train-go-models` trains one tuned binary classifier per GO term on ESM-2 35M plus the
handcrafted features. The pipeline is imputer → scaler → optional LightGBM feature selection →
XGBoost. It uses `GridSearchCV` on ROC-AUC and a shared 80/20 protein split. Only scores, grid
results and feature importances are saved; no models are persisted.

```bash
uv run cafa6-train-go-models --top-n 100 --output-dir data/models/go_term_xgboost
```

## Evaluation

**Internal (during training).** A protein-centric, IA-weighted CAFA Fmax per aspect (BP/MF/CC)
on held-out proteins (`modeling/evaluation/go.py`):
- predictions are limited to the top 1,500 per protein and max-propagated to ancestors;
- precision is averaged over proteins with a prediction, recall over all proteins with truth;
- recall counts true terms outside the model vocabulary, so a small output space is penalised.

Per-term ROC-AUC and average precision are recorded alongside. A random protein split cannot
reproduce the challenge's prospective target groups, so treat these numbers as a proxy.

**Official (CAFA-native).** `cafa6-evaluate <predictions_dir>` runs the official
CAFA-evaluator-PK against the May 2026 ground truth. It scores each of:
- **NK**: no prior knowledge;
- **LK**: limited knowledge;
- **PK**: partial knowledge, with known annotations excluded.

It uses CAFA normalisation, fill propagation and the terms of interest, and summarises weighted
F per regime and aspect.
