# CAFA6 protein function prediction

Predicting Gene Ontology (GO) terms for proteins in the
[CAFA6 Kaggle challenge](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction).
A multi-label neural network combines a protein language model (ESM-2) with handcrafted
sequence features, taxonomy, and embeddings of each gene's **promoter DNA**. A controlled
ablation measures what each feature group adds, scored with ROC-AUC and the official CAFA
evaluator.

<!-- results:start -->
**Results** (6 feature configurations × 3 seeds, paired):
- **ESM-2 carries most of the signal.** Kingdom features add a small, consistent gain; handcrafted
  features do not transfer to the official CAFA6 truth.
- **Promoter embeddings consistently improve per-term ranking (macro ROC-AUC)**, internally and
  officially, but not CAFA Fmax.
- **Promoters help "response to …" terms most:** twice the gain of other processes without ESM-2
  (p = 0.0001). With ESM-2 that extra benefit largely disappears.

| Config | Feature groups | Internal macro AUC | Official macro AUC |
|---|---|---|---|
| `esm2` | ESM-2 | 0.887 ± 0.002 | 0.838 ± 0.003 |
| `esm2_prot` | + handcrafted | 0.889 ± 0.002 | 0.834 ± 0.004 |
| `esm2_prot_tax` | + kingdom | 0.898 ± 0.001 | 0.844 ± 0.006 |
| `all` | + promoters | **0.906 ± 0.003** | **0.850 ± 0.002** |
| `prot_tax` | handcrafted + kingdom only | 0.829 ± 0.006 | 0.794 ± 0.007 |
| `prot_tax_promoters` | + promoters, no ESM-2 | 0.843 ± 0.001 | 0.799 ± 0.005 |

Official CAFA evaluator: see [results/README.md](results/README.md#official-cafa-scores).
Full analysis, figures and per-config outputs are in [results/](results/README.md).
<!-- results:end -->

## Approach

```mermaid
flowchart LR
    fasta[Protein sequences<br/>+ taxon IDs] --> esm[ESM-2 150M<br/>640-d]
    fasta --> hand[Handcrafted features<br/>56-d]
    fasta --> tax[Kingdom<br/>7-d]
    fasta --> prom[Promoter extraction<br/>Ensembl / NCBI + DIAMOND rescue]
    prom --> nt[Nucleotide Transformer<br/>512-d]
    prom --> agro[AgroNT, plants only<br/>1500-d]
    esm --> b1[Projection 256]
    hand --> b2[Projection 32]
    tax --> b2
    nt --> b3[Promoter projections<br/>fused to 128]
    agro --> b3
    b1 --> trunk[Shared MLP trunk]
    b2 --> trunk
    b3 --> trunk
    trunk --> out[2,180 GO terms<br/>ancestor-closed]
    out --> ch[Optional score channels:<br/>taxon prior, ESM-2 KNN]
    ch --> sub[Propagate to ancestors<br/>-> submission]
```

**Feature groups.** Each is built separately, keyed by protein ID, so it can be rebuilt or
dropped on its own. See [docs/features.md](docs/features.md).

| Group | Dim | What it captures | Coverage |
|---|---|---|---|
| ESM-2 150M embedding | 640 | evolutionary and structural signal learned from protein sequence | all proteins |
| Handcrafted | 56 | ProtParam biophysics, topology heuristics, terminal motifs, PROSITE patterns, amino-acid composition | all proteins |
| Kingdom | 7 | coarse taxonomy from NCBI | all proteins |
| Promoter, NT-v2 100M | 512 | the 3 kb upstream of each protein's gene, embedded with a multi-species DNA model | 97% train / 88% test |
| Promoter, AgroNT 1B | 1500 | the same promoters embedded with a plant genome model | plants only |

Promoters come from Ensembl and NCBI genomes for 55 anchor species. Proteins without a direct
gene link are rescued by DIAMOND alignment, labelled as an exact (same-taxon) or proxy
(same-phylum) match. The remaining gaps are imputed from the nearest training proteins in
ESM-2 space.

**Model.**
- One projection per feature group feeds a shared MLP that predicts the 1,000 most frequent
  terms plus all their ancestors.
- Training uses class-weighted BCE, with early stopping on IA-weighted CAFA Fmax.
- Predictions are propagated up the ontology before scoring.
- Two optional score channels can be blended in per aspect: an exact-taxon prior and ESM-2
  nearest-neighbour label transfer.

See [docs/modeling.md](docs/modeling.md).

**Evaluation.**
- **Internal:** 20% of labelled proteins are held out (seeded). Metrics are macro/micro ROC-AUC
  and protein-centric, IA-weighted Fmax per aspect.
- **Official:** predictions for the [released CAFA6 ground truth](https://doi.org/10.6084/m9.figshare.32861969)
  are scored with the [CAFA evaluator](https://github.com/claradepaolis/CAFA-evaluator-PK) under
  its three settings:
  - no knowledge (NK);
  - limited knowledge (LK);
  - partial knowledge (PK).

  It reports weighted Fmax, weighted micro-F and S-min; ROC-AUC is computed against the same
  truth.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Data is never committed; the
scripts below download or rebuild it. See [docs/DATA.md](docs/DATA.md).

```bash
uv sync
scripts/data/03_tools.sh                  # cafaeval, kaggle, diamond
scripts/data/01_competition.sh            # Kaggle data (needs a Kaggle API token)
scripts/data/02_ground_truth.sh           # CAFA6 ground truth from Figshare, checksum-verified
scripts/data/04_features.sh all           # every feature block (long: genomes and DNA models)

uv run cafa6-train-multilabel-nn --top-n 1000 --output-dir data/models/nn_1000
uv run cafa6-predict-multilabel-nn data/models/nn_1000 --output data/submissions/nn_1000.tsv
uv run cafa6-evaluate data/submissions     # official CAFA scores for every file in the folder
```

## Reproducing the ablation

```bash
uv run cafa6-ablation run --sweep feature-groups             # 6 configs x 3 seeds, ROC-AUC scored (~3.5 h on an M4 Pro)
uv run cafa6-ablation cafaeval --sweep feature-groups --best # official CAFA evaluator on the best config (~20 min per seed)
uv run cafa6-ablation report --sweep feature-groups          # tables and figures -> results/
```

Each run trains with the score channels off, so only the feature groups differ. Seeds are shared
across configs, so every comparison is paired.
- **Every run** is scored by ROC-AUC, on the internal holdout and against the CAFA6 truth.
- **The official evaluator** costs about 20 minutes per run, so it runs only on the config with
  the best internal score. Predictions for every run are kept, so
  `cafa6-ablation cafaeval --sweep feature-groups` can score the rest later. The write-up in [results/README.md](results/README.md)
is produced with the `document-results` Claude Code skill
([.claude/skills/document-results](.claude/skills/document-results/SKILL.md)).

## Repository layout

```text
src/feature_engineering/   feature blocks: embeddings, handcrafted, taxonomy, promoters/
src/modeling/              data assembly, NN trainer and predictor, XGBoost baseline,
                           GO evaluation, official scoring, ablation runner and report
scripts/                   data download and rebuild (scripts/data/), repo utilities
tests/                     unit tests (uv run pytest)
docs/                      data sources, feature and model details
results/                   committed ablation tables, figures and write-up
notebooks/                 ESM-2 vs handcrafted feature EDA
```

## Status and future work

- **Dropped:** homolog-transferred expression features (OrthoDB plus expression atlases). They
  needed a manual upload step and had thin coverage outside model organisms.
- **Next:**
  - ensemble the network with the per-term XGBoost baseline;
  - label-embedding / GO-graph output layers;
  - larger ESM-2 checkpoints.
