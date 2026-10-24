# Data

Nothing under `data/` is committed. The competition data cannot be redistributed under
Kaggle's terms, and the derived features, references and models total about 40 GB. Every
file can be rebuilt with the scripts in `scripts/data/`, run from the repository root.

```bash
scripts/data/03_tools.sh          # cafaeval, kaggle, diamond
scripts/data/01_competition.sh    # Kaggle competition data -> data/Train, data/Test, data/IA.tsv
scripts/data/02_ground_truth.sh   # final CAFA6 ground truth (Figshare), checksum-verified
scripts/data/04_features.sh all   # every feature block -> data/features/
uv run cafa6-repair-metadata      # only if embedding metadata.json files were lost
```

## Sources

| Data | Source | Fetched by | Size |
|---|---|---|---|
| Competition data (sequences, train terms, taxonomy, GO, IA) | [Kaggle `cafa-6-protein-function-prediction`](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction) | `01_competition.sh` | 96 MB zip |
| Final ground truth (NK/LK/PK, terms of interest, IA, GO) | [CAFA6/Kaggle Evaluation Data](https://doi.org/10.6084/m9.figshare.32861969), Figshare, CC BY 4.0 | `02_ground_truth.sh` (SHA-256 pinned) | 135 MB |
| NCBI taxonomy | `https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz` | automatically on first use | 60 MB |
| ESM-2 weights (35M / 150M) | `fair-esm` (`dl.fbaipublicfiles.com`) | automatically on first use | 0.1 / 0.6 GB |
| Ensembl / Ensembl Genomes genomes and GFF3 | `https://ftp.ensembl.org/pub/current_*`, `https://ftp.ensemblgenomes.ebi.ac.uk/pub` | `cafa6-build-and-extract-promoters` | GBs, deleted per species |
| Ensembl REST | `https://rest.ensembl.org` | promoter extraction fallback | — |
| UniProt ID mapping | `https://rest.uniprot.org/idmapping` | reference builder | — |
| NCBI Datasets assemblies | `https://api.ncbi.nlm.nih.gov/datasets/v2` | `cafa6-rescue-promoters` | GBs, cached proteomes |
| Nucleotide Transformer v2 100M / AgroNT 1B | Hugging Face `InstaDeepAI/*` | automatically on first use | 0.4 / 7.4 GB |
| CAFA evaluator | `github.com/claradepaolis/CAFA-evaluator-PK` | `03_tools.sh` | — |

## Layout

```text
data/
├── Train/                        train_sequences.fasta, train_terms.tsv, train_taxonomy.tsv, go-basic.obo
├── Test/                         testsuperset.fasta, testsuperset-taxon-list.tsv, test_taxonomy.tsv (derived)
├── IA.tsv                        information accretion weights
├── evaluation/
│   └── release_kaggle_final_May2026/   NK/LK/PK ground truth for official scoring
├── external/                     downloaded references: taxdump, proteomes, DIAMOND panel
├── features/
│   ├── sequence/                 {train,test}_features.tsv
│   ├── taxonomy/                 train_taxonomy_features.tsv, kingdom_map.json
│   ├── promoters/{train,test}/   per-species promoter TSVs, combined.tsv, alignment_rescue.tsv
│   └── embeddings/               {split}_esm2_150M, {split}_promoter_nt100m, {split}_promoter_agront1b
├── models/                       training runs
└── ablation/                     feature-group sweep runs (results are summarised in results/)
```

Embedding directories hold a row-major `float32` memmap (`embeddings.f32.dat`), row order in
`ids.tsv`, a completion bitmap (`done.bool.dat`) and `metadata.json` with the shape.

## Build order

```text
01_competition ──► sequence features ─────────────────────────────┐
              ├──► ESM-2 150M embeddings ──┬──────────────────────┤
              ├──► taxonomy (kingdoms) ────┼──────────────────────┤
              └──► promoter extraction ──► DIAMOND rescue ──► NT / AgroNT embeddings
                                           (uses ESM-2)           │
                                                                  ▼
                                    cafa6-train-multilabel-nn (imputes missing promoters)
                                                                  ▼
                       cafa6-predict-multilabel-nn ──► cafa6-evaluate (needs 02_ground_truth)
```

The imputed promoter cache and the inference artifacts (`inference/*.npz`) are produced by
the trainer itself, so no separate step builds them.
