# CAFA6 Project Plan

## Competition Overview

**Task:** Predict Gene Ontology (GO) term annotations for protein sequences — a large-scale, multi-label, hierarchical classification problem.

**Data structure** (same layout as CAFA5):
- `train_sequences.fasta` — protein amino-acid sequences
- `train_terms.tsv` — protein ID → GO term labels (multi-label)
- `train_taxonomy.tsv` — species/taxon ID per protein
- `go-basic.obo` — the GO ontology graph (DAG structure between terms)
- `testsuperset.fasta` + `testsuperset-taxon-list.tsv` — test proteins + species
- `IA.tsv` — information-accretion weights per GO term, used for scoring

**Species composition:** training spans 3000+ species (broad, long-tailed: human, mouse, yeast, plants, bacteria, etc.), while the test superset concentrates on a much smaller set (~90 species last cycle) of well-studied organisms. Taxonomy is a strong, cheap signal since some GO terms are species-specific.

**Compute constraints:** local M4 Mac, no dedicated GPU — plan assumes CPU/MPS-only, avoiding anything that requires CUDA or heavy structure prediction (e.g., running AlphaFold or full 3D modeling ourselves).

---

## Feature Engineering Pipeline

Full implementation: `cafa6_feature_pipeline.py`. Six feature blocks, merged into one table keyed by `protein_id`.

### 1. PLM Embeddings
- ESM2 (default: 35M-param checkpoint, mean-pooled), batched, MPS-aware
- Slowest step in the pipeline but still tractable locally; larger checkpoints (150M/650M) are a fallback only if the small model is confirmed to be the bottleneck

### 2. Biophysical / Physicochemical Properties
Via Biopython ProtParam + a few hand-rolled additions:
- Molecular weight, length, isoelectric point (pI), GRAVY (hydrophobicity), aromaticity, instability index, aliphatic index
- Net charge at physiological pH, charge density
- Fraction cysteine (redox/disulfide proxy), fraction charged, fraction polar
- Secondary structure fraction (helix/turn/sheet)

### 3. Amino Acid Composition
- 20-dim AA frequency vector (always on)
- Optional 400-dim dipeptide composition (off by default — turn on once the baseline pipeline is validated)

### 4. Topology Heuristics
Dependency-free approximations (swap for TMHMM/DeepTMHMM/IUPred2A later if useful):
- **Transmembrane helices**: Kyte-Doolittle sliding-window hydropathy, count of segments above threshold
- **Signal peptide**: N-terminal hydrophobicity heuristic
- **Disorder**: Uversky-style charge-hydropathy heuristic
- **Coiled-coil**: heptad-repeat periodicity via autocorrelation at lag 7

### 5. Taxonomy Encoding
- One-hot the top-N most common taxa, bucket the long tail as "other" — keeps dimensionality tractable while preserving the strongest signal

### 6. Homology-Transferred Expression
Converts a normally sparse feature (direct mRNA expression, only available for a handful of model organisms) into a near-universal one via orthology:

- **Ortholog mapping**: precomputed via eggNOG-mapper / OrthoDB / Ensembl Compara — map each protein to its best-match ortholog among a set of "anchor" species with rich expression atlases (human, mouse, rat, zebrafish, fly, yeast, Arabidopsis)
- **Expression atlases used**: GTEx (human, ~50 tissues), Human Protein Atlas (mRNA + IHC + subcellular localization — the localization calls are especially valuable for GO cellular-component terms), Expression Atlas (EBI, broader species), Tabula Sapiens/Muris (single-cell resolution), FANTOM5 CAGE
- **Degrades gracefully**: proteins with no confident ortholog match get NaN + `has_ortholog_expression = 0` rather than an imputed guess; ortholog identity/confidence is kept as a feature so the model can discount distant transfers (expression conservation degrades faster than sequence divergence, especially past ~100-200M years)
- **Derived features**: mean/max expression, tissue-specificity index (tau), anchor species (categorical), raw per-tissue vector

### Considered but deprioritized
- **AlphaGenome**: DNA-level regulatory model (expression, splicing, chromatin from sequence) — mismatched to CAFA6's protein-sequence task by default, but could supply auxiliary regulatory-context features for human/mouse proteins via the free API (no local inference — needs H100-class GPU). Lower priority than homology-expression given narrower coverage and higher engineering cost.

---

## Model Architecture Plan

Ordered by effort-to-payoff:

### Baseline: Gradient-boosted trees
- LightGBM/XGBoost, restricted to the ~3,000-5,000 most frequent GO terms (one-vs-rest or grouped)
- Handles missing values natively (important for the sparse homology-expression columns)
- Fast on CPU, strong tabular-data performance, past CAFA top solutions have used GPU-boosted variants (Py-Boost) as a core component

### Middle ground: Small MLP
- All features concatenated → 2-3 hidden layers → sigmoid multi-label output over top-K terms
- Lets embeddings, biophysical, and expression features interact rather than living in separate models
- Cheap to train with MPS, easy to add dropout/label smoothing for incomplete annotations

### Ontology-aware refinement
- **Post-hoc propagation** (cheap, do this regardless of base model): propagate predicted probabilities up the GO DAG (max/noisy-OR over children) — directly helps the CAFA metric, which is itself weighted by information accretion along the ontology
- **GNN over the GO graph** (stretch goal): model term-term relationships explicitly, as in CAFA5's top solutions — meaningfully more score, meaningfully more engineering

### Label-embedding / dual-tower approach (stretch goal)
- Embed GO term names/definitions with a small LLM, embed proteins via the same pipeline, score compatibility via similarity instead of training 30k+ separate output heads
- Generalizes better to rare terms since semantically related terms share embedding structure

### Final step: Ensemble
- Average predictions across model types (boosted trees + MLP + PLM-only baseline) — consistently what separates top CAFA leaderboard entries, since different architectures err differently on rare vs. common terms

---

## Recommended Build Order

1. Feature pipeline (blocks 1-5) → validated on train set
2. LightGBM baseline on top ~3-5k GO terms
3. Ontology propagation post-processing (cheap, real lift)
4. Small MLP variant, averaged with LightGBM
5. *(Stretch)* GNN or dual-tower label-embedding approach, once baseline error patterns are understood