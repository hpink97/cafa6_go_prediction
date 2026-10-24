# Feature blocks

Every block is a separate table or embedding directory keyed by `protein_id`. Blocks stay
separate until training, so each one can be rebuilt, inspected or ablated on its own.
`scripts/data/04_features.sh` builds them all; the commands below show each step.

| Block | Columns | Built by | Output |
|---|---|---|---|
| ESM-2 150M embedding | 640 | `cafa6-embeddings` | `embeddings/{split}_esm2_150M/` |
| Handcrafted protein features | 56 | `cafa6-features` | `sequence/{split}_features.tsv` |
| Kingdom | 7 | `cafa6-taxonomy-features` | `taxonomy/train_taxonomy_features.tsv` |
| Promoter, Nucleotide Transformer v2 100M | 512 | `cafa6-embed-promoters` | `embeddings/{split}_promoter_nt100m/` |
| Promoter, AgroNT 1B (plants only) | 1500 | `cafa6-embed-promoters` | `embeddings/{split}_promoter_agront1b/` |

## ESM-2 embeddings

Mean-pooled last-layer ESM-2 representations. Proteins longer than 1,022 residues are split
into chunks overlapping by 128 residues, and the chunk means are averaged. Runs are resumable
and use MPS on Apple Silicon automatically.

```bash
uv run cafa6-embeddings data/Train/train_sequences.fasta data/features/embeddings/train_esm2_150M --model-size 150M
uv run cafa6-embeddings ... --resume          # continue an interrupted run (same input and settings)
uv run cafa6-embeddings ... --limit 100       # smoke test
```

The 35M model (480-d, the default) is faster and is used by the XGBoost baseline. The neural
network uses 150M.

## Handcrafted protein features

`cafa6-features <fasta> <output.tsv>` writes 56 columns per protein:

- **Biophysical** (Biopython ProtParam): length, molecular weight, isoelectric point, GRAVY,
  aromaticity, instability and aliphatic indices, net charge, charge density, and cysteine,
  charged, polar and secondary-structure fractions.
- **Topology heuristics**: hydropathy-window transmembrane helix count, signal-peptide score and
  flag, charge/hydropathy disorder score and flag, and coiled-coil heptad periodicity.
- **Terminal motifs**: KDEL-like, PTS1-like, CAAX-like and N-myristoylation-like.
- **PROSITE-style patterns**: kinase ATP-binding, Ser/Thr kinase active site, ABC transporter,
  leucine zipper, P-loop, zinc metallopeptidase, cytochrome c heme, 4Fe-4S ferredoxin,
  EGF-like, thioredoxin active site, and bipartite NLS.
- **Composition**: 20 amino-acid fractions.

`notebooks/embedding_vs_manual_features.ipynb` measures how much of this ESM-2 already encodes.

## Taxonomy

`cafa6-taxonomy-features` maps each taxon to one of seven kingdoms (bacteria, archaea, metazoa,
fungi, plants, other eukaryotes, unknown) by walking NCBI `nodes.dmp`. The taxdump is
downloaded once into `data/external/ncbi_taxdump/`. The pipeline uses kingdoms only
(`--skip-taxon-vocabulary`); a top-N taxon one-hot is available with `--top-n`. At prediction
time, test kingdoms are encoded on the fly.

## Promoters

The 3 kb window upstream of each protein's gene start, embedded with DNA language models. It is
built in three steps.

**1. Extraction from reference genomes.** `species_map.py` lists 55 anchor species and 9
operon-aware bacteria. For bacteria, genes inside an operon (gap ≤ 50 bp) take the operon
leader's promoter. `cafa6-build-and-extract-promoters` handles one species at a time:

- builds a temporary reference (Ensembl or Ensembl Genomes GFF3 plus genome FASTA, with
  UniProt ID mapping from protein to gene);
- slices the promoters locally;
- deletes the genome and moves on.

Finished species are skipped on rerun.

```bash
uv run cafa6-build-and-extract-promoters data/Test/testsuperset.fasta data/features/promoters/test \
    --reference-work-dir data/external/promoter_reference_tmp
```

`cafa6-extract-promoters --local-reference-dir DIR` slices from a persistent local reference
(`{species}.protein_gene.tsv`, `{species}.genes.tsv`, `{species}.fa`). It falls back to Ensembl
REST unless `--offline-local-only` is set.

**2. Alignment rescue.** `cafa6-rescue-promoters` aligns proteins still missing a promoter with
DIAMOND. The panel is longest-isoform proteomes from Ensembl plus NCBI Datasets assemblies for
any taxon with at least 20 missing proteins. Hits are labelled by how close they are:

| Label | Rule |
|---|---|
| `alignment_exact` | same taxon, ≥ 95% identity, ≥ 80% coverage |
| `alignment_proxy` | best same-phylum hit, ≥ 40% identity, ≥ 60% coverage; another organism's promoter, not the protein's own |

Existing promoters are never overwritten. Use `--max-queries`, `--ncbi-max-taxa` or
`--ncbi-taxa` for pilot runs.

Resulting coverage:

| Split | Ensembl xref | Exact | Proxy | None | Covered |
|---|---|---|---|---|---|
| Train (82,404) | 64,596 | 11,880 | 3,444 | 2,484 | 97.0% |
| Test superset (224,309) | 110,320 | 44,536 | 43,446 | 26,007 | 88.4% |

**3. Embedding.** `cafa6-embed-promoters` mean-pools Nucleotide Transformer v2 100M (multi-species)
over every promoter. AgroNT 1B, a plant genome model, runs only on plant proteins
(`--embed-kingdoms eukaryota_viridiplantae`); other rows are zero. Proteins with no promoter
are imputed at training time (see [modeling.md](modeling.md)).

## Dropped: homolog-transferred expression

An OrthoDB-mapping plus expression-atlas feature block was prototyped. It never reached
training and has been removed: it needed a manual OrthoDB upload per protein set, and atlas
coverage outside model organisms was thin.
