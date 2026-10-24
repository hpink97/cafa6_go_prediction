#!/usr/bin/env bash
# Rebuild every feature block under data/features/ from the competition data.
#
# Usage: scripts/data/04_features.sh [sequence|esm|taxonomy|promoters|promoter-embeddings|all]...
#
# Stages, in dependency order (each one is resumable or skips finished outputs):
#   sequence             56 handcrafted protein features        minutes
#   esm                  ESM-2 150M embeddings (640-d)          hours on Apple Silicon / GPU
#   taxonomy             kingdom one-hot + kingdom map          minutes (downloads NCBI taxdump once)
#   promoters            3 kb promoters: Ensembl references, then DIAMOND rescue
#                        (needs esm first; downloads genomes per species and can take a day)
#   promoter-embeddings  NT-v2-100M (all) and AgroNT-1B (plants only)   hours; AgroNT needs ~8 GB of weights
#
# Run scripts/data/01_competition.sh and 03_tools.sh first.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

FEATURES=data/features
EMBEDDINGS=$FEATURES/embeddings

resume_flag() { [ -f "$1/metadata.json" ] && echo --resume || true; }

stage_sequence() {
  mkdir -p "$FEATURES/sequence"
  [ -s "$FEATURES/sequence/train_features.tsv" ] || uv run cafa6-features data/Train/train_sequences.fasta "$FEATURES/sequence/train_features.tsv"
  [ -s "$FEATURES/sequence/test_features.tsv" ] || uv run cafa6-features data/Test/testsuperset.fasta "$FEATURES/sequence/test_features.tsv"
}

stage_esm() {
  uv run cafa6-embeddings data/Train/train_sequences.fasta "$EMBEDDINGS/train_esm2_150M" --model-size 150M $(resume_flag "$EMBEDDINGS/train_esm2_150M")
  uv run cafa6-embeddings data/Test/testsuperset.fasta "$EMBEDDINGS/test_esm2_150M" --model-size 150M $(resume_flag "$EMBEDDINGS/test_esm2_150M")
}

stage_taxonomy() {
  mkdir -p "$FEATURES/taxonomy"
  # Test kingdoms are encoded on the fly at prediction time from data/Test/test_taxonomy.tsv.
  uv run cafa6-taxonomy-features data/Train/train_taxonomy.tsv "$FEATURES/taxonomy/train_taxonomy_features.tsv" \
    --skip-taxon-vocabulary --kingdom-map-out "$FEATURES/taxonomy/kingdom_map.json"
}

stage_promoters() {
  # Train taxa come from the taxonomy TSV (the rescue finds train_sequences.fasta next to it);
  # test headers already carry the taxon ID.
  local split input
  for split in train test; do
    if [ "$split" = train ]; then input=data/Train/train_taxonomy.tsv; else input=data/Test/testsuperset.fasta; fi
    uv run cafa6-build-and-extract-promoters "$input" "$FEATURES/promoters/$split" \
      --reference-work-dir data/external/promoter_reference_tmp
    uv run cafa6-rescue-promoters "$input" "$FEATURES/promoters/$split" \
      --embedding-dir "$EMBEDDINGS/${split}_esm2_150M"
  done
}

stage_promoter_embeddings() {
  local split
  for split in train test; do
    uv run cafa6-embed-promoters "$FEATURES/promoters/$split/combined.tsv" "$EMBEDDINGS/${split}_promoter_nt100m" \
      $(resume_flag "$EMBEDDINGS/${split}_promoter_nt100m")
    uv run cafa6-embed-promoters "$FEATURES/promoters/$split/combined.tsv" "$EMBEDDINGS/${split}_promoter_agront1b" \
      --model-name InstaDeepAI/agro-nucleotide-transformer-1b --embed-kingdoms eukaryota_viridiplantae \
      $(resume_flag "$EMBEDDINGS/${split}_promoter_agront1b")
  done
}

[ "$#" -gt 0 ] || set -- all
for stage in "$@"; do
  case "$stage" in
    sequence) stage_sequence ;;
    esm) stage_esm ;;
    taxonomy) stage_taxonomy ;;
    promoters) stage_promoters ;;
    promoter-embeddings) stage_promoter_embeddings ;;
    all) stage_sequence; stage_esm; stage_taxonomy; stage_promoters; stage_promoter_embeddings ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
done
