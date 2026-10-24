#!/usr/bin/env bash
# Download and unpack the CAFA6 Kaggle competition data into data/.
#
# Needs the Kaggle CLI (`uv tool install kaggle`) with an API token in ~/.kaggle/
# and the competition rules accepted on
# https://www.kaggle.com/competitions/cafa-6-protein-function-prediction.
# The data stays local: Kaggle's terms do not allow redistributing it.
#
# Produces:
#   data/Train/{train_sequences.fasta,train_terms.tsv,train_taxonomy.tsv,go-basic.obo}
#   data/Test/{testsuperset.fasta,testsuperset-taxon-list.tsv}
#   data/IA.tsv, data/sample_submission.tsv
#   data/Test/test_taxonomy.tsv   (derived: protein_id<TAB>taxon_id from the FASTA headers)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

COMPETITION=cafa-6-protein-function-prediction
ZIP="data/$COMPETITION.zip"
mkdir -p data

if [ ! -f "$ZIP" ]; then
  kaggle competitions download -c "$COMPETITION" -p data
fi

# -n: never overwrite files that are already unpacked.
unzip -n -q "$ZIP" -d data

if [ ! -s data/Test/test_taxonomy.tsv ]; then
  awk '/^>/ {sub(/^>/, ""); print $1 "\t" $2}' data/Test/testsuperset.fasta > data/Test/test_taxonomy.tsv
fi

echo "Competition data ready:"
wc -l data/Train/train_terms.tsv data/Train/train_taxonomy.tsv data/Test/test_taxonomy.tsv
