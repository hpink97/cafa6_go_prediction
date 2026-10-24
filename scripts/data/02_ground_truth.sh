#!/usr/bin/env bash
# Download and verify the CAFA6 final evaluation release used for official (CAFA-native) scoring:
# No/Limited/Partial-knowledge ground truth, PK known annotations, terms of interest, IA weights,
# the GO snapshot and the evaluated sequences.
#
# Source: "CAFA6/Kaggle Evaluation Data", Figshare, doi:10.6084/m9.figshare.32861969 (CC BY 4.0).
# Checksums pin the exact files the numbers in results/ were scored against.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RELEASE_DIR=data/evaluation/release_kaggle_final_May2026
FIGSHARE=https://ndownloader.figshare.com/files

# sha256  figshare-file-id  name
FILES="\
ca48860e6fbcf0ab65a51a9a275b6d184c3a36dd80cd02b1f193e03fcfd4d00b 66194237 go-basic.obo
e5421386844b537dc400840d467ada7ec21215d79cc2c9d47af67d83ed15535d 66194243 IA.tsv
50635c0c2d080d0a4eab860c2d78ddbee57aaf468bcf6e7c645af3522cfc51d6 66194258 May2026_groundtruth_LK.tsv
97842db61700e89284a2356aed2906e4f9c89c89b2d1f006ef2f74e5f8ff41c1 66194255 May2026_groundtruth_NK.tsv
66d1134abacd9e57befc6645e8ecbfa7b7b502bf710088b1514cb8892c48f4a5 66194240 May2026_groundtruth_PK_known.tsv
b2359f01c8efe7643b0709d2eca1d4da8629ec0956b1c69acbe4920bc5c1c912 66194252 May2026_groundtruth_PK.tsv
c04f3c07bb6a267e3e23faaa95343d4ab81e7a4d9a41b248984b7567fbacff7f 66194246 May2026_groundtruth_terms_of_interest.txt
bbee44126e9078b1bc50ba7476854c5ff28b3c708e7d443a8ad4d49f81882b8b 66194249 test_sequences.fasta"

mkdir -p "$RELEASE_DIR"
while read -r _ file_id name; do
  target="$RELEASE_DIR/$name"
  [ -s "$target" ] && continue
  echo "downloading $name"
  curl -fsSL --retry 3 -o "$target.part" "$FIGSHARE/$file_id"
  mv "$target.part" "$target"
done <<< "$FILES"

awk '{print $1 "  '"$RELEASE_DIR"'/" $3}' <<< "$FILES" | shasum -a 256 -c
echo "Ground truth verified in $RELEASE_DIR"
