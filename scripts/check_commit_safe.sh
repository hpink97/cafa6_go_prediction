#!/usr/bin/env bash
# Guard for this public repo: refuse to commit data, large files or secrets.
#
# Usage:
#   scripts/check_commit_safe.sh              # check staged files
#   scripts/check_commit_safe.sh PATH...      # check the given files/directories
#   ln -s ../../scripts/check_commit_safe.sh .git/hooks/pre-commit   # run on every commit
set -euo pipefail

MAX_BYTES=$((1024 * 1024))
cd "$(git rev-parse --show-toplevel)"

if [ "$#" -gt 0 ]; then
  files=$(find "$@" -type f -not -path '*/__pycache__/*' -not -name '.DS_Store' | sort)
else
  files=$(git diff --cached --name-only --diff-filter=ACMR)
fi

status=0
fail() { echo "BLOCKED: $1" >&2; status=1; }

while IFS= read -r file; do
  [ -z "$file" ] && continue
  [ -f "$file" ] || continue
  case "$file" in
    data/*) fail "$file is under data/ (rebuild with scripts/data/, never commit)" ;;
    *.fasta|*.fa|*.fa.gz|*.dmnd|*.zip|*.npz|*.pt|*.dat|*.parquet) fail "$file is a data/model artifact" ;;
    *train_terms.tsv|*train_taxonomy.tsv|*testsuperset*|*IA.tsv|*sample_submission.tsv|*groundtruth*) fail "$file looks like competition data" ;;
  esac
  size=$(wc -c < "$file" | tr -d ' ')
  if [ "$size" -gt "$MAX_BYTES" ]; then
    fail "$file is $((size / 1024)) KB (limit $((MAX_BYTES / 1024)) KB)"
  fi
  if grep -IEn '(AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"key"[[:space:]]*:[[:space:]]*"[0-9a-f]{32}")' "$file" >/dev/null 2>&1; then
    fail "$file contains something that looks like a secret"
  fi
done <<< "$files"

count=$(printf '%s\n' "$files" | grep -c . || true)
if [ "$status" -eq 0 ]; then
  echo "OK: $count file(s) checked"
fi
exit "$status"
