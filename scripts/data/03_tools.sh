#!/usr/bin/env bash
# Install the external tools the pipeline shells out to.
#   cafaeval  official CAFA evaluator (No/Limited/Partial-knowledge regimes), used by cafa6-evaluate
#   diamond   protein aligner, used by cafa6-rescue-promoters
#   kaggle    competition download, used by 01_competition.sh
set -euo pipefail

uv tool install --upgrade git+https://github.com/claradepaolis/CAFA-evaluator-PK
uv tool install --upgrade kaggle

if ! command -v diamond >/dev/null; then
  if command -v brew >/dev/null; then
    brew install diamond
  else
    echo "Install DIAMOND (https://github.com/bbuchfink/diamond), e.g. conda install -c bioconda diamond" >&2
    exit 1
  fi
fi

cafaeval --help >/dev/null && diamond version
