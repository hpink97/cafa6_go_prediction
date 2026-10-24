---
name: document-results
description: Document a finished CAFA6 experiment sweep (e.g. the feature-group ablation) as committed results - regenerate tables and figures with `cafa6-ablation report`, write the analysis in results/README.md, and refresh the README headline. Use when the user asks to write up, summarise, document or publish experiment/ablation results, or after a sweep finishes.
---

# Document experiment results

Turns a finished sweep under `data/ablation/<sweep>/` into the committed `results/` directory:

- **`results/README.md`:** the analysis.
- **`results/summary/`:** cross-run CSVs and figures.
- **`results/<config>/`:** one directory per configuration, with CSVs and plots.

Numbers in prose and tables must come from the generated CSVs, never from memory or logs.

## 1. Check the sweep is complete

```bash
ls data/ablation/<sweep>/*/seed*/official_auc.json | wc -l    # expect configs x seeds
tail -20 data/ablation/<sweep>.log
```

Every config × seed pair needs `model/summary.json` and `official_auc.json`.

The official CAFA evaluator (`cafaeval/{NK,LK,PK}/`) is opt-in because it takes about 20 minutes
per run. At minimum, run it for the best config (chosen on the internal holdout):

```bash
uv run cafa6-ablation cafaeval --sweep <sweep> --best
```

Its scores land in `summary/cafaeval.csv`. Say in the write-up which runs it covers.
- If any are missing, rerun `uv run cafa6-ablation run --sweep <sweep>`. It is resumable and
  skips finished stages.
- If a run keeps failing, read its `logs/*.log` and report the failure to the user instead of
  documenting a partial sweep as if it were complete.

## 2. Regenerate tables and figures

```bash
uv run cafa6-ablation report --sweep <sweep> --out results
```

This rewrites everything under `results/summary/` and `results/<config>/`. Then open each PNG
and look at it: check for label collisions, empty panels, and a sensible colour scale on the
heatmap.

## 3. Write `results/README.md`

Start from [template.md](template.md) and fill in every section from the CSVs.

**Tables.** Load the CSV and format it, e.g. with
`uv run --with tabulate python -c "import pandas as pd; print(pd.read_csv(...).to_markdown(floatfmt='.3f'))"`.
Never type numbers by hand.
- **Main table:** one row per config, mean ± sd over seeds for each headline metric, from
  `summary/summary.csv`.
- **Paired deltas:** from `summary/deltas.csv`. Show `delta_mean ± delta_sd` and
  `n_seeds_improved / n_seeds`.

**Findings.** State each effect as paired change ± sd plus how many seeds agree on the sign.
- Call an effect **consistent** only when every seed agrees and |mean| > sd.
- Otherwise call it **inconclusive**. Don't say "significant" with three seeds.

**Do promoters help?** This section is required. Answer it from:
- the **+ all promoters (with ESM-2)** and **+ all promoters (no ESM-2)** deltas, which separate
  "redundant with ESM-2" from "no signal";
- `summary/promoter_strata.csv`, which shows whether gains concentrate in plants, or in proteins
  with a directly linked promoter (`xref`) versus rescued or imputed ones.

**Which GO terms do promoters help?** This section is also required. Answer it from
`summary/response_terms.csv` and `summary/response_terms.png`.
- The hypothesis is that promoters help most for "response to …" processes, which are defined by
  inducible expression.
- Report:
  - the mean per-term AUC change for response terms vs other process terms, under both
    definitions (`by_name`, `by_ontology`);
  - the prevalence-stratified permutation p-value (the primary test; Mann-Whitney is secondary);
  - how many seeds agree.
- Report the with-ESM-2 and without-ESM-2 comparisons separately. A difference that appears
  only without ESM-2 means the signal is real but redundant with the protein embedding.
- Name a few of the most-helped response terms from `summary/response_term_deltas.csv`.

**Internal vs official.** Say where the internal 20% holdout and the official CAFA evaluation
disagree, and which metric each conclusion rests on. `f_w` and `f_micro_w` can rank models
differently.

**Caveats.** Always include:
- **Model capacity:** it changes with the feature groups (`n_parameters`).
- **Split:** the internal split is random by protein. Homologous proteins can straddle it, which
  inflates absolute internal scores but not paired deltas.
- **Official numbers:** there is one release, with NK/LK/PK sizes from the ground truth.
- **Score channels:** these were off. Production models blend in a taxon prior and ESM-2 KNN,
  which may absorb some feature-group gains.

Record the git revision from `results/<config>/config.json`.

## 4. Update the top-level README

Replace only the block between `<!-- results:start -->` and `<!-- results:end -->` with:
- a 2–3 line headline;
- a compact table (config × internal and official macro ROC-AUC, mean ± sd), plus the official
  CAFA scores (`f_w`, `f_micro_w`) of the configs `cafaeval` covered;
- a link to `results/README.md`.

Leave the rest of the README alone.

## 5. Check and hand over

```bash
scripts/check_commit_safe.sh results README.md
```

All files must be under 1 MB with nothing from `data/`. If `per_term_auc.csv` or a PNG is too
large, lower the DPI or round harder rather than skipping the file.

**Do not commit or push.** The user writes their own commits and PR messages. Hand over:
- the branch name to use;
- the exact `git add` paths;
- a short factual summary of the findings they can draw on.
