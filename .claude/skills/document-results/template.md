# <Sweep title>: which feature groups help CAFA6 GO prediction?

<!-- One paragraph: the answer first. Which groups help, by how much, and do promoters help. -->

## Setup

- **Model:** multi-label NN, top 1,000 GO terms plus ancestors (2,180 outputs), score channels off.
- **Configs:** <list each config and its feature groups, from results/<config>/config.json>
- **Seeds:** <seeds>. The same seed gives every config the same train/validation/test split, so
  comparisons are paired.
- **Internal evaluation:** the held-out 20% of labelled proteins.
- **Official evaluation:** CAFA6 final release (NK / LK / PK), scored with CAFA-evaluator-PK.
- **Code:** git revision `<revision>`. Reproduce with `uv run cafa6-ablation run --sweep <sweep>`.

### Metrics

| Metric | Where | Meaning |
|---|---|---|
| Macro ROC-AUC | internal, official | mean per-term AUC over terms with positives and negatives |
| Micro ROC-AUC | internal, official | AUC over all pooled protein-term pairs |
| IA-weighted Fmax | internal | protein-centric, information-accretion-weighted Fmax, mean of BP/MF/CC |
| f_w | official | CAFA weighted Fmax, mean over aspects, then over NK/LK/PK |
| f_micro_w | official | CAFA weighted micro-averaged F |
| s_w | official | weighted S-min (lower is better) |

## Results

<!-- Main table from summary/summary.csv: config x headline metrics, mean ± sd. -->

![Metric by configuration](summary/metric_by_config.png)

### Paired effect of each feature group

<!-- Table from summary/deltas.csv: comparison x key metrics, delta ± sd (seeds improved / seeds). -->

![Paired deltas](summary/delta_heatmap.png)

## Do promoters help?

<!-- With ESM-2 vs without ESM-2 deltas; strata by provenance and kingdom; plain-language verdict. -->

![Promoter effect by stratum](summary/promoter_strata.png)

### Are promoters most useful for "response to" terms?

<!-- Hypothesis, test (per-term AUC change, prevalence-stratified permutation), both definitions,
with vs without ESM-2, verdict, a few example terms. -->

![Per-term promoter effect, response vs other process terms](summary/response_terms.png)

## Internal vs official evaluation

<!-- Where the two disagree, and which metric each conclusion depends on. -->

## Caveats

<!-- Capacity confound, random split, single release, score channels off. -->

## Per-configuration details

<!-- Link each results/<config>/ directory: metrics.csv, per_term_auc.csv, training_curves.png, pr_curves.png. -->
