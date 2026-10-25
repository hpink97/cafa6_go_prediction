# Which feature groups help CAFA6 GO-term prediction?

**ESM-2 does nearly all the work.** It adds +0.065 official weighted Fmax (p < 0.001) over
handcrafted and kingdom features alone. Nothing else comes close.

**Promoters improve ranking but hurt the official CAFA score.**
- Promoter DNA embeddings raise per-term ranking (macro ROC-AUC) on every seed, internally and
  against the CAFA6 truth.
- On the official CAFA evaluator, adding them *lowers* weighted Fmax. In the protein bootstrap,
  the promoter model (`all`) is significantly worse than every ESM-2 model without promoters,
  by 0.010–0.014 weighted Fmax (p ≤ 0.006).
- **So `all`, the best model by AUC, is not the best model.** By the official metric the best are
  the three ESM-2 models without promoters, which are within seed noise of each other. They
  weren't tested against each other pairwise.

**As hypothesised, promoters help "response to …" processes most.** Without ESM-2 they gain twice
as much as other process terms (p = 0.0001). With ESM-2 the extra benefit is weak. Promoters carry
real expression-linked signal, but the protein language model already captures most of it.

**Internal evaluation would have chosen the wrong model.** On a random 20% holdout, promoters look
neutral to helpful. On the prospective CAFA6 targets they hurt. The follow-up
[architecture search](hparam-search/README.md) shows the same reversal. The best final model is a
wider network on ESM-2 + handcrafted + kingdom features, without promoters: official weighted
Fmax 0.326.

## Setup

- **Model:** the multi-label neural network ([docs/modeling.md](../docs/modeling.md)):
  - outputs the 1,000 most frequent GO terms plus all their ancestors (2,180 outputs);
  - one projection per feature group;
  - score channels (taxon prior, ESM-2 nearest neighbours) are **off**, so only the input
    features differ between configs.
- **Configs:** six, each a set of feature groups ([docs/features.md](../docs/features.md)):

  | Config | Feature groups |
  |---|---|
  | `esm2` | ESM-2 150M embedding (640-d) |
  | `esm2_prot` | + 56 handcrafted sequence features |
  | `esm2_prot_tax` | + 7-way kingdom one-hot |
  | `all` | + promoter embeddings: Nucleotide Transformer v2 (all species) and AgroNT (plants) |
  | `prot_tax` | handcrafted + kingdom, no ESM-2 |
  | `prot_tax_promoters` | handcrafted + kingdom + promoters, no ESM-2 |

- **Seeds:** 0, 1 and 2. A seed fixes the train/validation/test split and the initialisation.
  Every config sees the same split for a given seed, so all comparisons below are **paired**.
- **Internal evaluation:** a random 20% of the 82,404 labelled proteins is held out (16,481
  proteins). Early stopping uses a further validation split.
- **Official evaluation:** each model predicts the 8,023 proteins of the
  [CAFA6 final release](https://doi.org/10.6084/m9.figshare.32861969) under three settings:
  - no knowledge (NK);
  - limited knowledge (LK);
  - partial knowledge (PK), where annotations already known are excluded.

  ROC-AUC against this truth is computed for every run. The official CAFA evaluator, which takes
  about 20 minutes per run, was run for the best config; see [below](#official-cafa-scores).
- **Code:** git `2c147c8`. Reproduce with `uv run cafa6-ablation run --sweep feature-groups`, then
  `uv run cafa6-ablation report --sweep feature-groups`.

### Metrics

| Metric | Where | Meaning |
|---|---|---|
| Macro ROC-AUC | internal, official | mean per-term AUC over terms with both positives and negatives; every term counts equally |
| Micro ROC-AUC | internal, official | one AUC over all pooled protein–term pairs; dominated by common terms |
| IA-weighted Fmax | internal | protein-centric CAFA Fmax weighted by information accretion, mean of BP/MF/CC |
| f_w, f_micro_w, s_w | official | CAFA evaluator weighted Fmax, weighted micro-F, weighted S-min (lower is better), mean over aspects then NK/LK/PK |

## Results

Mean ± sd over 3 seeds. Official values are means over NK, LK and PK.

| Config               | Feature groups                            | Internal macro AUC   | Internal micro AUC   | Internal Fmax   | Official macro AUC   | Official micro AUC   | Parameters   |
|:---------------------|:------------------------------------------|:---------------------|:---------------------|:----------------|:---------------------|:---------------------|:-------------|
| `esm2`               | ESM-2                                     | 0.887 ± 0.002        | 0.945 ± 0.001        | 0.451 ± 0.002   | 0.838 ± 0.003        | 0.939 ± 0.002        | 0.99M        |
| `esm2_prot`          | ESM-2 + handcrafted                       | 0.889 ± 0.002        | 0.946 ± 0.000        | 0.452 ± 0.001   | 0.834 ± 0.004        | 0.938 ± 0.002        | 1.01M        |
| `esm2_prot_tax`      | ESM-2 + handcrafted + kingdom             | 0.898 ± 0.001        | 0.950 ± 0.000        | 0.457 ± 0.002   | 0.844 ± 0.006        | 0.940 ± 0.001        | 1.01M        |
| `all`                | ESM-2 + handcrafted + kingdom + promoters | 0.906 ± 0.003        | 0.952 ± 0.001        | 0.457 ± 0.001   | 0.850 ± 0.002        | 0.941 ± 0.001        | 1.37M        |
| `prot_tax`           | handcrafted + kingdom                     | 0.829 ± 0.006        | 0.927 ± 0.002        | 0.391 ± 0.005   | 0.794 ± 0.007        | 0.920 ± 0.002        | 0.71M        |
| `prot_tax_promoters` | handcrafted + kingdom + promoters         | 0.843 ± 0.001        | 0.928 ± 0.000        | 0.389 ± 0.002   | 0.799 ± 0.005        | 0.918 ± 0.002        | 1.07M        |

![Metric by configuration](summary/metric_by_config.png)

Official macro ROC-AUC by knowledge setting:

| Config               | NK macro AUC   | LK macro AUC   | PK macro AUC   |
|:---------------------|:---------------|:---------------|:---------------|
| `esm2`               | 0.841 ± 0.005  | 0.851 ± 0.007  | 0.822 ± 0.006  |
| `esm2_prot`          | 0.836 ± 0.004  | 0.843 ± 0.010  | 0.823 ± 0.006  |
| `esm2_prot_tax`      | 0.846 ± 0.006  | 0.855 ± 0.008  | 0.832 ± 0.003  |
| `all`                | 0.849 ± 0.003  | 0.865 ± 0.004  | 0.837 ± 0.004  |
| `prot_tax`           | 0.805 ± 0.011  | 0.802 ± 0.003  | 0.775 ± 0.007  |
| `prot_tax_promoters` | 0.812 ± 0.007  | 0.800 ± 0.008  | 0.786 ± 0.004  |

### Paired effect of each feature group

Each cell is the mean paired change from adding a group, then the paired t-test p-value across
seeds and (seeds improved / seeds). The comparisons are defined in
[`summary/deltas.csv`](summary/deltas.csv), which also holds the sd of each change. Official f_w
and f_micro_w come from the CAFA evaluator and average NK, LK and PK.

With 3 seeds the t-test has 2 degrees of freedom. p < 0.05 needs a large, very consistent effect,
and about 30 tests are shown uncorrected, so isolated p ≈ 0.02–0.05 values are weak evidence. An
effect is called **significant** here only when p < 0.05 and every seed agrees.

| Adding                       | Internal macro AUC    | Internal Fmax         | Official macro AUC    | Official f_w          | Official f_micro_w    |
|:-----------------------------|:----------------------|:----------------------|:----------------------|:----------------------|:----------------------|
| + handcrafted                | +0.0017 (p=0.15, 3/3) | +0.0017 (p=0.08, 3/3) | -0.0041 (p=0.05, 0/3) | +0.0002 (p=0.96, 2/3) | -0.0021 (p=0.42, 1/3) |
| + kingdom                    | +0.0099 (p=0.02, 3/3) | +0.0052 (p=0.11, 3/3) | +0.0105 (p=0.04, 3/3) | +0.0020 (p=0.65, 2/3) | -0.0006 (p=0.75, 1/3) |
| + handcrafted & kingdom      | +0.0116 (p=0.02, 3/3) | +0.0069 (p=0.09, 3/3) | +0.0064 (p=0.09, 3/3) | +0.0023 (p=0.54, 1/3) | -0.0027 (p=0.15, 0/3) |
| + all promoters (with ESM-2) | +0.0077 (p=0.03, 3/3) | -0.0007 (p=0.71, 1/3) | +0.0058 (p=0.16, 3/3) | -0.0080 (p=0.10, 0/3) | -0.0109 (p=0.06, 0/3) |
| + all promoters (no ESM-2)   | +0.0139 (p=0.04, 3/3) | -0.0017 (p=0.46, 1/3) | +0.0054 (p=0.23, 2/3) | -0.0074 (p=0.06, 0/3) | -0.0210 (p=0.02, 0/3) |
| + ESM-2                      | +0.0689 (p=0.00, 3/3) | +0.0666 (p=0.00, 3/3) | +0.0503 (p=0.01, 3/3) | +0.0648 (p=0.00, 3/3) | +0.0531 (p=0.00, 3/3) |

![Paired deltas](summary/delta_heatmap.png)

- **ESM-2 is the foundation.** It is significant on every metric and an order of magnitude larger
  than any other effect.
- **Kingdom features help ranking but not the CAFA score.** They significantly raise macro AUC,
  internally and officially, but leave official f_w and f_micro_w unchanged.
- **Handcrafted features don't help on the official truth.** They give a small internal gain, but
  official macro AUC falls (−0.004, 0/3 seeds) and the CAFA metrics don't move. Much of what they
  encode (hydrophobicity, charge, disorder) is predictable from ESM-2 embeddings (see
  `notebooks/embedding_vs_manual_features.ipynb`).
- **Promoters lower the official CAFA metrics on every seed**, with and without ESM-2: 0/3 seeds
  improve f_w or f_micro_w. Seed-level p ranges from 0.02 to 0.10; the
  [protein bootstrap](#is-all-the-best-model) is clearer.

## Do promoters help?

**For per-term ranking, yes. For the official CAFA score, no: they make it worse.**

- **With ESM-2,** adding promoter embeddings consistently raises macro ROC-AUC:
  - internal: +0.0077 ± 0.0023, 3/3 seeds;
  - official: +0.0058 ± 0.0046, 3/3 seeds, with the biggest gain in the limited-knowledge
    setting.

  Internal Fmax and micro AUC are unchanged (inconclusive).
- **Without ESM-2,** the internal macro gain is larger (+0.0139), so promoters carry real signal
  of their own. Part of it overlaps with what ESM-2 provides.
- **Reading:** macro AUC weights every term equally, while micro AUC and Fmax are dominated by
  common, high-information-content annotations. So promoters help rank the long tail of
  specific terms, and the protein-level metrics barely register it.
- **On the official CAFA evaluator they cost 0.008–0.014 weighted Fmax** (see
  [below](#is-all-the-best-model)).

  A likely cause is distribution shift in the promoter features:
  - training proteins: 3% have no promoter and need imputation, and 4% borrow one from a
    related species;
  - test superset: 12% and 19% respectively.

  The network learns to rely on a feature that is noisier for exactly the proteins CAFA
  evaluates.
- **Promoter source doesn't explain much.** Splitting test proteins by how their promoter was
  obtained or by kingdom, the Fmax change is within noise for every group. The exception is
  proteins with no promoter at all, whose imputed vector slightly hurts on average (−0.013,
  inconclusive). The plant-only AgroNT model brings no visible extra gain for plants.

| Test proteins     |   Proteins | Δ Fmax (with − without promoters)   |
|:------------------|-----------:|:------------------------------------|
| xref              |      12929 | -0.0003 ± 0.0017 (1/3)              |
| alignment_exact   |       2372 | +0.0005 ± 0.0102 (2/3)              |
| alignment_proxy   |        685 | +0.0011 ± 0.0153 (2/3)              |
| none (imputed)    |        494 | -0.0130 ± 0.0136 (0/3)              |
| plants            |       3079 | +0.0000 ± 0.0051 (2/3)              |
| non-plants        |      13402 | -0.0011 ± 0.0026 (1/3)              |
| all test proteins |      16481 | -0.0007 ± 0.0030 (1/3)              |

The groups in the first four rows are:
- **xref:** promoter from the protein's own gene (Ensembl cross-reference);
- **alignment_exact:** same-species gene found by DIAMOND;
- **alignment_proxy:** related-species gene;
- **none (imputed):** no promoter, imputed from ESM-2 neighbours.

![Promoter effect by stratum](summary/promoter_strata.png)

### Are promoters most useful for "response to" terms?

**Hypothesis.** Promoters encode transcriptional regulation, so they should help most for
processes defined by *inducible expression*: "response to heat", "response to glucose", and so
on.

**Test.**
- For each biological-process term, take the change in held-out ROC-AUC from adding promoters,
  averaged over seeds.
- Compare response terms against all other process terms, with response terms defined two ways:
  - **by name:** "response to …" or "cellular response to …";
  - **by ontology:** any descendant of *response to stimulus*, GO:0050896. This set is broader,
    for example "protein refolding".
- Rare terms have noisier AUCs, so the primary p-value comes from a permutation test that
  shuffles the response label only among terms of similar test-set frequency (quintiles). The
  Mann-Whitney test is secondary.

| Adding                  | Definition                 | Response terms   | Other process terms   | Improved (response / other)   |   p, prevalence-stratified |   p, Mann-Whitney | Seeds where response > other   |
|:------------------------|:---------------------------|:-----------------|:----------------------|:------------------------------|---------------------------:|------------------:|:-------------------------------|
| promoters with ESM-2    | name: 'response to …'      | +0.0115 (n=164)  | +0.0085 (n=1299)      | 74% / 72%                     |                     0.0193 |           0.013   | 2/3                            |
| promoters with ESM-2    | ontology: under GO:0050896 | +0.0095 (n=253)  | +0.0088 (n=1210)      | 74% / 72%                     |                     0.5272 |           0.15    | 2/3                            |
| promoters without ESM-2 | name: 'response to …'      | +0.0313 (n=164)  | +0.0153 (n=1299)      | 88% / 69%                     |                     0.0001 |           0       | 3/3                            |
| promoters without ESM-2 | ontology: under GO:0050896 | +0.0243 (n=253)  | +0.0156 (n=1210)      | 77% / 70%                     |                     0.0006 |           0.00044 | 3/3                            |

![Per-term promoter effect, response vs other process terms](summary/response_terms.png)

- **Without ESM-2 the hypothesis holds clearly.** Promoters raise response-term AUC twice as much
  as other process terms (+0.031 vs +0.015), and 88% of response terms improve, against 69% of
  others. The result holds under both definitions, on every seed, and after controlling for term
  frequency.
- **With ESM-2 the extra benefit mostly disappears.** It is significant only for the narrower
  name-based set (p = 0.019, 2/3 seeds) and absent for the broader ontology set (p = 0.53).
  Promoter DNA carries real stimulus-response information, but much of it is redundant with
  protein sequence. Plausibly that's because inducible protein families (heat-shock proteins,
  hormone-response factors) are themselves recognisable from sequence.
- **The most-helped terms are textbook inducible expression programmes:** responses to leukemia
  inhibitory factor, ischemia, glucose and glucose starvation, corticosteroids and cAMP.

Most-helped "response to" terms when promoters are added to the ESM-2 model (at least 20 positive
test proteins):

| GO term    | Name                                            |   Test positives |   Δ AUC |
|:-----------|:------------------------------------------------|-----------------:|--------:|
| GO:1990823 | response to leukemia inhibitory factor          |               26 |   0.105 |
| GO:1990830 | cellular response to leukemia inhibitory factor |               25 |   0.103 |
| GO:0002931 | response to ischemia                            |               20 |   0.068 |
| GO:0042149 | cellular response to glucose starvation         |               22 |   0.066 |
| GO:0071333 | cellular response to glucose stimulus           |               36 |   0.048 |
| GO:0071331 | cellular response to hexose stimulus            |               37 |   0.046 |
| GO:0031960 | response to corticosteroid                      |               63 |   0.045 |
| GO:0009749 | response to glucose                             |               72 |   0.045 |

…and to the model without ESM-2:

| GO term    | Name                                            |   Test positives |   Δ AUC |
|:-----------|:------------------------------------------------|-----------------:|--------:|
| GO:0002931 | response to ischemia                            |               20 |   0.17  |
| GO:1990830 | cellular response to leukemia inhibitory factor |               25 |   0.161 |
| GO:1990823 | response to leukemia inhibitory factor          |               26 |   0.16  |
| GO:0014823 | response to activity                            |               34 |   0.112 |
| GO:0048678 | response to axon injury                         |               26 |   0.103 |
| GO:0042149 | cellular response to glucose starvation         |               22 |   0.096 |
| GO:0051591 | response to cAMP                                |               33 |   0.092 |
| GO:0071322 | cellular response to carbohydrate stimulus      |               49 |   0.086 |

Every term's effect is in [`summary/response_term_deltas.csv`](summary/response_term_deltas.csv).

## Internal vs official evaluation

<a id="official-cafa-scores"></a>
**The two agree that ESM-2 dominates, but disagree on everything added to it:**
- **handcrafted features** help internally but not officially;
- **promoters** look neutral to helpful internally but hurt the official CAFA metrics.

The internal split is random by protein, so close homologues often land on both sides of it, and
promoter coverage looks like training. The CAFA6 targets were annotated after training and have
much patchier promoter coverage, so they reward features that generalise.

**Official CAFA evaluator scores** (all configs, all seeds; mean ± sd):

<!-- cafaeval-table -->
| Config               | Seeds   | f_w NK        | f_w LK        | f_w PK        | f_w overall   | f_micro_w NK   | f_micro_w LK   | f_micro_w PK   | f_micro_w overall   | s_w (lower better) overall   |
|:---------------------|:--------|:--------------|:--------------|:--------------|:--------------|:---------------|:---------------|:---------------|:--------------------|:-----------------------------|
| `esm2`               | 0, 1, 2 | 0.394 ± 0.009 | 0.401 ± 0.007 | 0.161 ± 0.002 | 0.319 ± 0.006 | 0.310 ± 0.007  | 0.343 ± 0.003  | 0.111 ± 0.001  | 0.255 ± 0.003       | 10.766 ± 0.061               |
| `esm2_prot`          | 0, 1, 2 | 0.392 ± 0.011 | 0.402 ± 0.004 | 0.162 ± 0.004 | 0.319 ± 0.003 | 0.306 ± 0.008  | 0.338 ± 0.005  | 0.114 ± 0.003  | 0.253 ± 0.000       | 10.777 ± 0.038               |
| `esm2_prot_tax`      | 0, 1, 2 | 0.401 ± 0.002 | 0.401 ± 0.008 | 0.161 ± 0.003 | 0.321 ± 0.003 | 0.312 ± 0.004  | 0.333 ± 0.006  | 0.111 ± 0.003  | 0.252 ± 0.003       | 10.815 ± 0.020               |
| `all`                | 0, 1, 2 | 0.379 ± 0.006 | 0.394 ± 0.004 | 0.165 ± 0.005 | 0.313 ± 0.002 | 0.295 ± 0.007  | 0.323 ± 0.004  | 0.106 ± 0.003  | 0.241 ± 0.002       | 10.881 ± 0.028               |
| `prot_tax`           | 0, 1, 2 | 0.303 ± 0.008 | 0.332 ± 0.008 | 0.133 ± 0.003 | 0.256 ± 0.004 | 0.248 ± 0.006  | 0.262 ± 0.010  | 0.087 ± 0.002  | 0.199 ± 0.005       | 11.351 ± 0.060               |
| `prot_tax_promoters` | 0, 1, 2 | 0.305 ± 0.003 | 0.316 ± 0.001 | 0.124 ± 0.001 | 0.249 ± 0.002 | 0.232 ± 0.005  | 0.226 ± 0.001  | 0.077 ± 0.002  | 0.178 ± 0.002       | 11.598 ± 0.010               |
<!-- /cafaeval-table -->

AUC and the CAFA metrics measure different things:
- **AUC** measures how well each term's scores rank proteins, over the whole score range, with
  every term weighted equally.
- **The CAFA metrics** score each protein's predicted set of terms at the single best threshold,
  weighted by information accretion.

The feature groups beyond ESM-2 help the first without helping the second.

### Is `all` the best model?

**No.** It has the best AUC, but on the official evaluator it is the *worst* of the four ESM-2
models.

**Protein bootstrap.** Within each seed, evaluation proteins are resampled 1,000 times, the same
resample is used for both models, and the differences are averaged over the 3 seeds. That gives a
95% CI and a two-sided p ([`summary/significance_bootstrap.csv`](summary/significance_bootstrap.csv)).
- **Internal Fmax** is the trainer's own metric on the held-out 20%.
- **Official f_w** is recomputed from each run's saved predictions with the evaluator's
  settings. It reproduces the CAFA evaluator to within 3e-4 on NK and LK. PK is left out
  because its known-annotation handling isn't reproduced yet, so PK relies on the seed-level
  tests above.

| all minus          | Internal Fmax                         | Official f_w (NK + LK)                |
|:-------------------|:--------------------------------------|:--------------------------------------|
| esm2               | +0.0061 [+0.0049, +0.0073], p = 0.001 | -0.0103 [-0.0177, -0.0037], p = 0.002 |
| esm2_prot          | +0.0044 [+0.0032, +0.0057], p = 0.001 | -0.0099 [-0.0176, -0.0034], p = 0.006 |
| esm2_prot_tax      | -0.0007 [-0.0018, +0.0006], p = 0.326 | -0.0137 [-0.0206, -0.0066], p = 0.001 |
| prot_tax           | +0.0658 [+0.0641, +0.0676], p = 0.001 | +0.0692 [+0.0595, +0.0787], p = 0.001 |
| prot_tax_promoters | +0.0675 [+0.0659, +0.0693], p = 0.001 | +0.0762 [+0.0679, +0.0854], p = 0.001 |

- **On the held-out 20%,** `all` beats ESM-2 alone and ties with the same model without
  promoters (`esm2_prot_tax`).
- **On the official CAFA6 targets,** every ESM-2 model without promoters significantly beats
  `all`.
- **Among those three, the official evaluator can't separate them:**
  - `esm2_prot_tax` has the best f_w (0.321);
  - `esm2` has the best f_micro_w (0.255) and S-min (10.77);
  - every difference between them is within seed noise.
- **The exception is PK** (proteins with prior annotations), the one setting where `all`
  scores highest (f_w 0.165 vs 0.161–0.162), though not significantly.

The [architecture search](hparam-search/README.md) repeats this test with a wider network, and the
ranking holds there: without promoters is significantly better on official f_w.

Every run's predictions and evaluator outputs are kept under `data/ablation/` for later analysis.

## Caveats

- **Capacity:** promoter configs have about 0.36M more parameters (1.37M vs 1.01M). The
  [architecture search](hparam-search/README.md) varies width directly. A much larger
  no-promoter network (2.1M parameters) is still the best on official metrics, so the promoter
  penalty is not a capacity artefact.
- **Internal split:** random by protein, so homologues leak across it. That inflates absolute
  internal scores. Paired deltas are less affected, but the handcrafted-feature result shows
  internal deltas can still mislead.
- **Official truth:** one release, and only 8,023 proteins across NK/LK/PK, so many terms have
  few positives and official per-term AUCs are noisy.
- **Score channels off:** production models blend in a taxon prior and ESM-2 nearest-neighbour
  transfer. These may absorb part of the kingdom and promoter gains.
- **Three seeds:** seed-level t-tests have little power, and none of the ~30 shown are corrected
  for multiple testing. Config-level claims therefore rest on the protein bootstrap, which
  resamples evaluation proteins but not training runs. The response-term tests are per GO term,
  with hundreds of terms per group.

## Per-configuration details

Each directory holds:
- `config.json`: feature groups, trainer flags, seeds, git revision;
- `metrics.csv`: every internal and official metric per seed;
- `per_term_auc.csv`: held-out ROC-AUC and average precision per GO term, mean over seeds;
- `training_curves.png`: validation loss and Fmax by epoch;
- `pr_curves.png`: IA-weighted precision–recall per aspect.

[`esm2`](esm2/) · [`esm2_prot`](esm2_prot/) · [`esm2_prot_tax`](esm2_prot_tax/) · [`all`](all/) ·
[`prot_tax`](prot_tax/) · [`prot_tax_promoters`](prot_tax_promoters/)

Cross-run tables are in [`summary/`](summary/): `runs.csv`, `summary.csv`, `deltas.csv`,
`promoter_strata.csv`, `response_terms.csv`, `response_term_deltas.csv` and `cafaeval.csv`.
