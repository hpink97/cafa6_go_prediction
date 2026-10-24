"""Score prediction files on the released CAFA6 final ground truth with the official CAFA-evaluator-PK.

Runs `cafaeval` once per knowledge setting (NK, LK, PK) over every file in a predictions folder,
using the configuration published with the release, then writes weighted micro-F, weighted Fmax
and weighted S-min per setting and GO aspect, plus the overall mean (the mean of the three
setting means).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.go_evaluation import NAMESPACE_TO_ASPECT, GeneOntology, roc_auc_summary

LOGGER = logging.getLogger("cafa6.modeling.evaluate_cafa6")
REGIMES = ("NK", "LK", "PK")
ASPECTS = {"biological_process": "BP", "molecular_function": "MF", "cellular_component": "CC"}
# cafaeval result file and column for each reported metric; S-min is lower-is-better.
METRICS = {
    "f_micro_w": ("evaluation_best_f_micro_w.tsv", "f_micro_w"),
    "f_w": ("evaluation_best_f_w.tsv", "f_w"),
    "s_w": ("evaluation_best_s.tsv", "s_w"),
}


def run_cafaeval(predictions_dir: Path, release_dir: Path, out_dir: Path, regime: str, threads: int, executable: str) -> Path:
    command = [
        executable,
        str(release_dir / "go-basic.obo"),
        str(predictions_dir),
        str(release_dir / f"May2026_groundtruth_{regime}.tsv"),
        "-ia", str(release_dir / "IA.tsv"),
        "-no_orphans",
        "-norm", "cafa",
        "-prop", "fill",
        "-toi", str(release_dir / "May2026_groundtruth_terms_of_interest.txt"),
        "-max_terms", "500",
        "-th_step", "0.001",
        "-threads", str(threads),
        "-out_dir", str(out_dir / regime),
    ]
    if regime == "PK":
        command += ["-known", str(release_dir / "May2026_groundtruth_PK_known.tsv")]
    LOGGER.info("running cafaeval for %s", regime)
    subprocess.run(command, check=True)
    return out_dir / regime


def summarize(regime_dirs: dict[str, Path], metric: str = "f_micro_w") -> pd.DataFrame:
    """Pivot one cafaeval metric to prediction file x (regime, aspect), with regime and overall means."""
    file_name, column = METRICS[metric]
    frames = []
    for regime, regime_dir in regime_dirs.items():
        frame = pd.read_csv(regime_dir / file_name, sep="\t")
        frame["regime"] = regime
        frames.append(frame)
    results = pd.concat(frames)
    results["aspect"] = results["ns"].map(ASPECTS)
    table = results.pivot_table(index="filename", columns=["regime", "aspect"], values=column)
    table = table.reindex(columns=pd.MultiIndex.from_product([REGIMES, ["BP", "MF", "CC"]]))
    for regime in REGIMES:
        table[(regime, "mean")] = table[regime].mean(axis=1)
    table[("overall", "mean")] = table[[(regime, "mean") for regime in REGIMES]].mean(axis=1)
    return table.sort_values(("overall", "mean"), ascending=metric == "s_w")


def official_roc_auc(predictions_path: Path, release_dir: Path, regime: str, terms: list[str]) -> dict[str, float | int]:
    """ROC-AUC of a submission against one regime's ground truth, over the model's terms of interest.

    Truth and PK known annotations are propagated to ancestors. A protein counts for a term only if
    it has truth in that term's aspect; in PK, pairs already known before the release are excluded.
    Unlisted protein-term pairs score 0.
    """
    ontology = GeneOntology.from_obo(release_dir / "go-basic.obo")
    terms_of_interest = set(pd.read_csv(release_dir / "May2026_groundtruth_terms_of_interest.txt", header=None)[0])
    terms = [term for term in dict.fromkeys(ontology.canonical_id(term) for term in terms) if term in terms_of_interest]
    term_index = {term: index for index, term in enumerate(terms)}
    aspects = np.asarray([NAMESPACE_TO_ASPECT[ontology.namespaces[term]] for term in terms])

    def propagated_pairs(path: Path) -> pd.DataFrame:
        table = pd.read_csv(path, sep="\t", usecols=["EntryID", "term"]).drop_duplicates()
        rows = [
            (protein_id, ancestor)
            for protein_id, term in table.itertuples(index=False, name=None)
            for ancestor in ontology.ancestors(term)
            if ancestor in term_index
        ]
        return pd.DataFrame(rows, columns=["EntryID", "term"]).drop_duplicates()

    truth_table = pd.read_csv(release_dir / f"May2026_groundtruth_{regime}.tsv", sep="\t", usecols=["EntryID", "aspect"])
    proteins = sorted(truth_table["EntryID"].unique())
    protein_index = {protein_id: index for index, protein_id in enumerate(proteins)}
    targets = np.zeros((len(proteins), len(terms)), dtype=np.uint8)
    truth = propagated_pairs(release_dir / f"May2026_groundtruth_{regime}.tsv")
    targets[truth["EntryID"].map(protein_index), truth["term"].map(term_index)] = 1

    mask = np.zeros_like(targets, dtype=bool)
    for protein_id, aspect in truth_table.drop_duplicates().itertuples(index=False, name=None):
        mask[protein_index[protein_id], aspects == aspect] = True
    if regime == "PK":
        known = propagated_pairs(release_dir / "May2026_groundtruth_PK_known.tsv")
        known = known[known["EntryID"].isin(protein_index)]
        mask[known["EntryID"].map(protein_index), known["term"].map(term_index)] = False

    scores = np.zeros(targets.shape, dtype=np.float32)
    predictions = pd.read_csv(predictions_path, sep="\t", header=None, names=["EntryID", "term", "score"])
    predictions = predictions[predictions["EntryID"].isin(protein_index) & predictions["term"].isin(term_index)]
    scores[predictions["EntryID"].map(protein_index), predictions["term"].map(term_index)] = predictions["score"]
    summary, _ = roc_auc_summary(targets, scores, aspects, mask=mask)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions_dir", type=Path, help="Folder of headerless protein<TAB>term<TAB>score files")
    parser.add_argument("--release-dir", type=Path, default=Path("data/evaluation/release_kaggle_final_May2026"))
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <predictions_dir>/../results")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cafaeval", default="cafaeval", help="Official evaluator executable")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if shutil.which(args.cafaeval) is None:
        parser.error(f"{args.cafaeval} not found; install with: uv tool install git+https://github.com/claradepaolis/CAFA-evaluator-PK")
    out_dir = args.out_dir or args.predictions_dir.parent / "results"
    regime_dirs = {
        regime: run_cafaeval(args.predictions_dir, args.release_dir, out_dir, regime, args.threads, args.cafaeval)
        for regime in REGIMES
    }
    for metric in METRICS:
        table = summarize(regime_dirs, metric)
        table.round(6).to_csv(out_dir / f"summary_{metric}.tsv", sep="\t")
    with pd.option_context("display.width", 250):
        print(summarize(regime_dirs, "f_micro_w").round(4).to_string())


if __name__ == "__main__":
    main()
