"""Feature-group ablation sweep for the multi-label network.

Every config x seed trains on a seeded 80% of the labelled proteins and is scored internally on
the held-out 20% (macro/micro ROC-AUC, IA-weighted CAFA Fmax) and by ROC-AUC against the CAFA6
ground truth. The official CAFA evaluator (weighted F, S-min) takes ~20 minutes per run, so it is
opt-in: run it for the best config, or later for every saved prediction. Score channels (taxon
prior, ESM-2 KNN) are off so only the feature groups differ. Seeds are shared across configs, so
comparisons are paired.

    cafa6-ablation run --sweep feature-groups            # all configs x seeds 0 1 2, resumable
    cafa6-ablation cafaeval --sweep feature-groups --best   # official CAFA scores, best config only
    cafa6-ablation cafaeval --sweep feature-groups          # ... or every finished run
    cafa6-ablation run --sweep smoke --smoke --seeds 0   # minutes-long end-to-end check
    cafa6-ablation report --sweep feature-groups         # tables and figures -> results/
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from modeling.evaluation.official import METRICS, REGIMES, official_roc_auc, run_cafaeval, summarize

LOGGER = logging.getLogger("cafa6.modeling.experiments.ablation")
DEFAULT_ROOT = Path("data/ablation")
DEFAULT_RELEASE_DIR = Path("data/evaluation/release_kaggle_final_May2026")


@dataclass(frozen=True)
class AblationConfig:
    name: str
    groups: tuple[str, ...]
    flags: tuple[str, ...]
    question: str


CONFIGS = {
    config.name: config
    for config in (
        AblationConfig("esm2", ("esm2",), ("--no-protein-features", "--no-taxonomy-features", "--no-promoter-features"), "baseline"),
        AblationConfig("esm2_prot", ("esm2", "protein"), ("--no-taxonomy-features", "--no-promoter-features"), "value of handcrafted features"),
        AblationConfig("esm2_prot_tax", ("esm2", "protein", "taxonomy"), ("--no-promoter-features",), "value of handcrafted + kingdom features"),
        AblationConfig("all", ("esm2", "protein", "taxonomy", "promoter_nt", "promoter_agront"), (), "value of promoters (NT + AgroNT)"),
        AblationConfig("prot_tax", ("protein", "taxonomy"), ("--no-esm2-features", "--no-promoter-features"), "floor without a protein language model"),
        AblationConfig("prot_tax_promoters", ("protein", "taxonomy", "promoter_nt", "promoter_agront"), ("--no-esm2-features",), "do promoters carry signal without ESM-2"),
    )
}
# Paired comparisons reported as (with - without): each isolates one feature-group addition.
COMPARISONS = (
    ("esm2_prot", "esm2", "+ handcrafted"),
    ("esm2_prot_tax", "esm2_prot", "+ kingdom"),
    ("esm2_prot_tax", "esm2", "+ handcrafted & kingdom"),
    ("all", "esm2_prot_tax", "+ all promoters (with ESM-2)"),
    ("prot_tax_promoters", "prot_tax", "+ all promoters (no ESM-2)"),
    ("esm2_prot_tax", "prot_tax", "+ ESM-2"),
)
COMMON_TRAIN_FLAGS = (
    "--no-knn",
    "--no-taxon-prior",
    "--checkpoint-taxon-weight", "0",
    "--test-size", "0.2",
    "--epochs", "100",
    "--overwrite",
)
SMOKE_TRAIN_FLAGS = (
    "--limit-terms", "20",
    "--max-train-samples", "2000",
    "--max-validation-samples", "500",
    "--max-test-samples", "500",
    "--epochs", "2",
)


def run_dir(root: Path, sweep: str, config: str, seed: int) -> Path:
    return root / sweep / config / f"seed{seed}"


def train_command(config: AblationConfig, seed: int, directory: Path, smoke: bool) -> list[str]:
    command = [
        sys.executable, "-m", "modeling.nn.train",
        *COMMON_TRAIN_FLAGS,
        *config.flags,
        "--random-state", str(seed),
        "--output-dir", str(directory / "model"),
        "--promoter-feature-cache", str(directory / "promoter_cache.csv"),
    ]
    return command + list(SMOKE_TRAIN_FLAGS) if smoke else command


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("running %s (log: %s)", " ".join(command[2:5]), log_path)
    with log_path.open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def train_and_predict(config: AblationConfig, seed: int, root: Path, sweep: str, release_dir: Path, smoke: bool) -> Path:
    """Train one config x seed and predict the ground-truth proteins; finished stages are skipped on rerun."""
    directory = run_dir(root, sweep, config.name, seed)
    model_dir = directory / "model"
    if not (model_dir / "summary.json").exists():
        _run_logged(train_command(config, seed, directory, smoke), directory / "logs" / "train.log")
    for cache in directory.glob("promoter_cache.csv*"):
        cache.unlink()

    predictions_dir = directory / "predictions"
    predictions = predictions_dir / f"{config.name}_seed{seed}.tsv"
    if not predictions.exists():
        truth_files = [str(release_dir / f"May2026_groundtruth_{regime}.tsv") for regime in REGIMES]
        command = [sys.executable, "-m", "modeling.nn.predict", str(model_dir), "--output", str(predictions), "--protein-ids", *truth_files]
        _run_logged(command, directory / "logs" / "predict.log")
    return directory


def has_cafaeval(directory: Path) -> bool:
    return all((directory / "cafaeval" / regime / METRICS["s_w"][0]).exists() for regime in REGIMES)


def score_cafaeval(directory: Path, release_dir: Path, threads: int) -> Path:
    """Official CAFA evaluator for one run's saved predictions, per knowledge regime."""
    if not has_cafaeval(directory):
        for regime in REGIMES:
            run_cafaeval(directory / "predictions", release_dir, directory / "cafaeval", regime, threads, "cafaeval")
    return directory


def score_auc(directory: Path, release_dir: Path) -> Path:
    """ROC-AUC of one run's saved predictions against each regime's ground truth."""
    predictions = next((directory / "predictions").glob("*.tsv"))
    auc_path = directory / "official_auc.json"
    if not auc_path.exists():
        terms = pd.read_csv(directory / "model" / "terms.tsv", sep="\t")["term"].tolist()
        auc = {regime: official_roc_auc(predictions, release_dir, regime, terms) for regime in REGIMES}
        auc_path.write_text(json.dumps(auc, indent=2) + "\n")
    return directory


def collect_run(directory: Path) -> dict[str, float | int | str]:
    """Flatten one finished run into a single row: internal test metrics and official metrics."""
    summary = json.loads((directory / "model" / "summary.json").read_text())
    run_config = json.loads((directory / "model" / "run_config.json").read_text())
    row: dict[str, float | int | str] = {
        "n_parameters": run_config["n_parameters"],
        "best_epoch": run_config["best_epoch"],
        **{f"internal_{key.removeprefix('test_nn_')}": value for key, value in summary.items() if key.startswith("test_nn_")},
    }
    if has_cafaeval(directory):
        regime_dirs = {regime: directory / "cafaeval" / regime for regime in REGIMES}
        for metric in METRICS:
            table = summarize(regime_dirs, metric)
            for (regime, aspect), value in table.iloc[0].items():
                row[f"official_{metric}_{regime}_{aspect}"] = value
    for regime, auc in json.loads((directory / "official_auc.json").read_text()).items():
        for key, value in auc.items():
            row[f"official_{key}_{regime}"] = value
    for key in ("macro_roc_auc", "micro_roc_auc"):
        row[f"official_{key}_overall"] = sum(row[f"official_{key}_{regime}"] for regime in REGIMES) / len(REGIMES)
    return row


def select_runs(sweep_dir: Path, configs: list[str], seeds: list[int] | None, best: bool) -> list[Path]:
    """Finished runs to evaluate; with `best`, only the config with the highest mean internal macro ROC-AUC.

    Selection uses the internal holdout, never the official truth the evaluator then scores.
    """
    finished = [
        directory
        for name in configs
        for directory in sorted((sweep_dir / name).glob("seed*"))
        if (directory / "official_auc.json").exists() and (seeds is None or int(directory.name.removeprefix("seed")) in seeds)
    ]
    if best and finished:
        auc = pd.Series(
            {directory: json.loads((directory / "model" / "summary.json").read_text())["test_nn_macro_roc_auc"] for directory in finished}
        )
        by_config = auc.groupby(lambda directory: directory.parent.name).mean()
        LOGGER.info("internal macro ROC-AUC by config: %s; best: %s", by_config.round(4).to_dict(), by_config.idxmax())
        finished = [directory for directory in finished if directory.parent.name == by_config.idxmax()]
    return finished


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Train, predict, and score configs x seeds")
    run_parser.add_argument("--configs", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    run_parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    run_parser.add_argument("--smoke", action="store_true", help="Tiny term/sample caps and 2 epochs, for an end-to-end check")
    run_parser.add_argument("--cafaeval", action="store_true", help="Also run the official CAFA evaluator (~20 min per run, overlapped with training)")
    run_parser.add_argument("--threads", type=int, default=8, help="cafaeval threads")
    cafaeval_parser = commands.add_parser("cafaeval", help="Official CAFA evaluator for finished runs")
    cafaeval_parser.add_argument("--best", action="store_true", help="Only the config with the highest mean internal macro ROC-AUC")
    cafaeval_parser.add_argument("--configs", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    cafaeval_parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Default: every finished seed")
    cafaeval_parser.add_argument("--threads", type=int, default=10)
    significance_parser = commands.add_parser("significance", help="Paired protein bootstrap: reference config vs each other")
    significance_parser.add_argument("--reference", choices=tuple(CONFIGS), default="all")
    significance_parser.add_argument("--n-bootstrap", type=int, default=1000)
    significance_parser.add_argument("--out", type=Path, default=Path("results/summary/significance_bootstrap.csv"))
    report_parser = commands.add_parser("report", help="Write tables and figures for a finished sweep")
    report_parser.add_argument("--out", type=Path, default=Path("results"))
    for sub in (run_parser, cafaeval_parser, significance_parser, report_parser):
        sub.add_argument("--sweep", required=True, help="Sweep name; runs live in <root>/<sweep>/")
        sub.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        sub.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
        sub.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.command == "run":
        # Training runs on the GPU and cafaeval on the CPU, so with --cafaeval each run is scored in
        # the background while the next one trains. Seed-major order: an interrupted sweep keeps
        # complete, paired seeds.
        with ThreadPoolExecutor(max_workers=1) as scorer:
            pending = []
            for seed in args.seeds:
                for name in args.configs:
                    directory = train_and_predict(CONFIGS[name], seed, args.root, args.sweep, args.release_dir, args.smoke)
                    score_auc(directory, args.release_dir)
                    LOGGER.info("trained and AUC-scored %s seed %d", name, seed)
                    if args.cafaeval:
                        pending.append(scorer.submit(score_cafaeval, directory, args.release_dir, args.threads))
            for future in pending:
                LOGGER.info("cafaeval scored %s", future.result())
    elif args.command == "cafaeval":
        for directory in select_runs(args.root / args.sweep, args.configs, args.seeds, args.best):
            score_cafaeval(directory, args.release_dir, args.threads)
            LOGGER.info("cafaeval scored %s", directory)
    elif args.command == "significance":
        from modeling.evaluation.significance import compare_to_reference

        others = [name for name in CONFIGS if name != args.reference and (args.root / args.sweep / name).exists()]
        table = compare_to_reference(args.root / args.sweep, args.reference, others, args.release_dir, args.n_bootstrap)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.round(6).to_csv(args.out, index=False)
        LOGGER.info("wrote %s", args.out)
    else:
        from modeling.experiments.ablation_report import write_report

        write_report(args.root / args.sweep, args.out, args.release_dir)


if __name__ == "__main__":
    main()
