"""Score prediction files on the released CAFA6 final ground truth with the official CAFA-evaluator-PK.

Runs `cafaeval` once per knowledge setting (NK, LK, PK) over every file in a predictions folder,
using the configuration published with the release, then writes the weighted micro-F per setting
and GO aspect plus the overall mean (the mean of the three setting means).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("cafa6.modeling.evaluate_cafa6")
REGIMES = ("NK", "LK", "PK")
ASPECTS = {"biological_process": "BP", "molecular_function": "MF", "cellular_component": "CC"}


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
    return out_dir / regime / "evaluation_best_f_micro_w.tsv"


def summarize(result_paths: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for regime, path in result_paths.items():
        frame = pd.read_csv(path, sep="\t")
        frame["regime"] = regime
        frames.append(frame)
    results = pd.concat(frames)
    results["aspect"] = results["ns"].map(ASPECTS)
    table = results.pivot_table(index="filename", columns=["regime", "aspect"], values="f_micro_w")
    table = table.reindex(columns=pd.MultiIndex.from_product([REGIMES, ["BP", "MF", "CC"]]))
    for regime in REGIMES:
        table[(regime, "mean")] = table[regime].mean(axis=1)
    table[("overall", "mean")] = table[[(regime, "mean") for regime in REGIMES]].mean(axis=1)
    return table.sort_values(("overall", "mean"), ascending=False)


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
    result_paths = {
        regime: run_cafaeval(args.predictions_dir, args.release_dir, out_dir, regime, args.threads, args.cafaeval)
        for regime in REGIMES
    }
    table = summarize(result_paths)
    table.round(6).to_csv(out_dir / "summary_f_micro_w.tsv", sep="\t")
    with pd.option_context("display.width", 250):
        print(table.round(4).to_string())


if __name__ == "__main__":
    main()
