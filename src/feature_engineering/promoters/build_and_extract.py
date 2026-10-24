"""Build temporary local promoter references and extract promoters species-by-species."""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from feature_engineering.promoters.build_reference import build_reference
from feature_engineering.promoters.extract_promoters import load_protein_taxa, run as extract_promoters
from feature_engineering.promoters.species_map import ANCHOR_SPECIES, OPERON_AWARE_SPECIES

LOGGER = logging.getLogger("cafa6.promoters.build_and_extract")


def run(
    input_path: Path,
    output_dir: Path,
    species_filter: list[str] | None,
    window: int,
    n_workers: int,
    timeout: float,
    reference_work_dir: Path | None,
    reprocess_existing: bool,
    retry_failed: bool,
    stop_on_error: bool,
    keep_reference: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    species_names = species_to_process(input_path, species_filter)
    if not species_names:
        LOGGER.warning("No configured anchor species found in %s", input_path)
        return

    work_parent = reference_work_dir or output_dir / "_local_reference_tmp"
    work_parent.mkdir(parents=True, exist_ok=True)
    previous_failures = set() if retry_failed else load_previous_failures(output_dir / "local_reference_batch_summary.tsv")
    summary_rows: list[dict[str, str]] = []

    for species in species_names:
        species_output = output_dir / f"{species}.tsv"
        if species_output.exists() and not reprocess_existing and species_output_complete(input_path, species, species_output):
            LOGGER.info("[%s] skipping existing output %s", species, species_output)
            summary_rows.append({"species": species, "status": "skipped_existing", "detail": str(species_output)})
            continue
        if species in previous_failures:
            LOGGER.info("[%s] skipping previous failure; pass --retry-failed to try again", species)
            summary_rows.append({"species": species, "status": "skipped_failed", "detail": "previous failure"})
            continue

        reference_dir = Path(tempfile.mkdtemp(prefix=f"{species}.", dir=work_parent))
        try:
            LOGGER.info("[%s] building temporary local reference in %s", species, reference_dir)
            build_reference(species, reference_dir, input_path=input_path, timeout=timeout)
            LOGGER.info("[%s] extracting promoters from temporary local reference", species)
            extract_promoters(
                input_path,
                output_dir,
                window,
                n_workers,
                [species],
                reference_dir,
                allow_rest_fallback=False,
            )
            summary_rows.append({"species": species, "status": "extracted", "detail": str(species_output)})
        except Exception as exc:
            LOGGER.exception("[%s] failed", species)
            summary_rows.append({"species": species, "status": "failed", "detail": str(exc)})
            if stop_on_error:
                raise
        finally:
            if keep_reference:
                LOGGER.info("[%s] keeping temporary reference at %s", species, reference_dir)
            else:
                shutil.rmtree(reference_dir, ignore_errors=True)
                LOGGER.info("[%s] deleted temporary reference %s", species, reference_dir)

    summary_path = output_dir / "local_reference_batch_summary.tsv"
    pd.DataFrame(summary_rows, columns=["species", "status", "detail"]).to_csv(summary_path, sep="\t", index=False)
    LOGGER.info("wrote batch summary to %s", summary_path)
    try:
        work_parent.rmdir()
    except OSError:
        pass


def species_to_process(input_path: Path, species_filter: list[str] | None) -> list[str]:
    taxonomy = load_protein_taxa(input_path)
    all_species = {**ANCHOR_SPECIES, **OPERON_AWARE_SPECIES}
    requested = None if species_filter is None or species_filter == ["all"] else set(species_filter)
    species_names = []
    for taxon_id in taxonomy["taxon_id"].drop_duplicates():
        species = all_species.get(taxon_id)
        if species is not None and (requested is None or species in requested):
            species_names.append(species)
    return species_names


def load_previous_failures(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    try:
        summary = pd.read_csv(summary_path, sep="\t", dtype=str)
    except pd.errors.EmptyDataError:
        return set()
    if not {"species", "status"}.issubset(summary.columns):
        return set()
    return set(summary.loc[summary["status"].isin(["failed", "skipped_failed"]), "species"].dropna())


def species_output_complete(input_path: Path, species: str, species_output: Path) -> bool:
    all_species = {**ANCHOR_SPECIES, **OPERON_AWARE_SPECIES}
    taxon_ids = {taxon_id for taxon_id, mapped_species in all_species.items() if mapped_species == species}
    taxonomy = load_protein_taxa(input_path)
    expected_ids = set(taxonomy.loc[taxonomy["taxon_id"].isin(taxon_ids), "protein_id"].dropna())
    if not expected_ids:
        return True
    try:
        existing = pd.read_csv(species_output, sep="\t", dtype=str, usecols=["protein_id"])
    except (pd.errors.EmptyDataError, ValueError):
        return False
    existing_ids = set(existing["protein_id"].dropna())
    missing = expected_ids.difference(existing_ids)
    if missing:
        LOGGER.info("[%s] existing output is missing %d/%d mapped input proteins", species, len(missing), len(expected_ids))
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="protein_id<TAB>taxon_id TSV, or FASTA with '>id taxon' headers")
    parser.add_argument("output_dir", type=Path, help="Directory for per-species promoter TSVs and combined.tsv")
    parser.add_argument("--species", nargs="*", default=None, help="Species to process, or all configured species present in the input")
    parser.add_argument("--window", type=int, default=3000, help="Promoter window size in bp, upstream of the TSS")
    parser.add_argument("--n-workers", type=int, default=10, help="REST worker count if a lower-level fallback is ever enabled")
    parser.add_argument("--timeout", type=float, default=180, help="HTTP timeout in seconds for reference downloads")
    parser.add_argument("--reference-work-dir", type=Path, default=None, help="Parent directory for temporary per-species reference directories")
    parser.add_argument("--reprocess-existing", action="store_true", help="Rebuild/re-extract species even when {species}.tsv already exists")
    parser.add_argument("--force", action="store_true", help="Deprecated alias for --reprocess-existing")
    parser.add_argument("--retry-failed", action="store_true", help="Retry species marked failed in the previous batch summary")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first species whose reference build or extraction fails")
    parser.add_argument("--keep-reference", action="store_true", help="Keep each temporary species reference directory instead of deleting it")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run(
        args.input,
        args.output_dir,
        args.species,
        args.window,
        args.n_workers,
        args.timeout,
        args.reference_work_dir,
        reprocess_existing=args.reprocess_existing or args.force,
        retry_failed=args.retry_failed,
        stop_on_error=args.stop_on_error,
        keep_reference=args.keep_reference,
    )


if __name__ == "__main__":
    main()