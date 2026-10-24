from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feature_engineering.promoters.extract_promoters import extract_for_species
from feature_engineering.promoters.local_reference import find_local_reference


class PromoterLocalReferenceTests(unittest.TestCase):
    def test_offline_local_only_without_reference_skips_rest_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = extract_for_species(
                ["P1"],
                "toy_species",
                4,
                1,
                Path(directory) / "gene_id_cache.tsv",
                Path(directory) / "sequence_cache.tsv",
                local_reference=None,
                allow_rest_fallback=False,
            )
        self.assertTrue(result.empty)

    def test_local_reference_extracts_forward_and_reverse_promoters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_dir = root / "reference"
            output_dir = root / "output"
            reference_dir.mkdir()
            output_dir.mkdir()
            species = "toy_species"

            (reference_dir / f"{species}.protein_gene.tsv").write_text(
                "protein_id\tensembl_gene_id\n"
                "P1\tG1\n"
                "P2\tG2\n"
            )
            (reference_dir / f"{species}.genes.tsv").write_text(
                "ensembl_gene_id\tseq_region_name\tstart\tend\tstrand\n"
                "G1\tchr1\t10\t20\t1\n"
                "G2\tchr1\t30\t40\t-1\n"
            )
            (reference_dir / f"{species}.fa").write_text(
                ">chr1\n"
                "AAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTT\n"
            )

            reference = find_local_reference(reference_dir, species)
            result = extract_for_species(
                ["P1", "P2"],
                species,
                4,
                1,
                output_dir / "gene_id_cache.tsv",
                output_dir / "sequence_cache.tsv",
                reference,
            )

        rows = result.set_index("protein_id")
        self.assertEqual(rows.loc["P1", "promoter_start"], 6)
        self.assertEqual(rows.loc["P1", "promoter_end"], 9)
        self.assertEqual(rows.loc["P1", "promoter_sequence"], "CCCG")
        self.assertEqual(rows.loc["P2", "promoter_start"], 41)
        self.assertEqual(rows.loc["P2", "promoter_end"], 44)
        self.assertEqual(rows.loc["P2", "promoter_sequence"], "CCCC")


if __name__ == "__main__":
    unittest.main()