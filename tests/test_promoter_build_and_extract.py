from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feature_engineering.promoters.build_and_extract import load_previous_failures, species_output_complete, species_to_process


class PromoterBuildAndExtractTests(unittest.TestCase):
    def test_species_to_process_uses_configured_species_in_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "taxonomy.tsv"
            path.write_text("protein1\t10090\nprotein2\t9606\nprotein3\t10090\nprotein4\tnope\n")
            self.assertEqual(species_to_process(path, None), ["mus_musculus", "homo_sapiens"])
            self.assertEqual(species_to_process(path, ["homo_sapiens"]), ["homo_sapiens"])

    def test_load_previous_failures_tracks_failed_and_skipped_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.tsv"
            path.write_text(
                "species\tstatus\tdetail\n"
                "homo_sapiens\tskipped_existing\tout.tsv\n"
                "arabidopsis_thaliana\tfailed\tmissing mappings\n"
                "salmonella_enterica\tskipped_failed\tprevious failure\n"
            )
            self.assertEqual(load_previous_failures(path), {"arabidopsis_thaliana", "salmonella_enterica"})

    def test_species_output_complete_detects_new_taxon_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            taxonomy = root / "taxonomy.tsv"
            output = root / "oryza_sativa.tsv"
            taxonomy.write_text("p1\t39947\np2\t39946\n")
            output.write_text("protein_id\tspecies\np1\toryza_sativa\n")
            self.assertFalse(species_output_complete(taxonomy, "oryza_sativa", output))
            output.write_text("protein_id\tspecies\np1\toryza_sativa\np2\toryza_sativa\n")
            self.assertTrue(species_output_complete(taxonomy, "oryza_sativa", output))


if __name__ == "__main__":
    unittest.main()