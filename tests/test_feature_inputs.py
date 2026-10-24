from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.data import build_feature_table
from modeling.train_multilabel_nn import resolve_feature_inputs


def _write_embeddings(path: Path, protein_ids: list[str], dim: int) -> None:
    path.mkdir(parents=True)
    pd.DataFrame({"protein_id": protein_ids}).to_csv(path / "ids.tsv", sep="\t", index=False)
    np.arange(len(protein_ids) * dim, dtype=np.float32).reshape(len(protein_ids), dim).tofile(path / "embeddings.f32.dat")
    np.ones(len(protein_ids), dtype=bool).tofile(path / "done.bool.dat")
    (path / "metadata.json").write_text(f'{{"embedding_dim": {dim}, "protein_count": {len(protein_ids)}}}')


class ResolveFeatureInputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.paths = {
            "esm2": root / "esm2",
            "protein": root / "protein.tsv",
            "taxonomy": root / "taxonomy.tsv",
            "promoter_nt": root / "nt",
            "promoter_agront": root / "agront",
        }
        for block in ("esm2", "promoter_nt", "promoter_agront"):
            _write_embeddings(self.paths[block], ["P1"], 2)
        for block in ("protein", "taxonomy"):
            self.paths[block].write_text("protein_id\nP1\n")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_disabled_blocks_resolve_to_none(self) -> None:
        resolved = resolve_feature_inputs(self.paths, {"promoter_agront", "taxonomy"})
        self.assertIsNone(resolved["promoter_agront"])
        self.assertIsNone(resolved["taxonomy"])
        self.assertEqual(resolved["promoter_nt"], self.paths["promoter_nt"])

    def test_missing_enabled_block_is_an_error_not_a_silent_drop(self) -> None:
        (self.paths["promoter_nt"] / "metadata.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "--no-promoter-nt-features"):
            resolve_feature_inputs(self.paths, set())

    def test_needs_esm2_or_protein_features(self) -> None:
        with self.assertRaises(ValueError):
            resolve_feature_inputs(self.paths, {"esm2", "protein"})

    def test_imputed_agront_needs_taxonomy(self) -> None:
        with self.assertRaises(ValueError):
            resolve_feature_inputs(self.paths, {"taxonomy"})
        # Without ESM-2 nothing is imputed, so AgroNT no longer needs the plant flag.
        self.assertIsNone(resolve_feature_inputs(self.paths, {"taxonomy", "esm2"})["esm2"])


class FeatureTableWithoutEsmTests(unittest.TestCase):
    def test_protein_features_define_rows_and_promoters_are_zero_filled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protein_path = root / "protein.tsv"
            pd.DataFrame({"protein_id": ["P1", "P2", "P3"], "length": [10, 20, 30]}).to_csv(protein_path, sep="\t", index=False)
            _write_embeddings(root / "nt", ["P2"], 2)

            table = build_feature_table(None, protein_path, promoter_embedding_dirs={"promoter_nt": root / "nt"})

            self.assertEqual(table.index.tolist(), ["P1", "P2", "P3"])
            self.assertEqual(table.columns.tolist(), ["length", "promoter_nt_0", "promoter_nt_1"])
            np.testing.assert_array_equal(table.loc["P2", ["promoter_nt_0", "promoter_nt_1"]], [0.0, 1.0])
            np.testing.assert_array_equal(table.loc[["P1", "P3"], ["promoter_nt_0", "promoter_nt_1"]], 0.0)


if __name__ == "__main__":
    unittest.main()
