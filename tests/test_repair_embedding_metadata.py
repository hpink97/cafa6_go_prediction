import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import repair_embedding_metadata


def _write_embedding_dir(path: Path, rows: int, dim: int) -> None:
    path.mkdir(parents=True)
    pd.DataFrame({"protein_id": [f"P{index}" for index in range(rows)]}).to_csv(path / "ids.tsv", sep="\t", index=False)
    np.zeros((rows, dim), dtype=np.float32).tofile(path / "embeddings.f32.dat")
    np.ones(rows, dtype=bool).tofile(path / "done.bool.dat")


class RepairEmbeddingMetadataTest(unittest.TestCase):
    def test_infers_esm_and_promoter_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_embedding_dir(root / "train_esm2_150M", rows=3, dim=5)
            _write_embedding_dir(root / "test_promoter_agront1b", rows=2, dim=7)

            written = repair_embedding_metadata.repair(root, project_root=root)

            self.assertEqual(len(written), 2)
            esm = json.loads((root / "train_esm2_150M" / "metadata.json").read_text())
            self.assertEqual((esm["protein_count"], esm["embedding_dim"], esm["model_size"]), (3, 5, "150M"))
            promoter = json.loads((root / "test_promoter_agront1b" / "metadata.json").read_text())
            self.assertEqual((promoter["protein_count"], promoter["embedding_dim"]), (2, 7))
            self.assertEqual(promoter["model_name"], "InstaDeepAI/agro-nucleotide-transformer-1b")

    def test_keeps_existing_metadata_unless_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_embedding_dir(root / "train_esm2_35M", rows=2, dim=4)
            (root / "train_esm2_35M" / "metadata.json").write_text("{}")

            self.assertEqual(repair_embedding_metadata.repair(root, project_root=root), [])
            self.assertEqual(len(repair_embedding_metadata.repair(root, project_root=root, overwrite=True)), 1)

    def test_rejects_size_that_is_not_a_whole_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_embedding_dir(root / "train_esm2_35M", rows=3, dim=4)
            with open(root / "train_esm2_35M" / "embeddings.f32.dat", "ab") as handle:
                handle.write(b"\x00\x00\x00\x00")

            with self.assertRaises(ValueError):
                repair_embedding_metadata.repair(root, project_root=root)


if __name__ == "__main__":
    unittest.main()
