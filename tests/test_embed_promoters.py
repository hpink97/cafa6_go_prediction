import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from feature_engineering.promoters import embed_promoters


class _FakeModel:
    config = SimpleNamespace(hidden_size=2)

    def eval(self):
        return self

    def to(self, _device):
        return self


def _fake_embed(_model, _tokenizer, sequences, _device):
    return np.ones((len(sequences), 2), dtype=np.float32)


class EmbedKingdomSelectionTest(unittest.TestCase):
    def test_reuse_reopens_zero_rows_in_selected_kingdom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            promoters = root / "combined.tsv"
            pd.DataFrame(
                {
                    "protein_id": ["plant_ncbi", "plant_old", "animal"],
                    "taxon_id": ["4081", "3702", "9606"],
                    "species": ["ncbi_GCF_1", "arabidopsis_thaliana", "homo_sapiens"],
                    "promoter_sequence": ["ACGT", "ACGT", "ACGT"],
                }
            ).to_csv(promoters, sep="\t", index=False)
            kingdoms = {"4081": "eukaryota_viridiplantae", "3702": "eukaryota_viridiplantae", "9606": "eukaryota_metazoa"}
            with (
                patch.object(embed_promoters.AutoTokenizer, "from_pretrained", return_value=object()),
                patch.object(embed_promoters.AutoModelForMaskedLM, "from_pretrained", return_value=_FakeModel()),
                patch.object(embed_promoters, "_embed_batch", side_effect=_fake_embed),
                patch("feature_engineering.sequence.taxonomy.classify_kingdoms", return_value=kingdoms),
            ):
                embed_promoters.generate_promoter_embeddings(
                    promoters, root / "out", model_name="fake", device_name="cpu", embed_species=["arabidopsis_thaliana"]
                )
                first = np.memmap(root / "out" / "embeddings.f32.dat", dtype="float32", mode="r", shape=(3, 2))
                self.assertEqual(first.sum(axis=1).tolist(), [0.0, 0.0, 2.0])
                del first

                embed_promoters.generate_promoter_embeddings(
                    promoters,
                    root / "out",
                    model_name="fake",
                    device_name="cpu",
                    reuse_existing=True,
                    embed_kingdoms=["eukaryota_viridiplantae"],
                )
            ids = pd.read_csv(root / "out" / "ids.tsv", sep="\t")["protein_id"].tolist()
            vectors = np.memmap(root / "out" / "embeddings.f32.dat", dtype="float32", mode="r", shape=(3, 2))
            self.assertEqual(dict(zip(ids, vectors.sum(axis=1).tolist())), {"animal": 0.0, "plant_ncbi": 2.0, "plant_old": 2.0})


if __name__ == "__main__":
    unittest.main()
