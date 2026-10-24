import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.data import PromoterImputer, fit_promoter_imputer


class PromoterImputerTest(unittest.TestCase):
    def test_fit_save_load_and_plant_specific_fill(self) -> None:
        esm2 = pd.DataFrame(
            {"protein_id": ["a", "b", "c", "held_out"], "e0": [1.0, 0.0, 1.0, 0.0], "e1": [0.0, 1.0, 0.1, 1.0]}
        )
        nt = pd.DataFrame({"protein_id": ["a", "b", "held_out"], "promoter_nt_0": [1.0, 3.0, 99.0]})
        agront = pd.DataFrame({"protein_id": ["a", "b", "c"], "promoter_agront_0": [5.0, 0.0, 7.0]})
        imputer = fit_promoter_imputer(esm2, {"promoter_nt": nt, "promoter_agront": agront}, ["a", "b", "c"], k=1)

        self.assertEqual(imputer.donor_features["promoter_nt"].ravel().tolist(), [1.0, 3.0])
        self.assertEqual(imputer.donor_features["promoter_agront"].ravel().tolist(), [5.0, 7.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "imputer.npz"
            imputer.save(path)
            loaded = PromoterImputer.load(path)

        query = np.asarray([[0.0, 1.0], [1.0, 0.05]], dtype=np.float32)
        self.assertEqual(loaded.impute("promoter_nt", query).ravel().tolist(), [3.0, 1.0])
        plant_scores = loaded.impute("promoter_agront", query, np.asarray([True, False]))
        self.assertEqual(plant_scores.ravel().tolist(), [7.0, 0.0])


if __name__ == "__main__":
    unittest.main()
