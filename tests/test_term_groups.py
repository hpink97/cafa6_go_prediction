from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.go_evaluation import GeneOntology
from modeling.term_groups import classify_terms, load_term_names, stratified_permutation_test

OBO = """format-version: 1.2

[Term]
id: GO:0050896
name: response to stimulus
namespace: biological_process

[Term]
id: GO:0009408
name: response to heat
namespace: biological_process
is_a: GO:0050896 ! response to stimulus

[Term]
id: GO:0042026
name: protein refolding
namespace: biological_process
is_a: GO:0050896 ! response to stimulus

[Term]
id: GO:0006412
name: translation
namespace: biological_process

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function
"""


class ClassifyTermsTests(unittest.TestCase):
    def test_name_and_ontology_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "go.obo"
            path.write_text(OBO)
            ontology = GeneOntology.from_obo(path)
            names = load_term_names(path)
        terms = pd.DataFrame(
            {
                "term": ["GO:0050896", "GO:0009408", "GO:0042026", "GO:0006412", "GO:0003674"],
                "aspect": ["P", "P", "P", "P", "F"],
            }
        )

        classes = classify_terms(terms, ontology, names)

        self.assertNotIn("GO:0003674", classes.index)  # process terms only
        self.assertEqual(classes["by_name"].to_dict(), {"GO:0050896": True, "GO:0009408": True, "GO:0042026": False, "GO:0006412": False})
        # The root itself is excluded; a descendant counts even when its name does not say "response".
        self.assertEqual(classes["by_ontology"].to_dict(), {"GO:0050896": False, "GO:0009408": True, "GO:0042026": True, "GO:0006412": False})


class StratifiedPermutationTests(unittest.TestCase):
    def test_detects_a_real_shift_and_not_a_prevalence_artefact(self) -> None:
        rng = np.random.default_rng(1)
        prevalence = rng.integers(10, 5000, size=400).astype(float)
        in_group = rng.random(400) < 0.2

        shifted = rng.normal(0, 0.01, size=400) + 0.02 * in_group
        difference, p_value = stratified_permutation_test(shifted, in_group, prevalence, n_permutations=2000)
        self.assertGreater(difference, 0.01)
        self.assertLess(p_value, 0.01)

        # The effect tracks prevalence, and the group is drawn from the common terms: within
        # prevalence bins the group is not special, so the stratified test should not fire.
        common = prevalence > np.median(prevalence)
        confounded_group = common & (rng.random(400) < 0.4)
        by_prevalence = 0.02 * (prevalence > np.quantile(prevalence, 0.8)) + rng.normal(0, 0.001, size=400)
        _, p_value = stratified_permutation_test(by_prevalence, confounded_group, prevalence, n_bins=5, n_permutations=2000)
        self.assertGreater(p_value, 0.05)


if __name__ == "__main__":
    unittest.main()
