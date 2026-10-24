from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from feature_engineering.promoters.alignment_rescue import (
    HIT_FIELDS,
    assign_reference_genes,
    canonical_protein_id,
    choose_queries,
    lineage,
    longest_isoform_per_gene,
    ncbi_panel,
    shared_lineage_depth,
    write_query_fasta,
)
from feature_engineering.promoters.extract_promoters import write_combined

# root(1) -> Eukaryota(2) -> Chordata(3, phylum) -> {mouse(10), rat(11)}; Arthropoda(4, phylum) -> fly(20)
PARENTS = {"1": "1", "2": "1", "3": "2", "4": "2", "10": "3", "11": "3", "20": "4"}
RANKS = {"1": "no rank", "2": "superkingdom", "3": "phylum", "4": "phylum", "10": "species", "11": "species", "20": "species"}
SPECIES_BY_TAXON = {"10": "mouse", "11": "rat", "20": "fly"}
TAXON_BY_SPECIES = {"mouse": "10", "rat": "11", "fly": "20"}


def hit(query: str, target: str, pident: float, coverage: float, bitscore: float) -> dict:
    return dict(zip(HIT_FIELDS, [query, target, pident, 300, 300, 300, 1e-50, bitscore, coverage, coverage]))


def assign(hits: list[dict], taxa: dict[str, str], proxy_rank: str | None = "phylum") -> pd.DataFrame:
    return assign_reference_genes(
        pd.DataFrame(hits, columns=HIT_FIELDS),
        taxa,
        SPECIES_BY_TAXON,
        TAXON_BY_SPECIES,
        PARENTS,
        RANKS,
        exact_identity=95.0,
        exact_coverage=80.0,
        proxy_identity=40.0,
        proxy_coverage=60.0,
        proxy_rank=proxy_rank,
    ).set_index("protein_id")


class AlignmentRescueTests(unittest.TestCase):
    def test_canonical_protein_id_matches_bare_uniprot_accessions(self) -> None:
        self.assertEqual(canonical_protein_id("sp|F4JRB0|HHO5_ARATH"), "F4JRB0")
        self.assertEqual(canonical_protein_id("tr|A0A123|ENTRY"), "A0A123")
        self.assertEqual(canonical_protein_id("plain_id"), "plain_id")

    def test_write_query_fasta_resolves_taxonomy_tsv_to_sibling_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            taxonomy = root / "train_taxonomy.tsv"
            taxonomy.write_text("P1\t9606\n")
            (root / "train_sequences.fasta").write_text(">sp|P1|ENTRY description\nMAAA\n>P2 other\nMKK\n")
            output = root / "queries.fa"
            write_query_fasta(taxonomy, ["P1"], output)
            self.assertEqual(output.read_text(), ">P1\nMAAA\n")


    def test_longest_isoform_per_gene_uses_gene_token(self) -> None:
        records = [
            SeqRecord(Seq("MAAA"), id="ENSP1.1", description="ENSP1.1 pep gene:ENSG1.4 transcript:ENST1"),
            SeqRecord(Seq("MAAAAAAA*"), id="ENSP2.1", description="ENSP2.1 pep gene:ENSG1.4 transcript:ENST2"),
            SeqRecord(Seq("MKK"), id="SPAC1.1:pep", description="SPAC1.1:pep pep gene:SPAC1.01"),
        ]
        genes = longest_isoform_per_gene(records)
        self.assertEqual(genes["ENSG1"], ("ENSP2.1", "MAAAAAAA"))
        self.assertEqual(genes["SPAC1.01"][0], "SPAC1.1:pep")

    def test_lineage_depth(self) -> None:
        self.assertEqual(lineage("10", PARENTS), ("1", "2", "3", "10"))
        self.assertEqual(shared_lineage_depth(lineage("10", PARENTS), lineage("11", PARENTS)), 3)
        self.assertEqual(shared_lineage_depth(lineage("10", PARENTS), lineage("20", PARENTS)), 2)

    def test_own_species_near_identity_is_exact(self) -> None:
        hits = [hit("q1", "rat|G_rat|p", 99.0, 95.0, 500), hit("q1", "mouse|G_mouse|p", 92.0, 95.0, 600)]
        result = assign(hits, {"q1": "11"})
        self.assertEqual(result.loc["q1", "promoter_source"], "alignment_exact")
        self.assertEqual(result.loc["q1", "reference_gene_id"], "G_rat")

    def test_ncbi_same_taxon_exact_even_when_ensembl_is_configured(self) -> None:
        hits = pd.DataFrame([hit("q1", "ncbi_GCF_rat|G_rat|p", 99.0, 95.0, 500)], columns=HIT_FIELDS)
        result = assign_reference_genes(
            hits, {"q1": "11"}, SPECIES_BY_TAXON, {**TAXON_BY_SPECIES, "ncbi_GCF_rat": "11"},
            PARENTS, RANKS, exact_identity=95.0, exact_coverage=80.0,
            proxy_identity=40.0, proxy_coverage=60.0, proxy_rank="phylum",
        )
        self.assertEqual(result.loc[0, "promoter_source"], "alignment_exact")

    def test_ncbi_different_taxon_never_counts_as_exact(self) -> None:
        hits = pd.DataFrame([hit("q1", "ncbi_GCF_mouse|G_mouse|p", 99.0, 95.0, 500)], columns=HIT_FIELDS)
        result = assign_reference_genes(
            hits, {"q1": "11"}, SPECIES_BY_TAXON, {**TAXON_BY_SPECIES, "ncbi_GCF_mouse": "10"},
            PARENTS, RANKS, exact_identity=95.0, exact_coverage=80.0,
            proxy_identity=40.0, proxy_coverage=60.0, proxy_rank="phylum",
        )
        self.assertEqual(result.loc[0, "promoter_source"], "alignment_proxy")

    def test_ncbi_panel_falls_back_to_parent_species_as_proxy(self) -> None:
        parents = {"1": "1", "2": "1", "100": "2", "101": "100"}
        ranks = {"100": "species", "101": "strain"}
        assembly = {"accession": "GCF_parent", "taxon_id": "100"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("feature_engineering.promoters.alignment_rescue.select_assemblies", side_effect=[{"101": None}, {"100": assembly}]) as select:
                with patch("feature_engineering.promoters.alignment_rescue.build_ncbi_proteome", return_value=(root / "p.gz", root / "g.gz")):
                    proteomes, exact_taxa, reference_taxa, sources = ncbi_panel(
                        pd.Series({"101": 100}), 20, None, root, parents, ranks, 30
                    )
        self.assertEqual(select.call_args_list[1].args[0], ["100"])
        self.assertIn("ncbi_GCF_parent", proteomes)
        self.assertEqual(exact_taxa, {})
        self.assertEqual(reference_taxa["ncbi_GCF_parent"], "100")
        self.assertEqual(sources["ncbi_GCF_parent"]["accession"], "GCF_parent")

    def test_proxy_prefers_closest_lineage_and_skips_own_species_paralogs(self) -> None:
        hits = [
            hit("q1", "rat|paralog|p", 70.0, 90.0, 900),
            hit("q1", "fly|G_fly|p", 60.0, 90.0, 800),
            hit("q1", "mouse|G_mouse|p", 55.0, 90.0, 300),
        ]
        result = assign(hits, {"q1": "11"}, proxy_rank=None)
        self.assertEqual(result.loc["q1", "promoter_source"], "alignment_proxy")
        self.assertEqual(result.loc["q1", "reference_gene_id"], "G_mouse")

    def test_proxy_rank_blocks_cross_phylum_transfer(self) -> None:
        hits = [hit("q1", "fly|G_fly|p", 80.0, 90.0, 900)]
        self.assertNotIn("q1", assign(hits, {"q1": "10"}).index)
        self.assertIn("q1", assign(hits, {"q1": "10"}, proxy_rank=None).index)

    def test_choose_queries_weights_toward_poorly_covered_clusters(self) -> None:
        ids = [f"p{index}" for index in range(200)]
        priority = pd.DataFrame({"protein_id": ids, "priority": [0.99] * 100 + [0.01] * 100})
        chosen = choose_queries(priority, ids, max_queries=50, seed=1)
        self.assertEqual(len(chosen), 50)
        self.assertGreater(np.mean([int(pid[1:]) < 100 for pid in chosen]), 0.9)
        self.assertEqual(len(choose_queries(priority, ids, max_queries=None, seed=1)), 200)

    def test_write_combined_ignores_non_promoter_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"protein_id": ["a"], "taxon_id": ["1"], "promoter_sequence": ["ACGT"]}).to_csv(root / "sp.tsv", sep="\t", index=False)
            pd.DataFrame({"species": ["sp"], "status": ["extracted"]}).to_csv(root / "local_reference_batch_summary.tsv", sep="\t", index=False)
            write_combined(root)
            combined = pd.read_csv(root / "combined.tsv", sep="\t")
        self.assertEqual(combined["protein_id"].tolist(), ["a"])
        self.assertEqual(combined["promoter_source"].tolist(), ["xref"])


if __name__ == "__main__":
    unittest.main()
