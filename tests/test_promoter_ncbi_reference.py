import unittest
import gzip
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from feature_engineering.promoters.ncbi_reference import build_ncbi_proteome, parse_ncbi_gff

GFF = """##gff-version 3
NC_1\tRefSeq\tregion\t1\t5000\t.\t+\t.\tID=NC_1:1..5000
NC_1\tRefSeq\tgene\t100\t900\t.\t+\t.\tID=gene-B1;locus_tag=B1;gene_biotype=protein_coding
NC_1\tProtein Homology\tCDS\t100\t900\t.\t+\t0\tID=cds-WP_1.1;Parent=gene-B1;Name=WP_1.1;protein_id=WP_1.1
NC_1\tGnomon\tgene\t2000\t4000\t.\t-\t.\tID=gene-LOC5;gene_biotype=protein_coding
NC_1\tGnomon\tmRNA\t2000\t4000\t.\t-\t.\tID=rna-XM_5.1;Parent=gene-LOC5
NC_1\tGnomon\tCDS\t2100\t2300\t.\t-\t0\tID=cds-XP_5.1;Parent=rna-XM_5.1;protein_id=XP_5.1
NC_1\tGnomon\tCDS\t2500\t2700\t.\t-\t0\tID=cds-XP_5.1;Parent=rna-XM_5.1;protein_id=XP_5.1
"""


class ParseNcbiGffTest(unittest.TestCase):
    def test_links_proteins_to_genes_through_parent_chain(self) -> None:
        genes, protein_to_gene = parse_ncbi_gff(GFF.splitlines())
        self.assertEqual(protein_to_gene, {"WP_1.1": "B1", "XP_5.1": "LOC5"})
        self.assertEqual(genes["ensembl_gene_id"].tolist(), ["B1", "LOC5"])
        self.assertEqual(genes["strand"].tolist(), [1, -1])
        self.assertEqual(genes.loc[1, "start"], 2000)

    def test_build_proteome_reads_fasta_from_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "GCF_test.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("ncbi_dataset/data/GCF_test/genomic.gff", GFF)
                archive.writestr("ncbi_dataset/data/GCF_test/protein.faa", ">WP_1.1 protein\nMKK*\n>XP_5.1 protein\nMVVV\n")
            with patch("feature_engineering.promoters.ncbi_reference.download_package", return_value=archive_path):
                result = build_ncbi_proteome("GCF_test", root / "cache", 30)
            self.assertIsNotNone(result)
            proteome, genes_path = result
            with gzip.open(proteome, "rt") as handle:
                self.assertEqual(handle.read(), ">ncbi_GCF_test|B1|WP_1.1\nMKK\n>ncbi_GCF_test|LOC5|XP_5.1\nMVVV\n")
            self.assertEqual(pd.read_csv(genes_path, sep="\t")["ensembl_gene_id"].tolist(), ["B1", "LOC5"])


if __name__ == "__main__":
    unittest.main()
