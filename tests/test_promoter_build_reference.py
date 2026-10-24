from __future__ import annotations

import unittest

from feature_engineering.promoters.build_reference import choose_fasta_name, choose_gff3_name, gff3_directory_url, normalize_ensembl_gene_id, parse_gff3_attributes, parse_gff3_gene_id


class PromoterBuildReferenceTests(unittest.TestCase):
    def test_choose_fasta_prefers_toplevel_then_primary_assembly(self) -> None:
        html = """
        <a href="Mus_musculus.GRCm39.dna.primary_assembly.fa.gz">primary</a>
        <a href="Mus_musculus.GRCm39.dna.toplevel.fa.gz">toplevel</a>
        """
        self.assertEqual(choose_fasta_name(html), "Mus_musculus.GRCm39.dna.toplevel.fa.gz")

    def test_choose_fasta_falls_back_to_primary_assembly(self) -> None:
        html = '<a href="Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz">primary</a>'
        self.assertEqual(choose_fasta_name(html), "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz")

    def test_choose_fasta_supports_ensembl_organisms_bgz(self) -> None:
        html = '<a href="hardmasked.fa.bgz">hard</a><a href="unmasked.fa.bgz">unmasked</a>'
        self.assertEqual(choose_fasta_name(html), "unmasked.fa.bgz")

    def test_choose_gff3_prefers_whole_genome_chr_file(self) -> None:
        html = """
        <a href="Mus_musculus.GRCm39.116.abinitio.gff3.gz">abinitio</a>
        <a href="Mus_musculus.GRCm39.116.chr.gff3.gz">chr</a>
        <a href="Mus_musculus.GRCm39.116.chromosome.1.gff3.gz">1</a>
        """
        self.assertEqual(choose_gff3_name(html), "Mus_musculus.GRCm39.116.chr.gff3.gz")

    def test_gene_id_parsing_strips_prefix_and_version(self) -> None:
        self.assertEqual(normalize_ensembl_gene_id("ENSMUSG00000002147.19"), "ENSMUSG00000002147")
        self.assertEqual(normalize_ensembl_gene_id("SPCC16C4.22"), "SPCC16C4.22")
        self.assertEqual(parse_gff3_gene_id(parse_gff3_attributes("ID=gene:ENSMUSG00000002147;Name=foo")), "ENSMUSG00000002147")

    def test_ensembl_genomes_species_use_division_ftp_roots(self) -> None:
        self.assertEqual(
            gff3_directory_url("oryza_sativa", 1),
            "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/current/gff3/oryza_sativa/",
        )


if __name__ == "__main__":
    unittest.main()