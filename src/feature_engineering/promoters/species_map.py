"""Anchor species available for Ensembl REST-based promoter extraction.

Maps NCBI taxon ID (as seen in train_taxonomy.tsv / test FASTA headers) to the
Ensembl species name used by the unified rest.ensembl.org API.
"""

from __future__ import annotations

ANCHOR_SPECIES: dict[str, str] = {
    "9606": "homo_sapiens",
    "10090": "mus_musculus",
    "10116": "rattus_norvegicus",
    "7955": "danio_rerio",
    "7227": "drosophila_melanogaster",
    "6239": "caenorhabditis_elegans",
    "559292": "saccharomyces_cerevisiae",
    "3702": "arabidopsis_thaliana",
    "284812": "schizosaccharomyces_pombe",
    "39947": "oryza_sativa",
    "9913": "bos_taurus",
    "9031": "gallus_gallus",
    "44689": "dictyostelium_discoideum",
    "237561": "candida_albicans",
    "227321": "aspergillus_nidulans",
    "9823": "sus_scrofa",
    "9986": "oryctolagus_cuniculus",
    # High-value missing taxa from the test set that are supported in Ensembl and
    # can be used without operon-specific handling.
    "8355": "xenopus_laevis",
    "8364": "xenopus_tropicalis",
    "9541": "macaca_fascicularis",
    "9598": "pan_troglodytes",
    "9601": "pongo_abelii",
    "9615": "canis_lupus_familiaris",
    # Additional high-count test taxa with exact Ensembl / Ensembl Genomes FTP support.
    "367110": "neurospora_crassa",
    "330879": "aspergillus_fumigatus",
    "39946": "oryza_sativa",
    "4577": "zea_mays",
    "284591": "yarrowia_lipolytica",
    "510516": "aspergillus_oryzae",
    "6238": "caenorhabditis_briggsae",
    "4081": "solanum_lycopersicum_gca000188115v5cm",
    "9940": "ovis_aries",
    "341663": "aspergillus_terreus",
    "3847": "glycine_max",
    "307796": "saccharomyces_cerevisiae",
    "4113": "solanum_tuberosum",
    "229533": "fusarium_graminearum",
    "425011": "aspergillus_niger",
    "4565": "triticum_aestivum",
    "242507": "pyricularia_oryzae",
    "9544": "macaca_mulatta",
    "331117": "aspergillus_fischeri",
    "3055": "chlamydomonas_reinhardtii",
    "344612": "aspergillus_clavatus",
    "4513": "hordeum_vulgare",
    "46245": "drosophila_pseudoobscura_gca009870125v2rs",
    "10141": "cavia_porcellus",
    "36329": "plasmodium_falciparum",
    "7165": "anopheles_gambiae",
    "9796": "equus_caballus",
    "7159": "aedes_aegypti_lvpagwg",
    "9595": "gorilla_gorilla",
    "451804": "aspergillus_fumigatus",
    "10036": "mesocricetus_auratus",
    "332952": "aspergillus_flavus",
    "10029": "cricetulus_griseus_crigri",
    "7091": "bombyx_mori_gca030269925v1rs",
    "9685": "felis_catus",
    "7244": "drosophila_virilis_gca030788295v1rs",
    "7245": "drosophila_yakuba_gca016746365v2rs",
    "8022": "oncorhynchus_mykiss",
}

# Species where genes are commonly organized into polycistronic operons: a naive
# per-gene upstream window mostly grabs the neighboring gene's coding sequence,
# not a promoter. These need operon-leader resolution (see operons.py) instead
# of the plain per-gene TSS window used for everything in ANCHOR_SPECIES.
#
# These were added because they are among the highest-count taxa in the test set
# and they are compact bacterial genomes where the upstream 3 kb window is often
# not a regulatory region but the trailing coding sequence of the neighboring gene.
OPERON_AWARE_SPECIES: dict[str, str] = {
    "83333": "escherichia_coli_str_k_12_substr_mg1655_gca_000005845",
    "208964": "pseudomonas_aeruginosa_pao1_gca_000006765",
    "224308": "bacillus_subtilis_subsp_subtilis_str_168_gca_000009045",
    "83332": "mycobacterium_tuberculosis_h37rv",
    "99287": "salmonella_enterica",
    "321314": "salmonella_enterica_subsp_enterica_serovar_choleraesuis_str_sc_b67",
    "316385": "escherichia_coli_str_k_12_substr_dh10b",
    "595496": "escherichia_coli_bw2952",
    "160488": "pseudomonas_putida_kt2440",
}

