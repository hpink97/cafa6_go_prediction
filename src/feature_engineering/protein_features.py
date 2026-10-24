"""Biophysical, amino-acid composition, and topology features."""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("AILMFWVYV")
CHARGED = set("DEKR")
POLAR = set("DEHKNQRST")
HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
BASIC_3_OF_5_WINDOW = (
    r"(?:[KR]{3}..|[KR]{2}.[KR].|[KR]{2}..[KR]|[KR].[KR]{2}.|"
    r"[KR].[KR].[KR]|[KR]..[KR]{2}|.[KR]{3}.|.[KR]{2}.[KR]|"
    r".[KR].[KR]{2}|..[KR]{3})"
)
# Curated high-specificity PROSITE signatures (accession noted), translated to regex.
# Restricted to domain/active-site/binding patterns; PTM sites (PS00001-PS00009) are excluded as too promiscuous.
PROSITE_PATTERNS = {
    "prosite_protein_kinase_atp_binding": re.compile(  # PS00107
        r"[LIV]G[^P]G[^P][FYWMGSTNH][SGA][^PW][LIVCAT][^PD].[GSTACLIVMFY].{5,18}[LIVMFYWCSTAR][AIVP][LIVMFAGCKR]K"
    ),
    "prosite_st_kinase_active_site": re.compile(r"[LIVMFYC].[HY].D[LIVMFY]K.{2}N[LIVMFYCT]{3}"),  # PS00108
    "prosite_abc_transporter": re.compile(  # PS00211
        r"[LIVMFYC][SA][SAPGLVFYKQH]G[DENQMW][KRQASPCLIMFW][KRNQSTAVM][KRACLVM][LIVMFYPAN][^PHY][LIVMFW][SAGCLIVP][^FYWHP][^KRHP][LIVMFYWSTA]"
    ),
    "prosite_leucine_zipper": re.compile(r"L.{6}L.{6}L.{6}L"),  # PS00029
    "prosite_ploop_ntp_binding": re.compile(r"[AG].{4}GK[ST]"),  # PS00017 ATP/GTP-binding (Walker A)
    "prosite_zinc_metallopeptidase": re.compile(  # PS00142 neutral zinc metallopeptidase HExxH
        r"[GSTALIVN][^PCHR][^KND]HE[LIVMFYW][^DEHRKP]H[^EKPC][LIVMFYWGSPQ]"
    ),
    "prosite_cytochrome_c_heme": re.compile(r"C[^CPWHF][^CPWR]CH[^CFYW]"),  # PS00190 CxxCH heme attachment
    "prosite_4fe4s_ferredoxin": re.compile(r"C.{2}C.{2}C.{3}C[PEG]"),  # PS00198 4Fe-4S iron-sulfur cluster
    "prosite_egf_like_domain": re.compile(r"C.C.{2}[GP][FYW].{4,8}C"),  # PS00022 EGF-like cysteine spacing
    "prosite_thioredoxin_active_site": re.compile(  # PS00194 redox-active CxxC
        r"[LIVMF][LIVMSTA].[LIVMFYC][FYWSTHE].{2}[FYWGTN]C[GATPLVE][PHYWSTA]C"
    ),
    "prosite_bipartite_nuclear_localization_signal": re.compile(  # PS50079; NLS_BP
        r"(?=([KR]{2}.{10}" + BASIC_3_OF_5_WINDOW + r"))"
    ),
}


def _clean_sequence(sequence: str) -> str:
    """Keep canonical residues so ProtParam and composition are well-defined."""
    return "".join(residue for residue in sequence.upper() if residue in AMINO_ACIDS)


def _fraction(sequence: str, residues: set[str]) -> float:
    return sum(residue in residues for residue in sequence) / len(sequence)


def _hydrophobic_segments(sequence: str, window: int = 19, threshold: float = 1.6) -> int:
    if len(sequence) < window:
        return 0

    covered = [False] * len(sequence)
    for start in range(len(sequence) - window + 1):
        mean_hydropathy = sum(HYDROPATHY[residue] for residue in sequence[start : start + window]) / window
        if mean_hydropathy >= threshold:
            for index in range(start, start + window):
                covered[index] = True

    segments = 0
    index = 0
    while index < len(covered):
        if not covered[index]:
            index += 1
            continue
        end = index
        while end < len(covered) and covered[end]:
            end += 1
        if end - index >= 15:
            segments += 1
        index = end
    return segments


def _signal_peptide_score(sequence: str) -> float:
    """Score N-terminal hydrophobicity; this is a screening heuristic, not a predictor."""
    n_terminal = sequence[:30]
    if len(n_terminal) < 8:
        return 0.0
    best = 0.0
    for start in range(min(8, len(n_terminal))):
        window = n_terminal[start : start + 12]
        if len(window) == 12:
            best = max(best, sum(HYDROPATHY[residue] for residue in window) / len(window))
    return max(0.0, min(1.0, (best - 0.5) / 3.5))


def _heptad_periodicity(sequence: str) -> float:
    if len(sequence) < 14:
        return 0.0
    hydrophobic = [residue in HYDROPHOBIC for residue in sequence]
    matches = sum(left == right for left, right in zip(hydrophobic, hydrophobic[7:]))
    return matches / (len(hydrophobic) - 7)


def _motif_features(sequence: str) -> dict[str, float]:
    """Return terminal-signal and curated PROSITE motif flags as sequence-only proxies."""
    features: dict[str, float] = {}
    terminal_features = {
        "kdel_like": sequence.endswith("KDEL"),
        "pts1_like": bool(re.search(r"[SAC][KRH][LM]$", sequence)),
        "caax_like": bool(re.search(r"C[AILMVPF][AILMVPF].$", sequence)),
    }
    features.update({f"motif_{name}_present": float(present) for name, present in terminal_features.items()})
    features["motif_n_myristoylation_like_present"] = float(len(sequence) >= 6 and sequence[1] == "G" and sequence[5] in "ASTCN")

    for name, pattern in PROSITE_PATTERNS.items():
        features[f"{name}_present"] = float(pattern.search(sequence) is not None)
    return features


def extract_features(sequence: str, protein_id: str | None = None) -> dict[str, float | str]:
    """Extract one feature row from a protein sequence."""
    cleaned = _clean_sequence(sequence)
    if not cleaned:
        raise ValueError("Sequence contains no canonical amino-acid residues")

    analysis = ProteinAnalysis(cleaned)
    length = len(cleaned)
    charge_fraction = _fraction(cleaned, CHARGED)
    mean_hydropathy = analysis.gravy()
    net_charge = cleaned.count("K") + cleaned.count("R") - cleaned.count("D") - cleaned.count("E")
    disorder_score = charge_fraction - max(mean_hydropathy, 0.0) / 10.0

    features: dict[str, float | str] = {
        "protein_id": protein_id or "",
        "length": length,
        "molecular_weight": analysis.molecular_weight(),
        "isoelectric_point": analysis.isoelectric_point(),
        "gravy": mean_hydropathy,
        "aromaticity": analysis.aromaticity(),
        "instability_index": analysis.instability_index(),
        "aliphatic_index": sum(cleaned.count(residue) * coefficient for residue, coefficient in {"A": 1, "V": 2.9, "I": 3.9, "L": 3.9}.items()) / length * 100,
        "net_charge": net_charge,
        "charge_density": net_charge / length,
        "fraction_cysteine": cleaned.count("C") / length,
        "fraction_charged": charge_fraction,
        "fraction_polar": _fraction(cleaned, POLAR),
        "fraction_helix": analysis.secondary_structure_fraction()[0],
        "fraction_turn": analysis.secondary_structure_fraction()[1],
        "fraction_sheet": analysis.secondary_structure_fraction()[2],
        "transmembrane_helix_count": _hydrophobic_segments(cleaned),
        "signal_peptide_score": _signal_peptide_score(cleaned),
        "signal_peptide_like": float(_signal_peptide_score(cleaned) >= 0.65),
        "disorder_score": disorder_score,
        "disorder_like": float(disorder_score >= 0.18),
        "coiled_coil_heptad_periodicity": _heptad_periodicity(cleaned),
    }
    features.update(_motif_features(cleaned))
    features.update({f"aa_fraction_{residue}": cleaned.count(residue) / length for residue in AMINO_ACIDS})
    return features


def extract_features_from_fasta(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Extract features for every FASTA record, preserving record order."""
    rows = []
    for index, record in enumerate(SeqIO.parse(str(path), "fasta")):
        if limit is not None and index >= limit:
            break
        protein_id = record.id.split("|")[1] if "|" in record.id else record.id
        rows.append(extract_features(str(record.seq), protein_id))
    if not rows:
        raise ValueError(f"No FASTA records found in {path}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input FASTA file")
    parser.add_argument("output", type=Path, help="Output feature table (TSV or CSV)")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N records")
    args = parser.parse_args()

    table = extract_features_from_fasta(args.input, args.limit)
    if args.output.suffix.lower() == ".csv":
        table.to_csv(args.output, index=False)
    else:
        table.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()