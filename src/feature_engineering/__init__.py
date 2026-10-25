"""Feature extraction utilities for CAFA6 protein sequences."""

from .sequence.protein_features import extract_features, extract_features_from_fasta

__all__ = ["extract_features", "extract_features_from_fasta"]