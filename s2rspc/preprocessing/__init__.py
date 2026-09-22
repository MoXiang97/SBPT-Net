"""Candidate preservation, superline construction, and tokenization."""

from .lcc import preserve_lcc_candidates
from .superline import construct_superlines
from .tokenization import tokenize_superlines


def preprocess_point_cloud(*args, **kwargs):
    """Import the end-to-end pipeline lazily to avoid a data/package cycle."""
    from .pipeline import preprocess_point_cloud as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "construct_superlines",
    "preserve_lcc_candidates",
    "preprocess_point_cloud",
    "tokenize_superlines",
]
