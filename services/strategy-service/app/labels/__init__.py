"""
Labels module for trading strategy labeling systems.

This module provides advanced labeling techniques for machine learning
in quantitative finance, including:

- Triple Barrier Labeling: Create balanced datasets with profit-taking,
  stop-loss, and time-based barriers
- Meta-Labeling: Secondary models to improve signal quality
- Sample Weighting: Handle overlapping labels and imbalanced datasets
"""

from .triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierLabel,
    TripleBarrierLabeler,
    MetaLabelResult,
    MetaLabeler,
    create_meta_labels
)

__all__ = [
    'TripleBarrierConfig',
    'TripleBarrierLabel',
    'TripleBarrierLabeler',
    'MetaLabelResult',
    'MetaLabeler',
    'create_meta_labels'
]