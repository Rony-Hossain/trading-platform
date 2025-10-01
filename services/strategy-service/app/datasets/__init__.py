"""
Dataset building module for machine learning in trading strategies.

Provides tools for creating high-quality labeled datasets using
advanced techniques from quantitative finance.
"""

from .builder import (
    DatasetConfig,
    DatasetMetadata,
    FeatureEngineer,
    DatasetBuilder,
    create_training_dataset,
    create_meta_labeling_dataset
)

__all__ = [
    'DatasetConfig',
    'DatasetMetadata', 
    'FeatureEngineer',
    'DatasetBuilder',
    'create_training_dataset',
    'create_meta_labeling_dataset'
]