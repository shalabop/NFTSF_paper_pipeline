"""
data/
=====
Unified data module for the NFTSF comparison pipeline.

All data loading, preprocessing, normalisation, and splitting live here.
The single public entry point is :func:`data.loader.get_dataloader`, which
accepts a merged config dict and returns model-ready data loaders.

Sub-modules
-----------
canonical       Build / load the canonical .npz dataset format.
preprocessing   Z-score normalisation and sliding-window segmentation.
splitter        Trajectory-level and segment-level train/val/test splits.
loader          Unified get_dataloader(config) dispatcher.
adapters        Format adapters for individual model families.
"""
