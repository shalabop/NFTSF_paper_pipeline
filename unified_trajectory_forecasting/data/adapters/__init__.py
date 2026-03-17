"""
data/adapters/
==============
Format adapters that convert the canonical ``.npz`` dataset into the
model-family-specific data structures expected by each training engine.

nftsf_adapter    → PyTorch DataLoaders of (context, target) segment tensors
gluonts_adapter  → GluonTS InMemoryDataset objects
"""
