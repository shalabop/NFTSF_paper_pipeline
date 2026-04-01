"""
models/architecture.py
======================
Defines ``create_nfm``, the factory function for the NFTSF Conditional
Normalising Flow model.

Architecture
------------
- Base distribution : DiagGaussian(latent_size)
- K flow blocks, each containing:
    - len(hidden_layers_list) Autoregressive RQS transforms
      (the i-th transform uses ``hidden_layers_list[i]`` hidden layers in its
      conditioner network)
    - 1 LU Linear Permute layer
- Context (x_past, size = context_size) is fed to every spline conditioner.

The returned object is a ``normflows.ConditionalNormalizingFlow`` instance
moved to *device*.
"""

from __future__ import annotations

import torch

import normflows as nf


def create_nfm(
    device: torch.device,
    latent_size: int,
    context_size: int,
    K: int = 6,
    hidden_units: int = 64,
    hidden_layers_list: tuple[int, ...] = (1, 2),
    tail_bound: float = 30.0,
) -> nf.ConditionalNormalizingFlow:
    """
    Build and return a conditional normalising flow model.

    Parameters
    ----------
    device            : Target device (cpu / cuda).
    latent_size       : Dimensionality of the predicted output (n_future).
    context_size      : Dimensionality of the conditioning input (n_past).
    K                 : Number of flow blocks.
    hidden_units      : Width of each conditioner hidden layer.
    hidden_layers_list: Tuple whose *length* = number of RQS transforms per
                        block; each *value* = number of hidden layers in that
                        transform's conditioner network.
    tail_bound        : Tail-bound for the rational-quadratic spline.

    Returns
    -------
    model : ConditionalNormalizingFlow moved to *device*.
    """
    base = nf.distributions.DiagGaussian(latent_size, trainable=False)

    flows: list = []
    for _ in range(K):
        for n_blocks in hidden_layers_list:
            flows.append(
                nf.flows.AutoregressiveRationalQuadraticSpline(
                    latent_size,
                    n_blocks,
                    hidden_units,
                    num_context_channels=context_size,
                    tail_bound=tail_bound,
                )
            )
        flows.append(nf.flows.LULinearPermute(latent_size))

    model = nf.ConditionalNormalizingFlow(q0=base, flows=flows)
    return model.to(device)
