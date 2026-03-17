"""
models/registry.py
==================
Maps model name strings to their wrapper classes, enabling the unified
training script to instantiate any model by name.
"""

from __future__ import annotations

from typing import Type

from models.base import BaseModel


def get_model(name: str) -> Type[BaseModel]:
    """
    Return the wrapper class for *name*.

    Parameters
    ----------
    name : Model identifier string, e.g. ``"nftsf"``, ``"tsdiff_q"``.

    Returns
    -------
    Subclass of BaseModel (not yet instantiated).

    Raises
    ------
    ValueError for unknown model names.
    """
    name = name.lower()

    if name == "nftsf":
        from models.nftsf import NFTSFModel
        return NFTSFModel

    if name in ("tsdiff_q", "tsdiff_ms", "tsdiff_cond"):
        from models.tsdiff import TSDiffModel
        return TSDiffModel

    if name == "csdi":
        from models.csdi import CSDIModel
        return CSDIModel

    if name == "ratd":
        from models.ratd import RATDModel
        return RATDModel

    if name == "nsdiff":
        from models.nsdiff import NsDiffModel
        return NsDiffModel

    if name == "arima":
        from models.arima import ARIMAModel
        return ARIMAModel

    raise ValueError(
        f"Unknown model '{name}'.  Available models: "
        "nftsf, tsdiff_q, tsdiff_ms, tsdiff_cond, csdi, ratd, nsdiff, arima."
    )
