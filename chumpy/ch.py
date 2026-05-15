"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license

``chumpy.ch`` namespace expected by legacy SMAL pickles.
"""

from __future__ import annotations

import numpy as np


class Ch:
    """Minimal stand-in for ``chumpy.ch.Ch`` (unpickling only)."""

    def __init__(self, *args, **kwargs):
        self._data = None
        if args:
            self._data = np.asarray(args[0])

    def _resolve(self) -> np.ndarray:
        # Real chumpy Ch instances store the underlying ndarray on attribute ``x``;
        # legacy pickles unpickle by restoring ``__dict__`` without calling ``__init__``,
        # so try common attribute names before falling back to ``_data``.
        for attr in ("x", "_x", "_data"):
            val = self.__dict__.get(attr)
            if val is not None:
                return np.asarray(val)
        return np.zeros((), dtype=np.float32)

    def __array__(self, dtype=None):
        arr = self._resolve()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    @property
    def r(self) -> np.ndarray:
        return self._resolve()


class ChArray(np.ndarray):
    """Minimal stand-in for ``chumpy.ch.ChArray``."""

    pass


__all__ = ["Ch", "ChArray"]
