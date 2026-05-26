from __future__ import annotations
"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""


"""``chumpy.ch`` namespace expected by legacy SMAL pickles."""

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
        if self._data is not None:
            return np.asarray(self._data)
        return np.zeros((), dtype=np.float32)

    @property
    def r(self) -> np.ndarray:
        return self._resolve()

    def __array__(self, dtype=None):
        arr = self.r()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr


class ChArray(np.ndarray):
    """Minimal stand-in for ``chumpy.ch.ChArray``."""


def materialize(value, dtype=np.float32) -> np.ndarray:
    """Recursively unwrap ``Ch`` / object arrays from legacy SMAL pickles."""
    if isinstance(value, Ch):
        return np.asarray(value.r(), dtype=dtype)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            flat = [materialize(x, dtype=dtype) for x in value.ravel()]
            return np.stack(flat).reshape(value.shape)
        return np.asarray(value, dtype=dtype)
    if isinstance(value, (list, tuple)):
        return np.asarray([materialize(x, dtype=dtype) for x in value], dtype=dtype)
    return np.asarray(value, dtype=dtype)


__all__ = ["Ch", "ChArray", "materialize"]
