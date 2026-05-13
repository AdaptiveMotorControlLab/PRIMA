"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""

"""``chumpy.ch`` namespace expected by legacy SMAL pickles."""

from __future__ import annotations

import numpy as np


class Ch:
    """Minimal stand-in for ``chumpy.ch.Ch`` (unpickling only)."""

    def __init__(self, *args, **kwargs):
        self._data = None
        if args:
            self._data = np.asarray(args[0])

    def r(self):
        if self._data is None:
            return np.zeros((), dtype=np.float32)
        return np.asarray(self._data)


class ChArray(np.ndarray):
    """Minimal stand-in for ``chumpy.ch.ChArray``."""

    pass


__all__ = ["Ch", "ChArray"]
