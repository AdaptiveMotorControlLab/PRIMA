"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license

Minimal ``chumpy`` compatibility for unpickling legacy SMAL model configs.
"""

from __future__ import annotations

from .ch import Ch, ChArray

__all__ = ["Ch", "ChArray"]
