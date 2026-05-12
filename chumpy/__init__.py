"""
Minimal ``chumpy`` compatibility for unpickling legacy SMAL model configs.

Real ``chumpy`` is optional; SMAL pickles reference ``chumpy.ch.Ch``.
"""

from __future__ import annotations

from .ch import Ch, ChArray

__all__ = ["Ch", "ChArray"]
