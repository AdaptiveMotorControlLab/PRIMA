from __future__ import annotations

"""Minimal ``chumpy`` compatibility for unpickling legacy SMAL model configs."""

from .ch import Ch, ChArray, materialize

__all__ = ["Ch", "ChArray", "materialize"]
