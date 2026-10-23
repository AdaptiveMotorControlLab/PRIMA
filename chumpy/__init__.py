from __future__ import annotations
"""
PRIMA: Boosting Animal Mesh Recovery with Taxonomic Priors and Adaptation-Driven Data Bootstrapping

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Taxonomic Priors and Adaptation-Driven Data Bootstrapping"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""


"""Minimal ``chumpy`` compatibility for unpickling legacy SMAL model configs."""

from .ch import Ch, ChArray, materialize

__all__ = ["Ch", "ChArray", "materialize"]
