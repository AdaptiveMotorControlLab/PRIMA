"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""

"""Top-level package for PRIMA.

This package contains models, datasets and utilities for
3D animal pose and shape estimation.
"""

from importlib.metadata import PackageNotFoundError, version


try:  # pragma: no cover - best effort during development
	__version__ = version("prima-animal")
except PackageNotFoundError:  # pragma: no cover
	__version__ = "0.0.0"


__all__ = ["__version__"]
