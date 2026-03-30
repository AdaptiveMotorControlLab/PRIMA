"""Top-level package for PRIMA.

This package contains models, datasets and utilities for
3D animal pose and shape estimation.
"""

from importlib.metadata import PackageNotFoundError, version


try:  # pragma: no cover - best effort during development
	__version__ = version("prima")
except PackageNotFoundError:  # pragma: no cover
	__version__ = "0.0.0"


__all__ = ["__version__"]
