"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""

from typing import Any


def recursive_to(x: Any, target: Any):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transferred to the target device.
    """
    import torch

    def move(value: Any):
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        if isinstance(value, torch.Tensor):
            return value.to(target)
        if isinstance(value, list):
            return [move(i) for i in value]
        return value

    return move(x)


def __getattr__(name: str):
    if name == "MeshRenderer":
        from .mesh_renderer import MeshRenderer

        globals()[name] = MeshRenderer
        return MeshRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MeshRenderer", "recursive_to"]
