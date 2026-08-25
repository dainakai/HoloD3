"""Restricted PyTorch checkpoint loading for state-dictionary artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.torch_version import TorchVersion


def load_state_checkpoint(path: str | Path, *, map_location: Any) -> dict[str, Any]:
    """Load tensor/state metadata without allowing arbitrary pickle globals.

    Preserved training checkpoints contain ``TorchVersion`` metadata, so that
    inert value class is explicitly allow-listed. Model classes and executable
    callables remain disallowed.
    """

    resolved = Path(path)
    with torch.serialization.safe_globals([TorchVersion]):
        value = torch.load(resolved, map_location=map_location, weights_only=True)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Checkpoint must contain a mapping: {resolved}")
    return dict(value)


__all__ = ["load_state_checkpoint"]
