from __future__ import annotations

from pathlib import Path

import pytest
import torch

from holod3.checkpoints import load_state_checkpoint


def test_restricted_checkpoint_loader_accepts_state_mappings(tmp_path: Path) -> None:
    path = tmp_path / "state.pt"
    torch.save({"model": {"weight": torch.ones(2)}, "epoch": 3}, path)
    loaded = load_state_checkpoint(path, map_location="cpu")
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["model"]["weight"], torch.ones(2))


def test_restricted_checkpoint_loader_rejects_executable_globals(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.pt"
    torch.save({"callable": len}, path)
    with pytest.raises(Exception, match="Weights only load failed|Unsupported global"):
        load_state_checkpoint(path, map_location="cpu")
