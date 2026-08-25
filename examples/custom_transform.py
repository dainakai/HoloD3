"""Example HoloD3 acquisition transform.

Copy this file next to your acquisition.yaml and reference a function as
`custom_transform.py:subtract_offset`. Transform inputs and outputs are 2D
float32 arrays in the inclusive range [0, 1].
"""

from __future__ import annotations

import numpy as np


def subtract_offset(image: np.ndarray, *, offset: float = 0.0) -> np.ndarray:
    """Subtract a fixed normalized intensity offset and preserve the valid range."""

    return np.clip(image - float(offset), 0.0, 1.0).astype(np.float32)
