#!/usr/bin/env python3
"""Download and verify HoloD3 checkpoints from the private Hugging Face repo."""

from __future__ import annotations

import sys

from holod3.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["fetch-models", *sys.argv[1:]]))
