"""Bounded, equality-checked CUDA Graph replay for one inference shape."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable

import torch


class CudaGraphScorer:
    """Own at most one graph for one model, frame, and full slice-block shape.

    The caller must consume the returned scores before the next call; replay
    reuses their storage. Different shapes (including partial final blocks)
    run eagerly. Create a new scorer for each frame to bound graph memory.
    Use from one CUDA stream in inference mode with an evaluation-mode model.
    """

    def __init__(self, score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], *, enabled: bool) -> None:
        self.score = score
        self.enabled = enabled
        self.graph: torch.cuda.CUDAGraph | None = None
        self.signature: tuple | None = None
        self.inputs: tuple[torch.Tensor, torch.Tensor] | None = None
        self.output: torch.Tensor | None = None
        self.captures = 0
        self.replays = 0
        self.capture_sec = 0.0
        self.fallback_reason: str | None = None

    @staticmethod
    def _signature(crops: torch.Tensor, diameters: torch.Tensor) -> tuple:
        return (
            crops.device, crops.dtype, tuple(crops.shape), tuple(crops.stride()),
            diameters.device, diameters.dtype, tuple(diameters.shape), tuple(diameters.stride()),
            torch.cuda.current_stream(crops.device).cuda_stream,
        )

    def __call__(self, crops: torch.Tensor, diameters: torch.Tensor) -> torch.Tensor:
        if not self.enabled or crops.device.type != "cuda" or torch.is_grad_enabled() or crops.numel() == 0:
            return self.score(crops, diameters)
        signature = self._signature(crops, diameters)
        if self.signature is not None and signature != self.signature:
            return self.score(crops, diameters)
        if self.graph is None:
            self.signature = signature
            started = time.perf_counter()
            try:
                self._capture(crops, diameters)
            except RuntimeError as exc:
                # Graph availability never changes the requested model backend.
                self.enabled = False
                self.graph = self.inputs = self.output = None
                self.fallback_reason = type(exc).__name__
                warnings.warn(
                    f"CUDA Graph capture unavailable; using the same model eagerly ({type(exc).__name__}).",
                    RuntimeWarning, stacklevel=2,
                )
            finally:
                self.capture_sec += time.perf_counter() - started
            if self.graph is None:
                return self.score(crops, diameters)
            # _capture already replayed and validated this input.
            return self.output
        assert self.inputs is not None
        self.inputs[0].copy_(crops)
        self.inputs[1].copy_(diameters)
        self.graph.replay()
        self.replays += 1
        return self.output

    def _capture(self, crops: torch.Tensor, diameters: torch.Tensor) -> None:
        with torch.cuda.device(crops.device):
            inputs = (crops.clone(), diameters.clone())
            caller = torch.cuda.current_stream(crops.device)
            stream = torch.cuda.Stream(device=crops.device)
            stream.wait_stream(caller)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    reference = self.score(*inputs)
            caller.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.score(*inputs)
            graph.replay()
            if not torch.equal(reference, output):
                raise RuntimeError("CUDA Graph output differs from eager execution")
            self.inputs, self.output, self.graph = inputs, output, graph
            self.captures += 1
            self.replays += 1

    def stats(self) -> dict[str, int | float | str | None]:
        return {
            "captures": self.captures, "replays": self.replays,
            "capture_sec": self.capture_sec, "fallback_reason": self.fallback_reason,
        }
