"""Optional, scoped cuFFT inverse with normalization fused into its final store."""

from __future__ import annotations

import ctypes
import time
import warnings

import torch


class NormalizedInversePlan:
    """One square complex64 transform, one output, and one owning CUDA stream.

    Results borrow the plan's output until the next execution. Consume them on
    the owning stream before reuse. No input mutation, global cache or load
    callback is involved. Close after all uses, including exceptional exits.
    """

    def __init__(self, side: int, device: torch.device) -> None:
        from cuda.bindings import nvrtc
        from nvmath.bindings import cufft

        from src.common.cuda_roi_kernels import _check_cuda, _enum_value, _one

        self.handle = None
        self.side, self.device = side, torch.device(device)
        self.cufft = cufft
        if side < 1 or self.device.type != "cuda" or self.device.index is None:
            raise ValueError("A positive FFT side and explicit CUDA device are required")
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.current_stream(self.device)
            self.output = torch.empty((side, side), dtype=torch.complex64, device=self.device)
            scale = float(torch.tensor(1.0 / (side * side), dtype=torch.float32)).hex() + "f"
            # Match Torch's complex multiplication by (scale, 0), including
            # Inf * 0 cross-products, signed values and subnormal arithmetic.
            source = f"""
            __device__ void normalize_store(void* output, unsigned long long i,
                                           float2 value, void*, void*) {{
                ((float2*)output)[i] = make_float2(
                    __fmaf_rn(value.x, {scale}, -__fmul_rn(value.y, 0.0f)),
                    __fmaf_rn(value.x, 0.0f, __fmul_rn(value.y, {scale})));
            }}
            """
            major, minor = torch.cuda.get_device_capability(self.device)
            program = _one(nvrtc.nvrtcCreateProgram(source.encode(), b"ifft_store.cu", 0, [], []), "NVRTC")
            try:
                options = [b"--std=c++17", f"--gpu-architecture=compute_{major}{minor}".encode(),
                           b"-dlto", b"--fmad=false", b"--ftz=false"]
                status = nvrtc.nvrtcCompileProgram(program, len(options), options)
                if _enum_value(status[0]) != 0:
                    log = bytearray(int(_one(nvrtc.nvrtcGetProgramLogSize(program), "NVRTC")))
                    _check_cuda(nvrtc.nvrtcGetProgramLog(program, log), "NVRTC")
                    raise RuntimeError(log.decode(errors="replace"))
                data = bytearray(int(_one(nvrtc.nvrtcGetLTOIRSize(program), "NVRTC")))
                _check_cuda(nvrtc.nvrtcGetLTOIR(program, data), "NVRTC")
                self.lto = bytes(data)
            finally:
                _check_cuda(nvrtc.nvrtcDestroyProgram(program), "NVRTC")
            # The symbol and LTO bytes must remain alive through plan creation.
            self.symbol = ctypes.create_string_buffer(b"normalize_store")
            self.handle = cufft.create()
            try:
                cufft.set_auto_allocation(self.handle, 0)
                cufft.xt_set_jit_callback(self.handle, ctypes.addressof(self.symbol), self.lto,
                                         len(self.lto), cufft.XtCallbackType.ST_COMPLEX, [0])
                # CUDA_C_32F=4; match Torch's contiguous cufftXtMakePlanMany layout.
                workspace_bytes = cufft.xt_make_plan_many(
                    self.handle, 2, [side, side], 0, 1, 1, 4, 0, 1, 1, 4, 1, 4,
                )
                self.workspace = torch.empty(workspace_bytes, device=self.device, dtype=torch.uint8)
                cufft.set_work_area(self.handle, self.workspace.data_ptr())
                cufft.set_stream(self.handle, self.stream.cuda_stream)
            except Exception:
                self.close()
                raise

    def execute(self, frequency: torch.Tensor) -> torch.Tensor:
        if self.handle is None:
            raise RuntimeError("The inverse FFT plan is closed")
        if (frequency.device != self.device or frequency.dtype != torch.complex64
                or frequency.shape != (self.side, self.side) or not frequency.is_contiguous()
                or frequency.requires_grad):
            raise ValueError("Expected a contiguous complex64 field on the plan's device without gradients")
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("The inverse FFT plan belongs to a different CUDA stream")
        size = frequency.numel() * frequency.element_size()
        if abs(frequency.data_ptr() - self.output.data_ptr()) < size:
            raise ValueError("Inverse FFT input must not overlap its output")
        with torch.cuda.device(self.device):
            self.cufft.xt_exec(self.handle, frequency.data_ptr(), self.output.data_ptr(), 1)
        return self.output

    def close(self) -> None:
        if self.handle is not None:
            with torch.cuda.device(self.device):
                self.stream.synchronize()
                self.cufft.destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ReconstructionIFFT:
    """Select and validate the optional plan before processing any frame.

    Unsupported libraries, shapes or differing preflight values select Torch.
    A failure during actual reconstruction is propagated, never retried against
    a partially processed frame. CPU, other dtypes and autograd retain Torch.
    """

    def __init__(self, side: int, device: torch.device, *, enabled: bool) -> None:
        self.side, self.device, self.enabled = side, torch.device(device), enabled
        self.plan: NormalizedInversePlan | None = None
        self.active = False
        self.setup_sec = 0.0
        self.fallback_reason: str | None = None

    def __enter__(self):
        if not self.enabled or self.device.type != "cuda":
            return self
        started = time.perf_counter()
        try:
            with torch.inference_mode(), torch.cuda.device(self.device):
                self.plan = NormalizedInversePlan(self.side, self.device)
                generator = torch.Generator(device=self.device).manual_seed(522)
                sample = torch.randn((self.side, self.side), dtype=torch.complex64,
                                     device=self.device, generator=generator)
                if not torch.equal(self.plan.execute(sample), torch.fft.ifft2(sample)):
                    raise RuntimeError("Normalized IFFT differs from Torch in the preflight check")
                self.active = True
        except Exception as exc:
            if self.plan is not None:
                self.plan.close()
                self.plan = None
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"Fused IFFT normalization unavailable; using Torch ({self.fallback_reason}).",
                          RuntimeWarning, stacklevel=2)
        finally:
            self.setup_sec = time.perf_counter() - started
        return self

    def __call__(self, frequency: torch.Tensor) -> torch.Tensor:
        if (self.plan is None or frequency.dtype != torch.complex64 or frequency.requires_grad
                or torch.is_grad_enabled() or frequency.shape != (self.side, self.side)
                or not frequency.is_contiguous() or frequency.device != self.device):
            return torch.fft.ifft2(frequency)
        return self.plan.execute(frequency)

    def __exit__(self, *_):
        if self.plan is not None:
            self.plan.close()
            self.plan = None
