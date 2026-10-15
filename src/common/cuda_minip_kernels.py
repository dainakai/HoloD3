"""Optional CUDA reduction of a complex reconstruction into its central MinIP."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc

from src.common.cuda_roi_kernels import _check_cuda, _enum_value, _one

_SOURCE = r"""
extern "C" __global__ void minip_update(
    const float2* __restrict__ field, float* __restrict__ projection,
    int image_side, int field_side, int offset
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)image_side * image_side) return;
    const long long row = i / image_side + offset;
    const long long col = i % image_side + offset;
    const float2 v = field[row * field_side + col];
    // Match Torch's separate square/add operations, including subnormal values.
    const float intensity = __fadd_rn(__fmul_rn(v.x, v.x), __fmul_rn(v.y, v.y));
    const float clipped = isnan(intensity) ? intensity : fminf(fmaxf(intensity, 0.0f), 1.0f);
    const float previous = projection[i];
    // torch.minimum propagates NaNs; fminf on its own does not.
    projection[i] = isnan(previous) ? previous : (isnan(clipped) ? clipped : fminf(previous, clipped));
}
"""


@dataclass
class CudaMinIPUpdater:
    device_index: int
    module: object
    function: object

    @classmethod
    def build(cls, device_index: int) -> CudaMinIPUpdater:
        with torch.cuda.device(device_index):
            torch.empty(1, device=f"cuda:{device_index}")
            major, minor = torch.cuda.get_device_capability(device_index)
            program = _one(nvrtc.nvrtcCreateProgram(_SOURCE.encode(), b"minip.cu", 0, [], []), "NVRTC")
            try:
                options = [
                    b"--std=c++17",
                    f"--gpu-architecture=sm_{major}{minor}".encode(),
                    b"--fmad=false",
                    b"--ftz=false",
                ]
                result = nvrtc.nvrtcCompileProgram(program, len(options), options)
                if _enum_value(result[0]) != 0:
                    log = bytearray(int(_one(nvrtc.nvrtcGetProgramLogSize(program), "NVRTC")))
                    _check_cuda(nvrtc.nvrtcGetProgramLog(program, log), "NVRTC")
                    raise RuntimeError(log.decode(errors="replace"))
                data = bytearray(int(_one(nvrtc.nvrtcGetCUBINSize(program), "NVRTC")))
                _check_cuda(nvrtc.nvrtcGetCUBIN(program, data), "NVRTC")
            finally:
                _check_cuda(nvrtc.nvrtcDestroyProgram(program), "NVRTC")
            module = _one(cu.cuModuleLoadData(bytes(data)))
            function = _one(cu.cuModuleGetFunction(module, b"minip_update"))
        return cls(device_index, module, function)

    def update(self, field: torch.Tensor, projection: torch.Tensor, offset: int) -> None:
        device = torch.device(f"cuda:{self.device_index}")
        if field.device != device or projection.device != device:
            raise ValueError("MinIP buffers must be on the kernel's CUDA device")
        if field.dtype != torch.complex64 or projection.dtype != torch.float32:
            raise ValueError("MinIP requires complex64 fields and float32 projections")
        if not field.is_contiguous() or not projection.is_contiguous():
            raise ValueError("MinIP buffers must be contiguous")
        if field.ndim != 2 or projection.ndim != 2 or field.shape[0] != field.shape[1]:
            raise ValueError("MinIP requires square two-dimensional buffers")
        side = int(projection.shape[0])
        if side < 1 or projection.shape[1] != side or offset < 0 or offset + side > field.shape[0]:
            raise ValueError("MinIP crop lies outside the field")
        params = (
            (field.data_ptr(), projection.data_ptr(), side, int(field.shape[1]), int(offset)),
            (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int),
        )
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device).cuda_stream
            _check_cuda(cu.cuLaunchKernel(self.function, (side * side + 255) // 256, 1, 1, 256, 1, 1, 0, stream, params, 0))
