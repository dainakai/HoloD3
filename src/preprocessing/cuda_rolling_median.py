#!/usr/bin/env python3
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
import torch
from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc
from cuda.bindings import runtime as cudart

from src.preprocessing.cuda_median_anchor import check_cuda, check_nvrtc


CUDA_SOURCE = r"""
extern "C" __global__ void update_hist_u8_kernel(
    const unsigned char* __restrict__ frames,
    unsigned char* __restrict__ hist,
    int frame_count,
    int total,
    int delta
) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= total) return;
    unsigned char* pixel_hist = hist + pixel * 256;
    for (int t = 0; t < frame_count; ++t) {
        unsigned char value = frames[t * total + pixel];
        if (delta > 0) {
            pixel_hist[value] += 1;
        } else {
            pixel_hist[value] -= 1;
        }
    }
}

extern "C" __global__ void median_from_hist_u8_kernel(
    const unsigned char* __restrict__ hist,
    unsigned char* __restrict__ out,
    int total,
    int frame_count
) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= total) return;
    const unsigned char* pixel_hist = hist + pixel * 256;
    int rank = (frame_count - 1) >> 1;
    int cumulative = 0;
    #pragma unroll
    for (int v = 0; v < 256; ++v) {
        cumulative += pixel_hist[v];
        if (cumulative > rank) {
            out[pixel] = (unsigned char)v;
            return;
        }
    }
    out[pixel] = 255;
}
"""


@dataclass
class RollingMedianU8:
    update_function: object
    median_function: object

    @classmethod
    def compile(cls, device_id: int = 0) -> "RollingMedianU8":
        check_cuda(cudart.cudaFree(0))
        major = check_cuda(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device_id))
        minor = check_cuda(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device_id))
        program = check_nvrtc(nvrtc.nvrtcCreateProgram(CUDA_SOURCE.encode(), b"rolling_median_u8.cu", 0, None, None))
        options = [
            f"--gpu-architecture=sm_{major}{minor}".encode(),
            b"--std=c++17",
            b"--use_fast_math",
        ]
        try:
            check_nvrtc(nvrtc.nvrtcCompileProgram(program, len(options), options))
        except RuntimeError as exc:
            log_size = check_nvrtc(nvrtc.nvrtcGetProgramLogSize(program))
            log = b" " * log_size
            check_nvrtc(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(log.decode()) from exc

        cubin_size = check_nvrtc(nvrtc.nvrtcGetCUBINSize(program))
        cubin = b" " * cubin_size
        check_nvrtc(nvrtc.nvrtcGetCUBIN(program, cubin))
        module = check_cuda(cuda.cuModuleLoadData(np.frombuffer(cubin, dtype=np.uint8)))
        update_function = check_cuda(cuda.cuModuleGetFunction(module, b"update_hist_u8_kernel"))
        median_function = check_cuda(cuda.cuModuleGetFunction(module, b"median_from_hist_u8_kernel"))
        return cls(update_function=update_function, median_function=median_function)

    def update_hist(self, frames: torch.Tensor, hist: torch.Tensor, delta: int, block_size: int = 256) -> None:
        if not frames.is_cuda or not hist.is_cuda:
            raise ValueError("frames and hist must be CUDA tensors")
        if frames.dtype != torch.uint8 or hist.dtype != torch.uint8:
            raise ValueError("frames and hist must be torch.uint8")
        if frames.ndim != 3:
            raise ValueError(f"frames must have shape (n, height, width), got {tuple(frames.shape)}")
        if not frames.is_contiguous():
            frames = frames.contiguous()
        frame_count, height, width = (int(v) for v in frames.shape)
        total = height * width
        if hist.numel() != total * 256:
            raise ValueError(f"hist has {hist.numel()} elements, expected {total * 256}")

        frame_ptr = ctypes.c_void_p(frames.data_ptr())
        hist_ptr = ctypes.c_void_p(hist.data_ptr())
        frame_count_arg = ctypes.c_int(frame_count)
        total_arg = ctypes.c_int(total)
        delta_arg = ctypes.c_int(delta)
        args = (ctypes.c_void_p * 5)()
        args[0] = ctypes.cast(ctypes.byref(frame_ptr), ctypes.c_void_p)
        args[1] = ctypes.cast(ctypes.byref(hist_ptr), ctypes.c_void_p)
        args[2] = ctypes.cast(ctypes.byref(frame_count_arg), ctypes.c_void_p)
        args[3] = ctypes.cast(ctypes.byref(total_arg), ctypes.c_void_p)
        args[4] = ctypes.cast(ctypes.byref(delta_arg), ctypes.c_void_p)

        grid = (total + block_size - 1) // block_size
        stream = torch.cuda.current_stream(frames.device).cuda_stream
        check_cuda(cuda.cuLaunchKernel(self.update_function, grid, 1, 1, block_size, 1, 1, 0, stream, args, 0))

    def median_from_hist(self, hist: torch.Tensor, height: int, width: int, frame_count: int, block_size: int = 256) -> torch.Tensor:
        if not hist.is_cuda or hist.dtype != torch.uint8:
            raise ValueError("hist must be a CUDA torch.uint8 tensor")
        total = int(height) * int(width)
        if hist.numel() != total * 256:
            raise ValueError(f"hist has {hist.numel()} elements, expected {total * 256}")
        out = torch.empty((height, width), device=hist.device, dtype=torch.uint8)

        hist_ptr = ctypes.c_void_p(hist.data_ptr())
        out_ptr = ctypes.c_void_p(out.data_ptr())
        total_arg = ctypes.c_int(total)
        frame_count_arg = ctypes.c_int(int(frame_count))
        args = (ctypes.c_void_p * 4)()
        args[0] = ctypes.cast(ctypes.byref(hist_ptr), ctypes.c_void_p)
        args[1] = ctypes.cast(ctypes.byref(out_ptr), ctypes.c_void_p)
        args[2] = ctypes.cast(ctypes.byref(total_arg), ctypes.c_void_p)
        args[3] = ctypes.cast(ctypes.byref(frame_count_arg), ctypes.c_void_p)

        grid = (total + block_size - 1) // block_size
        stream = torch.cuda.current_stream(hist.device).cuda_stream
        check_cuda(cuda.cuLaunchKernel(self.median_function, grid, 1, 1, block_size, 1, 1, 0, stream, args, 0))
        return out


_KERNEL_CACHE: dict[int, RollingMedianU8] = {}


def get_rolling_median_kernel(device_id: int) -> RollingMedianU8:
    kernel = _KERNEL_CACHE.get(device_id)
    if kernel is None:
        kernel = RollingMedianU8.compile(device_id=device_id)
        _KERNEL_CACHE[device_id] = kernel
    return kernel
