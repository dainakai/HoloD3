#!/usr/bin/env python3
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
import torch
from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc
from cuda.bindings import runtime as cudart


CUDA_SOURCE = r"""
extern "C" __global__ void median_u8_time_kernel_u8hist(
    const unsigned char* __restrict__ stack,
    unsigned char* __restrict__ out,
    int frames,
    int height,
    int width
) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int total = height * width;
    if (pixel >= total) return;

    unsigned char hist[256];
    #pragma unroll
    for (int v = 0; v < 256; ++v) {
        hist[v] = 0;
    }

    for (int t = 0; t < frames; ++t) {
        unsigned char value = stack[t * total + pixel];
        hist[value] += 1;
    }

    int rank = (frames - 1) >> 1;
    int cumulative = 0;
    #pragma unroll
    for (int v = 0; v < 256; ++v) {
        cumulative += hist[v];
        if (cumulative > rank) {
            out[pixel] = (unsigned char)v;
            return;
        }
    }
    out[pixel] = 255;
}

extern "C" __global__ void median_u8_time_kernel_u16hist(
    const unsigned char* __restrict__ stack,
    unsigned char* __restrict__ out,
    int frames,
    int height,
    int width
) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int total = height * width;
    if (pixel >= total) return;

    unsigned short hist[256];
    #pragma unroll
    for (int v = 0; v < 256; ++v) {
        hist[v] = 0;
    }

    for (int t = 0; t < frames; ++t) {
        unsigned char value = stack[t * total + pixel];
        hist[value] += 1;
    }

    int rank = (frames - 1) >> 1;
    int cumulative = 0;
    #pragma unroll
    for (int v = 0; v < 256; ++v) {
        cumulative += hist[v];
        if (cumulative > rank) {
            out[pixel] = (unsigned char)v;
            return;
        }
    }
    out[pixel] = 255;
}
"""


def check_cuda(result):
    if isinstance(result, tuple):
        err = result[0]
        values = result[1:]
    else:
        err = result
        values = ()
    if err:
        try:
            name = cuda.cuGetErrorName(err)[1].decode()
            desc = cuda.cuGetErrorString(err)[1].decode()
            raise RuntimeError(f"CUDA error {name}: {desc}")
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"CUDA error {err}") from exc
    if len(values) == 1:
        return values[0]
    return values


def check_nvrtc(result):
    if isinstance(result, tuple):
        err = result[0]
        values = result[1:]
    else:
        err = result
        values = ()
    if err:
        msg = nvrtc.nvrtcGetErrorString(err)[1].decode()
        raise RuntimeError(f"NVRTC error: {msg}")
    if len(values) == 1:
        return values[0]
    return values


@dataclass
class MedianKernelU8:
    function_u8hist: object
    function_u16hist: object

    @classmethod
    def compile(cls, device_id: int = 0) -> "MedianKernelU8":
        check_cuda(cudart.cudaFree(0))
        major = check_cuda(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device_id))
        minor = check_cuda(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device_id))
        program = check_nvrtc(nvrtc.nvrtcCreateProgram(CUDA_SOURCE.encode(), b"median_u8_time.cu", 0, None, None))
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
        function_u8hist = check_cuda(cuda.cuModuleGetFunction(module, b"median_u8_time_kernel_u8hist"))
        function_u16hist = check_cuda(cuda.cuModuleGetFunction(module, b"median_u8_time_kernel_u16hist"))
        return cls(function_u8hist=function_u8hist, function_u16hist=function_u16hist)

    def median(self, stack: torch.Tensor, block_size: int = 256) -> torch.Tensor:
        if not stack.is_cuda:
            raise ValueError("stack must be a CUDA tensor")
        if stack.dtype != torch.uint8:
            raise ValueError(f"stack must be torch.uint8, got {stack.dtype}")
        if stack.ndim != 3:
            raise ValueError(f"stack must have shape (frames, height, width), got {tuple(stack.shape)}")
        if not stack.is_contiguous():
            stack = stack.contiguous()

        frames, height, width = (int(v) for v in stack.shape)
        if not 1 <= frames <= 65535:
            raise ValueError("CUDA window median requires between 1 and 65535 frames")
        out = torch.empty((height, width), device=stack.device, dtype=torch.uint8)
        total = height * width
        grid = (total + block_size - 1) // block_size

        stack_ptr = ctypes.c_void_p(stack.data_ptr())
        out_ptr = ctypes.c_void_p(out.data_ptr())
        frames_arg = ctypes.c_int(frames)
        height_arg = ctypes.c_int(height)
        width_arg = ctypes.c_int(width)
        args = (ctypes.c_void_p * 5)()
        args[0] = ctypes.cast(ctypes.byref(stack_ptr), ctypes.c_void_p)
        args[1] = ctypes.cast(ctypes.byref(out_ptr), ctypes.c_void_p)
        args[2] = ctypes.cast(ctypes.byref(frames_arg), ctypes.c_void_p)
        args[3] = ctypes.cast(ctypes.byref(height_arg), ctypes.c_void_p)
        args[4] = ctypes.cast(ctypes.byref(width_arg), ctypes.c_void_p)

        stream = torch.cuda.current_stream(stack.device).cuda_stream
        function = self.function_u8hist if frames <= 255 else self.function_u16hist
        check_cuda(cuda.cuLaunchKernel(function, grid, 1, 1, block_size, 1, 1, 0, stream, args, 0))
        return out


_KERNEL_CACHE: dict[int, MedianKernelU8] = {}


def median_u8_time(stack: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    device_id = stack.device.index if stack.device.index is not None else torch.cuda.current_device()
    kernel = _KERNEL_CACHE.get(device_id)
    if kernel is None:
        kernel = MedianKernelU8.compile(device_id=device_id)
        _KERNEL_CACHE[device_id] = kernel
    return kernel.median(stack, block_size=block_size)
