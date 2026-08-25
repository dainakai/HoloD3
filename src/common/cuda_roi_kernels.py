from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc


def _enum_value(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except TypeError:
        return int(value.value)  # type: ignore[attr-defined]


def _check_cuda(result: object, label: str = "CUDA") -> tuple[object, ...]:
    if isinstance(result, tuple):
        err = result[0]
        values = result[1:]
    else:
        err = result
        values = ()
    if _enum_value(err) != 0:
        raise RuntimeError(f"{label} call failed: {err}")
    return values


def _one(result: object, label: str = "CUDA") -> object:
    values = _check_cuda(result, label)
    if len(values) != 1:
        raise RuntimeError(f"{label} call returned {len(values)} values, expected 1")
    return values[0]


_KERNEL_SRC = r"""
extern "C" __global__ void crop_abs2_meanpad_kernel(
    const float2* __restrict__ field,
    const int* __restrict__ x0,
    const int* __restrict__ y0,
    float* __restrict__ out,
    int nroi,
    int field_width,
    int datlen,
    int offset,
    int crop_size
) {
    __shared__ float sums[256];
    __shared__ int counts[256];

    const int roi = blockIdx.x;
    const int tid = threadIdx.x;
    if (roi >= nroi) {
        return;
    }

    const int base_x = x0[roi];
    const int base_y = y0[roi];
    const int pixels = crop_size * crop_size;

    float local_sum = 0.0f;
    int local_count = 0;
    for (int p = tid; p < pixels; p += blockDim.x) {
        const int yy = p / crop_size;
        const int xx = p - yy * crop_size;
        const int sx = base_x + xx;
        const int sy = base_y + yy;
        if ((unsigned int)sx < (unsigned int)datlen && (unsigned int)sy < (unsigned int)datlen) {
            const float2 v = field[(sy + offset) * field_width + (sx + offset)];
            float intensity = v.x * v.x + v.y * v.y;
            intensity = fminf(fmaxf(intensity, 0.0f), 1.0f);
            local_sum += intensity;
            local_count += 1;
        }
    }

    sums[tid] = local_sum;
    counts[tid] = local_count;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
            counts[tid] += counts[tid + stride];
        }
        __syncthreads();
    }

    const float mean_value = counts[0] > 0 ? sums[0] / (float)counts[0] : 0.0f;
    float* dst = out + (long long)roi * pixels;
    for (int p = tid; p < pixels; p += blockDim.x) {
        const int yy = p / crop_size;
        const int xx = p - yy * crop_size;
        const int sx = base_x + xx;
        const int sy = base_y + yy;
        float value = mean_value;
        if ((unsigned int)sx < (unsigned int)datlen && (unsigned int)sy < (unsigned int)datlen) {
            const float2 v = field[(sy + offset) * field_width + (sx + offset)];
            value = v.x * v.x + v.y * v.y;
            value = fminf(fmaxf(value, 0.0f), 1.0f);
        }
        dst[p] = value;
    }
}
"""


@dataclass
class CudaRoiAbs2Cropper:
    device_index: int
    module: object
    function: object

    @classmethod
    def build(cls, device: torch.device | int | None = None) -> CudaRoiAbs2Cropper:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if device is None:
            device_index = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            device_index = device.index if device.index is not None else torch.cuda.current_device()
        else:
            device_index = int(device)

        with torch.cuda.device(device_index):
            torch.empty(1, device=f"cuda:{device_index}")
            major, minor = torch.cuda.get_device_capability(device_index)
            arch = f"--gpu-architecture=sm_{major}{minor}".encode("ascii")
            program = _one(
                nvrtc.nvrtcCreateProgram(_KERNEL_SRC.encode("utf-8"), b"roi_abs2_crop.cu", 0, [], []),
                "NVRTC",
            )
            options = [b"--std=c++17", arch, b"--use_fast_math"]
            compile_result = nvrtc.nvrtcCompileProgram(program, len(options), options)
            if _enum_value(compile_result[0] if isinstance(compile_result, tuple) else compile_result) != 0:
                log_size = int(_one(nvrtc.nvrtcGetProgramLogSize(program), "NVRTC"))
                log = bytearray(log_size)
                _check_cuda(nvrtc.nvrtcGetProgramLog(program, log), "NVRTC")
                raise RuntimeError(log.decode("utf-8", errors="replace"))

            cubin_size = int(_one(nvrtc.nvrtcGetCUBINSize(program), "NVRTC"))
            if cubin_size > 0:
                image = bytearray(cubin_size)
                _check_cuda(nvrtc.nvrtcGetCUBIN(program, image), "NVRTC")
                module_image = bytes(image)
            else:
                ptx_size = int(_one(nvrtc.nvrtcGetPTXSize(program), "NVRTC"))
                image = bytearray(ptx_size)
                _check_cuda(nvrtc.nvrtcGetPTX(program, image), "NVRTC")
                module_image = bytes(image)

            _check_cuda(cu.cuInit(0), "CUDA")
            module = _one(cu.cuModuleLoadData(module_image), "CUDA")
            function = _one(cu.cuModuleGetFunction(module, b"crop_abs2_meanpad_kernel"), "CUDA")
        return cls(device_index=device_index, module=module, function=function)

    def crop_abs2_meanpad(
        self,
        field: torch.Tensor,
        x0: torch.Tensor,
        y0: torch.Tensor,
        out: torch.Tensor,
        datlen: int,
        offset: int,
    ) -> None:
        if field.device.type != "cuda" or out.device.type != "cuda":
            raise RuntimeError("field and out must be CUDA tensors")
        if field.dtype != torch.complex64:
            raise RuntimeError(f"field dtype must be complex64, got {field.dtype}")
        if out.dtype != torch.float32:
            raise RuntimeError(f"out dtype must be float32, got {out.dtype}")
        if x0.dtype != torch.int32 or y0.dtype != torch.int32:
            raise RuntimeError("x0 and y0 must be int32 CUDA tensors")
        if not field.is_contiguous():
            raise RuntimeError("field must be contiguous")
        if not x0.is_contiguous() or not y0.is_contiguous() or not out.is_contiguous():
            raise RuntimeError("x0, y0, and out must be contiguous")
        if out.ndim != 3 or out.shape[1] != out.shape[2]:
            raise RuntimeError("out must have shape [nroi, crop_size, crop_size]")
        nroi = int(out.shape[0])
        if nroi == 0:
            return
        crop_size = int(out.shape[1])
        field_width = int(field.shape[1])
        if field.ndim != 2 or field.shape[0] != field.shape[1]:
            raise RuntimeError("field must be a square 2-D tensor")
        if int(x0.numel()) != nroi or int(y0.numel()) != nroi:
            raise RuntimeError("x0/y0 length must match out.shape[0]")

        with torch.cuda.device(self.device_index):
            stream = torch.cuda.current_stream(field.device).cuda_stream
            params = (
                (
                    field.data_ptr(),
                    x0.data_ptr(),
                    y0.data_ptr(),
                    out.data_ptr(),
                    nroi,
                    field_width,
                    int(datlen),
                    int(offset),
                    crop_size,
                ),
                (
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                ),
            )
            _check_cuda(cu.cuLaunchKernel(self.function, nroi, 1, 1, 256, 1, 1, 0, stream, params, 0), "CUDA")
