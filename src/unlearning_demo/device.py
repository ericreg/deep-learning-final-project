"""Accelerator detection helpers.

PyTorch exposes ROCm devices through the ``torch.cuda`` namespace, so ROCm
support mostly means avoiding CUDA-only packages such as bitsandbytes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .imports import require


@dataclass(frozen=True)
class AcceleratorInfo:
    available: bool
    backend: str
    device_arg: str
    name: str
    total_gib: float
    detail: str


def get_accelerator_info() -> AcceleratorInfo:
    torch = require("torch", "Install a PyTorch build compatible with your accelerator.")
    hip_version = getattr(torch.version, "hip", None)
    cuda_version = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        backend = "rocm" if hip_version else "cuda"
        version = hip_version if hip_version else cuda_version
        return AcceleratorInfo(
            available=True,
            backend=backend,
            device_arg="cuda:0",
            name=props.name,
            total_gib=props.total_memory / (1024**3),
            detail=f"{backend}={version} gpu={props.name}",
        )
    backend = "rocm-build" if hip_version else "none"
    return AcceleratorInfo(
        available=False,
        backend=backend,
        device_arg="cpu",
        name="none",
        total_gib=0.0,
        detail="No PyTorch accelerator is available. For ROCm, install a ROCm-enabled PyTorch build.",
    )


def require_accelerator() -> AcceleratorInfo:
    info = get_accelerator_info()
    if not info.available:
        raise RuntimeError(info.detail)
    return info


def is_rocm() -> bool:
    return get_accelerator_info().backend == "rocm"


def empty_accelerator_cache() -> None:
    torch = require("torch")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def torch_dtype_from_name(dtype: str):
    torch = require("torch")
    normalized = dtype.lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")
