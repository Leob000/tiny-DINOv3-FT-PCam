from __future__ import annotations
import io
from typing import Mapping, Tuple, Optional, Dict

import torch
import torch.nn as nn


def state_dict_nbytes(obj) -> int:
    """
    Return the serialized size (bytes) of a model/state_dict saved with torch.save.

    Accepts either:
      - an nn.Module (we'll call .state_dict()), or
      - a state_dict mapping (str -> Tensor).

    Notes:
      - Moves tensors to CPU and drops gradients before saving.
      - Captures parameters + buffers only (no optimizer/activations/code).
    """
    if isinstance(obj, torch.nn.Module):
        sd = obj.state_dict()
    elif isinstance(obj, Mapping):
        sd = obj
    else:
        raise TypeError(
            "state_dict_nbytes expects an nn.Module or a state_dict mapping."
        )

    buf = io.BytesIO()
    cpu_sd = {k: v.detach().to("cpu") for k, v in sd.items()}
    torch.save(cpu_sd, buf)  # zipped pickle; same as saving to disk
    return buf.getbuffer().nbytes


def pretty_bytes(n: int):
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n)
    for u in units:
        if s < 1024 or u == units[-1]:
            return f"{s:.1f} {u}"
        s /= 1024.0


class _InputDTypeWrapper(nn.Module):
    """
    Wrap a model so that inputs passed as `pixel_values` are cast to a desired dtype,
    and outputs are (optionally) cast back to a target dtype for downstream code.
    """

    def __init__(
        self,
        base: nn.Module,
        input_dtype: torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.base = base
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

    def forward(self, *args, **kwargs):
        if "pixel_values" in kwargs:
            x = kwargs["pixel_values"]
            if isinstance(x, torch.Tensor):
                kwargs["pixel_values"] = x.to(dtype=self.input_dtype)
        elif args:
            x0 = args[0]
            if isinstance(x0, torch.Tensor):
                args = (x0.to(dtype=self.input_dtype),) + args[1:]
        out = self.base(*args, **kwargs)
        if self.output_dtype is not None and isinstance(out, torch.Tensor):
            out = out.to(self.output_dtype)
        return out


def _supports_bf16(device: str) -> bool:
    if device == "cuda":
        return bool(
            torch.cuda.is_available()
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
    if device == "cpu":
        # PyTorch supports bfloat16 tensors on CPU (performance varies).
        return True
    if device == "mps":
        # BFloat16 ops are not reliably supported on Apple MPS.
        return False
    return False


def _try_torchao_int8(model_cpu: nn.Module) -> Tuple[nn.Module, Dict[str, str]]:
    """
    Try TorchAO int8 dynamic activation + int8 weight quantization.
    Falls back to weight-only config if the dynamic config isn't available.
    """
    info: Dict[str, str] = {}
    try:
        from torchao.quantization import quantize_

        # Prefer dynamic activations + int8 weights (closest to eager dynamic quant)
        try:
            from torchao.quantization import Int8DynamicActivationInt8WeightConfig

            quantize_(model_cpu, Int8DynamicActivationInt8WeightConfig())
            info.update({"quantization": "int8", "backend": "torchao.dynamic"})
            return model_cpu, info
        except Exception:
            # Fallback to int8 weight-only if dynamic config not present/available
            from torchao.quantization import Int8WeightOnlyConfig

            quantize_(model_cpu, Int8WeightOnlyConfig())
            info.update({"quantization": "int8", "backend": "torchao.weight_only"})
            return model_cpu, info
    except Exception as e:
        # TorchAO not installed or failed — bubble up for legacy fallback
        raise e


def _fallback_eager_int8(model_cpu: nn.Module) -> Tuple[nn.Module, Dict[str, str]]:
    """
    Legacy eager-mode dynamic quantization (deprecated, but kept as a fallback).
    Quantizes nn.Linear with qint8 weights on CPU.
    """
    from torch.ao.quantization import quantize_dynamic  # type: ignore[import]

    qmodel = quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)
    info = {"quantization": "int8", "backend": "ao.quantize_dynamic(deprecated)"}
    return qmodel, info


def apply_quantization(
    model: nn.Module, mode: str, device: str
) -> Tuple[nn.Module, str, Dict[str, str]]:
    """
    Apply requested quantization to `model`.

    Args:
      model: nn.Module on any device/dtype (eval mode recommended).
      mode: 'none' | 'int8' | 'bf16'
      device: string device used by the caller ('cpu' | 'cuda' | 'mps')

    Returns:
      (quantized_model, eval_device, info_dict)

    Notes:
      - 'int8': prefers TorchAO (quantize_) on CPU; if TorchAO is unavailable,
        falls back to deprecated eager dynamic quantization. Model is moved to CPU.
      - 'bf16': casts weights to bfloat16 and wraps model so inputs are cast to bfloat16
        while outputs are converted back to float32 for downstream metrics.
      - Raises a RuntimeError when a mode is incompatible with the current device.
    """
    mode = (mode or "none").lower()
    info: Dict[str, str] = {}

    if mode == "none":
        info.update({"quantization": "none"})
        return model, device, info

    if mode == "int8":
        # CPU-only: move and eval
        model_cpu = model.to("cpu")
        model_cpu.eval()

        # Prefer TorchAO; if unavailable, fall back to eager dynamic
        try:
            qmodel, qinfo = _try_torchao_int8(model_cpu)
            qinfo["eval_device"] = "cpu"
            return qmodel, "cpu", qinfo
        except Exception:
            qmodel, qinfo = _fallback_eager_int8(model_cpu)
            qinfo["eval_device"] = "cpu"
            return qmodel, "cpu", qinfo

    if mode == "bf16":
        if not _supports_bf16(device):
            raise RuntimeError(
                f"BF16 is not supported on device '{device}'. "
                "Use CUDA (Ampere+/Hopper) or CPU, not Apple MPS."
            )
        model = model.to(device)
        model = model.to(dtype=torch.bfloat16)
        wrapped = _InputDTypeWrapper(
            model, input_dtype=torch.bfloat16, output_dtype=torch.float32
        )
        info.update({"quantization": "bf16", "eval_device": device})
        return wrapped, device, info

    raise ValueError(f"Unknown quantization mode: '{mode}'")
