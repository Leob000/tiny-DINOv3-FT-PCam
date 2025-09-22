from __future__ import annotations
import io
from typing import Mapping
import torch


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


def pretty_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n)
    for u in units:
        if s < 1024 or u == units[-1]:
            return f"{s:.1f} {u}"
        s /= 1024.0
