# src/utils/eval_utils.py
from __future__ import annotations
import time
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from src.utils.metrics import eval_binary_scores


def get_device() -> str:
    """
    Prefer Apple MPS on macOS, else CUDA if available, else CPU.
    """
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def try_flops(
    model: nn.Module, img_size: int = 224, device: str = "cpu"
) -> Optional[float]:
    """
    Returns GFLOPs (or None) using ptflops if available.
    Counts FLOPs of the forward(pixel_values=x) interface.
    """
    try:
        from ptflops import get_model_complexity_info

        class Wrap(nn.Module):
            def __init__(self, m: nn.Module):
                super().__init__()
                self.m = m

            def forward(self, x: torch.Tensor):
                return self.m(pixel_values=x)

        wrap = Wrap(model).to(device)
        macs, _ = get_model_complexity_info(
            wrap,
            (3, img_size, img_size),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        # MACs -> FLOPs (x2), then Giga
        assert isinstance(macs, (int, float)), f"macs type is {type(macs)}"
        return float(macs * 2 / 1e9)
    except Exception:
        return None


@torch.no_grad()
def _predict_proba(model: nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    # x: [B, 3, H, W]
    logits = model(pixel_values=x.to(device, non_blocking=True))
    return torch.softmax(logits, dim=1)[:, 1]


@torch.no_grad()
def predict_tta(model: nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    """
    Average probabilities over flips + 90° rotations.
    """
    outs = []
    outs.append(_predict_proba(model, x, device))
    outs.append(_predict_proba(model, torch.flip(x, dims=[-1]), device))  # H flip
    outs.append(_predict_proba(model, torch.flip(x, dims=[-2]), device))  # V flip
    outs.append(_predict_proba(model, torch.rot90(x, 1, dims=[-2, -1]), device))  # 90
    outs.append(_predict_proba(model, torch.rot90(x, 2, dims=[-2, -1]), device))  # 180
    outs.append(_predict_proba(model, torch.rot90(x, 3, dims=[-2, -1]), device))  # 270
    return torch.stack(outs, dim=0).mean(0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: str,
    max_batches: int = 0,
    use_tta: bool = False,
) -> Dict[str, float]:
    """
    Full metrics pass on a loader. Uses your eval_binary_scores() util.
    """
    model.eval()
    probs, labels = [], []
    iters = 0
    for x, y in loader:
        if use_tta:
            p = predict_tta(model, x, device).cpu().numpy()
        else:
            p = _predict_proba(model, x, device).cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
        iters += 1
        if max_batches and iters >= max_batches:
            break

    p = np.concatenate(probs) if probs else np.array([0.5], dtype=np.float32)
    y = np.concatenate(labels) if labels else np.array([0], dtype=np.int64)
    return eval_binary_scores(p, y)


@torch.no_grad()
def time_latency(
    model: nn.Module,
    loader,
    device: str,
    warmup: int = 20,
    iters: int = 100,
    max_batches: int = 0,
) -> Tuple[float, float]:
    """
    Returns (latency_ms_per_image, throughput_img_per_s).
    Uses bs from the provided loader.
    """

    def _next(it, ldr):
        try:
            return next(it)
        except StopIteration:
            return next(iter(ldr))

    model.eval()

    # Warmup
    it = iter(loader)
    for _ in range(warmup):
        x, _ = _next(it, loader)
        x = x.to(device)
        _ = model(pixel_values=x)
        if device == "cuda":
            torch.cuda.synchronize()
        if device == "mps":
            torch.mps.synchronize()
        if max_batches:
            break

    # Timed
    it = iter(loader)
    nimg, steps = 0, 0
    t0 = time.time()
    while steps < iters:
        x, _ = _next(it, loader)
        x = x.to(device)
        _ = model(pixel_values=x)
        if device == "cuda":
            torch.cuda.synchronize()
        if device == "mps":
            torch.mps.synchronize()
        nimg += x.size(0)
        steps += 1
        if max_batches:
            break

    dt = time.time() - t0
    if steps == 0 or nimg == 0:
        return float("nan"), float("nan")
    return (dt / nimg) * 1000.0, nimg / dt
