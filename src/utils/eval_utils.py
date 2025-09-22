# src/utils/eval_utils.py
from __future__ import annotations
import time
from typing import Dict, Tuple, Optional, List

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
def collect_binary_predictions(
    model: nn.Module,
    loader,
    device: str,
    max_batches: int = 0,
    use_tta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-sample probabilities, labels, and sequential indices for a loader.
    """

    model.eval()
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    indices: List[np.ndarray] = []
    iters = 0
    offset = 0

    for x, y in loader:
        if use_tta:
            batch_probs = predict_tta(model, x, device).cpu().numpy()
        else:
            batch_probs = _predict_proba(model, x, device).cpu().numpy()
        batch_labels = y.numpy()
        batch_size = batch_labels.shape[0]
        batch_indices = np.arange(offset, offset + batch_size, dtype=np.int64)

        probs.append(batch_probs)
        labels.append(batch_labels)
        indices.append(batch_indices)

        offset += batch_size
        iters += 1
        if max_batches and iters >= max_batches:
            break

    p = np.concatenate(probs) if probs else np.empty((0,), dtype=np.float32)
    y = np.concatenate(labels) if labels else np.empty((0,), dtype=np.int64)
    idx = np.concatenate(indices) if indices else np.empty((0,), dtype=np.int64)
    return p, y, idx


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
    p, y, _ = collect_binary_predictions(
        model, loader, device, max_batches=max_batches, use_tta=use_tta
    )
    if p.size == 0 or y.size == 0:
        # align with previous fallback behaviour when loader was empty
        p = np.array([0.5], dtype=np.float32)
        y = np.array([0], dtype=np.int64)
    return eval_binary_scores(p, y)


def evaluate_loss(model, loader, device, crit, max_batches=0):
    """Fast eval: average cross-entropy loss on GPU. No concatenation/CPU metrics."""
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader, start=1):
            time_batch_start = time.time()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(pixel_values=x)
            loss = crit(logits, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            print(
                f"Eval batch {bi} | time: {time.time() - time_batch_start:.3f}s",
                end="\r",
            )
            if max_batches and bi >= max_batches:
                break
    return total_loss / max(1, total_n)


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
