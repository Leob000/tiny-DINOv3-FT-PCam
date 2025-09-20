"""Shared data utilities for PCam training and evaluation."""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

from src.data.pcam_hf import PCamH5HF


class RandomRotate90:
    """Rotate PIL image by k*90 degrees with k in {0,1,2,3}."""

    def __call__(self, img: Image.Image) -> Image.Image:
        k = torch.randint(0, 4, ()).item()
        if k == 1:
            return img.transpose(Image.ROTATE_90)  # type:ignore[arg-type]
        if k == 2:
            return img.transpose(Image.ROTATE_180)  # type:ignore[arg-type]
        if k == 3:
            return img.transpose(Image.ROTATE_270)  # type:ignore[arg-type]
        return img


def _pcam_h5_path(data_dir: str, split: str, kind: str) -> str:
    return os.path.join(data_dir, f"camelyonpatch_level_2_split_{split}_{kind}.h5")


def _histology_transforms() -> T.Compose:
    return T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRotate90(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
        ]
    )


def build_pcam_datasets(
    data_dir: str,
    model_id: str,
    image_size: int,
    aug_histology: bool = False,
) -> Tuple[PCamH5HF, PCamH5HF, PCamH5HF]:
    """Create train/val/test PCam datasets with shared conventions."""

    train_tf = _histology_transforms() if aug_histology else None
    train_ds = PCamH5HF(
        _pcam_h5_path(data_dir, "train", "x"),
        _pcam_h5_path(data_dir, "train", "y"),
        model_id=model_id,
        image_size=image_size,
        transform=train_tf,
    )
    val_ds = PCamH5HF(
        _pcam_h5_path(data_dir, "valid", "x"),
        _pcam_h5_path(data_dir, "valid", "y"),
        model_id=model_id,
        image_size=image_size,
        transform=None,
    )
    test_ds = PCamH5HF(
        _pcam_h5_path(data_dir, "test", "x"),
        _pcam_h5_path(data_dir, "test", "y"),
        model_id=model_id,
        image_size=image_size,
        transform=None,
    )
    return train_ds, val_ds, test_ds


def build_eval_loaders(
    data_dir: str,
    model_id: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Tuple[DataLoader, DataLoader]:
    """Construct validation and test loaders sharing consistent settings."""

    _, val_ds, test_ds = build_pcam_datasets(
        data_dir=data_dir,
        model_id=model_id,
        image_size=image_size,
        aug_histology=False,
    )

    common_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, **common_kwargs)  # type:ignore
    test_loader = DataLoader(test_ds, **common_kwargs)  # type:ignore
    return val_loader, test_loader


def build_train_loader(
    data_dir: str,
    model_id: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
    aug_histology: bool,
) -> DataLoader:
    """Create the training dataloader following project defaults."""

    train_ds, _, _ = build_pcam_datasets(
        data_dir=data_dir,
        model_id=model_id,
        image_size=image_size,
        aug_histology=aug_histology,
    )
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
