# src/data/pcam_hf.py
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor


class PCamH5HF(Dataset):
    def __init__(
        self,
        h5_x,
        h5_y,
        model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        image_size=224,
    ):
        self.h5_x_path = h5_x
        self.h5_y_path = h5_y
        self.fx = None
        self.fy = None

        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.size = {"height": image_size, "width": image_size}

    def _ensure_open(self):
        if self.fx is None:
            self.fx = h5py.File(self.h5_x_path, "r")
        if self.fy is None:
            self.fy = h5py.File(self.h5_y_path, "r")

    def __len__(self):
        with h5py.File(self.h5_x_path, "r") as f:
            return len(f["x"])  # type:ignore

    def __getitem__(self, i):
        self._ensure_open()
        x = self.fx["x"][i]  # uint8 HxWx3 #type:ignore
        y = int(self.fy["y"][i])  # type:ignore
        img = Image.fromarray(x)  # type:ignore
        # pass size at call-time; let processor handle resize+normalize
        out = self.proc(images=img, return_tensors="pt", size=self.size)
        return out["pixel_values"].squeeze(0), torch.tensor(y, dtype=torch.long)

    def __del__(self):
        try:
            if self.fx is not None:
                self.fx.close()
            if self.fy is not None:
                self.fy.close()
        except Exception:
            pass
