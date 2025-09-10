# src/models/backbone_dinov3.py
import torch
from transformers import AutoModel


class DinoV3Backbone(torch.nn.Module):
    def __init__(
        self, model_id="facebook/dinov3-vits16-pretrain-lvd1689m", dtype=torch.float32
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype)
        self.hidden = self.model.config.hidden_size

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).pooler_output
