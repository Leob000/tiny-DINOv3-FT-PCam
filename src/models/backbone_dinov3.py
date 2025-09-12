# src/models/backbone_dinov3.py
import torch
import torch.nn as nn
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


class DinoV3PCam(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.hidden, num_classes)

    def forward(self, pixel_values):
        feats = self.backbone(pixel_values)
        return self.head(feats)
