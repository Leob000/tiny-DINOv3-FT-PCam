import torch.nn as nn


class DinoV3PCam(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.hidden, num_classes)

    def forward(self, pixel_values):
        feats = self.backbone(pixel_values)
        return self.head(feats)
