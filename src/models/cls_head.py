import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, use_weight_norm: bool = True, temperature: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        if use_weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)
        self.temperature = temperature
        # Label smoothing is handled in the loss function

    def forward(self, x):
        logits = self.linear(x)
        return logits / self.temperature