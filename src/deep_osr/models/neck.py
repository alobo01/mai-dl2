import torch.nn as nn

class EmbeddingNeck(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_batchnorm: bool = True, use_relu: bool = True):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_features))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        # ArcMargin could be added here if specified in config
        self.neck = nn.Sequential(*layers)

    def forward(self, x):
        return self.neck(x)