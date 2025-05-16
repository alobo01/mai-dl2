import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm

# Store output feature dimensions for common models
BACKBONE_OUTPUT_DIMS = {
    "resnet50": 2048,
    "efficientnet_b4": 1792, # efficientnet_b4 output features
    "vit_b16": 768,       # ViT-B/16 output features (classifier token)
    "dino_v2_vits14": 384, # DINOv2 ViT-S/14
    "dino_v2_vitb14": 768, # DINOv2 ViT-B/14
    "dino_v2_vitl14": 1024, # DINOv2 ViT-L/14
    "dino_v2_vitg14": 1536, # DINOv2 ViT-g/14
}


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def get_backbone(name: str, pretrained: bool = True, frozen: bool = False):
    num_output_features = BACKBONE_OUTPUT_DIMS.get(name)
    if num_output_features is None and "dino_v2" not in name: # DINOv2 handled separately
        raise ValueError(f"Number of output features not defined for backbone: {name}. Add to BACKBONE_OUTPUT_DIMS.")

    if name == "resnet50":
        model = tv_models.resnet50(pretrained=pretrained)
        model.fc = Identity() # Remove classifier
    elif name == "efficientnet_b4":
        model = timm.create_model("efficientnet_b4", pretrained=pretrained)
        model.classifier = Identity() # Remove classifier
    elif name == "vit_b16":
        model = timm.create_model("vit_base_patch16_224", pretrained=pretrained) # Assumes 224x224 input
        model.head = Identity() # Remove classifier head
    elif name.startswith("dino_v2_"): # e.g. dino_v2_vitb14
        # Full model name for torch.hub: e.g., dinov2_vits14, dinov2_vitb14, etc.
        hub_model_name = name.replace("dino_v2_", "dinov2_")
        try:
            # DINOv2 returns a dict of features, we typically want the CLS token
            # The model itself is just the feature extractor.
            model = torch.hub.load('facebookresearch/dinov2', hub_model_name)
            num_output_features = model.embed_dim # For ViT models in DINOv2
            BACKBONE_OUTPUT_DIMS[name] = num_output_features # Store dynamically
        except Exception as e:
            raise ValueError(f"Failed to load DINOv2 model {hub_model_name}. Check name and internet connection. Error: {e}")
        # DINOv2 models from torch.hub are feature extractors by default.
        # They output a dictionary of features, including x_norm_clstoken, x_norm_patchtokens, etc.
        # We need to wrap it to extract the CLS token.
        original_forward = model.forward
        def dino_cls_forward(x, **kwargs):
            # DINOv2's forward method has is_training argument, but typically for OSR we use backbone features
            # For DINOv2, might need to handle kwargs like `is_training` if the underlying model uses it.
            # Default behavior from hub load is often `model.eval()` implicitly for feature extraction.
            # The output is a dict, we want the CLS token.
            # Or use `model.forward_features(x)` which returns a dict.
            # `get_intermediate_layers` is more flexible.
            # For simplicity, let's assume we want the CLS token from the last layer.
            # The output of model(x) is the CLS token embedding if not using model.forward_features()
            # No, model(x) returns dict for some settings.
            # `model.get_intermediate_layers(x, n=1, return_class_token=True)[0]` is one way for CLS
            # Or just model(x) might be CLS token. This needs verification for specific DINOv2 model types.
            # The standard DINOv2 ViT models, when called directly, return the CLS token of the last block.
            features = original_forward(x) # This should be the CLS token for ViTs
            return features
        model.forward = dino_cls_forward

    else:
        raise ValueError(f"Backbone {name} not supported.")

    if frozen:
        for param in model.parameters():
            param.requires_grad = False
        model.eval() # Important for frozen backbones, esp. with BatchNorm

    return model, num_output_features