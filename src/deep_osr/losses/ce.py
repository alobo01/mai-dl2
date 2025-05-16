import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target): # input: (N, C) logits, target: (N) class indices
        if self.smoothing == 0.0:
            return F.cross_entropy(input, target)
            
        log_prob = F.log_softmax(input, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# Loss builder concept (can be a simple dict or a more complex class)
def get_loss_functions(cfg_model, cfg_train_loss):
    loss_fns = {}
    loss_weights = {}

    # CE for known classes
    loss_fns['ce_seen'] = LabelSmoothingCrossEntropy(smoothing=cfg_model.cls_head.label_smoothing)
    loss_weights['ce_seen'] = cfg_train_loss.ce_seen_weight
    
    # Placeholder for other losses based on OSR head type etc.
    # if cfg_model.osr_head.type == "kplus1" and cfg_train_loss.dummy_loss_weight > 0:
    #     loss_fns['dummy_loss'] = SomeOtherLoss()
    #     loss_weights['dummy_loss'] = cfg_train_loss.dummy_loss_weight
        
    return loss_fns, loss_weights