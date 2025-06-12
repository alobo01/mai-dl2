import torch
import torch.nn as nn
import torch.nn.functional as F


class ThresholdedNLLLoss(nn.Module):
    """
    Enhanced NLL Loss with confidence thresholding for open-set recognition.
    If the model's highest softmax probability falls below a threshold,
    the input is classified as unknown (dummy class).
    """
    
    def __init__(self, confidence_threshold=0.5, smoothing=0.0, dummy_class_penalty=1.0):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.smoothing = smoothing
        self.dummy_class_penalty = dummy_class_penalty
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=smoothing, reduction='none')
        
    def forward(self, logits, targets, is_known_mask=None):
        """
        Args:
            logits: (N, K) logits for K known classes
            targets: (N,) target labels for known classes
            is_known_mask: (N,) boolean mask indicating if sample is known
        """
        # Standard cross-entropy loss for known samples
        ce_loss = self.ce_loss(logits, targets)
        
        # Apply confidence thresholding
        probs = F.softmax(logits, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        
        # Samples with low confidence should be penalized more
        low_confidence_mask = max_probs < self.confidence_threshold
        
        # Apply higher penalty for low confidence predictions
        loss = ce_loss.clone()
        loss[low_confidence_mask] *= self.dummy_class_penalty
        
        return loss.mean()


class DummyClassNLLLoss(nn.Module):
    """
    Enhanced NLL Loss with dummy class for open-set recognition.
    Applies higher penalties when unknown samples are incorrectly 
    assigned to known classes.
    """
    
    def __init__(self, dummy_class_penalty=2.0, smoothing=0.0):
        super().__init__()
        self.dummy_class_penalty = dummy_class_penalty
        self.smoothing = smoothing
        
    def forward(self, logits_with_dummy, targets, is_known_mask):
        """
        Args:
            logits_with_dummy: (N, K+1) logits for K known classes + 1 dummy class
            targets: (N,) target labels (0 to K-1 for known, K for dummy)
            is_known_mask: (N,) boolean mask indicating if sample is known
        """
        # Use cross-entropy with label smoothing
        if self.smoothing == 0.0:
            ce_loss = F.cross_entropy(logits_with_dummy, targets, reduction='none')
        else:
            log_prob = F.log_softmax(logits_with_dummy, dim=-1)
            nll_loss = -log_prob.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
            smooth_loss = -log_prob.mean(dim=-1)
            ce_loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        # Apply higher penalty for misclassifying unknowns as knowns
        # Unknown samples should predict dummy class (K), not known classes (0 to K-1)
        unknown_mask = ~is_known_mask
        predicted_as_known = targets < (logits_with_dummy.shape[1] - 1)  # Not dummy class
        
        # Penalty for unknown samples predicted as known classes
        misclassified_unknown_mask = unknown_mask & predicted_as_known
        ce_loss[misclassified_unknown_mask] *= self.dummy_class_penalty
        
        return ce_loss.mean()


class CombinedThresholdPenaltyLoss(nn.Module):
    """
    Combines thresholding and penalty approaches for robust open-set training.
    """
    
    def __init__(self, confidence_threshold=0.5, dummy_class_penalty=2.0, 
                 threshold_weight=0.5, penalty_weight=0.5, smoothing=0.0):
        super().__init__()
        self.threshold_loss = ThresholdedNLLLoss(confidence_threshold, smoothing, dummy_class_penalty)
        self.penalty_loss = DummyClassNLLLoss(dummy_class_penalty, smoothing)
        self.threshold_weight = threshold_weight
        self.penalty_weight = penalty_weight
        
    def forward(self, logits_known, logits_with_dummy, targets_known, targets_with_dummy, is_known_mask):
        """
        Args:
            logits_known: (N, K) logits for K known classes only
            logits_with_dummy: (N, K+1) logits including dummy class
            targets_known: (N,) targets for known classes
            targets_with_dummy: (N,) targets including dummy class
            is_known_mask: (N,) boolean mask
        """
        threshold_loss = self.threshold_loss(logits_known, targets_known, is_known_mask)
        penalty_loss = self.penalty_loss(logits_with_dummy, targets_with_dummy, is_known_mask)
        
        return self.threshold_weight * threshold_loss + self.penalty_weight * penalty_loss


# Loss builder for different strategies
def get_enhanced_loss_functions(loss_type, **kwargs):
    """
    Factory function for enhanced loss functions.
    
    Args:
        loss_type: 'threshold', 'penalty', 'combined'
        **kwargs: parameters for the loss function
    """
    if loss_type == "threshold":
        # Filter parameters for ThresholdedNLLLoss
        threshold_params = {
            'confidence_threshold': kwargs.get('confidence_threshold', 0.5),
            'smoothing': kwargs.get('smoothing', 0.0),
            'dummy_class_penalty': kwargs.get('dummy_class_penalty', 1.0)
        }
        return ThresholdedNLLLoss(**threshold_params)    
    elif loss_type == "penalty":
        # Filter parameters for DummyClassNLLLoss
        penalty_params = {
            'dummy_class_penalty': kwargs.get('dummy_class_penalty', 2.0),
            'smoothing': kwargs.get('smoothing', 0.0)
        }
        return DummyClassNLLLoss(**penalty_params)
    elif loss_type == "combined":
        # Filter parameters for CombinedThresholdPenaltyLoss
        combined_params = {
            'confidence_threshold': kwargs.get('confidence_threshold', 0.5),
            'dummy_class_penalty': kwargs.get('dummy_class_penalty', 2.0),
            'threshold_weight': kwargs.get('threshold_weight', 0.5),
            'penalty_weight': kwargs.get('penalty_weight', 0.5),
            'smoothing': kwargs.get('smoothing', 0.0)
        }
        return CombinedThresholdPenaltyLoss(**combined_params)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
        
        
# Additional utility functions for OSR evaluation
def compute_osr_score(outputs, method='energy'):
    """
    Compute OSR score from model outputs.
    
    Args:
        outputs: dict with 'cls_logits' and optionally 'osr_score'
        method: 'energy', 'max_softmax', or 'entropy'
    """
    if isinstance(outputs, dict) and 'osr_score' in outputs:
        return outputs['osr_score']
    
    logits = outputs.get('cls_logits', outputs) if isinstance(outputs, dict) else outputs
    
    if method == 'energy':
        # Energy score (negative of log-sum-exp)
        return -torch.logsumexp(logits, dim=1)
    elif method == 'max_softmax':
        # Maximum softmax probability (higher = more confident = more known)
        probs = F.softmax(logits, dim=1)
        return torch.max(probs, dim=1)[0]
    elif method == 'entropy':
        # Entropy (higher = more uncertain = more unknown)
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy
    else:
        raise ValueError(f"Unknown OSR score method: {method}")
