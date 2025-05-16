import pytest
import torch
import torch.nn.functional as F
from deep_osr.losses.ce import LabelSmoothingCrossEntropy # Assuming src is in PYTHONPATH

def test_label_smoothing_cross_entropy_no_smoothing():
    """Test LabelSmoothingCrossEntropy with smoothing=0.0."""
    criterion_ls = LabelSmoothingCrossEntropy(smoothing=0.0)
    
    logits = torch.randn(4, 5) # Batch size 4, 5 classes
    targets = torch.tensor([0, 1, 2, 3])
    
    loss_ls = criterion_ls(logits, targets)
    loss_ce = F.cross_entropy(logits, targets)
    
    assert torch.isclose(loss_ls, loss_ce), "Loss with smoothing=0.0 should match F.cross_entropy"

def test_label_smoothing_cross_entropy_with_smoothing():
    """Test LabelSmoothingCrossEntropy with smoothing > 0.0."""
    num_classes = 5
    smoothing = 0.1
    criterion_ls = LabelSmoothingCrossEntropy(smoothing=smoothing)
    
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0],  # Strong prediction for class 0
                           [0.0, 10.0, 0.0, 0.0, 0.0]]) # Strong prediction for class 1
    targets = torch.tensor([0, 1])
    
    # Manual calculation for the first sample:
    # True label is 0. Target distribution: [1-0.1 + 0.1/5, 0.1/5, 0.1/5, 0.1/5, 0.1/5]
    # = [0.9 + 0.02, 0.02, 0.02, 0.02, 0.02] = [0.92, 0.02, 0.02, 0.02, 0.02]
    # log_softmax_probs for first sample: approx [0, -10, -10, -10, -10] after log(softmax(logits[0]))
    # loss = - sum(target_dist * log_softmax_probs)
    
    # Let's compare against PyTorch's own CrossEntropyLoss with soft labels if it existed,
    # or a known reference. For now, check basic properties.
    loss_ls = criterion_ls(logits, targets)
    loss_ce = F.cross_entropy(logits, targets) # Without smoothing

    assert loss_ls > 0, "Smoothed loss should be positive"
    # With strong correct predictions, smoothed loss might be slightly higher than non-smoothed CE if
    # the smoothing term penalizes overconfidence more than the NLL reduction for correct class.
    # Usually, LS helps generalization but might give slightly higher loss values on simple cases.

    # Example with one-hot targets for manual calculation using the formula:
    # loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss_term
    # nll_loss for sample 0 = -log_softmax(logits[0])[0] (approx 0 for this logit)
    # smooth_loss_term for sample 0 = -mean(log_softmax(logits[0])) (approx - (0 -10 -10 -10 -10)/5 = 40/5 = 8)
    # Expected loss for sample 0 = (1-0.1)*0 + 0.1*8 = 0.8. This is a rough approx.
    
    # A more direct check:
    # If logits strongly predict the correct class, NLL term is low.
    # If logits weakly predict, NLL term is high.
    # The smoothing term pushes towards uniform distribution.
    
    # Let's use the formula directly:
    log_prob = F.log_softmax(logits, dim=-1)
    nll_loss_manual = -log_prob.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    smooth_loss_manual = -log_prob.mean(dim=-1)
    expected_loss_batch = (1.0 - smoothing) * nll_loss_manual + smoothing * smooth_loss_manual
    expected_loss = expected_loss_batch.mean()

    assert torch.isclose(loss_ls, expected_loss), "Smoothed loss calculation does not match formula"

def test_label_smoothing_different_batch_sizes():
    criterion_ls = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    logits_batch1 = torch.randn(1, 5)
    targets_batch1 = torch.tensor([0])
    loss_batch1 = criterion_ls(logits_batch1, targets_batch1)
    assert loss_batch1.ndim == 0 # Scalar loss

    logits_batch10 = torch.randn(10, 5)
    targets_batch10 = torch.randint(0, 5, (10,))
    loss_batch10 = criterion_ls(logits_batch10, targets_batch10)
    assert loss_batch10.ndim == 0