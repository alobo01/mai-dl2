import pytest
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from deep_osr.open_set_metrics import OpenSetMetrics
from deep_osr.eval import recall_at_fixed_precision # Make sure eval.py is in src/

# Test OpenSetMetrics Class
@pytest.fixture
def os_metrics():
    return OpenSetMetrics(prefix='test')

def test_os_metrics_empty(os_metrics):
    metrics = os_metrics.compute(num_known_classes=2)
    assert len(metrics) == 0, "Metrics should be empty if no data updated"

def test_os_metrics_acc_known_perfect(os_metrics):
    # Probs: (Batch, K_classes)
    # OSR scores: (Batch,) - higher means more unknown
    # y_true_known_idx: (Batch,) - labels for knowns (0..K-1)
    # is_known_targets: (Batch,) - True if known, False if unknown
    
    probs = torch.tensor([[0.9, 0.1], [0.05, 0.95]]) # 2 samples, 2 known classes
    osr_scores = torch.tensor([0.1, 0.2]) # Low scores, indicating "known"
    y_true = torch.tensor([0, 1])
    is_known = torch.tensor([True, True])
    
    os_metrics.update(probs, osr_scores, y_true, is_known)
    metrics = os_metrics.compute(num_known_classes=2)
    
    assert 'test/acc_known' in metrics
    assert np.isclose(metrics['test/acc_known'], 1.0)

def test_os_metrics_acc_known_imperfect(os_metrics):
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.2]]) # 2 samples, 2 known classes. Second is wrong.
    osr_scores = torch.tensor([0.1, 0.2])
    y_true = torch.tensor([0, 1]) # True labels are 0 and 1
    is_known = torch.tensor([True, True])
    
    os_metrics.update(probs, osr_scores, y_true, is_known)
    metrics = os_metrics.compute(num_known_classes=2)
    
    assert 'test/acc_known' in metrics
    assert np.isclose(metrics['test/acc_known'], 0.5) # First correct, second incorrect

def test_os_metrics_auroc_perfect_separation(os_metrics):
    probs_k = torch.rand(4, 2) # Dummy probs
    # Knowns (low OSR score), Unknowns (high OSR score)
    osr_scores = torch.tensor([0.1, 0.2, 0.8, 0.9]) 
    y_true_idx = torch.tensor([0, 1, -1, -1]) # Two knowns, two unknowns
    is_known = torch.tensor([True, True, False, False])
    
    os_metrics.update(probs_k, osr_scores, y_true_idx, is_known)
    metrics = os_metrics.compute(num_known_classes=2)
    
    assert 'test/auroc' in metrics
    assert np.isclose(metrics['test/auroc'], 1.0)
    assert 'test/aupr_in' in metrics # Unknowns as positive
    assert np.isclose(metrics['test/aupr_in'], 1.0)
    assert 'test/aupr_out' in metrics # Knowns as positive
    assert np.isclose(metrics['test/aupr_out'], 1.0)


def test_os_metrics_auroc_no_separation(os_metrics):
    probs_k = torch.rand(4, 2)
    osr_scores = torch.tensor([0.5, 0.5, 0.5, 0.5]) # All scores same
    y_true_idx = torch.tensor([0, 1, -1, -1])
    is_known = torch.tensor([True, True, False, False]) # 2 known, 2 unknown

    os_metrics.update(probs_k, osr_scores, y_true_idx, is_known)
    metrics = os_metrics.compute(num_known_classes=2)
    
    assert 'test/auroc' in metrics
    assert np.isclose(metrics['test/auroc'], 0.5)
    # AUPR baseline depends on class balance. For 50/50, baseline is 0.5.
    num_unknown = (~is_known).sum().item()
    num_total = len(is_known)
    expected_aupr_in_baseline = num_unknown / num_total
    assert 'test/aupr_in' in metrics
    assert np.isclose(metrics['test/aupr_in'], expected_aupr_in_baseline)

def test_os_metrics_only_knowns(os_metrics):
    probs = torch.tensor([[0.9, 0.1], [0.05, 0.95]])
    osr_scores = torch.tensor([0.1, 0.2])
    y_true = torch.tensor([0, 1])
    is_known = torch.tensor([True, True]) # All known
    
    os_metrics.update(probs, osr_scores, y_true, is_known)
    metrics = os_metrics.compute(num_known_classes=2)
    
    assert 'test/acc_known' in metrics
    assert 'test/auroc' not in metrics # AUROC needs both classes
    assert 'test/aupr_in' not in metrics
    assert 'test/aupr_out' not in metrics


# Test recall_at_fixed_precision function (from eval.py)
@pytest.mark.parametrize("target_precision", [0.9, 0.95, 1.0])
def test_recall_at_fixed_precision_perfect_separation(target_precision):
    # Knowns have low OSR scores, Unknowns have high OSR scores
    y_true_is_known = np.array([True, True, False, False])
    osr_scores = np.array([0.1, 0.2, 0.8, 0.9]) # Higher = more unknown
    
    # With perfect separation, any reasonable target precision for knowns should be met.
    # This should allow all unknowns to be correctly recalled.
    u_recall = recall_at_fixed_precision(y_true_is_known, osr_scores, target_precision)
    assert np.isclose(u_recall, 1.0)

def test_recall_at_fixed_precision_no_separation():
    y_true_is_known = np.array([True, True, False, False])
    osr_scores = np.array([0.5, 0.5, 0.5, 0.5]) # All scores identical
    
    # Target precision likely won't be met, or if met (e.g. TP_known=0, FP_known=0 -> P=undef or 0)
    # result should be 0.0
    u_recall = recall_at_fixed_precision(y_true_is_known, osr_scores, 0.95)
    assert np.isclose(u_recall, 0.0), "With no separation, recall should be 0 if target precision is high"

def test_recall_at_fixed_precision_mixed_case():
    # Scenario: Knowns generally lower scores, Unknowns generally higher, but some overlap
    y_true_is_known = np.array([True, True, True, False, False, False]) # 3 known, 3 unknown
    osr_scores      = np.array([0.1, 0.2, 0.6,  0.4, 0.8, 0.9])
    # Known scores: 0.1, 0.2, 0.6
    # Unknown scores: 0.4, 0.8, 0.9
    # If threshold for "known" is <0.4 (e.g., osr_score <= 0.3):
    #   Pred Known: [0.1, 0.2]. Pred Unknown: [0.6, 0.4, 0.8, 0.9]
    #   TP_k = 2 (0.1, 0.2). FP_k = 0. Precision_k = 2/(2+0) = 1.0.
    #   With this threshold (-0.3 for score_for_known_as_positive):
    #   TP_u = 2 (0.8, 0.9). (if 0.4, 0.6 classified as unknown by this threshold)
    #   Total Unknowns = 3. U_Recall = 2/3.
    
    # If threshold for "known" is <0.5 (e.g., osr_score <= 0.45):
    #   Pred Known: [0.1, 0.2], [0.4 from unknowns]. Pred Unknown: [0.6 from knowns], [0.8, 0.9 from unknowns]
    #   TP_k = 2 (0.1, 0.2). FP_k = 1 (0.4). Precision_k = 2/(2+1) = 0.66. Does not meet 0.95.
    
    # The function should find the threshold that gives Precision_Known >= 0.95
    # and then calculate U_Recall.
    # The point where Precision_Known=1.0 (threshold for osr_score <= 0.3):
    #   Knowns predicted as known: those with osr_score <= 0.3 (samples 0.1, 0.2)
    #   Unknowns predicted as unknown: those with osr_score > 0.3 (samples 0.6(K), 0.4(U), 0.8(U), 0.9(U))
    #   True unknowns: 0.4, 0.8, 0.9. Predicted unknowns here (based on threshold): 0.4, 0.8, 0.9
    #   (also 0.6 from knowns is predicted as unknown)
    #   TP_u = 3. Total_u = 3. Recall_u = 1.0.
    
    u_recall = recall_at_fixed_precision(y_true_is_known, osr_scores, 0.95)
    assert np.isclose(u_recall, 1.0) # Based on manual walk-through above


def test_recall_at_fixed_precision_all_known():
    y_true_is_known = np.array([True, True, True])
    osr_scores = np.array([0.1, 0.2, 0.3])
    u_recall = recall_at_fixed_precision(y_true_is_known, osr_scores, 0.95)
    # No unknowns, so recall of unknowns is undefined or 0.
    # The function handles this by total_actual_unknowns == 0 returning 0.0 or 1.0
    # Check current implementation: if total_actual_unknowns == 0, returns 1.0 if tp_unknown == 0 else 0.0.
    # Here tp_unknown would be 0. So, should be 1.0 if using that specific logic.
    # A more standard return for this case would be 0.0 or np.nan. The prompt's description implies "U-Recall".
    # Current logic returns 0.0 if all known as per top of function.
    assert np.isclose(u_recall, 0.0)


def test_recall_at_fixed_precision_all_unknown():
    y_true_is_known = np.array([False, False, False])
    osr_scores = np.array([0.7, 0.8, 0.9])
    u_recall = recall_at_fixed_precision(y_true_is_known, osr_scores, 0.95)
    # Precision for knowns is 0/0 (undefined) or 0. Target precision of 0.95 cannot be met.
    # So, u_recall should be 0.0.
    assert np.isclose(u_recall, 0.0)

def test_recall_at_fixed_precision_invalid_target_precision():
    y_true_is_known = np.array([True, True, False, False])
    osr_scores = np.array([0.1, 0.2, 0.8, 0.9])
    
    with pytest.warns(UserWarning, match="target_precision_for_knowns must be in"): # If it warns
         u_recall_too_high = recall_at_fixed_precision(y_true_is_known, osr_scores, 1.1)
         assert np.isclose(u_recall_too_high, 0.0) # Corrected based on function's behavior

    with pytest.warns(UserWarning, match="target_precision_for_knowns must be in"):
         u_recall_too_low = recall_at_fixed_precision(y_true_is_known, osr_scores, 0.0)
         assert np.isclose(u_recall_too_low, 0.0)