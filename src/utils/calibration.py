import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # Initial temperature

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits_val, labels_val, max_iter=100, lr=0.01):
        """ Fits temperature on validation logits and labels. """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_():
            optimizer.zero_grad()
            loss = nll_criterion(logits_val / self.temperature, labels_val)
            loss.backward()
            return loss
        
        optimizer.step(eval_)
        print(f"Optimal temperature found: {self.temperature.item():.3f}")
        return self.temperature.item()


def calibrate_model_temperature(model_outputs_val, labels_val, device):
    """
    model_outputs_val: List of (logits_k_classes_batch, osr_score_batch) from validation set.
                       We only need logits_k_classes from known samples for temp scaling.
    labels_val: List of (y_true_for_knowns_batch, is_truly_known_mask_batch).
    
    Returns the optimal temperature.
    """
    all_logits_known = []
    all_labels_known = []

    for (logits_k_batch, _), (y_true_batch, is_known_mask_batch) in zip(model_outputs_val, labels_val):
        known_mask_b = is_known_mask_batch.to(device)
        logits_k_b = logits_k_batch.to(device)
        y_true_b = y_true_batch.to(device)

        all_logits_known.append(logits_k_b[known_mask_b])
        all_labels_known.append(y_true_b[known_mask_b])
    
    if not all_logits_known:
        print("Warning: No known samples in calibration data. Cannot fit temperature.")
        return 1.0

    logits_val_tensor = torch.cat(all_logits_known).to(device)
    labels_val_tensor = torch.cat(all_labels_known).to(device)

    if len(logits_val_tensor) == 0:
        print("Warning: No known samples in calibration data after filtering. Cannot fit temperature.")
        return 1.0

    scaler = TemperatureScaler().to(device)
    optimal_t = scaler.fit(logits_val_tensor, labels_val_tensor)
    return optimal_t

# PlattBinning and ECE minimization can be added here. They are more complex.