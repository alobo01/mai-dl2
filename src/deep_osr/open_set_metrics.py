import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc as sklearn_auc
import numpy as np

class OpenSetMetrics:
    def __init__(self, prefix='val'):
        self.prefix = prefix
        self.reset()

    def reset(self):
        self.all_k_class_probs = [] # Softmax probabilities over K known classes
        self.all_osr_scores = []    # Score for unknown detection (higher = more unknown)
        self.all_y_true_known_idx = [] # True labels for known samples (0 to K-1)
        self.all_is_known_targets = [] # Boolean mask: True if sample is known, False if unknown

    def update(self, k_class_probs, osr_scores, y_true_known_idx, is_known_targets):
        self.all_k_class_probs.append(k_class_probs.cpu())
        self.all_osr_scores.append(osr_scores.cpu())
        self.all_y_true_known_idx.append(y_true_known_idx.cpu())
        self.all_is_known_targets.append(is_known_targets.cpu())

    def compute(self, num_known_classes):
        if not self.all_is_known_targets: # No data updated
            return {} 

        k_class_probs = torch.cat(self.all_k_class_probs)
        osr_scores = torch.cat(self.all_osr_scores)
        y_true_known_idx = torch.cat(self.all_y_true_known_idx)
        is_known_targets = torch.cat(self.all_is_known_targets)

        metrics = {}

        # Ensure there are both known and unknown samples for open-set metrics
        has_known = torch.any(is_known_targets)
        has_unknown = torch.any(~is_known_targets)

        # --- Closed-set metrics (on known samples only) ---
        if has_known:
            known_samples_probs = k_class_probs[is_known_targets]
            known_samples_true_labels = y_true_known_idx[is_known_targets]
            
            if len(known_samples_probs) > 0:
                preds_known = torch.argmax(known_samples_probs, dim=1)
                closed_accuracy = (preds_known == known_samples_true_labels).float().mean().item()
                metrics[f'{self.prefix}/acc_known'] = closed_accuracy
                # F1 score can be added here (e.g., macro F1)
                # from sklearn.metrics import f1_score
                # closed_f1_macro = f1_score(known_samples_true_labels.numpy(), preds_known.numpy(), average='macro', zero_division=0)
                # metrics[f'{self.prefix}/f1_macro_known'] = closed_f1_macro


        # --- Open-set metrics (distinguishing known vs unknown) ---
        if has_known and has_unknown:
            # AUROC: Uses OSR scores. Unknowns are positive class.
            # sklearn expects y_true where 1=positive. Here, unknown is positive => ~is_known_targets
            # osr_scores: higher means more unknown (positive contribution to score)
            try:
                auroc = roc_auc_score(~is_known_targets.numpy(), osr_scores.numpy())
                metrics[f'{self.prefix}/auroc'] = auroc
            except ValueError as e: # Happens if only one class present in y_true after filtering
                metrics[f'{self.prefix}/auroc'] = 0.0 
                print(f"AUROC calculation error: {e}")


            # AUPR-In (Unknowns as positive class)
            try:
                aupr_in = average_precision_score(~is_known_targets.numpy(), osr_scores.numpy())
                metrics[f'{self.prefix}/aupr_in'] = aupr_in # In = Unknown
            except ValueError as e:
                metrics[f'{self.prefix}/aupr_in'] = 0.0
                print(f"AUPR-In calculation error: {e}")


            # AUPR-Out (Knowns as positive class, use -osr_scores or 1-osr_scores if osr_scores are normalized)
            # To keep consistency, let's use (1 - normalized_osr_score) or -osr_score
            try:
                aupr_out = average_precision_score(is_known_targets.numpy(), -osr_scores.numpy()) # Negative osr_score for knowns
                metrics[f'{self.prefix}/aupr_out'] = aupr_out # Out = Known
            except ValueError as e:
                metrics[f'{self.prefix}/aupr_out'] = 0.0
                print(f"AUPR-Out calculation error: {e}")

            # OSCR (Overall Score for Closed-set Recognition)
            # This requires `plot_oscr` logic, but adapted for calculation not plotting
            # auc_oscr = self.calculate_oscr_auc(k_class_probs, y_true_known_idx, is_known_targets, num_known_classes)
            # metrics[f'{self.prefix}/oscr_auc'] = auc_oscr
            # For simplicity, we will rely on the standalone script for OSCR plot & AUC value for now.

            # U-Recall @ 95% Seen-Precision (Recall of Unknowns at 95% Precision of Knowns)
            # Precision of knowns: TP_known / (TP_known + FP_known)
            # TP_known: known identified as known. FP_known: unknown identified as known.
            # Recall of unknowns: TP_unknown / (TP_unknown + FN_unknown)
            # TP_unknown: unknown identified as unknown. FN_unknown: known identified as unknown.
            
            # Sort by OSR score (ascending, if lower score means more "known")
            # Or descending, if higher score means "unknown" (our convention for osr_scores)
            # sklearn precision_recall_curve: y_true (1 for positive), probas_pred (score for positive class)
            # Here, positive class = unknown. So, use ~is_known_targets and osr_scores.
            precisions, recalls, _ = precision_recall_curve(~is_known_targets.numpy(), osr_scores.numpy()) # For "unknown" class detection

            # This is Recall of Unknowns vs Precision of Unknowns.
            # We want Recall of Unknowns vs Precision of Knowns.
            # Precision of Knowns = TN / (TN + FN) if Unknown is Positive class. (TN = known correctly rejected, FN = known incorrectly accepted as unknown)
            # This is tricky. Let's use the definition: "U-Recall@95-Seen-Precision"
            # Seen-Precision = CCR / (CCR + FPR_of_unknowns)
            # U-Recall = Recall of Unknowns
            # Find threshold where Seen-Precision = 0.95, then get U-Recall.
            
            # Simpler to implement directly:
            # Sort all samples by osr_score (descending, as higher means more unknown)
            sorted_indices = torch.argsort(osr_scores, descending=True)
            sorted_is_known = is_known_targets[sorted_indices]
            
            num_unknowns_total = (~is_known_targets).sum().item()
            num_knowns_total = is_known_targets.sum().item()

            if num_unknowns_total == 0 or num_knowns_total == 0:
                 metrics[f'{self.prefix}/u_recall_at_95_seen_prec'] = 0.0
            else:
                tp_unknown = 0 # Unknowns correctly identified as unknown
                fp_unknown_as_known = 0 # Unknowns misclassified as known (based on threshold moving)
                                     # This is not quite right.
                
                # Iterate through thresholds implicitly by taking top-N scoring samples as "unknown"
                # This is for U-Recall@X-TPR_known (not precision)
                # Let's implement the recall_at_fixed_precision directly as in the blueprint for eval.py.
                # Here, we log what's easy with sklearn.
                # The custom metric `recall_at_fixed_precision` is usually computed once on the final scores.
                pass # Placeholder for this specific metric in streaming fashion.

        self.reset() # Prepare for next epoch
        return metrics

    # This is part of the OSCR curve plotting script logic, can be adapted here if needed.
    # def calculate_oscr_auc(self, probs_k_all, y_true_all, is_known_all, num_known_classes):
    #     # probs_k_all: (N, K_classes)
    #     # y_true_all: (N,), labels for knowns are 0..K-1, for unknowns can be -1
    #     # is_known_all: (N,) boolean
        
    #     confidences = probs_k_all.max(1).values
    #     predictions = probs_k_all.argmax(1)

    #     # Sort by decreasing confidence
    #     order = confidences.argsort(descending=True)
        
    #     sorted_is_known = is_known_all[order]
    #     sorted_y_true = y_true_all[order]
    #     sorted_predictions = predictions[order]

    #     correct_classification_mask = (sorted_predictions == sorted_y_true)
        
    #     ccr_numerator = 0  # Correctly classified knowns
    #     fpr_numerator = 0  # Misclassified unknowns (as knowns)
        
    #     ccr_values = []
    #     fpr_values = []

    #     total_knowns = sorted_is_known.sum().item()
    #     total_unknowns = (~sorted_is_known).sum().item()

    #     if total_knowns == 0 or total_unknowns == 0:
    #         return 0.0

    #     for i in range(len(order)):
    #         if sorted_is_known[i]: # If it's a known sample
    #             if correct_classification_mask[i]:
    #                 ccr_numerator += 1
    #         else: # If it's an unknown sample (which is considered "accepted as known" here)
    #             fpr_numerator += 1
            
    #         ccr = ccr_numerator / total_knowns if total_knowns > 0 else 0.0
    #         fpr = fpr_numerator / total_unknowns if total_unknowns > 0 else 0.0
    #         ccr_values.append(ccr)
    #         fpr_values.append(fpr)
        
    #     # Ensure (0,0) and (1,1) points for AUC if not present.
    #     # fpr_values = [0.0] + fpr_values + [1.0]
    #     # ccr_values = [0.0] + ccr_values + [1.0]
    #     # The loop structure already covers implicit thresholds from 0 to N samples considered.
    #     # Sort by FPR for AUC calculation
    #     fpr_np = np.array(fpr_values)
    #     ccr_np = np.array(ccr_values)

    #     # Remove duplicates in fpr for auc calculation
    #     # Sort by fpr and then by ccr to break ties consistently
    #     sorted_indices = np.lexsort((ccr_np, fpr_np))
    #     fpr_np_sorted = fpr_np[sorted_indices]
    #     ccr_np_sorted = ccr_np[sorted_indices]
        
    #     # Add (0,0) and (1,1) if not already covered by the range of FPR/CCR
    #     # The plot_oscr.py script directly calculates this. It's fine to keep it there.
    #     # return sklearn_auc(fpr_np_unique, ccr_np_unique)
    #     return 0.0 # Placeholder for now, use script for final OSCR AUC.