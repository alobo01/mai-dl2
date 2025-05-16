import pickle, json, matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np # Added for safety with array operations
import pathlib # Added

def plot_oscr(scores_pkl_path, save_to):
    with open(scores_pkl_path, "rb") as f:
        d = pickle.load(f)
    # Ensure these are numpy arrays
    probs = np.array(d["probs"])
    y_true = np.array(d["y_true"])
    is_known = np.array(d["is_known"]).astype(bool) # Ensure boolean

    # Filter for known samples to calculate "correct" based on K-class prediction
    known_probs = probs[is_known]
    known_y_true = y_true[is_known]
    
    # Confidences are max softmax prob over K known classes, for ALL samples
    confidences = probs.max(1)
    
    # Order all samples by decreasing confidence
    order = confidences.argsort()[::-1]
    
    # Align all relevant arrays with this new order
    ordered_is_known = is_known[order]
    ordered_y_true = y_true[order] # y_true for unknowns is -1
    ordered_probs_argmax = probs.argmax(1)[order]

    # Calculate "correctness" only for samples that were originally known
    # A sample is "correct" if it's a known sample AND its K-class prediction matches its true K-class label.
    # For unknown samples, "correct" is False by this definition.
    
    # Mask for originally known samples in the ordered list
    mask_ordered_actually_known = ordered_is_known 
    
    # Check if K-class prediction is correct FOR THOSE THAT ARE KNOWN
    # Initialize `correct_pred_if_known` for all, then filter
    # `ordered_probs_argmax` is pred_label_idx (0 to K-1)
    # `ordered_y_true` is true_label_idx (0 to K-1 for knowns, -1 for unknowns)
    correct_pred_if_known_in_ordered = (ordered_probs_argmax == ordered_y_true)
    
    # This "correct" flag is only meaningful if the sample was actually known.
    # So, `c` in the loop should be `correct_pred_if_known_in_ordered[i] AND ordered_is_known[i]`
    
    tp_oscr = 0 # True Positive for OSCR: correctly classified known sample
    fp_oscr = 0 # False Positive for OSCR: unknown sample accepted (i.e. not rejected)
    
    tpr_values = [0.0] # CCR: Correct Classification Rate (of knowns)
    fpr_values = [0.0] # FPR: False Positive Rate (of unknowns accepted as knowns)

    total_knowns_in_dataset = ordered_is_known.sum()
    total_unknowns_in_dataset = (~ordered_is_known).sum()

    if total_knowns_in_dataset == 0 or total_unknowns_in_dataset == 0:
        print("Warning: OSCR cannot be computed. Not enough known or unknown samples.")
        auc_oscr = 0.0
    else:
        for i in range(len(order)):
            # A sample is considered "accepted" if its confidence is above the implicit threshold set by its rank.
            # If it's a known sample and correctly classified by K-class head:
            if ordered_is_known[i] and correct_pred_if_known_in_ordered[i]:
                tp_oscr += 1
            # If it's an unknown sample (it's "accepted" by this point in sorted list):
            elif not ordered_is_known[i]:
                fp_oscr += 1
            
            tpr_values.append(tp_oscr / total_knowns_in_dataset)
            fpr_values.append(fp_oscr / total_unknowns_in_dataset)

        # Ensure the curve spans the full range if needed, though usually not necessary if starting with (0,0)
        # tpr_values.append(1.0)
        # fpr_values.append(1.0)
        auc_oscr = auc(fpr_values, tpr_values)

    plt.figure()
    plt.plot(fpr_values, tpr_values, label=f"OSCR AUC={auc_oscr:.3f}")
    plt.xlabel("False Positive Rate (Unknowns Accepted as Known)")
    plt.ylabel("Correct Classification Rate (Knowns Correctly Classified)")
    plt.title("Open-Set Classification Recognition (OSCR) Curve")
    plt.legend(); plt.grid(); plt.tight_layout()
    
    save_dir = pathlib.Path(save_to).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to)
    print(f"OSCR curve saved to {save_to}")
    plt.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="Path to scores.pkl file")
    ap.add_argument("--out", default="oscr.png", help="Path to save the plot")
    args = ap.parse_args()
    
    plot_oscr(args.scores, args.out)