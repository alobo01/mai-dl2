import pickle, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sklearn_auc # Renamed to avoid conflict
import numpy as np
import pathlib

def plot_roc(scores_pkl_path, save_to):
    with open(scores_pkl_path, "rb") as f:
        d = pickle.load(f)
    
    # y_true for ROC: 1 if unknown (positive class), 0 if known
    y_true_roc = ~np.array(d["is_known"]).astype(bool) 
    # scores for ROC: higher means more likely unknown (positive class)
    osr_scores = np.array(d["osr_scores"])

    if len(np.unique(y_true_roc)) < 2:
        print("Warning: ROC curve cannot be computed. Only one class present in y_true_roc.")
        auc_roc = 0.0
        fpr, tpr = np.array([0,1]), np.array([0,1]) # Default line
    else:
        fpr, tpr, _ = roc_curve(y_true_roc, osr_scores)
        auc_roc = sklearn_auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={auc_roc:.3f} (Unknowns as Positive)")
    plt.plot([0, 1], [0, 1], 'k--') # Diagonal
    plt.xlabel("False Positive Rate (Knowns misclassified as Unknown)")
    plt.ylabel("True Positive Rate (Unknowns correctly identified)")
    plt.title("Receiver Operating Characteristic (ROC) Curve for Unknown Detection")
    plt.legend(); plt.grid(); plt.tight_layout()
    
    save_dir = pathlib.Path(save_to).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to)
    print(f"ROC curve saved to {save_to}")
    plt.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="Path to scores.pkl file")
    ap.add_argument("--out", default="roc.png", help="Path to save the plot")
    args = ap.parse_args()
    
    plot_roc(args.scores, args.out)