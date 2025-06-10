import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pathlib
import seaborn as sns
from typing import Dict, Any

def plot_roc_curve(y_true_is_unknown: np.ndarray, osr_scores: np.ndarray, save_to: str) -> float:
    """Plot ROC curve for unknown detection."""
    if len(np.unique(y_true_is_unknown)) < 2:
        print("Warning: ROC curve cannot be computed. Only one class present.")
        auroc = 0.0
        fpr, tpr = np.array([0,1]), np.array([0,1])
    else:
        fpr, tpr, _ = roc_curve(y_true_is_unknown, osr_scores)
        auroc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Unknown Detection')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_to)
    plt.close()
    return auroc

def plot_pr_curves(y_true_is_unknown: np.ndarray, osr_scores: np.ndarray, save_to: str) -> tuple[float, float]:
    """Plot Precision-Recall curves for both known and unknown classes."""
    # For unknowns (AUPR-In)
    precision_in, recall_in, _ = precision_recall_curve(y_true_is_unknown, osr_scores)
    aupr_in = auc(recall_in, precision_in)
    
    # For knowns (AUPR-Out)
    precision_out, recall_out, _ = precision_recall_curve(~y_true_is_unknown, -osr_scores)
    aupr_out = auc(recall_out, precision_out)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_in, precision_in, label=f'AUPR-In (Unknown) = {aupr_in:.3f}')
    plt.plot(recall_out, precision_out, label=f'AUPR-Out (Known) = {aupr_out:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(save_to)
    plt.close()
    return aupr_in, aupr_out

def plot_score_distributions(osr_scores: np.ndarray, is_known: np.ndarray, save_to: str):
    """Plot the distribution of OSR scores for known and unknown samples."""
    plt.figure(figsize=(10, 6))
    
    known_scores = osr_scores[is_known]
    unknown_scores = osr_scores[~is_known]
    
    # Use kernel density estimation for smooth distributions
    sns.kdeplot(data=known_scores, label='Known', fill=True, alpha=0.5)
    sns.kdeplot(data=unknown_scores, label='Unknown', fill=True, alpha=0.5)
    
    plt.xlabel('OSR Score')
    plt.ylabel('Density')
    plt.title('Distribution of OSR Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_to)
    plt.close()

def plot_confusion_matrix(y_true_is_known: np.ndarray, osr_scores: np.ndarray, threshold: float, save_to: str):
    """Plot confusion matrix for known/unknown classification at a specific threshold."""
    # Predict unknown if score > threshold
    y_pred = osr_scores > threshold
    
    # Compute confusion matrix
    tn = np.sum((~y_pred) & y_true_is_known)
    fp = np.sum(y_pred & y_true_is_known)
    fn = np.sum((~y_pred) & (~y_true_is_known))
    tp = np.sum(y_pred & (~y_true_is_known))
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Known', 'Unknown'],
                yticklabels=['Known', 'Unknown'])
    plt.title('Confusion Matrix at Operating Point')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_to)
    plt.close()

def generate_all_plots(scores_pkl_path: str, metrics_json_path: str, output_dir: str):
    """Generate all evaluation plots from scores and metrics files."""
    # Create output directory
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    with open(scores_pkl_path, 'rb') as f:
        scores_data = pickle.load(f)
    
    with open(metrics_json_path, 'r') as f:
        metrics = json.load(f)
    
    # Extract necessary arrays
    osr_scores = np.array(scores_data['osr_scores'])
    is_known = np.array(scores_data['is_known']).astype(bool)
    y_true_is_unknown = ~is_known
    
    # Generate plots
    plots_info = {
        'roc.png': lambda: plot_roc_curve(y_true_is_unknown, osr_scores, str(output_dir / 'roc.png')),
        'pr_curves.png': lambda: plot_pr_curves(y_true_is_unknown, osr_scores, str(output_dir / 'pr_curves.png')),
        'score_dist.png': lambda: plot_score_distributions(osr_scores, is_known, str(output_dir / 'score_dist.png')),
        'confusion_matrix.png': lambda: plot_confusion_matrix(is_known, osr_scores, np.median(osr_scores), str(output_dir / 'confusion_matrix.png'))
    }
    
    # Generate each plot
    for plot_name, plot_fn in plots_info.items():
        try:
            plot_fn()
            print(f"Generated {plot_name}")
        except Exception as e:
            print(f"Error generating {plot_name}: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate evaluation plots for open-set recognition')
    parser.add_argument('--scores', required=True, help='Path to scores pickle file')
    parser.add_argument('--metrics', required=True, help='Path to metrics JSON file')
    parser.add_argument('--output-dir', required=True, help='Directory to save plots')
    args = parser.parse_args()
    
    generate_all_plots(args.scores, args.metrics, args.output_dir) 