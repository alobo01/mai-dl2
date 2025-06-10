import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
import pathlib
import seaborn as sns
from typing import Dict, Any
import os

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

def plot_per_class_confusion_matrices(probs: np.ndarray, y_true: np.ndarray, is_known: np.ndarray, 
                                    output_dir: str, dataset_name: str = ""):
    """Plot confusion matrices for each individual known class."""
    # Only use known samples for per-class analysis
    known_mask = is_known
    if not np.any(known_mask):
        print("No known samples found for per-class confusion matrices.")
        return
    
    probs_known = probs[known_mask]
    y_true_known = y_true[known_mask]
    
    # Get predictions for known samples
    y_pred_known = np.argmax(probs_known, axis=1)
    
    # Get unique classes
    unique_classes = np.unique(y_true_known)
    num_classes = len(unique_classes)
    
    if num_classes < 2:
        print("Not enough classes for meaningful per-class confusion matrices.")
        return
    
    # Create a directory for per-class matrices
    per_class_dir = os.path.join(output_dir, "per_class_matrices")
    os.makedirs(per_class_dir, exist_ok=True)
    
    # Overall confusion matrix for all known classes
    cm_overall = confusion_matrix(y_true_known, y_pred_known, labels=unique_classes)
    
    plt.figure(figsize=(max(8, num_classes), max(6, num_classes)))
    sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in unique_classes],
                yticklabels=[f'Class {i}' for i in unique_classes])
    plt.title(f'Overall Confusion Matrix - Known Classes Only\n{dataset_name}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(os.path.join(per_class_dir, 'overall_known_classes_confusion_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-class accuracy and individual matrices
    class_accuracies = []
    
    # Calculate per-class metrics
    for class_idx in unique_classes:
        # Binary classification: current class vs all others
        y_true_binary = (y_true_known == class_idx).astype(int)
        y_pred_binary = (y_pred_known == class_idx).astype(int)
        
        # Calculate confusion matrix for this class
        cm_binary = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
        
        # Calculate accuracy for this class
        if np.sum(y_true_binary) > 0:  # Check if this class has any samples
            class_accuracy = cm_binary[1, 1] / np.sum(y_true_binary) if np.sum(y_true_binary) > 0 else 0
            class_accuracies.append((class_idx, class_accuracy))
            
            # Plot individual class confusion matrix
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[f'Not Class {class_idx}', f'Class {class_idx}'],
                        yticklabels=[f'Not Class {class_idx}', f'Class {class_idx}'])
            plt.title(f'Class {class_idx} vs Others\nAccuracy: {class_accuracy:.3f}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(per_class_dir, f'class_{class_idx}_confusion_matrix.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create a summary plot of per-class accuracies
    if class_accuracies:
        classes, accuracies = zip(*class_accuracies)
        
        plt.figure(figsize=(max(8, len(classes) * 0.8), 6))
        bars = plt.bar(range(len(classes)), accuracies, color='skyblue', alpha=0.7)
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title(f'Per-Class Accuracy\n{dataset_name}')
        plt.xticks(range(len(classes)), [f'Class {c}' for c in classes], rotation=45)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(per_class_dir, 'per_class_accuracies.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated per-class confusion matrices and accuracy summary in {per_class_dir}")

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
    probs = np.array(scores_data['probs'])
    y_true = np.array(scores_data['y_true'])
    
    # Try to determine dataset name from the path
    dataset_name = ""
    path_parts = str(scores_pkl_path).split(os.sep)
    for part in path_parts:
        if 'cifar10' in part.lower():
            dataset_name = "CIFAR-10"
            break
        elif 'gtsrb' in part.lower():
            dataset_name = "GTSRB"
            break
    
    # Generate existing plots
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
    
    # Generate per-class confusion matrices
    try:
        plot_per_class_confusion_matrices(probs, y_true, is_known, str(output_dir), dataset_name)
        print("Generated per-class confusion matrices")
    except Exception as e:
        print(f"Error generating per-class confusion matrices: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate evaluation plots for open-set recognition')
    parser.add_argument('--scores', required=True, help='Path to scores pickle file')
    parser.add_argument('--metrics', required=True, help='Path to metrics JSON file')
    parser.add_argument('--output-dir', required=True, help='Directory to save plots')
    args = parser.parse_args()
    
    generate_all_plots(args.scores, args.metrics, args.output_dir) 