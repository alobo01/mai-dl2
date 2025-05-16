from sklearn.metrics import confusion_matrix as sk_confusion_matrix # aliased
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pathlib

def plot_confusion_matrix(scores_pkl_path, num_known_classes, save_to, class_names=None):
    with open(scores_pkl_path, "rb") as f:
        d = pickle.load(f)

    probs = np.array(d["probs"])
    y_true = np.array(d["y_true"])
    is_known = np.array(d["is_known"]).astype(bool)

    # Filter for known samples only for closed-set confusion matrix
    known_mask = is_known
    if not np.any(known_mask):
        print("Warning: No known samples found. Skipping confusion matrix.")
        return

    y_true_known = y_true[known_mask]
    probs_known = probs[known_mask]
    y_pred_known = np.argmax(probs_known, axis=1)

    if class_names:
        assert len(class_names) == num_known_classes, "Number of class names must match num_known_classes"
        labels_for_plot = class_names
    else:
        labels_for_plot = [str(i) for i in range(num_known_classes)]
    
    # Ensure labels argument in sk_confusion_matrix covers all actual classes present
    unique_labels = np.unique(np.concatenate((y_true_known, y_pred_known)))
    if not all(ul < num_known_classes for ul in unique_labels):
        print(f"Warning: Predicted or true labels {unique_labels} exceed num_known_classes {num_known_classes}. Clamping or error.")
        # This indicates an issue upstream if labels are out of 0..K-1 range for knowns.
    
    cm = sk_confusion_matrix(y_true_known, y_pred_known, labels=range(num_known_classes))
    
    plt.figure(figsize=(max(8, num_known_classes*0.8), max(6, num_known_classes*0.7)))
    sns.heatmap(cm, xticklabels=labels_for_plot, yticklabels=labels_for_plot,
                square=True, annot=True, fmt="d", cmap="Blues") # Annot changed to True
    plt.xlabel("Predicted Known Class")
    plt.ylabel("True Known Class")
    plt.title("Confusion Matrix for Known Classes")
    plt.tight_layout()
    
    save_dir = pathlib.Path(save_to).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to)
    print(f"Confusion matrix saved to {save_to}")
    plt.close()

if __name__ == "__main__":
    import argparse
    # This script needs num_known_classes. It can be inferred from config or passed.
    # For standalone, it's better to pass it or get it from the scores_pkl if available.
    # Assuming scores_pkl doesn't directly store num_known_classes.
    # For simplicity, add as arg. Or load dataset config.
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="Path to scores.pkl file")
    ap.add_argument("--num_known_classes", required=True, type=int, help="Number of known classes (K)")
    ap.add_argument("--out", default="confusion_matrix.png", help="Path to save the plot")
    # Optional: --class_names "name1,name2,..."
    args = ap.parse_args()
    
    plot_confusion_matrix(args.scores, args.num_known_classes, args.out)