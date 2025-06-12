\
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

class ClassicEvaluator:
    def __init__(self, model, datamodule, cfg, output_dir):
        self.model = model
        self.datamodule = datamodule
        self.cfg = cfg
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.train.trainer.gpus > 0 else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Ensure output directories exist
        self.latex_dir = os.path.join(self.output_dir, "latex_reports")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.latex_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.num_known_classes = self.cfg.dataset.num_classes # Number of classes model trained on
        self.known_class_original_ids = list(self.cfg.dataset.known_classes_original_ids)
        self.unknown_class_original_ids = list(self.cfg.dataset.unknown_classes_original_ids)
        
        # Create a mapping from trained model's output index (0 to num_known_classes-1) to original class ID
        self.model_output_idx_to_original_id = {i: original_id for i, original_id in enumerate(self.known_class_original_ids)}
        # Create a reverse map for convenience if needed, though less direct for confusion matrix with original IDs
        self.original_id_to_model_output_idx = {original_id: i for i, original_id in enumerate(self.known_class_original_ids)}


    @torch.no_grad()
    def get_predictions(self, dataloader):
        all_preds = []
        all_probs = []
        all_labels = []
        all_entropies = []

        for batch in dataloader:
            images, labels = batch
            images = images.to(self.device)
            
            logits = self.model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            batch_entropy = entropy(probs.cpu().numpy(), axis=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_entropies.extend(batch_entropy)
            
        return np.array(all_labels), np.array(all_preds), np.array(all_probs), np.array(all_entropies)

    def evaluate_and_save_results(self):
        print("Starting custom evaluation...")

        # 1. Evaluate prediction performance on SEEN classes within the validation set
        print("Evaluating on KNOWN validation data...")
        val_known_loader = self.datamodule.eval_val_known_dataloader()
        if len(val_known_loader.dataset) > 0:
            labels_known_val, preds_known_val_mapped, probs_known_val, entropies_known_val = self.get_predictions(val_known_loader)
            
            # Map model's predicted indices (0 to N-1) back to original class IDs for evaluation
            preds_known_val_original_ids = np.array([self.model_output_idx_to_original_id[p] for p in preds_known_val_mapped])

            # a) Per-class accuracy, precision, recall for knowns
            # Ensure labels_known_val are original IDs, which they should be from eval_val_known_dataloader
            self.generate_known_class_performance_report(labels_known_val, preds_known_val_original_ids, probs_known_val, entropies_known_val, "validation_known")
        else:
            print("Validation set for known classes is empty. Skipping known class performance report.")


        # 2. Evaluate prediction performance on UNSEEN classes from the validation set
        print("Evaluating on UNKNOWN validation data...")
        val_unknown_loader = self.datamodule.eval_val_unknown_dataloader()
        if len(val_unknown_loader.dataset) > 0:
            labels_unknown_val, preds_unknown_val_mapped, probs_unknown_val, entropies_unknown_val = self.get_predictions(val_unknown_loader)
            # labels_unknown_val are original IDs of unknown classes
            # preds_unknown_val_mapped are model's output indices (0 to N-1, referring to known classes)
            preds_unknown_val_original_ids = np.array([self.model_output_idx_to_original_id[p] for p in preds_unknown_val_mapped])

            self.generate_unknown_class_analysis_report(labels_unknown_val, preds_unknown_val_original_ids, probs_unknown_val, entropies_unknown_val, "validation_unknown")
        else:
            print("Validation set for unknown classes is empty. Skipping unknown class analysis.")

        # 3. Combined Confusion Matrix (Validation Set)
        print("Generating combined confusion matrix for validation set...")
        if len(val_known_loader.dataset) > 0 and len(val_unknown_loader.dataset) > 0:
            all_val_labels = np.concatenate([labels_known_val, labels_unknown_val])
            all_val_preds_original_ids = np.concatenate([preds_known_val_original_ids, preds_unknown_val_original_ids])
            self.generate_combined_confusion_matrix(all_val_labels, all_val_preds_original_ids, "validation_combined")
        elif len(val_known_loader.dataset) > 0: # Only knowns available
             self.generate_combined_confusion_matrix(labels_known_val, preds_known_val_original_ids, "validation_known_only")
        else:
            print("Not enough data to generate a combined validation confusion matrix.")
            
        print(f"Custom evaluation reports saved in {self.latex_dir} and plots in {self.plots_dir}")

    def generate_known_class_performance_report(self, labels, preds_original_ids, probs, entropies, report_name_suffix):
        # Ensure labels are original IDs
        unique_labels = sorted(np.unique(labels)) # These are the original IDs of the known classes present in this dataset
        
        # Calculate overall metrics
        overall_accuracy = accuracy_score(labels, preds_original_ids)
        
        # Per-class metrics. `labels` in precision_recall_fscore_support should be the unique original class IDs present.
        # We need to ensure that `preds_original_ids` are also treated as referring to these same original IDs.
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, 
            preds_original_ids, 
            labels=unique_labels, # Specify all unique original labels present in `labels`
            zero_division=0
        )

        # Per-class accuracy (diagonal of confusion matrix / support per class)
        cm = confusion_matrix(labels, preds_original_ids, labels=unique_labels)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Confidence and Entropy Analysis
        avg_max_softmax_per_class = []
        avg_entropy_per_class = []
        
        for cls_original_id in unique_labels:
            cls_mask = (labels == cls_original_id)
            if np.sum(cls_mask) > 0:
                # For confidence, we look at the probability assigned to the *predicted* class (which is the max)
                # when the true class is cls_original_id.
                # However, it's more standard to report the probability of the *true* class, or the max probability.
                # Let's report max probability for correct predictions and overall for the class.
                
                # Max probability for samples of this true class
                avg_max_softmax_per_class.append(np.mean(np.max(probs[cls_mask], axis=1)))
                avg_entropy_per_class.append(np.mean(entropies[cls_mask]))
            else:
                avg_max_softmax_per_class.append(np.nan)
                avg_entropy_per_class.append(np.nan)

        # Create DataFrame for LaTeX
        report_df = pd.DataFrame({
            'Original Class ID': unique_labels,
            'Accuracy': per_class_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Support': support,
            'Avg Max Softmax': avg_max_softmax_per_class,
            'Avg Entropy': avg_entropy_per_class
        })
        report_df.set_index('Original Class ID', inplace=True)

        # Summary (Mean +/- Std)
        summary_metrics = {
            'Mean Accuracy': np.mean(per_class_accuracy), 'Std Accuracy': np.std(per_class_accuracy),
            'Mean Precision': np.mean(precision), 'Std Precision': np.std(precision),
            'Mean Recall': np.mean(recall), 'Std Recall': np.std(recall),
            'Mean F1-score': np.mean(f1), 'Std F1-score': np.std(f1),
            'Mean Avg Max Softmax': np.nanmean(avg_max_softmax_per_class), 'Std Avg Max Softmax': np.nanstd(avg_max_softmax_per_class),
            'Mean Avg Entropy': np.nanmean(avg_entropy_per_class), 'Std Avg Entropy': np.nanstd(avg_entropy_per_class),
            'Overall Accuracy': overall_accuracy
        }
        summary_df = pd.DataFrame([summary_metrics])

        latex_path = os.path.join(self.latex_dir, f"known_class_performance_{report_name_suffix}.tex")
        with open(latex_path, 'w') as f:
            f.write("\\documentclass{article}\n")
            f.write("\\usepackage{booktabs}\n")
            f.write("\\usepackage{geometry}\n")
            f.write("\\geometry{a4paper, margin=1in}\n")
            f.write("\\title{Known Class Performance (" + report_name_suffix.replace("_", " ").title() + ")}\n")
            f.write("\\date{\\today}\n")
            f.write("\\author{Automated Report}\n")
            f.write("\\begin{document}\n")
            f.write("\\maketitle\n")
            f.write("\\section*{Per-Class Performance}\n")
            f.write(report_df.to_latex(float_format="%.3f", escape=False, na_rep='-'))
            f.write("\\section*{Summary Metrics (Mean $\\pm$ Std)}\n")
            f.write(summary_df.to_latex(float_format="%.3f", escape=False, index=False, na_rep='-'))
            f.write("\\end{document}\n")
        print(f"Known class performance report saved to {latex_path}")
        
        # c) Describe if there are classes more difficult to recognize
        # This can be inferred from lower per-class accuracy, precision, recall.
        # Adding a textual summary to the LaTeX file:
        difficult_classes_desc = "\\section*{Observations on Difficult Classes}\n"
        min_acc_class = report_df['Accuracy'].idxmin()
        min_acc_val = report_df['Accuracy'].min()
        difficult_classes_desc += f"The class with the lowest accuracy is Original ID {min_acc_class} (Accuracy: {min_acc_val:.3f}).\n"
        
        # Identify classes with low recall (model misses them often)
        low_recall_threshold = 0.7 # Example threshold
        low_recall_classes = report_df[report_df['Recall'] < low_recall_threshold].index.tolist()
        if low_recall_classes:
            difficult_classes_desc += f"Classes with recall below {low_recall_threshold}: {', '.join(map(str, low_recall_classes))}.\n"
        else:
            difficult_classes_desc += "All classes have recall above {low_recall_threshold}.\n"

        # Identify classes with low precision (model often misclassifies other classes as them)
        low_precision_threshold = 0.7 # Example threshold
        low_precision_classes = report_df[report_df['Precision'] < low_precision_threshold].index.tolist()
        if low_precision_classes:
            difficult_classes_desc += f"Classes with precision below {low_precision_threshold}: {', '.join(map(str, low_precision_classes))}.\n"
        else:
            difficult_classes_desc += "All classes have precision above {low_precision_threshold}.\n"
            
        with open(latex_path, 'a') as f: # Append to existing file
            f.write(difficult_classes_desc)
            f.write("\\end{document}\n") # Re-add end document if it was overwritten by previous write's structure

    def generate_unknown_class_analysis_report(self, labels_unknown_original, preds_unknown_original_ids, probs_unknown, entropies_unknown, report_name_suffix):
        # labels_unknown_original: original IDs of the true unknown classes
        # preds_unknown_original_ids: original IDs of the *known* classes that the model predicted for these unknowns
        # probs_unknown: softmax probability vectors (over known classes) for these unknown inputs
        # entropies_unknown: entropy of the softmax vectors for these unknown inputs

        # Analyze predicted class distribution for unknowns
        predicted_known_class_ids_for_unknowns, counts = np.unique(preds_unknown_original_ids, return_counts=True)
        pred_dist_df = pd.DataFrame({
            'Predicted Known Class (Original ID)': predicted_known_class_ids_for_unknowns,
            'Count': counts,
            'Percentage': (counts / len(preds_unknown_original_ids)) * 100
        })
        pred_dist_df.set_index('Predicted Known Class (Original ID)', inplace=True)

        # Analyze confidence (max softmax probability) and entropy for unknowns
        max_softmax_for_unknowns = np.max(probs_unknown, axis=1)
        avg_max_softmax_overall = np.mean(max_softmax_for_unknowns)
        std_max_softmax_overall = np.std(max_softmax_for_unknowns)
        avg_entropy_overall = np.mean(entropies_unknown)
        std_entropy_overall = np.std(entropies_unknown)

        confidence_summary_df = pd.DataFrame([{
            'Avg Max Softmax': avg_max_softmax_overall,
            'Std Max Softmax': std_max_softmax_overall,
            'Avg Entropy': avg_entropy_overall,
            'Std Entropy': std_entropy_overall,
            'Total Unknown Samples': len(labels_unknown_original)
        }])

        latex_path = os.path.join(self.latex_dir, f"unknown_class_analysis_{report_name_suffix}.tex")
        with open(latex_path, 'w') as f:
            f.write("\\documentclass{article}\n")
            f.write("\\usepackage{booktabs}\n")
            f.write("\\usepackage{geometry}\n")
            f.write("\\geometry{a4paper, margin=1in}\n")
            f.write("\\title{Unknown Class Prediction Analysis (" + report_name_suffix.replace("_", " ").title() + ")}\n")
            f.write("\\date{\\today}\n")
            f.write("\\author{Automated Report}\n")
            f.write("\\begin{document}\n")
            f.write("\\maketitle\n")
            f.write("\\section*{Distribution of Predictions for Unknown Inputs}\n")
            f.write("Unknown inputs were predicted by the model (trained on known classes) as follows:\n")
            f.write(pred_dist_df.to_latex(float_format="%.2f%%", escape=False, columns=['Count', 'Percentage']))
            f.write("\\section*{Confidence and Entropy for Unknown Inputs}\n")
            f.write(confidence_summary_df.to_latex(float_format="%.3f", escape=False, index=False))
            f.write("\\end{document}\n")
        print(f"Unknown class analysis report saved to {latex_path}")

    def generate_combined_confusion_matrix(self, all_labels_original, all_preds_original_ids, plot_name_suffix):
        # all_labels_original: Contains original IDs of both known and unknown classes
        # all_preds_original_ids: Contains original IDs of *known* classes (as predicted by the model)
        
        # Define all unique original class IDs that appear in labels or predictions
        # This ensures the CM axes cover all relevant classes.
        # The predictions are only for known classes, so we need to define the full set of labels.
        
        # All true classes present in the dataset (both known and unknown original IDs)
        true_class_ids_present = sorted(list(set(all_labels_original)))
        
        # All classes the model *could* predict (these are the known original IDs)
        # This will define the columns of the confusion matrix.
        predicted_class_ids_possible = sorted(self.known_class_original_ids)

        # Create the confusion matrix
        # Rows: True original labels (can be known or unknown)
        # Columns: Predicted original labels (can only be one of the known classes)
        cm = confusion_matrix(all_labels_original, all_preds_original_ids, labels=true_class_ids_present, normalize=None)
        
        # If normalize='true', cm will be row-normalized.
        # cm_normalized_row = confusion_matrix(all_labels_original, all_preds_original_ids, labels=true_class_ids_present, normalize='true')

        # Create a DataFrame for better labeling with Seaborn
        # Rows (index) are true original labels
        # Columns are predicted original labels (these are the known classes the model outputs)
        # We need to ensure the columns of the CM correspond to `predicted_class_ids_possible`
        # The `confusion_matrix` function when `labels` is given, uses those for both rows and columns if `normalize` is not used for specific axis.
        # However, our predictions `all_preds_original_ids` only contain known class IDs.
        # So, the CM will have rows for all `true_class_ids_present` and columns implicitly for the unique values in `all_preds_original_ids`
        # or, if `labels` kwarg is used, for those specified labels.
        # To make it explicit:
        
        # We need a CM where rows are all true classes (known & unknown) and columns are all *possible* predicted known classes.
        # The issue is `confusion_matrix` expects `labels` to define both axes if specified.
        # Let's construct it carefully.
        
        # `true_class_ids_present` includes all original IDs (0-9 for CIFAR10 if all are in `all_labels_original`)
        # `predicted_class_ids_possible` includes only known original IDs (e.g., 0-7 for CIFAR10 8 known)
        
        # We want rows to be `true_class_ids_present` and columns to be `predicted_class_ids_possible`.
        # This is non-square if there are unknown classes.
        
        # Let's build the CM manually for this non-square case or ensure scikit-learn handles it.
        # Scikit-learn's confusion_matrix(y_true, y_pred, labels=L) will produce an L x L matrix.
        # This is not what we want if y_pred cannot take all values in L.
        
        # We will use `true_class_ids_present` for rows.
        # For columns, we use `predicted_class_ids_possible`.
        
        # Create a temporary mapping for unknown classes to a placeholder value if needed for CM,
        # or handle it by ensuring the plot distinguishes them.
        
        # Let's use the full set of original classes for rows and known original for columns.
        # This requires a bit more manual construction if we want specific column labels.
        
        # For simplicity with seaborn, we'll make a square CM based on all original IDs present in true labels.
        # The columns corresponding to unknown classes will be all zeros if model never predicts them (which it shouldn't).
        
        cm_display_labels = sorted(list(set(all_labels_original) | set(all_preds_original_ids)))

        cm_for_plot = confusion_matrix(all_labels_original, all_preds_original_ids, labels=cm_display_labels)
        cm_df = pd.DataFrame(cm_for_plot, index=cm_display_labels, columns=cm_display_labels)

        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(cm_df, annot=True, fmt='d', cmap="viridis")
        plt.title(f'Confusion Matrix - {plot_name_suffix.replace("_", " ").title()}')
        plt.ylabel('True Original Class ID')
        plt.xlabel('Predicted Original Class ID (Model Output Space)')
        
        # Highlight known vs unknown rows/columns if possible
        # Known classes: self.known_class_original_ids
        # Unknown classes: self.unknown_class_original_ids
        
        # Example: Color cell backgrounds or tick labels
        # This is complex with heatmap directly, might need post-processing or drawing rectangles.
        # For now, we rely on the user knowing which IDs are known/unknown from config.
        # A simpler visual cue:
        known_ticks_indices = [i for i, tick_label in enumerate(cm_display_labels) if tick_label in self.known_class_original_ids]
        unknown_ticks_indices = [i for i, tick_label in enumerate(cm_display_labels) if tick_label in self.unknown_class_original_ids]

        # Modify tick labels to indicate known/unknown status
        # This is just an example, could be more sophisticated.
        # y_tick_labels = [f\"{l} (K)\" if l in self.known_class_original_ids else f\"{l} (U)\" for l in cm_display_labels]
        # x_tick_labels = [f\"{l} (K)\" if l in self.known_class_original_ids else f\"{l} (U)\" for l in cm_display_labels]
        # ax.set_yticklabels(y_tick_labels)
        # ax.set_xticklabels(x_tick_labels)
        # Simpler: Add a legend or text description in the plot or caption.

        # Add colored rectangles to differentiate known/unknown sections if they are contiguous
        # This assumes known_class_original_ids are, e.g., [0,1,2] and unknown [3,4] and cm_display_labels is sorted.
        
        # Find boundaries for highlighting in the plot
        # This is tricky if class IDs are not contiguous or mixed up after sorting cm_display_labels.
        # A robust way is to iterate through tick labels.
        
        # Shade background of unknown class rows
        for i, label_val in enumerate(cm_display_labels): # Iterate over Y-axis ticks (True Labels)
            if label_val in self.unknown_class_original_ids:
                ax.add_patch(plt.Rectangle((0, i), len(cm_display_labels), 1, fill=True, color="lightcoral", alpha=0.3, lw=0))
        
        # Shade background of columns that are NOT part of the model's output space (i.e., if an unknown ID appears as a column)
        # Model only predicts known classes. So all columns should correspond to known_class_original_ids.
        # If cm_display_labels contains unknown IDs, those columns in cm_df should be zero.
        # We can highlight columns that *are* known predictions.
        for j, label_val in enumerate(cm_display_labels): # Iterate over X-axis ticks (Predicted Labels)
             if label_val not in self.known_class_original_ids: # Should not happen if preds are mapped correctly
                 ax.add_patch(plt.Rectangle((j, 0), 1, len(cm_display_labels), fill=True, color="lightgrey", alpha=0.5, lw=0))


        plot_path = os.path.join(self.plots_dir, f"confusion_matrix_{plot_name_suffix}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Combined confusion matrix plot saved to {plot_path}")

