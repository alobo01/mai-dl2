import json
import os
import pickle
from typing import Dict

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

# Assuming these modules are in paths recognizable by Python (e.g. in src/ and PYTHONPATH includes src root)
from data.dataset import OpenSetDataModule # If eval.py is in src/, then from .data.dataset
from train_module import OpenSetLightningModule # from .train_module
from utils.seed import seed_everything # from .utils.seed


# Custom metric function from blueprint, refined version
def recall_at_fixed_precision(
    y_true_is_known: np.ndarray,  # Boolean array: True if known, False if unknown
    osr_scores: np.ndarray,       # Scores: higher means more likely unknown
    target_precision_for_knowns: float = 0.95
) -> float:
    """
    Calculates the recall of unknown samples when the precision for identifying known samples
    is at least `target_precision_for_knowns`.

    Args:
        y_true_is_known: 1D boolean numpy array. True for known samples, False for unknown samples.
        osr_scores: 1D numpy array of OSR scores. Higher scores indicate a higher likelihood of being unknown.
        target_precision_for_knowns: The minimum desired precision for classifying known samples.

    Returns:
        The recall of unknown samples at the chosen operating point. Returns 0.0 if the
        target precision for knowns cannot be met or if input arrays are problematic.
    """
    if not (0 < target_precision_for_knowns <= 1.0):
        # raise ValueError("target_precision_for_knowns must be in (0, 1]")
        print("Warning (recall_at_fixed_precision): target_precision_for_knowns must be in (0, 1]. Using 0.0.")
        return 0.0

    # For precision of "knowns", "known" is the positive class.
    # Scores for PR curve should be such that higher means more likely "known". So, use -osr_scores.
    scores_for_known_as_positive = -osr_scores

    # Ensure there are both known and unknown samples for meaningful calculation
    unique_labels = np.unique(y_true_is_known)
    if len(unique_labels) < 2:
        # print(f"Warning (recall_at_fixed_precision): Only one class type ({'known' if unique_labels[0] else 'unknown'}) present. Cannot compute metric robustly.")
        # If all are known, and target precision is met (e.g. 1.0), recall of unknowns is 0 (no unknowns).
        if np.all(y_true_is_known): # All are known
             return 0.0 # No unknowns to recall.
        # If all are unknown, precision for knowns is undefined or 0.
        # If target_precision_for_knowns is 0 (allowed by check above if error wasn't raised),
        # it implies any threshold is fine, so all unknowns could be recalled.
        # However, this case usually means something is off. Let's return 0.0 for safety.
        return 0.0

    precisions_known, recalls_known, thresholds_pr_known = precision_recall_curve(
        y_true_is_known, scores_for_known_as_positive
    )
    # thresholds_pr_known are for scores_for_known_as_positive.
    # A sample is predicted "Known" if scores_for_known_as_positive >= threshold.

    valid_indices = np.where(precisions_known >= target_precision_for_knowns)[0]

    if len(valid_indices) == 0:
        # print(f"Warning (recall_at_fixed_precision): Target precision {target_precision_for_knowns:.2f} for knowns not met by any threshold.")
        return 0.0

    # Among valid thresholds, choose the one that maximizes recall_known (TPR_Known)
    # This prioritizes correctly identifying as many knowns as possible while meeting the precision target.
    best_idx_for_pr_array = valid_indices[np.argmax(recalls_known[valid_indices])]
    
    # Determine the OSR score threshold.
    # thresholds_pr_known has N elements, precisions_known and recalls_known have N+1.
    # thresholds_pr_known[i] corresponds to precisions_known[i+1] and recalls_known[i+1].
    # The point (recalls_known[0], precisions_known[0]) has no explicit threshold from the array;
    # it represents classifying all samples as positive (lowest possible threshold).
    if best_idx_for_pr_array == 0:
        # This corresponds to the point where recall_known is 1.0 (or highest possible).
        # The threshold is effectively min(scores_for_known_as_positive) or lower.
        chosen_threshold_for_known_positive = np.min(scores_for_known_as_positive) - 1e-6 # Ensure all are included
    else:
        # The threshold is thresholds_pr_known[best_idx_for_pr_array - 1]
        chosen_threshold_for_known_positive = thresholds_pr_known[best_idx_for_pr_array - 1]

    # Convert this threshold back for original osr_scores:
    # A sample is predicted "Known" if scores_for_known_as_positive >= chosen_threshold_for_known_positive
    # i.e., -osr_scores >= chosen_threshold_for_known_positive
    # i.e., osr_scores <= -chosen_threshold_for_known_positive
    final_osr_threshold_unknown_if_above = -chosen_threshold_for_known_positive

    # Predictions: sample is "Unknown" if osr_scores > final_osr_threshold_unknown_if_above
    predicted_as_unknown = osr_scores > final_osr_threshold_unknown_if_above

    # Calculate recall of unknowns
    y_true_is_unknown = ~y_true_is_known
    
    tp_unknown = np.sum(y_true_is_unknown & predicted_as_unknown)
    total_actual_unknowns = np.sum(y_true_is_unknown)

    if total_actual_unknowns == 0:
        # This case should ideally be caught by the len(unique_labels) < 2 check earlier.
        # If it's reached, it implies all samples were 'known'.
        # print("Warning (recall_at_fixed_precision): No unknown samples in the set to calculate recall for.")
        return 1.0 if tp_unknown == 0 else 0.0 # Or NaN, or 0.0
    
    recall_unknown = tp_unknown / total_actual_unknowns
    return recall_unknown


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    # Determine run_dir: either from cfg.eval.run_id or current hydra dir
    if cfg.eval.run_id:
        run_dir_abs = os.path.abspath(os.path.join(cfg.outputs_root_dir, "runs", cfg.eval.run_id))
        print(f"Evaluating existing run: {cfg.eval.run_id} (path: {run_dir_abs})")
        train_cfg_path = os.path.join(run_dir_abs, "config_resolved.yaml")
        if not os.path.exists(train_cfg_path):
             train_cfg_path = os.path.join(run_dir_abs, ".hydra", "config.yaml")
        
        if os.path.exists(train_cfg_path):
            cfg_model_data = OmegaConf.load(train_cfg_path)
        else:
            raise FileNotFoundError(f"Training config not found for run_id {cfg.eval.run_id} at {train_cfg_path}")
        
        if cfg.eval.checkpoint_path:
            ckpt_path = os.path.abspath(cfg.eval.checkpoint_path) # Ensure it's an absolute path or relative to cwd
        else:
            ckpt_dir = os.path.join(run_dir_abs, "checkpoints")
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
            if not ckpts:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
            # Try to find 'last.ckpt' or the one with highest epoch or best metric
            if "last.ckpt" in ckpts:
                ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
            else: # Heuristic: sort and pick the "latest" looking one. More robust would be to use ModelCheckpoint's best_model_path if stored.
                ckpts.sort(reverse=True) 
                ckpt_path = os.path.join(ckpt_dir, ckpts[0]) 
        print(f"Using checkpoint: {ckpt_path}")

    else: 
        run_dir_abs = os.path.abspath(hydra_cfg.run.dir)
        print(f"Evaluating current run (or run defined by hydra.run.dir): {run_dir_abs}")
        cfg_model_data = cfg 
        if not cfg.eval.checkpoint_path:
            raise ValueError("cfg.eval.checkpoint_path must be set if not providing cfg.eval.run_id and relying on current run.")
        ckpt_path = os.path.abspath(cfg.eval.checkpoint_path)

    if not os.path.exists(run_dir_abs):
        raise FileNotFoundError(f"Run directory {run_dir_abs} does not exist.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")

    # Load model
    model = OpenSetLightningModule.load_from_checkpoint(ckpt_path, cfg=cfg_model_data)
    model.eval()
    device = torch.device("cuda" if cfg_model_data.train.trainer.gpus > 0 else "cpu")
    model.to(device)

    # Data module
    datamodule = OpenSetDataModule(cfg_model_data.dataset)
    datamodule.prepare_data() # Downloads data if not present

    # MAV fitting for OpenMax (if applicable)
    if model.osr_head_type == "openmax":
        print("Setting up datamodule for MAV fitting (needs training data)...")
        datamodule.setup(stage='fit') # Ensures train_dataloader is available
        print("Fitting OpenMax MAVs for evaluation...")
        model.fit_openmax_if_needed(datamodule)

    # Setup datamodule for test sets
    print("Setting up datamodule for test sets...")
    datamodule.setup(stage='test')

    # Load calibration temperature
    calibration_info_path = os.path.join(run_dir_abs, "calibration_info.json")
    calibration_temperature = 1.0
    if os.path.exists(calibration_info_path):
        with open(calibration_info_path, 'r') as f:
            calib_info = json.load(f)
        calibration_temperature = calib_info.get('temperature', 1.0)
        model.calibration_temperature = calibration_temperature 
        print(f"Loaded calibration temperature: {calibration_temperature:.4f}")
    else:
        print("No calibration_info.json found in run directory. Using default temperature (1.0 or model's own).")

    # --- Evaluation Loop ---
    results = {
        'probs_k_classes': [], 
        'osr_scores': [],      
        'y_true_known_idx': [],
        'is_known_target': [], 
        'embeddings': []       
    }

    # Process known test set
    if datamodule.test_dataset_known and len(datamodule.test_dataset_known) > 0:
        test_known_loader = DataLoader(datamodule.test_dataset_known, batch_size=cfg.eval.batch_size, num_workers=cfg_model_data.dataset.num_workers)
        print(f"Evaluating on known test set ({len(datamodule.test_dataset_known)} samples)...")
        for batch in tqdm(test_known_loader, desc="Known Test Data"):
            x, y_true_k_idx, is_known_mask = batch 
            x = x.to(device)
            with torch.no_grad():
                closed_set_logits_k_raw, osr_head_raw_output = model(x)
                eval_temp = model.calibration_temperature
                probs_k = F.softmax(closed_set_logits_k_raw / eval_temp, dim=1)
                osr_score = model._get_osr_score_from_outputs(closed_set_logits_k_raw, osr_head_raw_output)
                
                results['probs_k_classes'].append(probs_k.cpu())
                results['osr_scores'].append(osr_score.cpu())
                results['y_true_known_idx'].append(y_true_k_idx.cpu())
                results['is_known_target'].append(is_known_mask.cpu())

                if cfg.eval.save_features_for_tsne:
                    features_bb = model.backbone(x)
                    embeddings_batch = model.neck(features_bb)
                    results['embeddings'].append(embeddings_batch.cpu())
    else:
        print("Known test dataset is empty or not available.")

    # Process unknown test set
    if datamodule.test_dataset_unknown and len(datamodule.test_dataset_unknown) > 0 :
        test_unknown_loader = DataLoader(datamodule.test_dataset_unknown, batch_size=cfg.eval.batch_size, num_workers=cfg_model_data.dataset.num_workers)
        print(f"Evaluating on unknown test set ({len(datamodule.test_dataset_unknown)} samples)...")
        for batch in tqdm(test_unknown_loader, desc="Unknown Test Data"):
            x, _, is_known_mask = batch 
            x = x.to(device)
            with torch.no_grad():
                closed_set_logits_k_raw, osr_head_raw_output = model(x)
                eval_temp = model.calibration_temperature
                probs_k = F.softmax(closed_set_logits_k_raw / eval_temp, dim=1)
                osr_score = model._get_osr_score_from_outputs(closed_set_logits_k_raw, osr_head_raw_output)

                results['probs_k_classes'].append(probs_k.cpu())
                results['osr_scores'].append(osr_score.cpu())
                results['y_true_known_idx'].append(torch.full_like(osr_score, -1, dtype=torch.long).cpu()) 
                results['is_known_target'].append(is_known_mask.cpu())

                if cfg.eval.save_features_for_tsne:
                    features_bb = model.backbone(x)
                    embeddings_batch = model.neck(features_bb)
                    results['embeddings'].append(embeddings_batch.cpu())
    else:
        print("Unknown test dataset is empty or not available.")

    # Concatenate all results
    for key in results:
        if results[key]: 
            results[key] = torch.cat(results[key]).numpy()
        else: 
             results[key] = np.array([])

    # --- Compute Metrics ---
    metrics: Dict[str, float] = {}
    y_true_combined = results['y_true_known_idx']
    is_known_combined = results['is_known_target'].astype(bool) # Ensure boolean
    osr_scores_combined = results['osr_scores']
    probs_k_combined = results['probs_k_classes']

    if len(is_known_combined) == 0:
        print("No samples processed. Skipping metrics calculation.")
    else:
        known_mask_np = is_known_combined
        if np.any(known_mask_np):
            preds_on_knowns = np.argmax(probs_k_combined[known_mask_np], axis=1)
            true_labels_on_knowns = y_true_combined[known_mask_np]
            metrics['acc_known'] = np.mean(preds_on_knowns == true_labels_on_knowns)

        if np.any(known_mask_np) and np.any(~known_mask_np): # Need both knowns and unknowns
            try:
                metrics['auroc'] = roc_auc_score(~is_known_combined, osr_scores_combined)
            except ValueError:
                metrics['auroc'] = 0.0 # Or np.nan
                print("AUROC calculation failed (likely only one class in y_true for OSR).")
            try:
                metrics['aupr_in'] = average_precision_score(~is_known_combined, osr_scores_combined)
            except ValueError:
                metrics['aupr_in'] = 0.0
                print("AUPR-In calculation failed.")
            try:
                metrics['aupr_out'] = average_precision_score(is_known_combined, -osr_scores_combined)
            except ValueError:
                metrics['aupr_out'] = 0.0
                print("AUPR-Out calculation failed.")
            
            metrics['u_recall_at_95_seen_prec'] = recall_at_fixed_precision(
                y_true_is_known=is_known_combined,
                osr_scores=osr_scores_combined,
                target_precision_for_knowns=0.95
            )
        else:
            print("Not enough sample diversity (known/unknown) to compute all open-set metrics.")

    print("\nFinal Metrics:")
    for k, v_metric in metrics.items():
        print(f"  {k}: {v_metric:.4f}")

    # --- Save Outputs ---
    # Output directory for this evaluation's results, stored within the original training run_dir_abs
    eval_output_dir = os.path.join(run_dir_abs, "eval_outputs") 
    # If eval is run for a "current" hydra run (not a specific past run_id), this eval will have its own hydra dir.
    # The `run_dir_abs` should consistently point to the *training* run's directory for storing associated eval results.
    # If `cfg.eval.run_id` was not set, `run_dir_abs` is `hydra_cfg.run.dir` which is the *eval script's own output dir*.
    # This might be confusing. The convention is: outputs of eval belong to the evaluated training run.
    # The `run_dir_abs` has been set to point to the training run's dir correctly.
    
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Determine a unique name for the metric/score files, e.g., based on checkpoint name or a timestamp
    eval_run_descriptor = cfg.eval.run_id if cfg.eval.run_id else f"ckpt_{os.path.basename(ckpt_path).replace('.ckpt','')}"

    metrics_filename = os.path.join(eval_output_dir, f"metrics_{eval_run_descriptor}.json")
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_filename}")

    if cfg.eval.save_scores:
        scores_data_for_pkl: Dict[str, np.ndarray] = {
            "probs": probs_k_combined,
            "y_true": y_true_combined, 
            "is_known": is_known_combined,
            "osr_scores": osr_scores_combined
        }
        if cfg.eval.save_features_for_tsne and len(results['embeddings']) > 0:
            scores_data_for_pkl["feats"] = results['embeddings']
        
        scores_pkl_filename = os.path.join(eval_output_dir, f"scores_{eval_run_descriptor}.pkl")
        with open(scores_pkl_filename, 'wb') as f:
            pickle.dump(scores_data_for_pkl, f)
        print(f"Scores for plotting saved to {scores_pkl_filename}")

if __name__ == "__main__":
    main()